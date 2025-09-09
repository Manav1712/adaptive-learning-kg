"""
Discover Content ↔ LO Links (Candidates + Scoring)

This module prepares candidate LO targets for each content item
to be later scored by an LLM. It reads processed inputs from the
prepare step and writes candidate pairs for downstream scoring.

Inputs (from prepare step):
- data/processed/lo_index.csv
- data/processed/content_items.csv

Outputs:
- data/processed/content_link_candidates.csv
 - data/processed/edges_content.csv (after scoring)

Candidate strategy:
- Start with LOs in the same unit and/or chapter as the content's parent LO
- Optionally add lexical shortlist by overlapping keywords with LO text

This module can both prepare candidates and (optionally) score them using an LLM
or a deterministic dry-run heuristic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import zlib
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

import pandas as pd

try:  # Optional import; only required for real LLM scoring
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - soft dependency
    OpenAI = None  # type: ignore


# ----------------------------
# Configuration
# ----------------------------


@dataclass
class DiscoveryConfig:
    """
    Configuration for content→LO candidate generation.

    Fields:
    - input_lo_index: Path to LO index CSV
    - input_content_items: Path to content items CSV
    - output_candidates: Path to write candidate pairs CSV
    - restrict_same_unit: If True, only consider LOs in same unit
    - restrict_same_chapter: If True, only consider LOs in same chapter
    - lexical_top_k: Limit of lexical matches to add per content (0 to disable)
    - lexical_min_overlap: Minimum keyword overlap to consider as lexical match
    """

    input_lo_index: str = "data/processed/lo_index.csv"
    input_content_items: str = "data/processed/content_items.csv"
    output_candidates: str = "data/processed/content_link_candidates.csv"

    restrict_same_unit: bool = True
    restrict_same_chapter: bool = False

    lexical_top_k: int = 5
    lexical_min_overlap: int = 1

    # Relation mapping by content type
    relation_concept: str = "explained_by"
    relation_example: str = "exemplified_by"
    relation_try_it: str = "practiced_by"

    # Scoring/LLM parameters
    model: str = "gpt-4o-mini"
    modality: str = "text_only"  # "text_only" | "multimodal"
    temperature: float = 0.0
    max_targets_per_call: int = 8
    max_retries: int = 3
    score_mode: str = "score"  # "score" | "yes_no"
    score_threshold: float = 0.6
    output_edges: str = "data/processed/edges_content.csv"
    # Progress logging
    progress_path: str = "data/processed/progress_links.jsonl"


def load_config(config_path: Optional[str]) -> DiscoveryConfig:
    """
    Loads candidate-generation configuration from YAML or returns defaults.

    Args:
        config_path: Path to config.yaml or None

    Returns:
        DiscoveryConfig populated from file or defaults

    Behavior:
        - Uses keys under output_paths/input_paths/pruning/relations if present
        - Falls back to defaults on missing file or missing YAML
    """
    if not config_path or not os.path.exists(config_path) or yaml is None:
        return DiscoveryConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    inputs = data.get("output_paths", {})  # prior config.yaml used output_paths for processed files
    # fallback to input_paths names as well (either is fine)
    inputs_alt = data.get("input_paths", {})
    pruning = data.get("pruning", {})
    relations = data.get("relations", {})
    cfg = DiscoveryConfig(
        input_lo_index=inputs.get("lo_index", inputs_alt.get("lo_index", "data/processed/lo_index.csv")),
        input_content_items=inputs.get("content_items", inputs_alt.get("content_items", "data/processed/content_items.csv")),
        output_candidates=data.get("output_candidates", "data/processed/content_link_candidates.csv"),
        restrict_same_unit=bool(pruning.get("same_unit", True)),
        restrict_same_chapter=bool(pruning.get("same_chapter", False)),
        lexical_top_k=int(pruning.get("lexical_top_k", 5)),
        lexical_min_overlap=int(pruning.get("lexical_min_overlap", 1)),
        relation_concept=relations.get("concept", "explained_by"),
        relation_example=relations.get("example", "exemplified_by"),
        relation_try_it=relations.get("try_it", "practiced_by"),
    )
    # Scoring/LLM params (optional)
    model = data.get("model")
    if isinstance(model, str):
        cfg.model = model
    modality = data.get("modality")
    if modality in {"text_only", "multimodal"}:
        cfg.modality = modality
    scoring = data.get("scoring", {})
    if isinstance(scoring, dict):
        cfg.score_mode = scoring.get("mode", cfg.score_mode)
        cfg.score_threshold = float(scoring.get("threshold", cfg.score_threshold))
    runtime = data.get("runtime", {})
    if isinstance(runtime, dict):
        cfg.max_targets_per_call = int(runtime.get("max_targets_per_call", cfg.max_targets_per_call))
        cfg.max_retries = int(runtime.get("max_retries", cfg.max_retries))
    outputs = data.get("output_paths", {})
    if isinstance(outputs, dict):
        cfg.output_edges = outputs.get("edges_content", cfg.output_edges)
    return cfg


# ----------------------------
# Utilities
# ----------------------------


_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into lowercase keywords (simple heuristic).

    Args:
        text: Input string

    Returns:
        List of lowercase tokens (alphanumeric words)
    """
    if not isinstance(text, str) or not text:
        return []
    return [m.group(0).lower() for m in _WORD_PATTERN.finditer(text)]


def _chapter_to_int(val: object) -> Optional[int]:
    """Attempt to parse chapter identifiers into integers for chronological ordering."""
    try:
        s = str(val).strip()
        digits = "".join(c for c in s if c.isdigit())
        return int(digits) if digits else int(s)
    except Exception:
        return None


def _ctype_order(value: object) -> int:
    mapping = {"concept": 0, "example": 1, "try_it": 2}
    return mapping.get(str(value).lower(), 99)


def select_diverse_chronological_content(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Select a diverse, chronologically ordered subset of content items.

    Strategy:
    - Sort by (book, unit, chapter_num, content_type order, content_id)
    - Build round-robin buckets per (book, unit) to maximize diversity
    - Pick up to `limit` rows in round robin order preserving chronology within buckets
    """
    if limit is None or limit <= 0 or len(df) <= limit:
        return df

    tmp = df.copy()
    tmp["_chapter_num"] = tmp.get("chapter", None).map(_chapter_to_int)
    tmp["_ctype_ord"] = tmp.get("content_type", None).map(_ctype_order)
    tmp.sort_values(["book", "unit", "_chapter_num", "_ctype_ord", "content_id"], inplace=True)

    # Build buckets by (book, unit)
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for idx, r in tmp.iterrows():
        key = (str(r.get("book") or ""), str(r.get("unit") or ""))
        buckets.setdefault(key, []).append(idx)

    selected_indices: List[int] = []
    # Round-robin selection across buckets
    while len(selected_indices) < limit and any(buckets.values()):
        for key in list(buckets.keys()):
            if buckets[key]:
                selected_indices.append(buckets[key].pop(0))
                if len(selected_indices) >= limit:
                    break
            else:
                buckets.pop(key, None)

    out = tmp.loc[selected_indices]
    out.sort_values(["book", "unit", "_chapter_num", "_ctype_ord", "content_id"], inplace=True)
    # Drop helper cols
    return out.drop(columns=[c for c in ["_chapter_num", "_ctype_ord"] if c in out.columns])


def build_prompt_for_content(
    content_row: pd.Series,
    candidate_los: List[Tuple[str, str]],
    lo_lookup: Dict[str, Dict[str, str]],
    config: DiscoveryConfig,
) -> Dict[str, object]:
    """
    Builds a prompt payload for the LLM (text-only or multimodal).

    Args:
        content_row: Content item row (text, image_urls, content_type)
        candidate_los: List of (lo_id, reason) pairs
        lo_lookup: Mapping lo_id -> {learning_objective, unit, chapter}
        config: DiscoveryConfig with modality settings

    Returns:
        Dict representing a prompt payload ready for LLM client

    Behavior:
        - Includes content text and optional image_url blocks
        - Packs multiple candidate LOs with identifiers for scoring
        - Asks model to return JSON with ids and scores or YES/NO
    """
    content_text = str(content_row.get("text") or "")
    image_urls: List[str] = content_row.get("image_urls") or []
    ct = str(content_row.get("content_type") or "")

    lo_items = [
        {
            "lo_id": lo_id,
            "objective": lo_lookup.get(lo_id, {}).get("learning_objective", ""),
            "unit": lo_lookup.get(lo_id, {}).get("unit", ""),
            "chapter": lo_lookup.get(lo_id, {}).get("chapter", ""),
            "reason": reason,
        }
        for lo_id, reason in candidate_los
    ]

    system = (
        "You are a precise educational graph builder. Given a content item and candidate learning objectives, "
        "decide if the content directly supports the LO. Output JSON: {results:[{lo_id, verdict|score, rationale}]}."
    )

    user_blocks: List[Dict[str, object]] = []
    user_blocks.append({"type": "text", "text": f"Content type: {ct}\n\nContent:\n{content_text}"})
    if config.modality == "multimodal" and image_urls:
        for url in image_urls:
            user_blocks.append({"type": "image_url", "image_url": url})
    user_blocks.append({"type": "text", "text": "Candidates:"})
    for item in lo_items:
        user_blocks.append({
            "type": "text",
            "text": f"- [{item['lo_id']}] {item['objective']} (unit: {item['unit']}, chapter: {item['chapter']}, reason: {item['reason']})",
        })

    # Returning a generic payload; the LLM client will adapt as needed
    return {"system": system, "user": user_blocks, "mode": config.score_mode}


def heuristic_score(content_text: str, lo_text: str) -> float:
    """
    Simple deterministic similarity heuristic for dry-run mode.

    Args:
        content_text: Text of the content item
        lo_text: Learning objective text

    Returns:
        Float score between 0 and 1 based on token overlap
    """
    ctoks = set(tokenize(content_text))
    ltoks = set(tokenize(lo_text))
    if not ctoks or not ltoks:
        return 0.0
    overlap = len(ctoks.intersection(ltoks))
    # Normalize by LO token count to measure "how much of LO is covered by content"
    denom = max(1, len(ltoks))
    return overlap / float(denom)


def score_candidates(
    candidates_df: pd.DataFrame,
    content_df: pd.DataFrame,
    lo_df: pd.DataFrame,
    config: DiscoveryConfig,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Scores candidate content→LO pairs using LLM or heuristic fallback.

    Args:
        candidates_df: DataFrame of candidate pairs
        content_df: DataFrame with content details (text, images)
        lo_df: DataFrame with LO details
        config: Scoring configuration
        dry_run: If True, use heuristic instead of LLM

    Returns:
        DataFrame of filtered edges with columns:
        source_lo_id, target_content_id, relation, score, rationale, modality, run_id
    """
    # Build quick lookups
    lo_lookup: Dict[str, Dict[str, str]] = {
        str(r["lo_id"]): {
            "learning_objective": str(r.get("learning_objective") or ""),
            "unit": str(r.get("unit") or ""),
            "chapter": str(r.get("chapter") or ""),
        }
        for _, r in lo_df.iterrows()
    }
    content_lookup: Dict[str, pd.Series] = {str(r["content_id"]): r for _, r in content_df.iterrows()}

    rows: List[Dict[str, object]] = []

    # Batch by content_id to pack multiple LOs per call
    grouped = candidates_df.groupby("target_content_id")
    total_groups = int(getattr(grouped, "ngroups", 0) or 0)
    processed_groups = 0
    started_at = time.time()
    progress_path = getattr(config, "progress_path", "data/processed/progress_links.jsonl")
    # Initialize/clear progress log
    try:
        ensure_parent_directory(progress_path)
        with open(progress_path, "w", encoding="utf-8") as _f:
            pass
    except Exception:
        pass

    for content_id, group in grouped:
        processed_groups += 1
        len_before = len(rows)
        content_row = content_lookup.get(str(content_id))
        if content_row is None:
            continue

        ctext = str(content_row.get("text") or "")
        candidate_list = [(str(r["source_lo_id"]), str(r.get("reason") or "")) for _, r in group.iterrows()]

        if dry_run:
            # Heuristic scoring
            for lo_id, _reason in candidate_list:
                lo_text = lo_lookup.get(lo_id, {}).get("learning_objective", "")
                score = heuristic_score(ctext, lo_text)
                if config.score_mode == "yes_no":
                    keep = score >= config.score_threshold
                    verdict = "YES" if keep else "NO"
                else:
                    keep = score >= config.score_threshold
                    verdict = f"{score:.2f}"
                if keep:
                    relation = relation_for_content_type(str(content_row.get("content_type") or ""), config)
                    rows.append(
                        {
                            "source_lo_id": lo_id,
                            "target_content_id": content_id,
                            "relation": relation,
                            "score": score,
                            "rationale": "heuristic token overlap",
                            "modality": config.modality,
                            "run_id": config.model,
                        }
                    )
            continue

        # Real LLM integration (if OPENAI_API_KEY is set and OpenAI client available)
        use_llm = (OpenAI is not None) and (os.environ.get("OPENAI_API_KEY") not in (None, ""))
        if not use_llm:
            # Fallback to heuristic if LLM not available
            for lo_id, _reason in candidate_list:
                lo_text = lo_lookup.get(lo_id, {}).get("learning_objective", "")
                score = heuristic_score(ctext, lo_text)
                if score >= config.score_threshold:
                    relation = relation_for_content_type(str(content_row.get("content_type") or ""), config)
                    rows.append(
                        {
                            "source_lo_id": lo_id,
                            "target_content_id": content_id,
                            "relation": relation,
                            "score": score,
                            "rationale": "heuristic token overlap (no OPENAI_API_KEY)",
                            "modality": config.modality,
                            "run_id": config.model,
                        }
                    )
            continue

        # Prepare LO lookup for prompt chunks of size max_targets_per_call
        def chunk_list(items: List[Tuple[str, str]], n: int) -> List[List[Tuple[str, str]]]:
            return [items[i : i + n] for i in range(0, len(items), n)]

        client = OpenAI()

        for chunk in chunk_list(candidate_list, max(1, int(config.max_targets_per_call))):
            prompt = build_prompt_for_content(content_row, chunk, lo_lookup, config)
            # Build OpenAI chat messages structure
            system_msg = {"role": "system", "content": prompt["system"]}
            # Convert our user blocks to OpenAI content blocks
            content_blocks: List[Dict[str, object]] = []
            for block in prompt["user"]:
                if block.get("type") == "text":
                    content_blocks.append({"type": "text", "text": str(block.get("text", ""))})
                elif block.get("type") == "image_url":
                    url = block.get("image_url")
                    # Support both string and dict forms
                    if isinstance(url, str):
                        content_blocks.append({"type": "image_url", "image_url": {"url": url}})
                    elif isinstance(url, dict):
                        content_blocks.append({"type": "image_url", "image_url": url})
            user_msg = {"role": "user", "content": content_blocks}

            # Ask model to output strict JSON
            instruction = (
                "Respond ONLY with JSON in this schema: {\n"
                "  \"results\": [ { \"lo_id\": string, \"score\": number, \"rationale\": string } ]\n"
                "} where score in [0,1]."
            )
            content_blocks.append({"type": "text", "text": instruction})

            # Retry with exponential backoff
            last_err: Optional[Exception] = None
            for attempt in range(int(config.max_retries) + 1):
                try:
                    resp = client.chat.completions.create(
                        model=config.model,
                        temperature=float(config.temperature),
                        messages=[system_msg, user_msg],
                    )
                    text = resp.choices[0].message.content if resp.choices else "{}"
                    try:
                        data = json.loads(text)
                    except Exception:
                        # Try to extract JSON blob if extra text wraps it
                        start = text.find("{")
                        end = text.rfind("}")
                        data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {"results": []}
                    results = data.get("results", []) if isinstance(data, dict) else []
                    for item in results:
                        lo_id = str(item.get("lo_id", ""))
                        score = float(item.get("score", 0.0))
                        rationale = str(item.get("rationale", ""))
                        if score >= config.score_threshold:
                            relation = relation_for_content_type(str(content_row.get("content_type") or ""), config)
                            rows.append(
                                {
                                    "source_lo_id": lo_id,
                                    "target_content_id": content_id,
                                    "relation": relation,
                                    "score": score,
                                    "rationale": rationale or "LLM decision",
                                    "modality": config.modality,
                                    "run_id": config.model,
                                }
                            )
                    break  # success
                except Exception as e:  # rate limits, network, etc.
                    last_err = e
                    time.sleep(2 ** attempt)
            # If all retries failed, fall back to heuristic for this chunk
            if last_err is not None and not results:
                for lo_id, _reason in chunk:
                    lo_text = lo_lookup.get(lo_id, {}).get("learning_objective", "")
                    score = heuristic_score(ctext, lo_text)
                    if score >= config.score_threshold:
                        relation = relation_for_content_type(str(content_row.get("content_type") or ""), config)
                        rows.append(
                            {
                                "source_lo_id": lo_id,
                                "target_content_id": content_id,
                                "relation": relation,
                                "score": score,
                                "rationale": f"heuristic fallback after error: {type(last_err).__name__}",
                                "modality": config.modality,
                                "run_id": config.model,
                            }
                        )

        # Per-content progress logging
        try:
            len_after = len(rows)
            kept_for_content = max(0, len_after - len_before)
            elapsed = max(0.0, time.time() - started_at)
            rate = (processed_groups / elapsed) if elapsed > 0 else 0.0
            remaining = max(0, total_groups - processed_groups)
            eta_sec = int(remaining / rate) if rate > 0 else 0
            print(
                f"[score] {processed_groups}/{total_groups} content | +{kept_for_content} edges (total {len_after}) | elapsed {elapsed:.1f}s | ETA {eta_sec/60:.1f}m",
                flush=True,
            )
            # Append JSONL record
            rec = {
                "content_id": str(content_id),
                "num_candidates": int(len(group)),
                "num_kept": int(kept_for_content),
                "edges_total": int(len_after),
                "processed": int(processed_groups),
                "total": int(total_groups),
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": int(eta_sec),
            }
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            # Best-effort logging; never fail the run due to progress issues
            pass

    return pd.DataFrame(rows)


def relation_for_content_type(content_type: str, config: DiscoveryConfig) -> str:
    """
    Maps content_type to the proposed relation name.

    Args:
        content_type: One of {concept, example, try_it}
        config: DiscoveryConfig providing mapping

    Returns:
        Relation string
    """
    ct = (content_type or "").strip().lower()
    if ct == "concept":
        return config.relation_concept
    if ct == "example":
        return config.relation_example
    return config.relation_try_it


def ensure_parent_directory(path: str) -> None:
    """
    Ensures parent directory exists for the given file path.

    Args:
        path: File path

    Returns:
        None
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _add_suffix_to_path(path: str, suffix: str) -> str:
    """
    Inserts a suffix before the file extension. If no extension, appends suffix.

    example: edges.csv + .shard0-of-4 => edges.shard0-of-4.csv
    """
    base = os.path.basename(path)
    parent = os.path.dirname(path)
    if "." in base:
        name, ext = base.rsplit(".", 1)
        new_base = f"{name}{suffix}.{ext}"
    else:
        new_base = f"{base}{suffix}"
    return os.path.join(parent, new_base)


# ----------------------------
# Candidate generation
# ----------------------------


def build_lo_metadata(lo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares LO metadata including tokenized learning objectives.

    Args:
        lo_df: DataFrame with columns lo_id, learning_objective, unit, chapter, book

    Returns:
        DataFrame with added 'lo_tokens' list column
    """
    lo_copy = lo_df.copy()
    lo_copy["learning_objective"] = lo_copy["learning_objective"].astype(str)
    lo_copy["lo_tokens"] = lo_copy["learning_objective"].map(tokenize)
    return lo_copy


def generate_candidates_for_row(
    content_row: pd.Series,
    lo_meta: pd.DataFrame,
    config: DiscoveryConfig,
) -> List[Tuple[str, str]]:
    """
    Generates candidate LO IDs for a single content row.

    Args:
        content_row: Row from content_items.csv
        lo_meta: LO metadata DataFrame with lo_tokens
        config: Candidate generation config

    Returns:
        List of tuples (candidate_lo_id, reason_tag)
        - reason_tag ∈ {"unit", "chapter", "lexical"}
    """
    candidates: List[Tuple[str, str]] = []

    parent_unit = str(content_row.get("unit") or "")
    parent_chapter = str(content_row.get("chapter") or "")
    content_text = str(content_row.get("text") or "")
    content_tokens: Set[str] = set(tokenize(content_text))

    # Same unit/chapter filtering
    pool = lo_meta
    if config.restrict_same_unit:
        pool = pool[pool["unit"].astype(str) == parent_unit]
    if config.restrict_same_chapter:
        pool = pool[pool["chapter"].astype(str) == parent_chapter]

    # Add unit/chapter matches
    for lo_id in pool["lo_id"].astype(str).tolist():
        candidates.append((lo_id, "unit" if config.restrict_same_unit else "chapter"))

    # Lexical shortlist across the full LO set to catch cross-unit links
    if config.lexical_top_k and len(content_tokens) > 0:
        overlaps: List[Tuple[str, int]] = []
        for _, lo_row in lo_meta.iterrows():
            lo_id = str(lo_row["lo_id"])  # type: ignore
            lo_tokens: Set[str] = set(lo_row.get("lo_tokens") or [])
            overlap = len(content_tokens.intersection(lo_tokens))
            if overlap >= config.lexical_min_overlap:
                overlaps.append((lo_id, overlap))
        # Sort by overlap desc and take top_k
        overlaps.sort(key=lambda t: t[1], reverse=True)
        for lo_id, _score in overlaps[: config.lexical_top_k]:
            candidates.append((lo_id, "lexical"))

    # Deduplicate keeping earliest reason
    seen: set = set()
    unique: List[Tuple[str, str]] = []
    for lo_id, reason in candidates:
        key = lo_id
        if key in seen:
            continue
        seen.add(key)
        unique.append((lo_id, reason))
    return unique


def write_candidates(
    content_df: pd.DataFrame,
    lo_meta: pd.DataFrame,
    config: DiscoveryConfig,
) -> pd.DataFrame:
    """
    Generates and writes candidate pairs for all content items.

    Args:
        content_df: DataFrame of content items
        lo_meta: DataFrame of LO metadata with tokens
        config: Config for candidate generation

    Returns:
        DataFrame of candidate edges written to CSV

    Behavior:
        - For each content item, proposes candidate LOs
        - Infers proposed relation from content_type
        - Writes CSV with columns: source_lo_id, target_content_id, proposed_relation, reason
    """
    rows: List[Dict[str, str]] = []
    for _, row in content_df.iterrows():
        content_id = str(row["content_id"])  # type: ignore
        content_type = str(row.get("content_type") or "")
        relation = relation_for_content_type(content_type, config)
        for lo_id, reason in generate_candidates_for_row(row, lo_meta, config):
            rows.append(
                {
                    "source_lo_id": str(lo_id),
                    "target_content_id": content_id,
                    "proposed_relation": relation,
                    "reason": reason,
                }
            )

    out_df = pd.DataFrame(rows)
    ensure_parent_directory(config.output_candidates)
    out_df.to_csv(config.output_candidates, index=False)
    return out_df


# ----------------------------
# Entrypoint
# ----------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    """
    CLI entrypoint to generate content→LO candidates.

    Args:
        argv: Optional CLI args for testing

    Returns:
        Exit code 0 on success
    """
    parser = argparse.ArgumentParser(description="Generate and/or score content→LO links")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of content items for a smoke run")
    parser.add_argument("--mode", type=str, default="candidates", choices=["candidates", "score", "both"], help="Run candidate generation, scoring, or both")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic heuristic scoring instead of LLM")
    parser.add_argument("--threshold", type=float, default=None, help="Override score threshold (0-1)")
    # Sharding flags for parallel processing
    parser.add_argument("--num-shards", type=int, default=1, help="Number of parallel shards")
    parser.add_argument("--shard-index", type=int, default=0, help="This shard index (0-based)")
    parser.add_argument("--no-suffix-outputs", action="store_true", help="Do not suffix output/progress paths with shard info")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    lo_df = pd.read_csv(config.input_lo_index)
    content_df = pd.read_csv(config.input_content_items)

    # Parse image_urls JSON column if present
    if "image_urls" in content_df.columns:
        def _safe_parse(s: str) -> List[str]:
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, list) else []
            except Exception:
                return []
        content_df["image_urls"] = content_df["image_urls"].astype(str).map(_safe_parse)

    # Apply sharding on content_ids deterministically (CRC32)
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(f"Invalid --shard-index {shard_index}; must be in [0,{num_shards-1}]")
    if num_shards > 1:
        def _belongs_to_shard(cid: str) -> bool:
            try:
                key = str(cid).encode("utf-8")
                return (zlib.crc32(key) % num_shards) == shard_index
            except Exception:
                return False
        before = len(content_df)
        content_df = content_df[content_df["content_id"].astype(str).map(_belongs_to_shard)].copy()
        after = len(content_df)
        print(f"Shard {shard_index}/{num_shards}: content items {after}/{before}")

        # Auto-suffix outputs/progress unless disabled
        if not args.no_suffix_outputs:
            suffix = f".shard{shard_index}-of-{num_shards}"
            config.output_candidates = _add_suffix_to_path(config.output_candidates, suffix)
            config.output_edges = _add_suffix_to_path(config.output_edges, suffix)
            config.progress_path = _add_suffix_to_path(getattr(config, "progress_path", "data/processed/progress_links.jsonl"), suffix)

    if args.limit is not None and args.limit > 0:
        # Use diverse, chronologically ordered selection instead of head()
        content_df = select_diverse_chronological_content(content_df, int(args.limit)).copy()

    lo_meta = build_lo_metadata(lo_df)
    # Allow CLI to override score threshold
    if args.threshold is not None:
        try:
            config.score_threshold = float(args.threshold)
        except Exception:
            pass

    if args.mode in {"candidates", "both"}:
        out_df = write_candidates(content_df, lo_meta, config)
        print(f"Wrote {config.output_candidates} ({len(out_df)} rows)")

    if args.mode in {"score", "both"}:
        # Load candidates (ensure exists if running score standalone)
        cand_path = config.output_candidates
        if not os.path.exists(cand_path):
            out_df = write_candidates(content_df, lo_meta, config)
        else:
            out_df = pd.read_csv(cand_path)

        # Initialize progress log file
        try:
            ensure_parent_directory(getattr(config, "progress_path", "data/processed/progress_links.jsonl"))
            if os.path.exists(config.progress_path):
                os.remove(config.progress_path)
            print(f"Progress log: {config.progress_path}")
        except Exception:
            pass

        edges_df = score_candidates(out_df, content_df, lo_meta, config, dry_run=args.dry_run)
        ensure_parent_directory(config.output_edges)
        edges_df.to_csv(config.output_edges, index=False)
        print(f"Wrote {config.output_edges} ({len(edges_df)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

