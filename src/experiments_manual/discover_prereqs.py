"""
Discover LO → LO Prerequisite Edges (Coach Graph)

This module compares Learning Objectives (LOs) to infer prerequisite
relationships using an LLM (with a deterministic fallback heuristic).

Inputs (from prepare step):
- data/processed/lo_index.csv
- data/processed/content_items.csv

Outputs:
- data/processed/prereq_link_candidates.csv (optional)
- data/processed/edges_prereqs.csv

Approach:
- Aggregate content per LO into a consolidated view (text + images)
- Generate candidate LO→LO pairs (restrict by unit/chapter, lexical shortlist)
- Score each candidate pair using LLM; fallback to heuristic if needed
- Write filtered edges with columns:
  source_lo_id, target_lo_id, relation, score, rationale, modality, run_id
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import os
import zlib

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:  # pragma: no cover - optional
    pass

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
class PrereqConfig:
    """
    Configuration for LO→LO prerequisite discovery.
    """
    input_lo_index: str = "data/processed/lo_index.csv"
    input_content_items: str = "data/processed/content_items.csv"

    output_candidates: str = "data/processed/prereq_link_candidates.csv"
    output_edges: str = "data/processed/edges_prereqs.csv"

    restrict_same_unit: bool = True
    restrict_same_chapter: bool = False

    lexical_top_k: int = 10
    lexical_min_overlap: int = 1

    model: str = "gpt-4o-mini"
    modality: str = "text_only"  # "text_only" | "multimodal"
    temperature: float = 0.0
    max_targets_per_call: int = 8  # number of source LOs per API call
    max_retries: int = 3
    score_mode: str = "score"
    score_threshold: float = 0.6

    progress_path: str = "data/processed/progress_prereqs.jsonl"


def load_config(config_path: Optional[str]) -> PrereqConfig:
    """
    Loads config from YAML or returns sane defaults.
    """
    if not config_path or not os.path.exists(config_path) or yaml is None:
        return PrereqConfig()
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = PrereqConfig()
    inputs = data.get("output_paths", {})
    inputs_alt = data.get("input_paths", {})

    cfg.input_lo_index = str(inputs.get("lo_index", inputs_alt.get("lo_index", cfg.input_lo_index)))
    cfg.input_content_items = str(
        inputs.get("content_items", inputs_alt.get("content_items", cfg.input_content_items))
    )

    out = data.get("output_paths", {})
    cfg.output_candidates = str(out.get("prereq_candidates", cfg.output_candidates))
    cfg.output_edges = str(out.get("edges_prereqs", cfg.output_edges))

    pruning = data.get("pruning", {})
    cfg.restrict_same_unit = bool(pruning.get("restrict_same_unit", cfg.restrict_same_unit))
    cfg.restrict_same_chapter = bool(pruning.get("restrict_same_chapter", cfg.restrict_same_chapter))
    cfg.lexical_top_k = int(pruning.get("lexical_top_k", cfg.lexical_top_k))
    cfg.lexical_min_overlap = int(pruning.get("lexical_min_overlap", cfg.lexical_min_overlap))

    llm = data.get("llm", {})
    cfg.model = str(llm.get("model", cfg.model))
    cfg.modality = str(llm.get("modality", cfg.modality))
    cfg.temperature = float(llm.get("temperature", cfg.temperature))
    cfg.max_targets_per_call = int(llm.get("max_targets_per_call", cfg.max_targets_per_call))
    cfg.max_retries = int(llm.get("max_retries", cfg.max_retries))
    cfg.score_mode = str(llm.get("score_mode", cfg.score_mode))
    cfg.score_threshold = float(llm.get("score_threshold", cfg.score_threshold))

    out_misc = data.get("progress", {})
    cfg.progress_path = str(out_misc.get("prereqs_progress", cfg.progress_path))

    return cfg


# ----------------------------
# Utilities
# ----------------------------


def ensure_parent_directory(path: str) -> None:
    """Ensures parent directory exists for a file path."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _add_suffix_to_path(path: str, suffix: str) -> str:
    base = os.path.basename(path)
    parent = os.path.dirname(path)
    if "." in base:
        name, ext = base.rsplit(".", 1)
        new_base = f"{name}{suffix}.{ext}"
    else:
        new_base = f"{base}{suffix}"
    return os.path.join(parent, new_base)


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer, lowercased."""
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def unique(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_parse_image_urls(val: object) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    s = str(val)
    try:
        obj = json.loads(s)
        return [str(x) for x in obj] if isinstance(obj, list) else []
    except Exception:
        return []


# TODO: Add LO view aggregation functions
# TODO: Add candidate generation functions  
# TODO: Add heuristic and prompting functions
# TODO: Add scoring functions
# TODO: Add CLI functions


# ----------------------------
# LO view aggregation
# ----------------------------


def build_lo_views(lo_df: pd.DataFrame, content_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates content per LO into a consolidated view.

    Returns DataFrame with columns:
    - lo_id, learning_objective, unit, chapter
    - aggregate_text: learning objective + concatenated content text
    - image_urls: list JSON-serializable
    """
    # Normalize image_urls column if present
    if "image_urls" in content_df.columns:
        content_df = content_df.copy()
        content_df["image_urls"] = content_df["image_urls"].map(safe_parse_image_urls)

    # Group content by lo_id
    texts_by_lo: Dict[str, List[str]] = {}
    images_by_lo: Dict[str, List[str]] = {}
    for _, row in content_df.iterrows():
        lo_id = str(row.get("lo_id") or "")
        if not lo_id:
            continue
        txt = str(row.get("text") or "")
        if txt:
            texts_by_lo.setdefault(lo_id, []).append(txt)
        imgs = row.get("image_urls")
        if isinstance(imgs, list) and imgs:
            images_by_lo.setdefault(lo_id, []).extend([str(u) for u in imgs])

    records: List[Dict[str, object]] = []
    for _, r in lo_df.iterrows():
        lo_id = str(r.get("lo_id") or "")
        lo_text = str(r.get("learning_objective") or "")
        unit = str(r.get("unit") or "")
        chapter = str(r.get("chapter") or "")
        pieces = [lo_text] + texts_by_lo.get(lo_id, [])
        agg_text = "\n\n".join([p for p in pieces if p])
        imgs = unique(images_by_lo.get(lo_id, []))
        records.append(
            {
                "lo_id": lo_id,
                "learning_objective": lo_text,
                "unit": unit,
                "chapter": chapter,
                "aggregate_text": agg_text,
                "image_urls": imgs,
            }
        )

    return pd.DataFrame(records)


# ----------------------------
# Candidate generation
# ----------------------------


def build_lo_metadata(lo_views: pd.DataFrame) -> pd.DataFrame:
    """Adds token columns to support lexical shortlist heuristics."""
    df = lo_views.copy()
    df["tokens"] = df["learning_objective"].astype(str).map(tokenize)
    df["token_set"] = df["tokens"].map(set)
    return df


def generate_prereq_candidates(lo_meta: pd.DataFrame, config: PrereqConfig) -> pd.DataFrame:
    """
    Generates candidate LO→LO pairs for scoring.

    Strategy:
    - For each target LO B, consider sources A in same unit/chapter
    - Rank A by lexical overlap with B (tokens of A ∩ tokens of B)
    - Keep top-K per target
    """
    rows: List[Dict[str, str]] = []

    # Index by unit/chapter for quick filtering
    by_unit: Dict[str, List[Tuple[str, Set[str]]]] = {}
    by_unit_chapter: Dict[Tuple[str, str], List[Tuple[str, Set[str]]]] = {}
    for _, r in lo_meta.iterrows():
        lo_id = str(r["lo_id"])  # type: ignore
        unit = str(r.get("unit") or "")
        chapter = str(r.get("chapter") or "")
        tset: Set[str] = set(r.get("tokens") or [])
        by_unit.setdefault(unit, []).append((lo_id, tset))
        by_unit_chapter.setdefault((unit, chapter), []).append((lo_id, tset))

    for _, target in lo_meta.iterrows():
        target_id = str(target["lo_id"])  # type: ignore
        unit = str(target.get("unit") or "")
        chapter = str(target.get("chapter") or "")
        target_tokens: Set[str] = set(target.get("tokens") or [])

        # Candidate pool based on structure restrictions
        if config.restrict_same_chapter:
            pool = by_unit_chapter.get((unit, chapter), [])
        elif config.restrict_same_unit:
            pool = by_unit.get(unit, [])
        else:
            pool = [(str(r["lo_id"]), set(r.get("tokens") or [])) for _, r in lo_meta.iterrows()]

        scored: List[Tuple[str, int]] = []
        for cand_id, cand_tokens in pool:
            if cand_id == target_id:
                continue
            overlap = len(target_tokens.intersection(cand_tokens))
            if overlap >= int(config.lexical_min_overlap):
                scored.append((cand_id, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: int(config.lexical_top_k)] if int(config.lexical_top_k) > 0 else scored
        for cand_id, ov in top:
            rows.append(
                {
                    "source_lo_id": cand_id,
                    "target_lo_id": target_id,
                    "reason": f"lexical_overlap={ov}",
                }
            )

    return pd.DataFrame(rows)


# ----------------------------
# Heuristic and Prompting
# ----------------------------


def heuristic_prereq_score(source_text: str, target_text: str) -> float:
    """
    Directional token-overlap heuristic: how much of the target appears in the source.

    score = |tokens(source) ∩ tokens(target)| / max(1, |tokens(target)|)
    """
    stoks = set(tokenize(source_text))
    ttoks = set(tokenize(target_text))
    if not stoks or not ttoks:
        return 0.0
    overlap = len(stoks.intersection(ttoks))
    return overlap / float(max(1, len(ttoks)))


def build_prompt_for_prereq(
    target_row: pd.Series,
    candidate_chunk: List[Tuple[str, str]],
    lo_lookup: Dict[str, Dict[str, object]],
    config: PrereqConfig,
) -> Dict[str, object]:
    """
    Builds a prompt asking whether each candidate source LO is a prerequisite for the target LO.
    Returns a dict compatible with the OpenAI messages builder used downstream.
    """
    target_id = str(target_row.get("lo_id") or "")
    target_text = str(target_row.get("aggregate_text") or target_row.get("learning_objective") or "")
    unit = str(target_row.get("unit") or "")
    chapter = str(target_row.get("chapter") or "")

    system = (
        "You are an expert math curriculum designer. Given a target Learning Objective (LO) and a list of "
        "candidate source LOs, decide if each source LO is a prerequisite for the target. "
        "Return a JSON object with results for each candidate including a confidence score in [0,1] and brief rationale."
    )

    user_blocks: List[Dict[str, object]] = []
    user_blocks.append(
        {
            "type": "text",
            "text": (
                f"Target LO (id={target_id}, unit={unit}, chapter={chapter}):\n"
                f"{target_text}\n\n"
                "For each candidate source LO below, decide if it is a prerequisite for the target:"
            ),
        }
    )
    # Include target LO images in multimodal mode
    if getattr(config, "modality", "text_only") == "multimodal":
        t_images = target_row.get("image_urls") or []
        for url in t_images:
            try:
                user_blocks.append({"type": "image_url", "image_url": url})
            except Exception:
                # best-effort; ignore malformed URLs
                pass

    # Append source candidates
    for lo_id, reason in candidate_chunk:
        src = lo_lookup.get(lo_id, {})
        src_text = str(src.get("aggregate_text") or src.get("learning_objective") or "")
        user_blocks.append({"type": "text", "text": f"Candidate source LO (id={lo_id}) [{reason}]:\n{src_text}"})
        if getattr(config, "modality", "text_only") == "multimodal":
            s_images = src.get("image_urls") or []
            for url in s_images:
                try:
                    user_blocks.append({"type": "image_url", "image_url": url})
                except Exception:
                    pass

    return {"system": system, "user": user_blocks}


# ----------------------------
# Scoring
# ----------------------------


def score_prereq_candidates(
    candidates_df: pd.DataFrame,
    lo_views: pd.DataFrame,
    config: PrereqConfig,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Scores candidate LO→LO pairs using LLM or heuristic fallback.
    """
    # Prepare lookup by lo_id
    lo_lookup: Dict[str, Dict[str, object]] = {
        str(r["lo_id"]): {
            "learning_objective": str(r.get("learning_objective") or ""),
            "aggregate_text": str(r.get("aggregate_text") or ""),
            "unit": str(r.get("unit") or ""),
            "chapter": str(r.get("chapter") or ""),
            "image_urls": list(r.get("image_urls") or []),
        }
        for _, r in lo_views.iterrows()
    }

    rows: List[Dict[str, object]] = []

    grouped = candidates_df.groupby("target_lo_id")
    total_groups = int(getattr(grouped, "ngroups", 0) or 0)
    processed_groups = 0
    started_at = time.time()
    progress_path = getattr(config, "progress_path", "data/processed/progress_prereqs.jsonl")
    try:
        ensure_parent_directory(progress_path)
        with open(progress_path, "w", encoding="utf-8") as _f:
            pass
    except Exception:
        pass

    for target_id, group in grouped:
        processed_groups += 1
        len_before = len(rows)
        target_row = lo_views[lo_views["lo_id"].astype(str) == str(target_id)].head(1)
        if target_row.empty:
            continue
        target_series = target_row.iloc[0]

        candidate_list = [(str(r["source_lo_id"]), str(r.get("reason") or "")) for _, r in group.iterrows()]

        if dry_run:
            for src_id, _reason in candidate_list:
                s_text = lo_lookup.get(src_id, {}).get("aggregate_text", "")
                t_text = lo_lookup.get(str(target_id), {}).get("aggregate_text", "")
                score = heuristic_prereq_score(str(s_text), str(t_text))
                keep = score >= float(config.score_threshold)
                if keep:
                    rows.append(
                        {
                            "source_lo_id": src_id,
                            "target_lo_id": str(target_id),
                            "relation": "prerequisite",
                            "score": float(score),
                            "rationale": "heuristic token overlap (prereq)",
                            "modality": config.modality,
                            "run_id": config.model,
                        }
                    )
            # progress
            try:
                len_after = len(rows)
                kept = max(0, len_after - len_before)
                elapsed = max(0.0, time.time() - started_at)
                rate = (processed_groups / elapsed) if elapsed > 0 else 0.0
                remaining = max(0, total_groups - processed_groups)
                eta_sec = int(remaining / rate) if rate > 0 else 0
                print(
                    f"[prereq] {processed_groups}/{total_groups} targets | +{kept} edges (total {len_after}) | elapsed {elapsed:.1f}s | ETA {eta_sec/60:.1f}m",
                    flush=True,
                )
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "target_lo_id": str(target_id),
                                "num_candidates": int(len(group)),
                                "num_kept": int(kept),
                                "edges_total": int(len_after),
                                "processed": int(processed_groups),
                                "total": int(total_groups),
                                "elapsed_sec": round(elapsed, 1),
                                "eta_sec": int(eta_sec),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            continue

        # Real LLM integration
        use_llm = (OpenAI is not None) and (os.environ.get("OPENAI_API_KEY") not in (None, ""))
        if not use_llm:
            # fallback to heuristic if no API key
            for src_id, _reason in candidate_list:
                s_text = lo_lookup.get(src_id, {}).get("aggregate_text", "")
                t_text = lo_lookup.get(str(target_id), {}).get("aggregate_text", "")
                score = heuristic_prereq_score(str(s_text), str(t_text))
                if score >= float(config.score_threshold):
                    rows.append(
                        {
                            "source_lo_id": src_id,
                            "target_lo_id": str(target_id),
                            "relation": "prerequisite",
                            "score": float(score),
                            "rationale": "heuristic token overlap (no OPENAI_API_KEY)",
                            "modality": config.modality,
                            "run_id": config.model,
                        }
                    )
            # progress
            try:
                len_after = len(rows)
                kept = max(0, len_after - len_before)
                elapsed = max(0.0, time.time() - started_at)
                rate = (processed_groups / elapsed) if elapsed > 0 else 0.0
                remaining = max(0, total_groups - processed_groups)
                eta_sec = int(remaining / rate) if rate > 0 else 0
                print(
                    f"[prereq] {processed_groups}/{total_groups} targets | +{kept} edges (total {len_after}) | elapsed {elapsed:.1f}s | ETA {eta_sec/60:.1f}m",
                    flush=True,
                )
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "target_lo_id": str(target_id),
                                "num_candidates": int(len(group)),
                                "num_kept": int(kept),
                                "edges_total": int(len_after),
                                "processed": int(processed_groups),
                                "total": int(total_groups),
                                "elapsed_sec": round(elapsed, 1),
                                "eta_sec": int(eta_sec),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            continue

        # LLM scoring
        def chunk_list(items: List[Tuple[str, str]], n: int) -> List[List[Tuple[str, str]]]:
            return [items[i : i + n] for i in range(0, len(items), n)]

        client = OpenAI()

        for chunk in chunk_list(candidate_list, max(1, int(config.max_targets_per_call))):
            prompt = build_prompt_for_prereq(target_series, chunk, lo_lookup, config)
            system_msg = {"role": "system", "content": prompt["system"]}
            content_blocks: List[Dict[str, object]] = []
            for block in prompt["user"]:
                if block.get("type") == "text":
                    content_blocks.append({"type": "text", "text": str(block.get("text", ""))})
                elif block.get("type") == "image_url":
                    url = block.get("image_url")
                    if isinstance(url, str):
                        content_blocks.append({"type": "image_url", "image_url": {"url": url}})
                    elif isinstance(url, dict):
                        content_blocks.append({"type": "image_url", "image_url": url})
            user_msg = {"role": "user", "content": content_blocks}

            instruction = (
                "Respond ONLY with JSON in this schema: {\n"
                "  \"results\": [ { \"lo_id\": string, \"score\": number, \"rationale\": string } ]\n"
                "} where score in [0,1]."
            )
            content_blocks.append({"type": "text", "text": instruction})

            last_err: Optional[Exception] = None
            results: List[Dict[str, object]] = []
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
                        start = text.find("{")
                        end = text.rfind("}")
                        data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {"results": []}
                    results = data.get("results", []) if isinstance(data, dict) else []
                    for item in results:
                        src_id = str(item.get("lo_id", ""))
                        score = float(item.get("score", 0.0))
                        rationale = str(item.get("rationale", ""))
                        if score >= float(config.score_threshold):
                            rows.append(
                                {
                                    "source_lo_id": src_id,
                                    "target_lo_id": str(target_id),
                                    "relation": "prerequisite",
                                    "score": float(score),
                                    "rationale": rationale or "LLM decision",
                                    "modality": config.modality,
                                    "run_id": config.model,
                                }
                            )
                    break  # success
                except Exception as e:  # rate limits, network, etc.
                    last_err = e
                    time.sleep(2 ** attempt)

            if last_err is not None and not results:
                # Heuristic fallback for this chunk
                for src_id, _reason in chunk:
                    s_text = lo_lookup.get(src_id, {}).get("aggregate_text", "")
                    t_text = lo_lookup.get(str(target_id), {}).get("aggregate_text", "")
                    score = heuristic_prereq_score(str(s_text), str(t_text))
                    if score >= float(config.score_threshold):
                        rows.append(
                            {
                                "source_lo_id": src_id,
                                "target_lo_id": str(target_id),
                                "relation": "prerequisite",
                                "score": float(score),
                                "rationale": f"heuristic fallback after error: {type(last_err).__name__}",
                                "modality": config.modality,
                                "run_id": config.model,
                            }
                        )

        # progress
        try:
            len_after = len(rows)
            kept = max(0, len_after - len_before)
            elapsed = max(0.0, time.time() - started_at)
            rate = (processed_groups / elapsed) if elapsed > 0 else 0.0
            remaining = max(0, total_groups - processed_groups)
            eta_sec = int(remaining / rate) if rate > 0 else 0
            print(
                f"[prereq] {processed_groups}/{total_groups} targets | +{kept} edges (total {len_after}) | elapsed {elapsed:.1f}s | ETA {eta_sec/60:.1f}m",
                flush=True,
            )
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "target_lo_id": str(target_id),
                            "num_candidates": int(len(group)),
                            "num_kept": int(kept),
                            "edges_total": int(len_after),
                            "processed": int(processed_groups),
                            "total": int(total_groups),
                            "elapsed_sec": round(elapsed, 1),
                            "eta_sec": int(eta_sec),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    return pd.DataFrame(rows)


# ----------------------------
# CLI
# ----------------------------


def write_candidates(lo_meta: pd.DataFrame, config: PrereqConfig) -> pd.DataFrame:
    out_df = generate_prereq_candidates(lo_meta, config)
    ensure_parent_directory(config.output_candidates)
    out_df.to_csv(config.output_candidates, index=False)
    return out_df


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Discover LO→LO prerequisites")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of target LOs for a smoke run")
    parser.add_argument("--mode", type=str, default="both", choices=["candidates", "score", "both"], help="Run candidate generation, scoring, or both")
    parser.add_argument("--dry-run", action="store_true", help="Use heuristic scoring instead of LLM")
    parser.add_argument("--threshold", type=float, default=None, help="Override score threshold (0-1)")
    # Sharding flags for parallel processing (shard by target_lo_id)
    parser.add_argument("--num-shards", type=int, default=1, help="Number of parallel shards")
    parser.add_argument("--shard-index", type=int, default=0, help="This shard index (0-based)")
    parser.add_argument("--no-suffix-outputs", action="store_true", help="Do not suffix output/progress paths with shard info")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    lo_df = pd.read_csv(config.input_lo_index)
    content_df = pd.read_csv(config.input_content_items)

    if args.threshold is not None:
        try:
            config.score_threshold = float(args.threshold)
        except Exception:
            pass

    lo_views = build_lo_views(lo_df, content_df)
    lo_meta = build_lo_metadata(lo_views)

    # Apply sharding on target LO ids (deterministic by CRC32)
    import zlib as _zlib
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(f"Invalid --shard-index {shard_index}; must be in [0,{num_shards-1}]")
    if num_shards > 1:
        def _belongs_to_shard(lo_id: str) -> bool:
            try:
                return (_zlib.crc32(str(lo_id).encode("utf-8")) % num_shards) == shard_index
            except Exception:
                return False
        before = len(lo_meta)
        lo_meta = lo_meta[lo_meta["lo_id"].astype(str).map(_belongs_to_shard)].copy()
        after = len(lo_meta)
        print(f"Shard {shard_index}/{num_shards}: target LOs {after}/{before}")

        # Auto-suffix outputs/progress unless disabled
        if not args.no_suffix_outputs:
            suffix = f".shard{shard_index}-of-{num_shards}"
            config.output_candidates = _add_suffix_to_path(config.output_candidates, suffix)
            config.output_edges = _add_suffix_to_path(config.output_edges, suffix)
            config.progress_path = _add_suffix_to_path(getattr(config, "progress_path", "data/processed/progress_prereqs.jsonl"), suffix)

    if args.limit is not None and args.limit > 0:
        # Limit target LOs only
        keep_ids = set(str(x) for x in lo_meta.head(args.limit)["lo_id"].tolist())
        lo_meta = lo_meta[lo_meta["lo_id"].astype(str).isin(keep_ids)].copy()

    if args.mode in {"candidates", "both"}:
        out_df = write_candidates(lo_meta, config)
        print(f"Wrote {config.output_candidates} ({len(out_df)} rows)")

    if args.mode in {"score", "both"}:
        cand_path = config.output_candidates
        if not os.path.exists(cand_path):
            out_df = write_candidates(lo_meta, config)
        else:
            out_df = pd.read_csv(cand_path)

        # Initialize progress log
        try:
            ensure_parent_directory(config.progress_path)
            if os.path.exists(config.progress_path):
                os.remove(config.progress_path)
            print(f"Progress log: {config.progress_path}")
        except Exception:
            pass

        edges_df = score_prereq_candidates(out_df, lo_views, config, dry_run=args.dry_run)
        ensure_parent_directory(config.output_edges)
        edges_df.to_csv(config.output_edges, index=False)
        print(f"Wrote {config.output_edges} ({len(edges_df)} rows)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
