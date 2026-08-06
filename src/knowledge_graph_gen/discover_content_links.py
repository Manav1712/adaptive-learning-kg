"""
Discover content ↔ LO links.

Reads lo_index.csv + content_items.csv from a run folder, then writes:
  intermediates/content_link_candidates.csv
  edges_content.csv

Candidate pool is global (every content item × every LO). The LLM scores
each pair; edges at or above the threshold are kept with a relation based
on content type: explained_by / exemplified_by / practiced_by.
"""

from __future__ import annotations
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

load_dotenv()


@dataclass
class DiscoveryConfig:
    input_lo_index: str = ""
    input_content_items: str = ""
    output_candidates: str = ""
    output_edges: str = ""

    relation_concept: str = "explained_by"
    relation_example: str = "exemplified_by"
    relation_try_it: str = "practiced_by"

    model: str = "gpt-4o-mini"
    modality: str = "multimodal"  # "text_only" | "multimodal"
    temperature: float = 0.0
    max_targets_per_call: int = 8
    max_response_tokens: int = 1500
    max_concurrency: int = 8
    image_detail: str = "low"
    max_retries: int = 3
    score_threshold: float = 0.8


def apply_run_dir(config: DiscoveryConfig, run_dir: str) -> None:
    """Wire all I/O paths to a versioned run folder."""
    config.input_lo_index = os.path.join(run_dir, "lo_index.csv")
    config.input_content_items = os.path.join(run_dir, "content_items.csv")
    config.output_candidates = os.path.join(run_dir, "intermediates", "content_link_candidates.csv")
    config.output_edges = os.path.join(run_dir, "edges_content.csv")

# ----------------------------
# Utilities
# ----------------------------

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

def select_chronological_content(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Select a chronologically ordered subset of content items.

    Prefers prepare_nodes order columns on the parent LO when present;
    otherwise falls back to book / content_type / content_id.
    """
    if limit is None or limit <= 0 or len(df) <= limit:
        return df

    tmp = df.copy()
    order_cols = [c for c in ["book_order", "chapter_order", "unit_order", "lo_order"] if c in tmp.columns]
    if order_cols:
        tmp["_ctype_ord"] = tmp.get("content_type", None).map(_ctype_order)
        tmp.sort_values([*order_cols, "_ctype_ord", "content_id"], inplace=True)
        return tmp.drop(columns=["_ctype_ord"]).head(limit)

    tmp["_chapter_num"] = tmp.get("chapter", None).map(_chapter_to_int)
    tmp["_ctype_ord"] = tmp.get("content_type", None).map(_ctype_order)
    tmp.sort_values(["book", "unit", "_chapter_num", "_ctype_ord", "content_id"], inplace=True)
    out = tmp.head(limit)
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
        "decide if the content directly supports the LO.\n\n"
        "EDGE CRITERIA:\n"
        "- Content teaches concepts/skills needed for the LO\n"
        "- Content provides examples/practice for LO objectives\n"
        "- Content prepares students for LO assessment\n\n"
        "SCORING:\n"
        "- score ∈ [-1, 1]; positive means the content supports the LO, negative means it does not\n"
        "- confidence ∈ [0, 1]; your certainty in the assigned score\n\n"
        "EXAMPLES:\n"
        "Derivative definition → LO: Apply derivative rules (score: 0.9, confidence: 0.85)\n"
        "Chain rule worked example → LO: Use chain rule (score: 0.8, confidence: 0.8)\n"
        "Algebra review → LO: Calculus concepts (score: -0.8, confidence: 0.9)\n"
        "Advanced multivariable topic → LO: Basic single-variable prerequisites (score: -0.7, confidence: 0.75)\n\n"
        "Output JSON: {results:[{lo_id, score, confidence, rationale}]}"
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
    return {"system": system, "user": user_blocks}


def score_candidates(
    candidates_df: pd.DataFrame,
    content_df: pd.DataFrame,
    lo_df: pd.DataFrame,
    config: DiscoveryConfig,
) -> pd.DataFrame:
    """
    Scores candidate content→LO pairs using LLM.

    Args:
        candidates_df: DataFrame of candidate pairs
        content_df: DataFrame with content details (text, images)
        lo_df: DataFrame with LO details
        config: Scoring configuration

    Returns:
        DataFrame of filtered edges with columns:
        source_lo_id, target_content_id, relation, score, confidence, rationale, modality, run_id
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

    use_llm = (OpenAI is not None) and (os.environ.get("OPENAI_API_KEY") not in (None, ""))
    if not use_llm:
        raise RuntimeError("LLM scoring is required. Please set OPENAI_API_KEY.")

    client = OpenAI(timeout=120.0)

    def chunk_list(items: List[Tuple[str, str]], n: int) -> List[List[Tuple[str, str]]]:
        return [items[i : i + n] for i in range(0, len(items), n)]

    # Batch by content_id to pack multiple LOs per call, then flatten to
    # (content, chunk) tasks so API calls can be issued concurrently.
    tasks: List[Tuple[str, pd.Series, List[Tuple[str, str]]]] = []
    for content_id, group in candidates_df.groupby("target_content_id"):
        content_row = content_lookup.get(str(content_id))
        if content_row is None:
            continue
        candidate_list = [(str(r["source_lo_id"]), str(r.get("reason") or "")) for _, r in group.iterrows()]
        for chunk in chunk_list(candidate_list, max(1, int(config.max_targets_per_call))):
            tasks.append((str(content_id), content_row, chunk))

    def score_chunk(
        task: Tuple[str, pd.Series, List[Tuple[str, str]]]
    ) -> Tuple[str, pd.Series, List[Dict[str, object]]]:
        content_id, content_row, chunk = task
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
                    image_url: Dict[str, object] = {"url": url}
                elif isinstance(url, dict):
                    image_url = dict(url)
                else:
                    continue
                # Low detail keeps diagram signal at a fraction of the image token cost.
                image_url.setdefault("detail", config.image_detail)
                content_blocks.append({"type": "image_url", "image_url": image_url})

        # Ask model to output strict JSON
        instruction = (
            "Respond ONLY with JSON in this schema: {\n"
            "  \"results\": [ { \"lo_id\": string, \"score\": number, \"confidence\": number, \"rationale\": string } ]\n"
            "} where score in [-1,1] and confidence in [0,1]. "
            "Return one entry per candidate and keep each rationale under 15 words."
        )
        content_blocks.append({"type": "text", "text": instruction})
        user_msg = {"role": "user", "content": content_blocks}

        # Retry with exponential backoff
        last_err: Optional[Exception] = None
        for attempt in range(int(config.max_retries) + 1):
            try:
                resp = client.chat.completions.create(
                    model=config.model,
                    temperature=float(config.temperature),
                    messages=[system_msg, user_msg],
                    max_tokens=int(config.max_response_tokens),
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content if resp.choices else "{}"
                try:
                    data = json.loads(text)
                except Exception:
                    # Try to extract JSON blob if extra text wraps it
                    start = text.find("{")
                    end = text.rfind("}")
                    data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {"results": []}
                return content_id, content_row, (data.get("results", []) if isinstance(data, dict) else [])
            except Exception as e:  # rate limits, network, truncated JSON, etc.
                last_err = e
                time.sleep(2 ** attempt)

        print(
            f"[content] content={content_id} chunk of {len(chunk)} failed after "
            f"{int(config.max_retries) + 1} attempts: {type(last_err).__name__}: {last_err}",
            flush=True,
        )
        return content_id, content_row, []

    total_tasks = len(tasks)
    processed_tasks = 0
    started_at = time.time()

    with ThreadPoolExecutor(max_workers=max(1, int(config.max_concurrency))) as pool:
        for content_id, content_row, results in pool.map(score_chunk, tasks):
            processed_tasks += 1
            len_before = len(rows)

            for item in results:
                lo_id = str(item.get("lo_id", ""))
                score = float(item.get("score", 0.0))
                confidence = item.get("confidence", None)
                try:
                    confidence = float(confidence) if confidence is not None else None
                except Exception:
                    confidence = None
                rationale = str(item.get("rationale", ""))
                if score >= config.score_threshold:
                    relation = relation_for_content_type(str(content_row.get("content_type") or ""), config)
                    rows.append(
                        {
                            "source_lo_id": lo_id,
                            "target_content_id": content_id,
                            "relation": relation,
                            "score": score,
                            "confidence": confidence,
                            "rationale": rationale or "LLM decision",
                            "modality": config.modality,
                            "run_id": config.model,
                        }
                    )

            # Per-content progress logging
            len_after = len(rows)
            kept_for_content = max(0, len_after - len_before)
            log_progress(processed_tasks, total_tasks, kept_for_content, len_after, started_at)

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


def log_progress(processed: int, total: int, edges_added: int, total_edges: int, started_at: float) -> None:
    """
    Log progress with simple timing information.
    
    Args:
        processed: Number of content items processed
        total: Total number of content items
        edges_added: Edges added in this batch
        total_edges: Total edges so far
        started_at: Start time timestamp
    """
    elapsed = max(0.0, time.time() - started_at)
    rate = (processed / elapsed) if elapsed > 0 else 0.0
    eta_sec = int((total - processed) / rate) if rate > 0 else 0
    print(f"[score] {processed}/{total} content | +{edges_added} edges (total {total_edges}) | elapsed {elapsed:.1f}s | ETA {eta_sec/60:.1f}m", flush=True)

# ----------------------------
# Candidate generation
# ----------------------------

def generate_candidates_for_row(
    content_row: pd.Series,
    lo_meta: pd.DataFrame,
    config: DiscoveryConfig,
) -> List[Tuple[str, str]]:
    """Propose every LO as a candidate for this content item (global pool)."""
    return [(str(lo_id), "all") for lo_id in lo_meta["lo_id"].astype(str).tolist()]


def write_candidates(
    content_df: pd.DataFrame,
    lo_meta: pd.DataFrame,
    config: DiscoveryConfig,
) -> pd.DataFrame:
    """Write candidate pairs: source_lo_id, target_content_id, proposed_relation, reason."""
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


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and/or score content→LO links")
    parser.add_argument("--run-dir", required=True, help="Run folder with lo_index.csv and content_items.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of content items for a smoke run")
    parser.add_argument("--mode", type=str, default="both", choices=["candidates", "score", "both"])
    parser.add_argument("--threshold", type=float, default=None, help="Override score threshold (0-1)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = DiscoveryConfig()
    apply_run_dir(config, args.run_dir)
    if args.threshold is not None:
        config.score_threshold = float(args.threshold)

    lo_df = pd.read_csv(config.input_lo_index)
    content_df = pd.read_csv(config.input_content_items)

    # Attach LO order columns onto content for chronological --limit selection.
    order_cols = [c for c in ["lo_id", "book_order", "chapter_order", "unit_order", "lo_order"] if c in lo_df.columns]
    if len(order_cols) > 1 and "lo_id_parent" in content_df.columns:
        content_df = content_df.merge(
            lo_df[order_cols].rename(columns={"lo_id": "lo_id_parent"}),
            on="lo_id_parent",
            how="left",
        )

    if "image_urls" in content_df.columns:
        def _safe_parse(s: str) -> List[str]:
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, list) else []
            except Exception:
                return []
        content_df["image_urls"] = content_df["image_urls"].astype(str).map(_safe_parse)

    if args.limit is not None and args.limit > 0:
        content_df = select_chronological_content(content_df, int(args.limit)).copy()

    lo_meta = lo_df.copy()
    lo_meta["learning_objective"] = lo_meta["learning_objective"].astype(str)

    if args.mode in {"candidates", "both"}:
        out_df = write_candidates(content_df, lo_meta, config)
        print(f"Wrote {config.output_candidates} ({len(out_df)} rows)")

    if args.mode in {"score", "both"}:
        cand_path = config.output_candidates
        if not os.path.exists(cand_path):
            out_df = write_candidates(content_df, lo_meta, config)
        else:
            out_df = pd.read_csv(cand_path)

        edges_df = score_candidates(out_df, content_df, lo_meta, config)
        ensure_parent_directory(config.output_edges)
        edges_df.to_csv(config.output_edges, index=False)
        print(f"Wrote {config.output_edges} ({len(edges_df)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
