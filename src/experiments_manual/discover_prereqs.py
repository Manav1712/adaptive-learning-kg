"""
Discover LO → LO Prerequisite Edges (Coach Graph)

This module compares Learning Objectives (LOs) to infer prerequisite
relationships using an LLM.

Inputs (from prepare step):
- data/processed/lo_index.csv
- data/processed/content_items.csv

Outputs:
- data/processed/prereq_link_candidates.csv (optional)
- data/processed/edges_prereqs.csv

Approach:
- Aggregate content per LO into a consolidated view (text + images)
- Generate candidate LO→LO pairs (restrict by unit/chapter)
- Score each candidate pair using LLM
- Write filtered edges with columns:
  source_lo_id, target_lo_id, relation, score, rationale, modality, run_id
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

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

    model: str = "gpt-4o-mini"
    modality: str = "text_only"  # "text_only" | "multimodal"
    temperature: float = 0.0
    max_targets_per_call: int = 8  # number of source LOs per API call
    max_retries: int = 3
    score_mode: str = "score"
    score_threshold: float = 0.6


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

    llm = data.get("llm", {})
    cfg.model = str(llm.get("model", cfg.model))
    cfg.modality = str(llm.get("modality", cfg.modality))
    cfg.temperature = float(llm.get("temperature", cfg.temperature))
    cfg.max_targets_per_call = int(llm.get("max_targets_per_call", cfg.max_targets_per_call))
    cfg.max_retries = int(llm.get("max_retries", cfg.max_retries))
    cfg.score_mode = str(llm.get("score_mode", cfg.score_mode))
    cfg.score_threshold = float(llm.get("score_threshold", cfg.score_threshold))


    return cfg


# ----------------------------
# Utilities
# ----------------------------


def ensure_parent_directory(path: str) -> None:
    """Ensures parent directory exists for a file path."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def log_prereq_progress(processed: int, total: int, edges_added: int, total_edges: int, started_at: float) -> None:
    """
    Log progress with simple timing information for prerequisite discovery.
    
    Args:
        processed: Number of target LOs processed
        total: Total number of target LOs
        edges_added: Edges added in this batch
        total_edges: Total edges so far
        started_at: Start time timestamp
    """
    elapsed = max(0.0, time.time() - started_at)
    rate = (processed / elapsed) if elapsed > 0 else 0.0
    eta_sec = int((total - processed) / rate) if rate > 0 else 0
    print(f"[prereq] {processed}/{total} targets | +{edges_added} edges (total {total_edges}) | elapsed {elapsed:.1f}s | ETA {eta_sec/60:.1f}m", flush=True)



def _chapter_to_int(val: object) -> Optional[int]:
    try:
        s = str(val).strip()
        digits = "".join(c for c in s if c.isdigit())
        return int(digits) if digits else int(s)
    except Exception:
        return None


def select_chronological_los(lo_meta: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Select a chronologically ordered subset of target LOs when limiting.
    - Sort by (book, unit, chapter_num, lo_id)
    - Take first `limit` rows in chronological order
    """
    if limit is None or limit <= 0 or len(lo_meta) <= limit:
        return lo_meta

    tmp = lo_meta.copy()
    tmp["_chapter_num"] = tmp.get("chapter", None).map(_chapter_to_int)
    tmp.sort_values(["book", "unit", "_chapter_num", "lo_id"], inplace=True)

    # Take first `limit` rows (chronologically ordered)
    out = tmp.head(limit)
    # Drop helper cols
    return out.drop(columns=[c for c in ["_chapter_num"] if c in out.columns])


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
        book = str(r.get("book") or "")
        pieces = [lo_text] + texts_by_lo.get(lo_id, [])
        agg_text = "\n\n".join([p for p in pieces if p])
        imgs = unique(images_by_lo.get(lo_id, []))

        records.append(
            {
                "lo_id": lo_id,
                "learning_objective": lo_text,
                "unit": unit,
                "chapter": chapter,
                "book": book,
                "aggregate_text": agg_text,
                "image_urls": imgs,
            }
        )

    return pd.DataFrame(records)


# ----------------------------
# Candidate generation
# ----------------------------




def generate_prereq_candidates(lo_meta: pd.DataFrame, config: PrereqConfig) -> pd.DataFrame:
    """
    Generates candidate LO→LO pairs for scoring.

    Strategy:
    - For each target LO B, consider sources A in same unit/chapter
    - Uses unit/chapter filtering only (lexical matching removed)
    """
    rows: List[Dict[str, str]] = []

    for _, target in lo_meta.iterrows():
        target_id = str(target["lo_id"])  # type: ignore
        unit = str(target.get("unit") or "")
        chapter = str(target.get("chapter") or "")

        # Candidate pool based on structure restrictions
        if config.restrict_same_chapter:
            pool = lo_meta[(lo_meta["unit"].astype(str) == unit) & (lo_meta["chapter"].astype(str) == chapter)]
        elif config.restrict_same_unit:
            pool = lo_meta[lo_meta["unit"].astype(str) == unit]
        else:
            pool = lo_meta

        for _, cand in pool.iterrows():
            cand_id = str(cand["lo_id"])
            if cand_id == target_id:
                continue
            
            rows.append(
                {
                    "source_lo_id": cand_id,
                    "target_lo_id": target_id,
                    "reason": "unit" if config.restrict_same_unit else "chapter",
                }
            )

    return pd.DataFrame(rows)


# ----------------------------
# Heuristic and Prompting
# ----------------------------




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
) -> pd.DataFrame:
    """
    Scores candidate LO→LO pairs using LLM.
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

    for target_id, group in grouped:
        processed_groups += 1
        len_before = len(rows)
        target_row = lo_views[lo_views["lo_id"].astype(str) == str(target_id)].head(1)
        if target_row.empty:
            continue
        target_series = target_row.iloc[0]

        candidate_list = [(str(r["source_lo_id"]), str(r.get("reason") or "")) for _, r in group.iterrows()]

        # Real LLM integration
        use_llm = (OpenAI is not None) and (os.environ.get("OPENAI_API_KEY") not in (None, ""))
        if not use_llm:
            # Skip if no LLM available
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

            # If all retries failed, skip this chunk
            if last_err is not None and not results:
                continue

        # Progress logging
        len_after = len(rows)
        kept = max(0, len_after - len_before)
        log_prereq_progress(processed_groups, total_groups, kept, len_after, started_at)

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
    parser.add_argument("--threshold", type=float, default=None, help="Override score threshold (0-1)")
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
    lo_meta = lo_views.copy()


    if args.limit is not None and args.limit > 0:
        # Limit target LOs using chronological selection
        lo_meta = select_chronological_los(lo_meta, int(args.limit)).copy()

    if args.mode in {"candidates", "both"}:
        out_df = write_candidates(lo_meta, config)
        print(f"Wrote {config.output_candidates} ({len(out_df)} rows)")

    if args.mode in {"score", "both"}:
        cand_path = config.output_candidates
        if not os.path.exists(cand_path):
            out_df = write_candidates(lo_meta, config)
        else:
            out_df = pd.read_csv(cand_path)


        edges_df = score_prereq_candidates(out_df, lo_views, config)
        ensure_parent_directory(config.output_edges)
        edges_df.to_csv(config.output_edges, index=False)
        print(f"Wrote {config.output_edges} ({len(edges_df)} rows)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
