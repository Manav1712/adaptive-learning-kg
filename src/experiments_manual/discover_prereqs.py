"""
Discover LO → LO Prerequisite Edges (Coach Graph)

This module compares Learning Objectives (LOs) to infer prerequisite
relationships using an LLM.

Inputs (from prepare step; configured in-code via PrereqConfig):
- data/processed/lo_index.csv
- data/processed/content_items.csv

Outputs:
- data/processed/prereq_link_candidates.csv (optional)
- data/processed/edges_prereqs.csv

Approach:
- Aggregate content per LO into a consolidated view (text + images)
- Generate candidate LO→LO pairs by considering all earlier LOs (cross-chapter/unit/book)
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
from dotenv import load_dotenv  # type: ignore
import pandas as pd
from openai import OpenAI  # type: ignore

# ----------------------------
# Configuration
# ----------------------------
load_dotenv()
@dataclass
class PrereqConfig:
    """
    Configuration for LO→LO prerequisite discovery.
    """
    input_lo_index: str = "data/processed/lo_index.csv"
    input_content_items: str = "data/processed/content_items.csv"

    output_candidates: str = "data/processed/prereq_link_candidates.csv"
    output_edges: str = "data/processed/edges_prereqs.csv"

    restrict_same_unit: bool = False
    restrict_same_chapter: bool = False

    model: str = "gpt-4o-mini"
    modality: str = "multimodal"  # "text_only" | "multimodal"
    temperature: float = 0.0
    max_targets_per_call: int = 8  # number of source LOs per API call
    max_retries: int = 3
    score_mode: str = "score"
    score_threshold: float = 0.7
    min_confidence: float = 0.6


def load_config() -> PrereqConfig:
    """
    Returns in-code defaults defined by PrereqConfig.
    """
    return PrereqConfig()


# ----------------------------
# Utilities
# ----------------------------

def _first_int_or_raise(val: object) -> int:
    """Extracts the first integer from a value or raises if none found."""
    import re
    m = re.search(r"(\d+)", str(val or ""))
    if not m:
        raise ValueError(f"no integer in: {val}")
    return int(m.group(1))
    
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


def select_chronological_los(lo_meta: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Select a chronologically ordered subset of target LOs when limiting.
    - Sort by (book, unit, chapter_num, lo_id)
    - Take first `limit` rows in chronological order
    """
    if limit is None or limit <= 0 or len(lo_meta) <= limit:
        return lo_meta

    tmp = lo_meta.copy()
    def _safe_chapter(v: object) -> Optional[int]:
        try:
            return _first_int_or_raise(v)
        except Exception:
            return None
    tmp["_chapter_num"] = tmp.get("chapter", None).map(_safe_chapter)
    bad = int(tmp["_chapter_num"].isnull().sum())
    if bad:
        print(f"[prereq] Skipped {bad} LOs in --limit due to unparsable chapter.", flush=True)
        tmp = tmp[tmp["_chapter_num"].notnull()].copy()
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

    # Group content by LO, accepting either lo_id or lo_id_parent as source key
    texts_by_lo: Dict[str, List[str]] = {}
    images_by_lo: Dict[str, List[str]] = {}
    for _, row in content_df.iterrows():
        lo_id = str(row.get("lo_id") or row.get("lo_id_parent") or "")
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


def create_chronological_key(row: pd.Series) -> Tuple[str, str, int, str]:
    """
    Creates a chronological ordering key for LOs.
    Returns (book, unit, chapter_num, lo_id) for sorting.
    """
    book = str(row.get("book") or "")
    unit = str(row.get("unit") or "")
    chapter = str(row.get("chapter") or "")
    lo_id = str(row.get("lo_id") or "")
    
    # Extract numeric chapter for proper sorting; raise if missing
    chapter_num = _first_int_or_raise(chapter)
    
    return (book, unit, chapter_num, lo_id)


def generate_prereq_candidates(lo_meta: pd.DataFrame, config: PrereqConfig) -> pd.DataFrame:
    """
    Generates candidate LO→LO pairs for scoring.

    Strategy:
    - For each target LO B, consider all earlier sources A across chapters/units/books
    - Enforce chronologically forward pairs only (source precedes target) via `_chrono_key`
    - Candidate pool always spans the entire set (cross-chapter/unit/book)
    """
    rows: List[Dict[str, str]] = []

    # Add chronological keys for sorting, skipping rows that can't be parsed
    lo_meta = lo_meta.copy()
    broken_los: List[Tuple[str, str, str]] = []
    def _safe_key(r: pd.Series) -> Optional[Tuple[str, str, int, str]]:
        try:
            return create_chronological_key(r)
        except Exception:
            broken_los.append((str(r.get("lo_id") or ""), str(r.get("unit") or ""), str(r.get("chapter") or "")))
            return None
    lo_meta['_chrono_key'] = lo_meta.apply(_safe_key, axis=1)
    bad = int(lo_meta['_chrono_key'].isnull().sum())
    if bad:
        examples = list({t for t in broken_los})[:5]
        print(f"[prereq] Skipped {bad} LOs with unparsable unit/chapter. Examples: {examples}", flush=True)
        lo_meta = lo_meta[lo_meta['_chrono_key'].notnull()].copy()

    # Candidate generation with chronological constraint:
    # - Always allow cross-chapter/unit/book by using the full pool
    # - For each target, only consider sources that precede it by _chrono_key (forward-only prerequisites)
    for _, target in lo_meta.iterrows():
        target_id = str(target["lo_id"])  # type: ignore
        target_key = target['_chrono_key']
        pool = lo_meta

        for _, cand in pool.iterrows():
            cand_id = str(cand["lo_id"])
            cand_key = cand['_chrono_key']
            
            if cand_id == target_id:
                continue
            
            # CHRONOLOGICAL ENFORCEMENT: Only allow forward direction
            if cand_key >= target_key:  # source must precede target
                continue
            
            rows.append(
                {
                    "source_lo_id": cand_id,
                    "target_lo_id": target_id,
                    # Tag reason uniformly since pool is global
                    "reason": "all",
                }
            )
    return pd.DataFrame(rows)


# ----------------------------
# Prompting
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
        "candidate source LOs, decide if each source LO is a prerequisite for the target.\n\n"
        "PREREQUISITE CRITERIA:\n"
        "- Source LO teaches concepts/skills needed BEFORE the target LO\n"
        "- Source LO provides foundational knowledge for the target LO\n"
        "- Target LO builds upon or extends concepts from the source LO\n"
        "- Cross-chapter, cross-unit, and cross-book prerequisites ARE allowed if the source precedes the target in the curriculum order\n"
        "- The prerequisite MUST be earlier in the curriculum order than the target\n\n"
        "DIRECTION CONSTRAINT:\n"
        "- Only output prerequisites from earlier to later in the curriculum.\n"
        "- If the source is the same position or later than the target, return a NEGATIVE score.\n\n"
        "SCORING:\n"
        "- score ∈ [-1, 1]; positive means source IS a prerequisite, negative means it is NOT\n"
        "- confidence ∈ [0, 1]; your certainty in the assigned score\n\n"
        "EXAMPLES:\n"
        "Positive Example: Function notation → Composite functions (score: 0.8, confidence: 0.85)\n"
        "Positive Example: Polynomial basics → Polynomial derivatives (score: 0.85, confidence: 0.8)\n"
        " Negative Example: Later chapter topic → Earlier chapter topic (score: -0.9, confidence: 0.95)\n"
        "Negative Example: Advanced integration → Basic differentiation (score: -0.9, confidence: 0.85)\n\n"
        "Output JSON: {results:[{lo_id, score, confidence, rationale}]}"
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
            # Include fields required by create_chronological_key
            "book": str(r.get("book") or ""),
            "lo_id": str(r.get("lo_id") or ""),
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

        # LLM scoring only
        use_llm = (OpenAI is not None) and (os.environ.get("OPENAI_API_KEY") not in (None, ""))
        if not use_llm:
            raise RuntimeError("LLM scoring is required (heuristic disabled). Please set OPENAI_API_KEY.")

        # LLM scoring
        def chunk_list(items: List[Tuple[str, str]], n: int) -> List[List[Tuple[str, str]]]:
            return [items[i : i + n] for i in range(0, len(items), n)]

        client = OpenAI() if use_llm else None

        for chunk in chunk_list(candidate_list, max(1, int(config.max_targets_per_call))):
            results: List[Dict[str, object]] = []
            last_err: Optional[Exception] = None

            if use_llm and client is not None:
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
                    "  \"results\": [ { \"lo_id\": string, \"score\": number, \"confidence\": number, \"rationale\": string } ]\n"
                    "} where score in [-1,1] and confidence in [0,1]."
                )
                content_blocks.append({"type": "text", "text": instruction})

                for attempt in range(int(config.max_retries) + 1):
                    try:
                        resp = client.chat.completions.create(
                            model=config.model,
                            temperature=float(config.temperature),
                            messages=[system_msg, user_msg],
                            max_tokens=300,
                            response_format={"type": "json_object"},
                        )
                        text = resp.choices[0].message.content if resp.choices else "{}"
                        try:
                            data = json.loads(text)
                        except Exception:
                            start = text.find("{")
                            end = text.rfind("}")
                            data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {"results": []}
                        results = data.get("results", []) if isinstance(data, dict) else []
                        break
                    except Exception as e:  # rate limits, network, etc.
                        last_err = e
                        time.sleep(2 ** attempt)

            # Materialize results into edges after guards
            for item in results:
                src_id = str(item.get("lo_id", ""))
                score = float(item.get("score", 0.0))
                confidence = item.get("confidence", None)
                try:
                    confidence = float(confidence) if confidence is not None else None
                except Exception:
                    confidence = None
                rationale = str(item.get("rationale", ""))

                # POST-SCORE GUARD: Ensure chronological direction
                src_row = lo_lookup.get(src_id, {})
                target_row_dict = lo_lookup.get(str(target_id), {})
                if src_row and target_row_dict:
                    src_key = create_chronological_key(pd.Series(src_row))
                    target_key = create_chronological_key(pd.Series(target_row_dict))
                    if src_key >= target_key:
                        continue

                # Confidence gate
                if config.min_confidence is not None and confidence is not None:
                    if float(confidence) < float(config.min_confidence):
                        continue

                if score >= float(config.score_threshold):
                    rows.append(
                        {
                            "source_lo_id": src_id,
                            "target_lo_id": str(target_id),
                            "relation": "prerequisite",
                            "score": float(score),
                            "confidence": confidence,
                            "rationale": rationale or "LLM decision",
                            "modality": config.modality,
                            "run_id": config.model,
                        }
                    )

        # Progress logging
        len_after = len(rows)
        kept = max(0, len_after - len_before)
        log_prereq_progress(processed_groups, total_groups, kept, len_after, started_at)

    # Remove reciprocals, keeping only forward (chronological) direction as a safety net
    df = pd.DataFrame(rows)
    if not df.empty:
        # Attach chronological keys for sorting
        lo_key_map: Dict[str, Tuple[str, str, int, str]] = {}
        for lo_id, rec in lo_lookup.items():
            lo_key_map[str(lo_id)] = create_chronological_key(pd.Series(rec))

        def is_forward(a: str, b: str) -> bool:
            return lo_key_map.get(a, ("", "", 0, "")) < lo_key_map.get(b, ("", "", 0, ""))

        # Identify reciprocal pairs
        pair_set: Set[Tuple[str, str]] = set(zip(df["source_lo_id"].astype(str), df["target_lo_id"].astype(str)))
        to_drop: Set[Tuple[str, str]] = set()
        for a, b in list(pair_set):
            if (b, a) in pair_set:
                # keep only forward chronological edge
                if is_forward(a, b):
                    to_drop.add((b, a))
                else:
                    to_drop.add((a, b))

        if to_drop:
            mask = df.apply(lambda r: (str(r["source_lo_id"]), str(r["target_lo_id"])) not in to_drop, axis=1)
            df = df[mask].reset_index(drop=True)

    return df


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
    parser.add_argument("--limit", type=int, default=None, help="Limit number of target LOs for a smoke run")
    parser.add_argument("--mode", type=str, default="both", choices=["candidates", "score", "both"], help="Run candidate generation, scoring, or both")
    parser.add_argument("--threshold", type=float, default=None, help="Override score threshold (0-1)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config()
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
