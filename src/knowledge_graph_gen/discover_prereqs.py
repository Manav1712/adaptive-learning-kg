"""
Discover LO → LO prerequisite edges.

Reads lo_index.csv + content_items.csv from a run folder, then writes:
  intermediates/prereq_link_candidates.csv
  edges_prereqs.csv

Approach:
- Aggregate content per LO (text + images)
- Candidate pairs are chronologically forward only (earlier LO → later LO)
- LLM scores each pair; keep edges at or above the score threshold
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from dotenv import load_dotenv  # type: ignore
import pandas as pd
from openai import OpenAI  # type: ignore

load_dotenv()


@dataclass
class PrereqConfig:
    input_lo_index: str = ""
    input_content_items: str = ""
    output_candidates: str = ""
    output_edges: str = ""

    model: str = "gpt-4o-mini"
    modality: str = "multimodal"  # "text_only" | "multimodal"
    temperature: float = 0.0
    max_targets_per_call: int = 8  # candidate sources packed per API call
    max_response_tokens: int = 1500
    max_concurrency: int = 8
    image_detail: str = "low"
    max_retries: int = 3
    score_threshold: float = 0.7
    min_confidence: float = 0.6


def apply_run_dir(config: PrereqConfig, run_dir: str) -> None:
    """Wire all I/O paths to a versioned run folder."""
    config.input_lo_index = os.path.join(run_dir, "lo_index.csv")
    config.input_content_items = os.path.join(run_dir, "content_items.csv")
    config.output_candidates = os.path.join(run_dir, "intermediates", "prereq_link_candidates.csv")
    config.output_edges = os.path.join(run_dir, "edges_prereqs.csv")


# ----------------------------
# Utilities
# ----------------------------

def _as_int(val: object, default: Optional[int] = None) -> Optional[int]:
    """Best-effort int coercion for order columns."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return int(val)
    except Exception:
        return default


def create_chronological_key(row: pd.Series) -> Tuple[int, int, int, int, str]:
    """
    Curriculum ordering key for an LO.

    Prefers explicit order columns written by prepare_nodes.py:
      (book_order, chapter_order, unit_order, lo_order, lo_id)
    Falls back to lo_id alone if order columns are missing.
    """
    lo_id = str(row.get("lo_id") or "")
    book_order = _as_int(row.get("book_order"))
    chapter_order = _as_int(row.get("chapter_order"))
    unit_order = _as_int(row.get("unit_order"), 0) or 0
    lo_order = _as_int(row.get("lo_order"), 0) or 0

    if book_order is None or chapter_order is None:
        # Legacy fallback: sort by numeric lo_id (works within a book, not across).
        try:
            numeric_id = int(float(lo_id))
        except Exception as exc:
            raise ValueError(
                f"LO {lo_id} missing book_order/chapter_order; re-run prepare_nodes.py"
            ) from exc
        return (99, 99, 0, numeric_id, lo_id)

    return (book_order, chapter_order, unit_order, lo_order, lo_id)


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
    Sort by prepare_nodes order columns when present.
    """
    if limit is None or limit <= 0 or len(lo_meta) <= limit:
        return lo_meta

    tmp = lo_meta.copy()
    order_cols = [c for c in ["book_order", "chapter_order", "unit_order", "lo_order", "lo_id"] if c in tmp.columns]
    if len(order_cols) >= 2:
        tmp.sort_values(order_cols, inplace=True)
    else:
        tmp.sort_values(["book", "lo_id"], inplace=True)
    return tmp.head(limit)


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
    - lo_id, learning_objective, unit, chapter, book
    - book_order, chapter_order, unit_order, lo_order (when present on lo_index)
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
                "book_order": r.get("book_order"),
                "chapter_order": r.get("chapter_order"),
                "unit_order": r.get("unit_order"),
                "lo_order": r.get("lo_order"),
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
    - For each target LO B, consider all earlier sources A across chapters/units/books
    - Enforce chronologically forward pairs only (source precedes target) via `_chrono_key`
    - Candidate pool always spans the entire set (cross-chapter/unit/book)
    """
    rows: List[Dict[str, str]] = []

    # Attach chronological keys; skip rows that cannot be ordered.
    lo_meta = lo_meta.copy()
    broken_los: List[Tuple[str, str, str]] = []

    def _safe_key(r: pd.Series) -> Optional[Tuple[int, int, int, int, str]]:
        try:
            return create_chronological_key(r)
        except Exception:
            broken_los.append(
                (str(r.get("lo_id") or ""), str(r.get("unit") or ""), str(r.get("chapter") or ""))
            )
            return None

    lo_meta["_chrono_key"] = lo_meta.apply(_safe_key, axis=1)
    bad = int(lo_meta["_chrono_key"].isnull().sum())
    if bad:
        examples = list({t for t in broken_los})[:5]
        print(
            f"[prereq] Skipped {bad} LOs with missing order columns. "
            f"Re-run prepare_nodes.py. Examples: {examples}",
            flush=True,
        )
        lo_meta = lo_meta[lo_meta["_chrono_key"].notnull()].copy()

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
            "book": str(r.get("book") or ""),
            "lo_id": str(r.get("lo_id") or ""),
            "book_order": r.get("book_order"),
            "chapter_order": r.get("chapter_order"),
            "unit_order": r.get("unit_order"),
            "lo_order": r.get("lo_order"),
            "image_urls": list(r.get("image_urls") or []),
        }
        for _, r in lo_views.iterrows()
    }

    rows: List[Dict[str, object]] = []

    use_llm = (OpenAI is not None) and (os.environ.get("OPENAI_API_KEY") not in (None, ""))
    if not use_llm:
        raise RuntimeError("LLM scoring is required (heuristic disabled). Please set OPENAI_API_KEY.")

    client = OpenAI(timeout=120.0)

    def chunk_list(items: List[Tuple[str, str]], n: int) -> List[List[Tuple[str, str]]]:
        return [items[i : i + n] for i in range(0, len(items), n)]

    # Flatten to (target, chunk) tasks so API calls can be issued concurrently.
    tasks: List[Tuple[str, pd.Series, List[Tuple[str, str]]]] = []
    for target_id, group in candidates_df.groupby("target_lo_id"):
        target_row = lo_views[lo_views["lo_id"].astype(str) == str(target_id)].head(1)
        if target_row.empty:
            continue
        target_series = target_row.iloc[0]
        candidate_list = [(str(r["source_lo_id"]), str(r.get("reason") or "")) for _, r in group.iterrows()]
        for chunk in chunk_list(candidate_list, max(1, int(config.max_targets_per_call))):
            tasks.append((str(target_id), target_series, chunk))

    def score_chunk(
        task: Tuple[str, pd.Series, List[Tuple[str, str]]]
    ) -> Tuple[str, List[Dict[str, object]]]:
        target_id, target_series, chunk = task
        prompt = build_prompt_for_prereq(target_series, chunk, lo_lookup, config)
        system_msg = {"role": "system", "content": prompt["system"]}
        content_blocks: List[Dict[str, object]] = []
        for block in prompt["user"]:
            if block.get("type") == "text":
                content_blocks.append({"type": "text", "text": str(block.get("text", ""))})
            elif block.get("type") == "image_url":
                url = block.get("image_url")
                if isinstance(url, str):
                    image_url: Dict[str, object] = {"url": url}
                elif isinstance(url, dict):
                    image_url = dict(url)
                else:
                    continue
                # Low detail keeps diagram signal at a fraction of the image token cost.
                image_url.setdefault("detail", config.image_detail)
                content_blocks.append({"type": "image_url", "image_url": image_url})

        instruction = (
            "Respond ONLY with JSON in this schema: {\n"
            "  \"results\": [ { \"lo_id\": string, \"score\": number, \"confidence\": number, \"rationale\": string } ]\n"
            "} where score in [-1,1] and confidence in [0,1]. "
            "Return one entry per candidate and keep each rationale under 15 words."
        )
        content_blocks.append({"type": "text", "text": instruction})
        user_msg = {"role": "user", "content": content_blocks}

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
                    start = text.find("{")
                    end = text.rfind("}")
                    data = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {"results": []}
                return target_id, (data.get("results", []) if isinstance(data, dict) else [])
            except Exception as e:  # rate limits, network, truncated JSON, etc.
                last_err = e
                time.sleep(2 ** attempt)

        print(
            f"[prereq] target={target_id} chunk of {len(chunk)} failed after "
            f"{int(config.max_retries) + 1} attempts: {type(last_err).__name__}: {last_err}",
            flush=True,
        )
        return target_id, []

    total_tasks = len(tasks)
    processed_tasks = 0
    started_at = time.time()

    with ThreadPoolExecutor(max_workers=max(1, int(config.max_concurrency))) as pool:
        for target_id, results in pool.map(score_chunk, tasks):
            processed_tasks += 1
            len_before = len(rows)

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
            log_prereq_progress(processed_tasks, total_tasks, kept, len_after, started_at)

    # Remove reciprocals, keeping only forward (chronological) direction as a safety net
    df = pd.DataFrame(rows)
    if not df.empty:
        # Attach chronological keys for sorting
        lo_key_map: Dict[str, Tuple[int, int, int, int, str]] = {}
        for lo_id, rec in lo_lookup.items():
            lo_key_map[str(lo_id)] = create_chronological_key(pd.Series(rec))

        def is_forward(a: str, b: str) -> bool:
            return lo_key_map.get(a, (99, 99, 0, 0, "")) < lo_key_map.get(b, (99, 99, 0, 0, ""))

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
    parser.add_argument("--run-dir", required=True, help="Run folder with lo_index.csv and content_items.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of target LOs for a smoke run")
    parser.add_argument("--mode", type=str, default="both", choices=["candidates", "score", "both"])
    parser.add_argument("--threshold", type=float, default=None, help="Override score threshold (0-1)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = PrereqConfig()
    apply_run_dir(config, args.run_dir)
    if args.threshold is not None:
        config.score_threshold = float(args.threshold)

    lo_df = pd.read_csv(config.input_lo_index)
    content_df = pd.read_csv(config.input_content_items)
    lo_views = build_lo_views(lo_df, content_df)
    lo_meta = lo_views.copy()

    if args.limit is not None and args.limit > 0:
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
