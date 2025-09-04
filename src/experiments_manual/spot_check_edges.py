
"""
Spot Check Edges (Phase 4 - Human Labeling)

Provides lightweight utilities to sample a subset of generated edges for
human spot checks. Supports two workflows:

- Export mode (default): writes a CSV with sampled edges and rich context for
  humans to label asynchronously in a spreadsheet.
- Interactive mode: presents sampled edges one-by-one in the terminal to
  collect quick human labels and saves results.

Additionally, a summarization mode can read a labeled CSV and report
aggregate keep/discard/unclear rates and related statistics.

Usage examples:

  # Export 30 sampled content→LO edges for labeling
  python3 src/experiments_manual/spot_check_edges.py \
    --edges data/processed/edges_content.csv \
    --lo-index data/processed/lo_index.csv \
    --content-items data/processed/content_items.csv \
    --sample-size 30 \
    --out-csv data/processed/spot_check_content_sample.csv

  # Export 30 sampled LO→LO edges for labeling
  python3 src/experiments_manual/spot_check_edges.py \
    --edges data/processed/edges_prereqs.csv \
    --lo-index data/processed/lo_index.csv \
    --sample-size 30 \
    --out-csv data/processed/spot_check_prereqs_sample.csv

  # Interactive terminal labeling (writes labels CSV)
  python3 src/experiments_manual/spot_check_edges.py \
    --edges data/processed/edges_prereqs.csv \
    --lo-index data/processed/lo_index.csv \
    --mode interactive \
    --sample-size 10 \
    --labels-out data/processed/spot_check_prereqs_labels.csv

  # Summarize a labeled CSV
  python3 src/experiments_manual/spot_check_edges.py --summarize data/processed/spot_check_prereqs_labels.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd


def read_csv_required(path: str) -> pd.DataFrame:
    """Reads a CSV file and raises FileNotFoundError if missing.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with CSV contents
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def infer_edge_type(edges_df: pd.DataFrame) -> str:
    """Infers edge type from columns.

    Args:
        edges_df: Edges DataFrame

    Returns:
        "content" for LO→content edges; "prereqs" for LO→LO edges
    """
    cols = {c.lower() for c in edges_df.columns}
    if {"source_lo_id", "target_content_id"}.issubset(cols):
        return "content"
    if {"source_lo_id", "target_lo_id"}.issubset(cols):
        return "prereqs"
    raise ValueError("Unrecognized edges schema (expect content or prereqs columns)")


def sample_edges(edges_df: pd.DataFrame, sample_size: int, seed: Optional[int]) -> pd.DataFrame:
    """Returns a random sample of edges.

    Args:
        edges_df: Full edges DataFrame
        sample_size: Number of rows to sample (capped by DataFrame length)
        seed: Optional random seed for reproducibility

    Returns:
        DataFrame with up to sample_size rows
    """
    n = min(int(sample_size), int(len(edges_df)))
    if n <= 0:
        return edges_df.head(0).copy()
    return edges_df.sample(n=n, random_state=seed).reset_index(drop=True)


def enrich_for_content_edges(edges_df: pd.DataFrame, lo_df: pd.DataFrame, content_df: pd.DataFrame) -> pd.DataFrame:
    """Adds human-friendly context for content→LO edges.

    Joins LO metadata on source and content metadata on target for easier
    human review.

    Returns a DataFrame containing the original columns plus:
      - lo_objective, lo_unit, lo_chapter
      - content_title, content_text, content_type, content_unit, content_chapter
    """
    lo_meta = lo_df.rename(columns={
        "lo_id": "source_lo_id",
        "learning_objective": "lo_objective",
        "unit": "lo_unit",
        "chapter": "lo_chapter",
    })[["source_lo_id", "lo_objective", "lo_unit", "lo_chapter"]]

    content_meta = content_df.rename(columns={
        "content_id": "target_content_id",
        "learning_objective": "content_title",
        "text": "content_text",
        "content_type": "content_type",
        "unit": "content_unit",
        "chapter": "content_chapter",
    })[[
        "target_content_id",
        "content_title",
        "content_text",
        "content_type",
        "content_unit",
        "content_chapter",
    ]]

    out = edges_df.merge(lo_meta, on="source_lo_id", how="left")
    out = out.merge(content_meta, on="target_content_id", how="left")
    return out


def enrich_for_prereq_edges(edges_df: pd.DataFrame, lo_df: pd.DataFrame) -> pd.DataFrame:
    """Adds human-friendly context for LO→LO prerequisite edges.

    Joins LO metadata for both source and target LOs for easier human review.

    Returns a DataFrame containing the original columns plus:
      - src_objective, src_unit, src_chapter
      - tgt_objective, tgt_unit, tgt_chapter
    """
    src_meta = lo_df.rename(columns={
        "lo_id": "source_lo_id",
        "learning_objective": "src_objective",
        "unit": "src_unit",
        "chapter": "src_chapter",
    })[["source_lo_id", "src_objective", "src_unit", "src_chapter"]]

    tgt_meta = lo_df.rename(columns={
        "lo_id": "target_lo_id",
        "learning_objective": "tgt_objective",
        "unit": "tgt_unit",
        "chapter": "tgt_chapter",
    })[["target_lo_id", "tgt_objective", "tgt_unit", "tgt_chapter"]]

    out = edges_df.merge(src_meta, on="source_lo_id", how="left")
    out = out.merge(tgt_meta, on="target_lo_id", how="left")
    return out


def export_sample_to_csv(sample_df: pd.DataFrame, edge_type: str, out_csv: str) -> str:
    """Writes a CSV with context and empty labeling fields for human review.

    Args:
        sample_df: Enriched, sampled edges
        edge_type: "content" or "prereqs"
        out_csv: Destination CSV file path

    Returns:
        Absolute path to the written CSV
    """
    # Labeling scaffold
    sample_df = sample_df.copy()
    sample_df.insert(0, "label_decision", "")  # keep | discard | unclear
    sample_df.insert(1, "label_notes", "")
    sample_df.insert(2, "label_strength_1to5", "")  # optional numeric

    # Reorder a bit for readability
    preferred_first: List[str] = [
        "label_decision",
        "label_strength_1to5",
        "label_notes",
        "relation" if "relation" in sample_df.columns else None,
        "score" if "score" in sample_df.columns else None,
        "modality" if "modality" in sample_df.columns else None,
        "run_id" if "run_id" in sample_df.columns else None,
    ]
    preferred_first = [c for c in preferred_first if c is not None]  # type: ignore
    remaining_cols = [c for c in sample_df.columns if c not in preferred_first]
    ordered_cols = preferred_first + remaining_cols
    sample_df = sample_df[ordered_cols]

    out_path = os.path.abspath(out_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sample_df.to_csv(out_path, index=False)
    return out_path


def interactive_labeling(sample_df: pd.DataFrame, edge_type: str) -> pd.DataFrame:
    """Runs a simple terminal-based labeling loop over the sampled edges.

    Args:
        sample_df: Enriched, sampled edges
        edge_type: "content" or "prereqs"

    Returns:
        DataFrame with three new columns: label_decision, label_strength_1to5, label_notes
    """
    rows: List[Dict[str, str]] = []
    for _, r in sample_df.iterrows():
        print("\n=========================")
        if edge_type == "content":
            print(f"LO [{r.get('source_lo_id','')}] {r.get('lo_objective','')}")
            print(f"Content [{r.get('target_content_id','')}] {r.get('content_title','')} | type={r.get('content_type','')}")
            txt = str(r.get("content_text", "") or "").strip().replace("\n", " ")
            print(f"Content text: {txt[:500]}")
            print(f"Unit/Chapter: LO({r.get('lo_unit','')}/{r.get('lo_chapter','')}), Content({r.get('content_unit','')}/{r.get('content_chapter','')})")
        else:
            print(f"Source LO [{r.get('source_lo_id','')}] {r.get('src_objective','')}")
            print(f"Target LO [{r.get('target_lo_id','')}] {r.get('tgt_objective','')}")
            print(f"Unit/Chapter: Src({r.get('src_unit','')}/{r.get('src_chapter','')}), Tgt({r.get('tgt_unit','')}/{r.get('tgt_chapter','')})")
        print(f"Relation: {r.get('relation','')} | Score: {r.get('score','')} | Modality: {r.get('modality','')}")
        rationale = str(r.get("rationale", "") or "").strip().replace("\n", " ")
        if rationale:
            print(f"Rationale: {rationale[:500]}")

        decision = input("Label [keep/discard/unclear]: ").strip().lower()
        if decision not in {"keep", "discard", "unclear"}:
            decision = "unclear"
        strength = input("Strength (1-5, optional): ").strip()
        notes = input("Notes (optional): ").strip()

        rr = dict(r)
        rr["label_decision"] = decision
        rr["label_strength_1to5"] = strength
        rr["label_notes"] = notes
        rows.append(rr)

    return pd.DataFrame(rows)


def summarize_labeled_csv(labeled_csv: str) -> Dict[str, object]:
    """Reads a labeled CSV and returns aggregate summary stats.

    Args:
        labeled_csv: Path to CSV produced by export or interactive modes

    Returns:
        Dictionary with counts and proportions per label_decision and
        simple score summaries by label bucket (if score present)
    """
    df = read_csv_required(labeled_csv)
    out: Dict[str, object] = {}

    # Decision distribution
    if "label_decision" in df.columns:
        counts = df["label_decision"].astype(str).str.lower().value_counts().to_dict()
        total = int(sum(counts.values())) if counts else 0
        props = {k: (v / total if total else 0.0) for k, v in counts.items()}
        out["decision_counts"] = counts
        out["decision_proportions"] = props

    # Score by decision
    if "score" in df.columns and "label_decision" in df.columns:
        try:
            df2 = df.copy()
            df2["score"] = pd.to_numeric(df2["score"], errors="coerce")
            df2 = df2.dropna(subset=["score"])
            stats: Dict[str, Dict[str, float]] = {}
            for dec, g in df2.groupby(df2["label_decision"].astype(str).str.lower()):
                s = sorted(g["score"].tolist())
                n = len(s)
                if n == 0:
                    continue
                def pct(p: float) -> float:
                    if n == 1:
                        return float(s[0])
                    idx = max(0, min(n - 1, int(round(p * (n - 1)))))
                    return float(s[idx])
                stats[dec] = {
                    "count": float(n),
                    "min": float(s[0]),
                    "max": float(s[-1]),
                    "mean": float(sum(s) / n),
                    "p25": pct(0.25),
                    "p50": pct(0.50),
                    "p75": pct(0.75),
                }
            out["score_stats_by_decision"] = stats
        except Exception:
            pass

    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for sampling, labeling, and summarizing spot checks.

    Typical flows:
      1) Export sample for spreadsheet labeling (default mode)
      2) Interactive terminal labeling (--mode interactive)
      3) Summarize a labeled CSV (--summarize path)
    """
    parser = argparse.ArgumentParser(description="Spot-check edges for human labeling")
    parser.add_argument("--edges", type=str, help="Path to edges CSV (content or prereqs)")
    parser.add_argument("--lo-index", default="data/processed/lo_index.csv", type=str)
    parser.add_argument("--content-items", default="data/processed/content_items.csv", type=str)
    parser.add_argument("--edge-type", choices=["content", "prereqs"], default=None, help="Optional override")
    parser.add_argument("--sample-size", default=30, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--mode", choices=["export", "interactive"], default="export")
    parser.add_argument("--out-csv", default="data/processed/spot_check_sample.csv", type=str, help="For export mode")
    parser.add_argument("--labels-out", default="data/processed/spot_check_labels.csv", type=str, help="For interactive mode")
    parser.add_argument("--summarize", default=None, type=str, help="Summarize a labeled CSV and exit")
    args = parser.parse_args(argv)

    if args.summarize:
        summary = summarize_labeled_csv(args.summarize)
        print(summary)
        return 0

    if not args.edges:
        raise SystemExit("--edges is required unless using --summarize")

    edges_df = read_csv_required(args.edges)
    lo_df = read_csv_required(args.lo_index)

    # Edge type inference/override
    edge_type = args.edge_type or infer_edge_type(edges_df)

    # Enrich with context
    if edge_type == "content":
        content_df = read_csv_required(args.content_items)
        enriched = enrich_for_content_edges(edges_df, lo_df, content_df)
    else:
        enriched = enrich_for_prereq_edges(edges_df, lo_df)

    # Sample
    sample_df = sample_edges(enriched, args.sample_size, args.seed)

    if args.mode == "interactive":
        labeled_df = interactive_labeling(sample_df, edge_type)
        out_path = os.path.abspath(args.labels_out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        labeled_df.to_csv(out_path, index=False)
        print(f"Wrote labels -> {out_path}")
    else:
        out_path = export_sample_to_csv(sample_df, edge_type, args.out_csv)
        print(f"Wrote sample -> {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


