"""
Evaluate Generated Edges (Content Links and LO→LO Prerequisites)

This utility summarizes edges produced by scripts in experiments_manual/:
- data/processed/edges_content.csv (content→LO links)
- data/processed/edges_prereqs.csv (LO→LO prerequisites)

It computes simple metrics, validates referential integrity against inputs,
and can export a machine-readable JSON summary.

CLI usage examples:
- python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_content.csv
- python3 src/experiments_manual/evaluate_outputs.py --edges data/processed/edges_prereqs.csv
- python3 src/experiments_manual/evaluate_outputs.py \
    --edges data/processed/edges_content.csv \
    --threshold 0.7 --top-n 5 --json-out data/processed/eval_content_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Configuration (paths + params)
# ----------------------------


@dataclass
class EvalConfig:
    """Configuration for evaluating generated edges.

    Inputs:
    - edges: Path to edges CSV (content or prereqs)
    - lo_index: Path to LO index CSV
    - content_items: Path to content items CSV (for content-link evaluation)

    Params:
    - threshold: Score threshold for counting "kept" edges
    - top_n: Top-N highest-scoring edges to display per relation
    - json_out: Optional path to write JSON summary
    """

    edges: str
    lo_index: str = "data/processed/lo_index.csv"
    content_items: str = "data/processed/content_items.csv"
    threshold: float = 0.6
    top_n: int = 5
    json_out: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------


def read_csv_safely(path: str) -> pd.DataFrame:
    """Reads a CSV if available; raises FileNotFoundError otherwise."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def infer_edge_type(edges_df: pd.DataFrame) -> str:
    """Infers edge type based on column names: 'content' or 'prereqs'."""
    cols = set(c.lower() for c in edges_df.columns)
    if {"source_lo_id", "target_content_id"}.issubset(cols):
        return "content"
    if {"source_lo_id", "target_lo_id"}.issubset(cols):
        return "prereqs"
    raise ValueError("Unrecognized edges schema. Expected content or prereqs columns.")


def basic_stats(values: List[float]) -> Dict[str, float]:
    """Computes simple descriptive statistics for a list of numbers."""
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    s = sorted(values)
    n = len(s)
    def pct(p: float) -> float:
        if n == 1:
            return float(s[0])
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(s[idx])
    return {
        "count": float(n),
        "min": float(s[0]),
        "max": float(s[-1]),
        "mean": float(sum(s) / n),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
    }


def summarize_edges_common(edges_df: pd.DataFrame, cfg: EvalConfig) -> Dict[str, object]:
    """Summarizes fields common to both content and prereqs edges."""
    out: Dict[str, object] = {}
    out["num_edges"] = int(len(edges_df))
    out["relations"] = (
        edges_df["relation"].astype(str).value_counts().sort_index().to_dict() if "relation" in edges_df.columns else {}
    )
    out["run_ids"] = (
        edges_df["run_id"].astype(str).value_counts().sort_index().to_dict() if "run_id" in edges_df.columns else {}
    )
    out["modalities"] = (
        edges_df["modality"].astype(str).value_counts().sort_index().to_dict() if "modality" in edges_df.columns else {}
    )
    if "score" in edges_df.columns:
        scores = [float(x) for x in edges_df["score"].tolist() if pd.notnull(x)]
        out["score_stats"] = basic_stats(scores)
        out["num_kept_ge_threshold"] = int(sum(1 for x in scores if x >= float(cfg.threshold)))
    else:
        out["score_stats"] = basic_stats([])
        out["num_kept_ge_threshold"] = 0
    return out


def summarize_top_edges(edges_df: pd.DataFrame, cfg: EvalConfig, group_field: str) -> Dict[str, List[Dict[str, object]]]:
    """Returns top-N edges per group (e.g., per relation) sorted by score desc."""
    results: Dict[str, List[Dict[str, object]]] = {}
    if "score" not in edges_df.columns:
        return results
    groups = edges_df.groupby(group_field) if group_field in edges_df.columns else [("all", edges_df)]
    for key, g in groups:
        g2 = g.copy()
        try:
            g2["score"] = g2["score"].astype(float)
        except Exception:
            continue
        g2 = g2.sort_values("score", ascending=False).head(max(0, int(cfg.top_n)))
        # Trim long rationales for readability
        rows: List[Dict[str, object]] = []
        for _, r in g2.iterrows():
            item = {
                "source": str(r.get("source_lo_id", "")),
                "target": str(r.get("target_content_id", r.get("target_lo_id", ""))),
                "relation": str(r.get("relation", "")),
                "score": float(r.get("score", 0.0)),
                "rationale": str(r.get("rationale", ""))[:280],
            }
            rows.append(item)
        results[str(key)] = rows
    return results


def validate_references_content(edges_df: pd.DataFrame, lo_df: pd.DataFrame, content_df: pd.DataFrame) -> Dict[str, object]:
    """Validates that content-link edges reference existing LO and content ids."""
    missing_lo = 0
    missing_content = 0
    lo_ids = set(lo_df["lo_id"].astype(str).tolist())
    content_ids = set(content_df["content_id"].astype(str).tolist())
    for _, r in edges_df.iterrows():
        if str(r.get("source_lo_id", "")) not in lo_ids:
            missing_lo += 1
        if str(r.get("target_content_id", "")) not in content_ids:
            missing_content += 1
    # coverage: proportion of content items that received >=1 edge
    content_with_edge = set(edges_df["target_content_id"].astype(str).tolist()) if "target_content_id" in edges_df.columns else set()
    coverage = 0.0
    if len(content_df) > 0:
        coverage = float(len(content_with_edge)) / float(len(content_df))
    return {
        "missing_source_lo_refs": int(missing_lo),
        "missing_target_content_refs": int(missing_content),
        "content_coverage": coverage,
        "num_content_items": int(len(content_df)),
        "num_content_with_edges": int(len(content_with_edge)),
    }


def validate_references_prereqs(edges_df: pd.DataFrame, lo_df: pd.DataFrame) -> Dict[str, object]:
    """Validates that LO→LO edges reference existing LO ids and computes coverage."""
    missing_src = 0
    missing_tgt = 0
    lo_ids = set(lo_df["lo_id"].astype(str).tolist())
    for _, r in edges_df.iterrows():
        if str(r.get("source_lo_id", "")) not in lo_ids:
            missing_src += 1
        if str(r.get("target_lo_id", "")) not in lo_ids:
            missing_tgt += 1
    # coverage: proportion of LOs that have >=1 incoming prereq
    targets_with_incoming = set(edges_df["target_lo_id"].astype(str).tolist()) if "target_lo_id" in edges_df.columns else set()
    coverage = 0.0
    if len(lo_df) > 0:
        coverage = float(len(targets_with_incoming)) / float(len(lo_df))
    return {
        "missing_source_lo_refs": int(missing_src),
        "missing_target_lo_refs": int(missing_tgt),
        "lo_incoming_coverage": coverage,
        "num_los": int(len(lo_df)),
        "num_los_with_incoming": int(len(targets_with_incoming)),
    }


def evaluate(edges_path: str, cfg: EvalConfig) -> Dict[str, object]:
    """Evaluates the specified edges CSV and returns a summary dict."""
    edges_df = read_csv_safely(edges_path)
    kind = infer_edge_type(edges_df)
    lo_df = read_csv_safely(cfg.lo_index)

    summary: Dict[str, object] = {
        "edges_path": edges_path,
        "edge_type": kind,
        "threshold": float(cfg.threshold),
    }

    if kind == "content":
        content_df = read_csv_safely(cfg.content_items)
        summary.update(summarize_edges_common(edges_df, cfg))
        summary["top_edges_by_relation"] = summarize_top_edges(edges_df, cfg, group_field="relation")
        summary["integrity"] = validate_references_content(edges_df, lo_df, content_df)
    else:  # prereqs
        summary.update(summarize_edges_common(edges_df, cfg))
        summary["top_edges_by_relation"] = summarize_top_edges(edges_df, cfg, group_field="relation")
        summary["integrity"] = validate_references_prereqs(edges_df, lo_df)

    # Additional unique counts
    if kind == "content":
        summary["unique_sources"] = int(edges_df["source_lo_id"].astype(str).nunique()) if "source_lo_id" in edges_df.columns else 0
        summary["unique_targets"] = int(edges_df["target_content_id"].astype(str).nunique()) if "target_content_id" in edges_df.columns else 0
    else:
        summary["unique_sources"] = int(edges_df["source_lo_id"].astype(str).nunique()) if "source_lo_id" in edges_df.columns else 0
        summary["unique_targets"] = int(edges_df["target_lo_id"].astype(str).nunique()) if "target_lo_id" in edges_df.columns else 0

    return summary


def print_human_readable(summary: Dict[str, object]) -> None:
    """Prints a compact, human-readable summary to stdout."""
    print(f"Edges: {summary.get('edges_path')} ({summary.get('edge_type')})")
    print(f"Total edges: {summary.get('num_edges')} | Unique sources: {summary.get('unique_sources')} | Unique targets: {summary.get('unique_targets')}")
    score_stats = summary.get("score_stats", {}) or {}
    if score_stats:
        print(
            "Score stats -> "
            f"min {score_stats.get('min'):.3f}, p25 {score_stats.get('p25'):.3f}, p50 {score_stats.get('p50'):.3f}, "
            f"p75 {score_stats.get('p75'):.3f}, max {score_stats.get('max'):.3f}, mean {score_stats.get('mean'):.3f}"
        )
    print(f"Kept (score >= {summary.get('threshold')}): {summary.get('num_kept_ge_threshold')}")
    relations = summary.get("relations", {}) or {}
    if relations:
        rel_txt = ", ".join(f"{k}={v}" for k, v in sorted(relations.items()))
        print(f"Relations: {rel_txt}")
    run_ids = summary.get("run_ids", {}) or {}
    if run_ids:
        run_txt = ", ".join(f"{k}={v}" for k, v in sorted(run_ids.items()))
        print(f"Run IDs: {run_txt}")
    modalities = summary.get("modalities", {}) or {}
    if modalities:
        mod_txt = ", ".join(f"{k}={v}" for k, v in sorted(modalities.items()))
        print(f"Modalities: {mod_txt}")

    integrity = summary.get("integrity", {}) or {}
    if summary.get("edge_type") == "content":
        print(
            "Integrity -> "
            f"missing source LOs {integrity.get('missing_source_lo_refs', 0)}, "
            f"missing content {integrity.get('missing_target_content_refs', 0)}, "
            f"content coverage {integrity.get('content_coverage', 0.0):.3f} "
            f"({integrity.get('num_content_with_edges', 0)}/{integrity.get('num_content_items', 0)})"
        )
    else:
        print(
            "Integrity -> "
            f"missing source LOs {integrity.get('missing_source_lo_refs', 0)}, "
            f"missing target LOs {integrity.get('missing_target_lo_refs', 0)}, "
            f"incoming coverage {integrity.get('lo_incoming_coverage', 0.0):.3f} "
            f"({integrity.get('num_los_with_incoming', 0)}/{integrity.get('num_los', 0)})"
        )

    # Top edges (by relation)
    print("Top edges by relation:")
    top = summary.get("top_edges_by_relation", {}) or {}
    if not top:
        print("  (no score data)")
    for rel, rows in sorted(top.items()):
        print(f"  {rel}:")
        for item in rows:
            src = item.get("source", "")
            tgt = item.get("target", "")
            sc = item.get("score", 0.0)
            rationale = (item.get("rationale", "") or "").replace("\n", " ")
            print(f"    {src} -> {tgt} | {sc:.3f} | {rationale}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate generated edges (content or prereqs)")
    parser.add_argument("--edges", required=True, type=str, help="Path to edges CSV to evaluate")
    parser.add_argument("--lo-index", default="data/processed/lo_index.csv", type=str, help="Path to lo_index.csv")
    parser.add_argument("--content-items", default="data/processed/content_items.csv", type=str, help="Path to content_items.csv (for content-links)")
    parser.add_argument("--threshold", default=0.6, type=float, help="Score threshold for kept counts")
    parser.add_argument("--top-n", default=5, type=int, help="Top-N edges to display per relation")
    parser.add_argument("--json-out", default=None, type=str, help="Optional path to write JSON summary")
    args = parser.parse_args(argv)

    cfg = EvalConfig(
        edges=args.edges,
        lo_index=args.lo_index,
        content_items=args.content_items,
        threshold=float(args.threshold),
        top_n=int(args.top_n),
        json_out=args.json_out,
    )

    summary = evaluate(cfg.edges, cfg)
    print_human_readable(summary)

    if cfg.json_out:
        directory = os.path.dirname(os.path.abspath(cfg.json_out))
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(cfg.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote JSON summary -> {cfg.json_out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



