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
from typing import Dict, List, Optional, Tuple, Set

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
    """Reads a CSV file safely, raising FileNotFoundError if missing.
    
    Args:
        path: Path to CSV file
        
    Returns:
        pandas DataFrame with CSV contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def infer_edge_type(edges_df: pd.DataFrame) -> str:
    """Infers edge type based on column names.
    
    Args:
        edges_df: DataFrame with edge data
        
    Returns:
        "content" if has source_lo_id + target_content_id columns
        "prereqs" if has source_lo_id + target_lo_id columns
        
    Raises:
        ValueError: If schema doesn't match expected patterns
    """
    cols = set(c.lower() for c in edges_df.columns)
    if {"source_lo_id", "target_content_id"}.issubset(cols):
        return "content"
    if {"source_lo_id", "target_lo_id"}.issubset(cols):
        return "prereqs"
    raise ValueError("Unrecognized edges schema. Expected content or prereqs columns.")


def basic_stats(values: List[float]) -> Dict[str, float]:
    """Computes descriptive statistics for a list of numeric values.
    
    Args:
        values: List of float values to analyze
        
    Returns:
        Dictionary with count, min, max, mean, p25, p50, p75 percentiles
    """
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
    """Summarizes common metrics across both content and prereq edges.
    
    Args:
        edges_df: DataFrame with edges (content or prereq schema)
        cfg: Evaluation configuration with threshold
        
    Returns:
        Dictionary with num_edges, relations, run_ids, modalities counts,
        score_stats, and num_kept_ge_threshold
    """
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
    """Returns top-N highest-scoring edges per group (e.g., per relation).
    
    Args:
        edges_df: DataFrame with edges and score column
        cfg: Evaluation configuration with top_n setting
        group_field: Column name to group by (e.g., "relation")
        
    Returns:
        Dictionary mapping group values to lists of top edges with
        source, target, relation, score, and truncated rationale
    """
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


# ----------------------------
# Extended analysis helpers
# ----------------------------


def _edge_keys(edges_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Extracts (source, target) pairs from edges DataFrame.
    
    Args:
        edges_df: DataFrame with either content or prereq edge schema
        
    Returns:
        List of (source_id, target_id) tuples for all edges
    """
    cols = set(c.lower() for c in edges_df.columns)
    if {"source_lo_id", "target_content_id"}.issubset(cols):
        return [(str(r.get("source_lo_id", "")), str(r.get("target_content_id", ""))) for _, r in edges_df.iterrows()]
    if {"source_lo_id", "target_lo_id"}.issubset(cols):
        return [(str(r.get("source_lo_id", "")), str(r.get("target_lo_id", ""))) for _, r in edges_df.iterrows()]
    return []


def compute_duplicates(edges_df: pd.DataFrame) -> int:
    """Counts duplicate edges by (source, target) pairs.
    
    Args:
        edges_df: DataFrame with edges
        
    Returns:
        Number of duplicate edges (total edges - unique edges)
    """
    keys = _edge_keys(edges_df)
    return max(0, len(keys) - len(set(keys)))


def structural_metrics_prereqs(edges_df: pd.DataFrame) -> Dict[str, object]:
    """Analyzes structural properties of LO→LO prerequisite graph.
    
    Args:
        edges_df: Prerequisite edges DataFrame
        
    Returns:
        Dictionary with is_dag, num_cycles, longest_path_len, reciprocal_pairs
        Falls back to reciprocal pairs only if NetworkX unavailable
    """
    try:
        import networkx as nx  # type: ignore
    except Exception:  # Optional dependency
        # Fallback: compute only reciprocal pairs without building a graph
        keys = _edge_keys(edges_df)
        edge_set = set(keys)
        reciprocal = 0
        seen: Set[Tuple[str, str]] = set()
        for u, v in keys:
            if (v, u) in edge_set and (v, u) not in seen and (u, v) not in seen:
                reciprocal += 1
                seen.add((u, v))
                seen.add((v, u))
        return {
            "is_dag": None,
            "num_cycles": None,
            "longest_path_len": None,
            "reciprocal_pairs": reciprocal,
        }

    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        u = str(r.get("source_lo_id", ""))
        v = str(r.get("target_lo_id", ""))
        if u and v:
            G.add_edge(u, v)

    is_dag = nx.is_directed_acyclic_graph(G)
    # Count reciprocal pairs
    reciprocal = 0
    visited: Set[Tuple[str, str]] = set()
    for u, v in G.edges():
        if (v, u) in G.edges() and (u, v) not in visited and (v, u) not in visited:
            reciprocal += 1
            visited.add((u, v))
            visited.add((v, u))

    if is_dag:
        try:
            lp = nx.dag_longest_path_length(G)
        except Exception:
            lp = None
        return {
            "is_dag": True,
            "num_cycles": 0,
            "longest_path_len": lp,
            "reciprocal_pairs": int(reciprocal),
        }
    else:
        # Limit cycle enumeration for safety
        try:
            cycles_iter = nx.simple_cycles(G)
            # Count up to a cap to avoid huge outputs
            cap = 1000
            count = 0
            for _ in cycles_iter:
                count += 1
                if count >= cap:
                    break
        except Exception:
            count = None  # unknown
        return {
            "is_dag": False,
            "num_cycles": count,
            "longest_path_len": None,
            "reciprocal_pairs": int(reciprocal),
        }


def curriculum_consistency_prereqs(edges_df: pd.DataFrame, lo_df: pd.DataFrame) -> Dict[str, float]:
    """Measures curriculum consistency for LO→LO prerequisite edges.
    
    Args:
        edges_df: Prerequisite edges DataFrame
        lo_df: LO index DataFrame with unit/chapter metadata
        
    Returns:
        Dictionary with intra_unit_ratio, intra_chapter_ratio, total_considered
    """
    unit = lo_df.set_index("lo_id")["unit"].astype(str).to_dict()
    chapter = lo_df.set_index("lo_id")["chapter"].astype(str).to_dict()
    same_unit = 0
    same_chapter = 0
    total = 0
    for _, r in edges_df.iterrows():
        u = str(r.get("source_lo_id", ""))
        v = str(r.get("target_lo_id", ""))
        if not u or not v:
            continue
        total += 1
        if unit.get(u, "") == unit.get(v, ""):
            same_unit += 1
        if chapter.get(u, "") == chapter.get(v, ""):
            same_chapter += 1
    return {
        "intra_unit_ratio": (same_unit / total) if total else 0.0,
        "intra_chapter_ratio": (same_chapter / total) if total else 0.0,
        "total_considered": int(total),
    }


def curriculum_consistency_content(edges_df: pd.DataFrame, lo_df: pd.DataFrame, content_df: pd.DataFrame) -> Dict[str, float]:
    """Measures curriculum consistency for content→LO edges.
    
    Args:
        edges_df: Content edges DataFrame
        lo_df: LO index DataFrame with unit/chapter metadata
        content_df: Content items DataFrame with unit/chapter metadata
        
    Returns:
        Dictionary with intra_unit_ratio, intra_chapter_ratio, total_considered
    """
    lo_unit = lo_df.set_index("lo_id")["unit"].astype(str).to_dict()
    lo_chapter = lo_df.set_index("lo_id")["chapter"].astype(str).to_dict()
    c_unit = content_df.set_index("content_id")["unit"].astype(str).to_dict()
    c_chapter = content_df.set_index("content_id")["chapter"].astype(str).to_dict()

    same_unit = 0
    same_chapter = 0
    total = 0
    for _, r in edges_df.iterrows():
        lo_id = str(r.get("source_lo_id", ""))
        cid = str(r.get("target_content_id", ""))
        if not lo_id or not cid:
            continue
        total += 1
        if lo_unit.get(lo_id, "") == c_unit.get(cid, ""):
            same_unit += 1
        if lo_chapter.get(lo_id, "") == c_chapter.get(cid, ""):
            same_chapter += 1
    return {
        "intra_unit_ratio": (same_unit / total) if total else 0.0,
        "intra_chapter_ratio": (same_chapter / total) if total else 0.0,
        "total_considered": int(total),
    }


def parsimony_prereqs(edges_df: pd.DataFrame) -> Dict[str, float]:
    """Measures parsimony and redundancy for LO→LO prerequisite edges.
    
    Args:
        edges_df: Prerequisite edges DataFrame
        
    Returns:
        Dictionary with duplicate_edges, redundancy_ratio, out_degree_p95, in_degree_p95
        Redundancy: edges (u,v) where 2-hop path u→w→v exists
    """
    # Duplicates
    duplicates = compute_duplicates(edges_df)

    # Build adjacency
    adj_out: Dict[str, Set[str]] = {}
    adj_in: Dict[str, Set[str]] = {}
    keys = []
    for _, r in edges_df.iterrows():
        u = str(r.get("source_lo_id", ""))
        v = str(r.get("target_lo_id", ""))
        if not u or not v:
            continue
        keys.append((u, v))
        adj_out.setdefault(u, set()).add(v)
        adj_in.setdefault(v, set()).add(u)

    # Redundancy via 2-hop presence: u->v considered redundant if exists w with u->w and w->v
    redundant = 0
    for u, v in keys:
        mids = adj_out.get(u, set())
        if any((w in adj_in.get(v, set())) for w in mids if w != v and w != u):
            redundant += 1

    def _p95(values: List[int]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
        return float(s[idx])

    out_degrees = [len(vs) for vs in adj_out.values()]
    in_degrees = [len(vs) for vs in adj_in.values()]

    total_edges = len(keys)
    return {
        "duplicate_edges": int(duplicates),
        "redundancy_ratio": (redundant / total_edges) if total_edges else 0.0,
        "out_degree_p95": _p95(out_degrees),
        "in_degree_p95": _p95(in_degrees),
    }


def parsimony_content(edges_df: pd.DataFrame) -> Dict[str, float]:
    """Measures parsimony for content→LO edges.
    
    Args:
        edges_df: Content edges DataFrame
        
    Returns:
        Dictionary with duplicate_edges, lo_out_degree_p95 (LO fan-out to content)
    """
    duplicates = compute_duplicates(edges_df)
    # LO out-degree to content
    out_counts: Dict[str, int] = {}
    for _, r in edges_df.iterrows():
        lo = str(r.get("source_lo_id", ""))
        if lo:
            out_counts[lo] = out_counts.get(lo, 0) + 1

    def _p95(values: List[int]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
        return float(s[idx])

    return {
        "duplicate_edges": int(duplicates),
        "lo_out_degree_p95": _p95(list(out_counts.values())),
    }


def validate_references_content(edges_df: pd.DataFrame, lo_df: pd.DataFrame, content_df: pd.DataFrame) -> Dict[str, object]:
    """Validates referential integrity for content→LO edges and computes coverage.
    
    Args:
        edges_df: Content edges DataFrame
        lo_df: LO index DataFrame
        content_df: Content items DataFrame
        
    Returns:
        Dictionary with missing refs counts, content coverage ratio, and totals
    """
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
    """Validates referential integrity for LO→LO edges and computes incoming coverage.
    
    Args:
        edges_df: Prerequisite edges DataFrame
        lo_df: LO index DataFrame
        
    Returns:
        Dictionary with missing refs counts, LO incoming coverage ratio, and totals
    """
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
    """Evaluates the specified edges CSV and returns a comprehensive summary.
    
    Args:
        edges_path: Path to edges CSV file (content or prereqs)
        cfg: Evaluation configuration with thresholds and settings
        
    Returns:
        Dictionary with comprehensive evaluation metrics including:
        - Basic counts and statistics
        - Top edges by relation
        - Referential integrity checks
        - Curriculum consistency metrics
        - Parsimony analysis
        - Structural analysis (for prereqs only)
    """
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
        # Curriculum consistency and parsimony for content-links
        summary["curriculum_consistency"] = curriculum_consistency_content(edges_df, lo_df, content_df)
        summary["parsimony"] = parsimony_content(edges_df)
    else:  # prereqs
        summary.update(summarize_edges_common(edges_df, cfg))
        summary["top_edges_by_relation"] = summarize_top_edges(edges_df, cfg, group_field="relation")
        summary["integrity"] = validate_references_prereqs(edges_df, lo_df)
        # Structure, curriculum consistency, parsimony for LO→LO
        summary["structure"] = structural_metrics_prereqs(edges_df)
        summary["curriculum_consistency"] = curriculum_consistency_prereqs(edges_df, lo_df)
        summary["parsimony"] = parsimony_prereqs(edges_df)

    # Additional unique counts
    if kind == "content":
        summary["unique_sources"] = int(edges_df["source_lo_id"].astype(str).nunique()) if "source_lo_id" in edges_df.columns else 0
        summary["unique_targets"] = int(edges_df["target_content_id"].astype(str).nunique()) if "target_content_id" in edges_df.columns else 0
    else:
        summary["unique_sources"] = int(edges_df["source_lo_id"].astype(str).nunique()) if "source_lo_id" in edges_df.columns else 0
        summary["unique_targets"] = int(edges_df["target_lo_id"].astype(str).nunique()) if "target_lo_id" in edges_df.columns else 0

    return summary


def print_human_readable(summary: Dict[str, object]) -> None:
    """Prints a compact, human-readable summary to stdout.
    
    Args:
        summary: Evaluation summary dictionary from evaluate() function
    """
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
        cc = summary.get("curriculum_consistency", {}) or {}
        pars = summary.get("parsimony", {}) or {}
        print(
            "Curriculum -> "
            f"intra-unit {cc.get('intra_unit_ratio', 0.0):.3f}, "
            f"intra-chapter {cc.get('intra_chapter_ratio', 0.0):.3f}"
        )
        print(
            "Parsimony -> "
            f"duplicates {pars.get('duplicate_edges', 0)}, "
            f"LO out-degree p95 {pars.get('lo_out_degree_p95', 0.0):.1f}"
        )
    else:
        print(
            "Integrity -> "
            f"missing source LOs {integrity.get('missing_source_lo_refs', 0)}, "
            f"missing target LOs {integrity.get('missing_target_lo_refs', 0)}, "
            f"incoming coverage {integrity.get('lo_incoming_coverage', 0.0):.3f} "
            f"({integrity.get('num_los_with_incoming', 0)}/{integrity.get('num_los', 0)})"
        )
        struct = summary.get("structure", {}) or {}
        cc = summary.get("curriculum_consistency", {}) or {}
        pars = summary.get("parsimony", {}) or {}
        print(
            "Structure -> "
            f"is_dag {struct.get('is_dag')}, "
            f"cycles {struct.get('num_cycles')}, "
            f"longest_path {struct.get('longest_path_len')}" 
        )
        print(
            "Curriculum -> "
            f"intra-unit {cc.get('intra_unit_ratio', 0.0):.3f}, "
            f"intra-chapter {cc.get('intra_chapter_ratio', 0.0):.3f}"
        )
        print(
            "Parsimony -> "
            f"duplicates {pars.get('duplicate_edges', 0)}, "
            f"redundancy {pars.get('redundancy_ratio', 0.0):.3f}, "
            f"out-degree p95 {pars.get('out_degree_p95', 0.0):.1f}, "
            f"in-degree p95 {pars.get('in_degree_p95', 0.0):.1f}"
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
    """Main entry point for edge evaluation script.
    
    Args:
        argv: Optional command line arguments for testing
        
    Returns:
        Exit code 0 on success
    """
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



