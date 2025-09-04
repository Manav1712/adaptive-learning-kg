"""
Comprehensive Graph Quality Scorecard (offline)

Reads processed CSVs and produces a single JSON report with key KPIs for:
- Content → LO links (edges_content.csv)
- LO → LO prerequisites (edges_prereqs.csv)

Outputs:
- data/processed/graph_quality_report.json

Usage:
  python3 src/experiments_manual/evaluate_graph_quality.py \
    --edges-content data/processed/edges_content.csv \
    --edges-prereqs data/processed/edges_prereqs.csv \
    --lo-index data/processed/lo_index.csv \
    --content-items data/processed/content_items.csv \
    --out-json data/processed/graph_quality_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    """Reads a CSV file safely, raising FileNotFoundError if missing.
    
    Args:
        path: Path to CSV file
        
    Returns:
        pandas DataFrame with CSV contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


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


def _basic_stats(values: List[float]) -> Dict[str, float]:
    """Computes descriptive statistics for a list of numeric values.
    
    Args:
        values: List of float values to analyze
        
    Returns:
        Dictionary with count, min, max, mean, p25, p50, p75 percentiles
    """
    if not values:
        return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
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


def _duplicates(edges_df: pd.DataFrame) -> int:
    """Counts duplicate edges by (source, target) pairs.
    
    Args:
        edges_df: DataFrame with edges
        
    Returns:
        Number of duplicate edges (total edges - unique edges)
    """
    keys = _edge_keys(edges_df)
    return max(0, len(keys) - len(set(keys)))


def summarize_common(edges_df: pd.DataFrame) -> Dict[str, object]:
    """Summarizes common metrics across both content and prereq edges.
    
    Args:
        edges_df: DataFrame with edges (content or prereq schema)
        
    Returns:
        Dictionary with num_edges, score_stats, relations, modalities, run_ids counts
    """
    out: Dict[str, object] = {}
    out["num_edges"] = int(len(edges_df))
    if "score" in edges_df.columns:
        scores = [float(x) for x in edges_df["score"].tolist() if pd.notnull(x)]
        out["score_stats"] = _basic_stats(scores)
    else:
        out["score_stats"] = _basic_stats([])
    out["relations"] = (
        edges_df["relation"].astype(str).value_counts().sort_index().to_dict() if "relation" in edges_df.columns else {}
    )
    out["modalities"] = (
        edges_df["modality"].astype(str).value_counts().sort_index().to_dict() if "modality" in edges_df.columns else {}
    )
    out["run_ids"] = (
        edges_df["run_id"].astype(str).value_counts().sort_index().to_dict() if "run_id" in edges_df.columns else {}
    )
    return out


def integrity_content(edges_df: pd.DataFrame, lo_df: pd.DataFrame, content_df: pd.DataFrame) -> Dict[str, object]:
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
    content_with_edge = set(edges_df["target_content_id"].astype(str).tolist()) if "target_content_id" in edges_df.columns else set()
    coverage = float(len(content_with_edge)) / float(len(content_df)) if len(content_df) > 0 else 0.0
    return {
        "missing_source_lo_refs": int(missing_lo),
        "missing_target_content_refs": int(missing_content),
        "content_coverage": coverage,
        "num_content_items": int(len(content_df)),
        "num_content_with_edges": int(len(content_with_edge)),
    }


def integrity_prereqs(edges_df: pd.DataFrame, lo_df: pd.DataFrame) -> Dict[str, object]:
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
    targets_with_incoming = set(edges_df["target_lo_id"].astype(str).tolist()) if "target_lo_id" in edges_df.columns else set()
    coverage = float(len(targets_with_incoming)) / float(len(lo_df)) if len(lo_df) > 0 else 0.0
    return {
        "missing_source_lo_refs": int(missing_src),
        "missing_target_lo_refs": int(missing_tgt),
        "lo_incoming_coverage": coverage,
        "num_los": int(len(lo_df)),
        "num_los_with_incoming": int(len(targets_with_incoming)),
    }


def structural_prereqs(edges_df: pd.DataFrame) -> Dict[str, object]:
    """Analyzes structural properties of LO→LO prerequisite graph.
    
    Args:
        edges_df: Prerequisite edges DataFrame
        
    Returns:
        Dictionary with is_dag, num_cycles, longest_path_len, reciprocal_pairs
        Falls back to reciprocal pairs only if NetworkX unavailable
    """
    try:
        import networkx as nx  # type: ignore
    except Exception:
        # Compute only reciprocal pairs without NX
        keys = _edge_keys(edges_df)
        edge_set = set(keys)
        reciprocal = 0
        seen: Set[Tuple[str, str]] = set()
        for u, v in keys:
            if (v, u) in edge_set and (v, u) not in seen and (u, v) not in seen:
                reciprocal += 1
                seen.add((u, v))
                seen.add((v, u))
        return {"is_dag": None, "num_cycles": None, "longest_path_len": None, "reciprocal_pairs": reciprocal}

    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        u = str(r.get("source_lo_id", ""))
        v = str(r.get("target_lo_id", ""))
        if u and v:
            G.add_edge(u, v)

    is_dag = nx.is_directed_acyclic_graph(G)
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
        return {"is_dag": True, "num_cycles": 0, "longest_path_len": lp, "reciprocal_pairs": int(reciprocal)}
    else:
        try:
            cycles_iter = nx.simple_cycles(G)
            cap = 1000
            count = 0
            for _ in cycles_iter:
                count += 1
                if count >= cap:
                    break
        except Exception:
            count = None
        return {"is_dag": False, "num_cycles": count, "longest_path_len": None, "reciprocal_pairs": int(reciprocal)}


def curriculum_prereqs(edges_df: pd.DataFrame, lo_df: pd.DataFrame) -> Dict[str, float]:
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


def curriculum_content(edges_df: pd.DataFrame, lo_df: pd.DataFrame, content_df: pd.DataFrame) -> Dict[str, float]:
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


def _p95(values: List[int]) -> float:
    """Computes 95th percentile of a list of integers.
    
    Args:
        values: List of integer values
        
    Returns:
        95th percentile value as float
    """
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return float(s[idx])


def parsimony_prereqs(edges_df: pd.DataFrame) -> Dict[str, float]:
    """Measures parsimony and redundancy for LO→LO prerequisite edges.
    
    Args:
        edges_df: Prerequisite edges DataFrame
        
    Returns:
        Dictionary with duplicate_edges, redundancy_ratio, out_degree_p95, in_degree_p95
        Redundancy: edges (u,v) where 2-hop path u→w→v exists
    """
    duplicates = _duplicates(edges_df)
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
    redundant = 0
    for u, v in keys:
        mids = adj_out.get(u, set())
        if any((w in adj_in.get(v, set())) for w in mids if w != v and w != u):
            redundant += 1
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
    duplicates = _duplicates(edges_df)
    out_counts: Dict[str, int] = {}
    for _, r in edges_df.iterrows():
        lo = str(r.get("source_lo_id", ""))
        if lo:
            out_counts[lo] = out_counts.get(lo, 0) + 1
    return {
        "duplicate_edges": int(duplicates),
        "lo_out_degree_p95": _p95(list(out_counts.values())),
    }


def build_report(
    edges_content: Optional[pd.DataFrame],
    edges_prereqs: Optional[pd.DataFrame],
    lo_df: pd.DataFrame,
    content_df: pd.DataFrame,
) -> Dict[str, object]:
    """Builds comprehensive quality report for both edge types.
    
    Args:
        edges_content: Content→LO edges DataFrame (optional)
        edges_prereqs: LO→LO prerequisite edges DataFrame (optional)
        lo_df: LO index DataFrame
        content_df: Content items DataFrame
        
    Returns:
        Dictionary with content_links, prerequisites, and overall sections
        Each section contains common metrics, integrity, curriculum consistency, parsimony
    """
    report: Dict[str, object] = {}

    if edges_content is not None and not edges_content.empty:
        content_section: Dict[str, object] = {}
        content_section.update(summarize_common(edges_content))
        content_section["unique_sources"] = int(edges_content["source_lo_id"].astype(str).nunique()) if "source_lo_id" in edges_content.columns else 0
        content_section["unique_targets"] = int(edges_content["target_content_id"].astype(str).nunique()) if "target_content_id" in edges_content.columns else 0
        content_section["integrity"] = integrity_content(edges_content, lo_df, content_df)
        content_section["curriculum_consistency"] = curriculum_content(edges_content, lo_df, content_df)
        content_section["parsimony"] = parsimony_content(edges_content)
        report["content_links"] = content_section

    if edges_prereqs is not None and not edges_prereqs.empty:
        prereq_section: Dict[str, object] = {}
        prereq_section.update(summarize_common(edges_prereqs))
        prereq_section["unique_sources"] = int(edges_prereqs["source_lo_id"].astype(str).nunique()) if "source_lo_id" in edges_prereqs.columns else 0
        prereq_section["unique_targets"] = int(edges_prereqs["target_lo_id"].astype(str).nunique()) if "target_lo_id" in edges_prereqs.columns else 0
        prereq_section["integrity"] = integrity_prereqs(edges_prereqs, lo_df)
        prereq_section["structure"] = structural_prereqs(edges_prereqs)
        prereq_section["curriculum_consistency"] = curriculum_prereqs(edges_prereqs, lo_df)
        prereq_section["parsimony"] = parsimony_prereqs(edges_prereqs)
        report["prerequisites"] = prereq_section

    # Overall highlights
    overall: Dict[str, object] = {}
    if edges_content is not None and not edges_content.empty:
        overall["content_edges"] = int(len(edges_content))
    if edges_prereqs is not None and not edges_prereqs.empty:
        overall["prereq_edges"] = int(len(edges_prereqs))
    report["overall"] = overall
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for graph quality scorecard.
    
    Args:
        argv: Optional command line arguments for testing
        
    Returns:
        Exit code 0 on success
    """
    parser = argparse.ArgumentParser(description="Graph quality scorecard (offline)")
    parser.add_argument("--edges-content", default="data/processed/edges_content.csv", type=str)
    parser.add_argument("--edges-prereqs", default="data/processed/edges_prereqs.csv", type=str)
    parser.add_argument("--lo-index", default="data/processed/lo_index.csv", type=str)
    parser.add_argument("--content-items", default="data/processed/content_items.csv", type=str)
    parser.add_argument("--out-json", default="data/processed/graph_quality_report.json", type=str)
    args = parser.parse_args(argv)

    lo_df = _read_csv(args.lo_index)
    content_df = _read_csv(args.content_items)

    edges_content = pd.read_csv(args.edges_content) if os.path.exists(args.edges_content) else None
    edges_prereqs = pd.read_csv(args.edges_prereqs) if os.path.exists(args.edges_prereqs) else None

    report = build_report(edges_content, edges_prereqs, lo_df, content_df)

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


