"""
Utility helpers for loading the demo knowledge graph CSVs and deriving fast lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class KnowledgeGraphData:
    """
    Container that holds the raw DataFrames plus handy lookup tables.
    """

    los: pd.DataFrame
    content: pd.DataFrame
    edges_prereqs: pd.DataFrame
    edges_content: pd.DataFrame
    prereq_in_map: Dict[int, List[int]]
    content_ids_map: Dict[int, List[str]]
    lo_lookup: Dict[int, dict]


def load_demo_frames(base_dir: Path) -> KnowledgeGraphData:
    """
    Load demo CSV files and pre-compute adjacency maps.

    Inputs:
        base_dir: Directory that contains `lo_index.csv`, `content_items.csv`,
                  `edges_prereqs.csv`, and `edges_content.csv`.

    Outputs:
        KnowledgeGraphData with pandas DataFrames and helper lookups ready for retrieval.
    """

    los_df = pd.read_csv(base_dir / "lo_index.csv")
    content_df = pd.read_csv(base_dir / "content_items.csv")
    edges_prereqs_df = pd.read_csv(base_dir / "edges_prereqs.csv")
    edges_content_df = pd.read_csv(base_dir / "edges_content.csv")

    prereq_in_map = _build_prereq_in_map(edges_prereqs_df)
    content_ids_map = _build_content_ids_map(edges_content_df)

    lo_lookup = {
        int(row.lo_id): row._asdict()
        for row in los_df.itertuples(index=False)
    }

    return KnowledgeGraphData(
        los=los_df,
        content=content_df,
        edges_prereqs=edges_prereqs_df,
        edges_content=edges_content_df,
        prereq_in_map=prereq_in_map,
        content_ids_map=content_ids_map,
        lo_lookup=lo_lookup,
    )


def _build_prereq_in_map(edges_df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Build mapping of LO → prerequisite LOs.

    Inputs:
        edges_df: DataFrame with `source_lo_id` and `target_lo_id`.

    Outputs:
        Dict where keys are target LO ids and values are sorted lists of prerequisite ids.
    """

    prereq_map: Dict[int, List[int]] = {}
    if edges_df.empty:
        return prereq_map

    grouped = edges_df.groupby("target_lo_id")
    for target_id, group in grouped:
        prereq_map[int(target_id)] = [
            int(source) for source in group["source_lo_id"].tolist()
        ]
    return prereq_map


def _build_content_ids_map(edges_content_df: pd.DataFrame) -> Dict[int, List[str]]:
    """
    Build mapping of LO → associated content ids.

    Inputs:
        edges_content_df: DataFrame linking `source_lo_id` to `target_content_id`.

    Outputs:
        Dict where keys are LO ids and values are ordered content id lists.
    """

    content_map: Dict[int, List[str]] = {}
    if edges_content_df.empty:
        return content_map

    grouped = edges_content_df.groupby("source_lo_id")
    for source_lo_id, group in grouped:
        content_map[int(source_lo_id)] = group["target_content_id"].astype(str).tolist()
    return content_map

