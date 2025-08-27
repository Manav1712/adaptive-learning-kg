"""
Discover Content ↔ LO Links (Candidate Generation)

This module prepares candidate LO targets for each content item
to be later scored by an LLM. It reads processed inputs from the
prepare step and writes candidate pairs for downstream scoring.

Inputs (from prepare step):
- data/processed/lo_index.csv
- data/processed/content_items.csv

Output (candidates for LLM step):
- data/processed/content_link_candidates.csv

Candidate strategy:
- Start with LOs in the same unit and/or chapter as the content's parent LO
- Optionally add lexical shortlist by overlapping keywords with LO text

Note: This module does NOT call the LLM. It only prepares candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

import pandas as pd


# ----------------------------
# Configuration
# ----------------------------


@dataclass
class DiscoveryConfig:
    """
    Configuration for content→LO candidate generation.

    Fields:
    - input_lo_index: Path to LO index CSV
    - input_content_items: Path to content items CSV
    - output_candidates: Path to write candidate pairs CSV
    - restrict_same_unit: If True, only consider LOs in same unit
    - restrict_same_chapter: If True, only consider LOs in same chapter
    - lexical_top_k: Limit of lexical matches to add per content (0 to disable)
    - lexical_min_overlap: Minimum keyword overlap to consider as lexical match
    """

    input_lo_index: str = "data/processed/lo_index.csv"
    input_content_items: str = "data/processed/content_items.csv"
    output_candidates: str = "data/processed/content_link_candidates.csv"

    restrict_same_unit: bool = True
    restrict_same_chapter: bool = False

    lexical_top_k: int = 5
    lexical_min_overlap: int = 1

    # Relation mapping by content type
    relation_concept: str = "explained_by"
    relation_example: str = "exemplified_by"
    relation_try_it: str = "practiced_by"


def load_config(config_path: Optional[str]) -> DiscoveryConfig:
    """
    Loads candidate-generation configuration from YAML or returns defaults.

    Args:
        config_path: Path to config.yaml or None

    Returns:
        DiscoveryConfig populated from file or defaults

    Behavior:
        - Uses keys under output_paths/input_paths/pruning/relations if present
        - Falls back to defaults on missing file or missing YAML
    """
    if not config_path or not os.path.exists(config_path) or yaml is None:
        return DiscoveryConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    inputs = data.get("output_paths", {})  # prior config.yaml used output_paths for processed files
    # fallback to input_paths names as well (either is fine)
    inputs_alt = data.get("input_paths", {})
    pruning = data.get("pruning", {})
    relations = data.get("relations", {})
    return DiscoveryConfig(
        input_lo_index=inputs.get("lo_index", inputs_alt.get("lo_index", "data/processed/lo_index.csv")),
        input_content_items=inputs.get("content_items", inputs_alt.get("content_items", "data/processed/content_items.csv")),
        output_candidates=data.get("output_candidates", "data/processed/content_link_candidates.csv"),
        restrict_same_unit=bool(pruning.get("same_unit", True)),
        restrict_same_chapter=bool(pruning.get("same_chapter", False)),
        lexical_top_k=int(pruning.get("lexical_top_k", 5)),
        lexical_min_overlap=int(pruning.get("lexical_min_overlap", 1)),
        relation_concept=relations.get("concept", "explained_by"),
        relation_example=relations.get("example", "exemplified_by"),
        relation_try_it=relations.get("try_it", "practiced_by"),
    )


# ----------------------------
# Utilities
# ----------------------------


_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """
    Tokenizes text into lowercase keywords (simple heuristic).

    Args:
        text: Input string

    Returns:
        List of lowercase tokens (alphanumeric words)
    """
    if not isinstance(text, str) or not text:
        return []
    return [m.group(0).lower() for m in _WORD_PATTERN.finditer(text)]


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


# ----------------------------
# Candidate generation
# ----------------------------


def build_lo_metadata(lo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares LO metadata including tokenized learning objectives.

    Args:
        lo_df: DataFrame with columns lo_id, learning_objective, unit, chapter, book

    Returns:
        DataFrame with added 'lo_tokens' list column
    """
    lo_copy = lo_df.copy()
    lo_copy["learning_objective"] = lo_copy["learning_objective"].astype(str)
    lo_copy["lo_tokens"] = lo_copy["learning_objective"].map(tokenize)
    return lo_copy


def generate_candidates_for_row(
    content_row: pd.Series,
    lo_meta: pd.DataFrame,
    config: DiscoveryConfig,
) -> List[Tuple[str, str]]:
    """
    Generates candidate LO IDs for a single content row.

    Args:
        content_row: Row from content_items.csv
        lo_meta: LO metadata DataFrame with lo_tokens
        config: Candidate generation config

    Returns:
        List of tuples (candidate_lo_id, reason_tag)
        - reason_tag ∈ {"unit", "chapter", "lexical"}
    """
    candidates: List[Tuple[str, str]] = []

    parent_unit = str(content_row.get("unit") or "")
    parent_chapter = str(content_row.get("chapter") or "")
    content_text = str(content_row.get("text") or "")
    content_tokens: Set[str] = set(tokenize(content_text))

    # Same unit/chapter filtering
    pool = lo_meta
    if config.restrict_same_unit:
        pool = pool[pool["unit"].astype(str) == parent_unit]
    if config.restrict_same_chapter:
        pool = pool[pool["chapter"].astype(str) == parent_chapter]

    # Add unit/chapter matches
    for lo_id in pool["lo_id"].astype(str).tolist():
        candidates.append((lo_id, "unit" if config.restrict_same_unit else "chapter"))

    # Lexical shortlist across the full LO set to catch cross-unit links
    if config.lexical_top_k and len(content_tokens) > 0:
        overlaps: List[Tuple[str, int]] = []
        for _, lo_row in lo_meta.iterrows():
            lo_id = str(lo_row["lo_id"])  # type: ignore
            lo_tokens: Set[str] = set(lo_row.get("lo_tokens") or [])
            overlap = len(content_tokens.intersection(lo_tokens))
            if overlap >= config.lexical_min_overlap:
                overlaps.append((lo_id, overlap))
        # Sort by overlap desc and take top_k
        overlaps.sort(key=lambda t: t[1], reverse=True)
        for lo_id, _score in overlaps[: config.lexical_top_k]:
            candidates.append((lo_id, "lexical"))

    # Deduplicate keeping earliest reason
    seen: set = set()
    unique: List[Tuple[str, str]] = []
    for lo_id, reason in candidates:
        key = lo_id
        if key in seen:
            continue
        seen.add(key)
        unique.append((lo_id, reason))
    return unique


def write_candidates(
    content_df: pd.DataFrame,
    lo_meta: pd.DataFrame,
    config: DiscoveryConfig,
) -> pd.DataFrame:
    """
    Generates and writes candidate pairs for all content items.

    Args:
        content_df: DataFrame of content items
        lo_meta: DataFrame of LO metadata with tokens
        config: Config for candidate generation

    Returns:
        DataFrame of candidate edges written to CSV

    Behavior:
        - For each content item, proposes candidate LOs
        - Infers proposed relation from content_type
        - Writes CSV with columns: source_lo_id, target_content_id, proposed_relation, reason
    """
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


# ----------------------------
# Entrypoint
# ----------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    """
    CLI entrypoint to generate content→LO candidates.

    Args:
        argv: Optional CLI args for testing

    Returns:
        Exit code 0 on success
    """
    parser = argparse.ArgumentParser(description="Generate content→LO candidate pairs")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of content items for a smoke run")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    lo_df = pd.read_csv(config.input_lo_index)
    content_df = pd.read_csv(config.input_content_items)

    # Parse image_urls JSON column if present
    if "image_urls" in content_df.columns:
        def _safe_parse(s: str) -> List[str]:
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, list) else []
            except Exception:
                return []
        content_df["image_urls"] = content_df["image_urls"].astype(str).map(_safe_parse)

    if args.limit is not None and args.limit > 0:
        content_df = content_df.head(args.limit).copy()

    lo_meta = build_lo_metadata(lo_df)
    out_df = write_candidates(content_df, lo_meta, config)
    print(f"Wrote {config.output_candidates} ({len(out_df)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

