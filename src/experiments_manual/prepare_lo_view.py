"""
Prepare LO Content View

Reads raw content CSVs (concepts, examples, try_its), normalizes into:
- data/processed/lo_index.csv
- data/processed/content_items.csv

Behaviors:
- Parses raw_content as JSON when possible; otherwise treats as plain text
- Extracts text and image URLs (from markdown and plain links)
- Generates stable content_id per (lo_id, content_type, sequence)
- Supports optional sampling via config.yaml if present
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

import pandas as pd


# ----------------------------
# Configuration
# ----------------------------


@dataclass
class PrepareConfig:
    concept_glob: str = "data/raw/concept*.csv"
    example_glob: str = "data/raw/example*.csv"
    try_it_glob: str = "data/raw/try_it*.csv"

    output_lo_index: str = "data/processed/lo_index.csv"
    output_content_items: str = "data/processed/content_items.csv"

    sample_max_los: Optional[int] = None
    sample_max_content_per_lo: Optional[int] = None


def load_config(config_path: Optional[str]) -> PrepareConfig:
    
    """
    Loads configuration from a YAML file or returns defaults.
    
    Args:
        config_path: Path to config.yaml file, or None for defaults
        
    Returns:
        PrepareConfig instance with loaded or default values
        
    Behavior:
        - Falls back to defaults if file doesn't exist or YAML unavailable
        - Maps config keys to PrepareConfig fields
    """

    if not config_path:
        return PrepareConfig()
    if not os.path.exists(config_path):
        return PrepareConfig()
    if yaml is None:
        return PrepareConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Navigate expected keys if present
    inputs = (data.get("input_paths") or {}) if isinstance(data, dict) else {}
    outputs = (data.get("output_paths") or {}) if isinstance(data, dict) else {}
    sample = (data.get("sample") or {}) if isinstance(data, dict) else {}
    return PrepareConfig(
        concept_glob=inputs.get("concept", "data/raw/concept*.csv"),
        example_glob=inputs.get("example", "data/raw/example*.csv"),
        try_it_glob=inputs.get("try_it", "data/raw/try_it*.csv"),
        output_lo_index=outputs.get("lo_index", "data/processed/lo_index.csv"),
        output_content_items=outputs.get("content_items", "data/processed/content_items.csv"),
        sample_max_los=sample.get("max_los"),
        sample_max_content_per_lo=sample.get("max_content_per_lo"),
    )


# ----------------------------
# Utilities
# ----------------------------


_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_URL_PATTERN = re.compile(r"https?://[^\s)]+")


def extract_image_urls_from_text(text: str) -> List[str]:
    
    """
    Extracts image URLs from text content.
    
    Args:
        text: Raw text that may contain markdown images or plain URLs
        
    Returns:
        List of unique image URLs found in the text
        
    Behavior:
        - Looks for markdown format: ![alt](url)
        - Also finds plain URLs starting with http/https
        - Removes duplicates while preserving order
    """

    if not text:
        return []
    urls: List[str] = []
    urls.extend(_MARKDOWN_IMAGE_PATTERN.findall(text))
    urls.extend(_URL_PATTERN.findall(text))
    # Deduplicate preserving order
    seen: set = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def strip_markdown_images(text: str) -> str:
    """
    Removes markdown image syntax from text.
    
    Args:
        text: Text that may contain markdown images
        
    Returns:
        Text with markdown image syntax removed
        
    Behavior:
        - Removes patterns like ![alt](url) from text
        - Preserves the rest of the text content
    """
    if not text:
        return text
    return _MARKDOWN_IMAGE_PATTERN.sub("", text)


def try_parse_json(text: str) -> Optional[object]:
    
    """
    Attempts to parse text as JSON with graceful fallback.
    
    Args:
        text: String that may contain JSON data
        
    Returns:
        Parsed JSON object if successful, None if parsing fails
        
    Behavior:
        - Returns None for invalid JSON instead of raising exceptions
        - Useful for handling mixed content types
    """

    try:
        return json.loads(text)
    except Exception:
        return None


def collect_strings_recursively(node: object) -> List[str]:
    
    """
    Recursively extracts all string values from nested data structures.
    
    Args:
        node: Object that may be a string, dict, list, or other type
        
    Returns:
        List of all string values found in the nested structure
        
    Behavior:
        - Traverses nested dictionaries and lists
        - Collects all string values into a flat list
        - Handles arbitrary nesting depth
    """

    collected: List[str] = []
    if isinstance(node, str):
        collected.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            collected.extend(collect_strings_recursively(value))
    elif isinstance(node, list):
        for item in node:
            collected.extend(collect_strings_recursively(item))
    return collected


def normalize_row_text_and_images(raw_content: str) -> Tuple[str, List[str]]:
    """
    Normalizes raw content by extracting clean text and image URLs.
    
    Args:
        raw_content: Raw content string that may be JSON or plain text
        
    Returns:
        Tuple of (cleaned_text, image_urls_list)
        
    Behavior:
        - First tries to parse as JSON and extract strings recursively
        - Falls back to plain text if JSON parsing fails
        - Extracts image URLs from both formats
        - Removes markdown image syntax from final text
    """
    # Attempt JSON parse first
    parsed = try_parse_json(raw_content)
    if parsed is not None:
        strings = collect_strings_recursively(parsed)
        joined_text = "\n\n".join(s for s in strings if isinstance(s, str))
        image_urls = extract_image_urls_from_text(joined_text)
        cleaned_text = strip_markdown_images(joined_text)
        return cleaned_text.strip(), image_urls
    # Fallback: plain text
    image_urls = extract_image_urls_from_text(raw_content)
    cleaned_text = strip_markdown_images(raw_content)
    return cleaned_text.strip(), image_urls


def ensure_parent_directory(path: str) -> None:
    """
    Creates parent directories for a file path if they don't exist.
    
    Args:
        path: File path whose parent directories should be created
        
    Returns:
        None
        
    Behavior:
        - Creates all necessary parent directories
        - Uses exist_ok=True to avoid errors if directories already exist
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def read_csvs_by_glob(glob_pattern: str) -> List[pd.DataFrame]:
    """
    Reads multiple CSV files matching a glob pattern.
    
    Args:
        glob_pattern: Glob pattern to match CSV files (e.g., "data/raw/*.csv")
        
    Returns:
        List of pandas DataFrames, one per CSV file
        
    Behavior:
        - Finds all matching files and reads them as CSVs
        - Skips unreadable files gracefully
        - Returns empty list if no files found
    """
    import glob

    file_paths = sorted(glob.glob(glob_pattern))
    dataframes: List[pd.DataFrame] = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
        except Exception:
            # Skip unreadable files
            continue
    return dataframes


def load_raw_frames(config: PrepareConfig) -> pd.DataFrame:
    """
    Loads and unifies raw CSV data from multiple content types.
    
    Args:
        config: Configuration object with file glob patterns
        
    Returns:
        Unified DataFrame with all raw content data
        
    Behavior:
        - Reads concept, example, and try_it CSV files
        - Harmonizes columns across different file formats
        - Normalizes lo_id to string and creates content_type column
        - Raises FileNotFoundError if no files found
    """
    frames: List[pd.DataFrame] = []
    frames.extend(read_csvs_by_glob(config.concept_glob))
    frames.extend(read_csvs_by_glob(config.example_glob))
    frames.extend(read_csvs_by_glob(config.try_it_glob))
    if not frames:
        raise FileNotFoundError(
            "No raw CSVs found. Expected patterns like data/raw/concept*.csv, example*.csv, try_it*.csv"
        )
    # Harmonize columns by selecting the known superset when present
    unified = pd.concat(frames, ignore_index=True, sort=False)
    # Expected columns (best effort)
    expected_cols = [
        "lo_id",
        "raw_content",
        "type",
        "book",
        "learning_objective",
        "unit",
        "chapter",
    ]
    for col in expected_cols:
        if col not in unified.columns:
            unified[col] = None
    # Normalize lo_id to string
    unified["lo_id"] = unified["lo_id"].astype(str)
    # Standardize content_type
    def map_content_type(value: object) -> str:
        text_value = (str(value).strip().lower() if value is not None else "")
        if "concept" in text_value:
            return "concept"
        if "example" in text_value:
            return "example"
        if "try" in text_value:
            return "try_it"
        return text_value or "unknown"

    unified["content_type"] = unified["type"].map(map_content_type)
    return unified


def apply_sampling(
    df: pd.DataFrame, max_los: Optional[int], max_content_per_lo: Optional[int]
) -> pd.DataFrame:
    """
    Applies sampling to reduce dataset size for experimentation.
    
    Args:
        df: Input DataFrame to sample
        max_los: Maximum number of learning objectives to keep
        max_content_per_lo: Maximum content items per (lo_id, content_type) combination
        
    Returns:
        Sampled DataFrame with reduced size
        
    Behavior:
        - First filters by max_los if specified
        - Then limits content per (lo_id, content_type) group
        - Returns original DataFrame if no sampling specified
    """
    if max_los is None and max_content_per_lo is None:
        return df
    sampled = df
    if max_los is not None:
        # Sample by lo_id deterministically (sort then head)
        lo_ids = sorted(sampled["lo_id"].dropna().unique())[: max_los]
        sampled = sampled[sampled["lo_id"].isin(lo_ids)].copy()
    if max_content_per_lo is not None:
        sampled = (
            sampled.sort_values(["lo_id", "content_type"]).groupby(["lo_id", "content_type"], as_index=False)
            .head(max_content_per_lo)
        )
    return sampled


def generate_content_ids(df: pd.DataFrame) -> pd.Series:
    """
    Generates unique content IDs for each content item.
    
    Args:
        df: DataFrame with lo_id and content_type columns
        
    Returns:
        Series of content IDs in format "lo_id_content_type_sequence"
        
    Behavior:
        - Creates sequence numbers within each (lo_id, content_type) group
        - Generates IDs like "100_concept_1", "100_concept_2", "100_example_1"
        - Ensures stable, predictable ID generation
    """
    # Sequence within (lo_id, content_type)
    df = df.copy()
    df["_seq"] = (
        df.sort_values(["lo_id", "content_type"]).groupby(["lo_id", "content_type"]).cumcount() + 1
    )
    return df.apply(lambda r: f"{r['lo_id']}_{r['content_type']}_{int(r['_seq'])}", axis=1)


def build_outputs(unified: pd.DataFrame, config: PrepareConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the final output DataFrames from unified raw data.
    
    Args:
        unified: Unified DataFrame with all raw content
        config: Configuration object with sampling parameters
        
    Returns:
        Tuple of (lo_index_df, content_items_df)
        
    Behavior:
        - Applies sampling first to reduce processing time
        - Normalizes text and extracts image URLs for all content
        - Generates content IDs and creates final column structure
        - Separates into LO index and content items tables
    """
    # Apply sampling first to speed up normalization
    unified_sampled = apply_sampling(unified, config.sample_max_los, config.sample_max_content_per_lo)

    # Normalize text and image URLs
    texts: List[str] = []
    images_list: List[List[str]] = []
    for raw in unified_sampled["raw_content"].astype(str).tolist():
        cleaned_text, image_urls = normalize_row_text_and_images(raw)
        texts.append(cleaned_text)
        images_list.append(image_urls)

    content_df = unified_sampled.copy()
    content_df["text"] = texts
    content_df["image_urls"] = [json.dumps(urls) for urls in images_list]

    # Generate content_id
    content_df["content_id"] = generate_content_ids(content_df)

    # Select content_items columns
    content_items = content_df[[
        "content_id",
        "content_type",
        "lo_id",
        "text",
        "image_urls",
        "book",
        "learning_objective",
        "unit",
        "chapter",
    ]].rename(columns={"lo_id": "lo_id_parent"})

    # Build lo_index
    lo_index = (
        unified_sampled[["lo_id", "learning_objective", "unit", "chapter", "book"]]
        .drop_duplicates(subset=["lo_id"])
        .rename(columns={"lo_id": "lo_id"})
    )

    return lo_index, content_items


def main(argv: Optional[Iterable[str]] = None) -> int:
    """
    Main entry point for the prepare_lo_view script.
    
    Args:
        argv: Command line arguments (optional, for testing)
        
    Returns:
        Exit code (0 for success)
        
    Behavior:
        - Parses command line arguments
        - Loads configuration
        - Processes raw data and generates outputs
        - Creates output directories if needed
        - Writes CSV files and reports results
    """
    parser = argparse.ArgumentParser(description="Prepare LO content view")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (optional)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)

    unified = load_raw_frames(config)
    lo_index_df, content_items_df = build_outputs(unified, config)

    # Ensure output directories
    ensure_parent_directory(config.output_lo_index)
    ensure_parent_directory(config.output_content_items)

    lo_index_df.to_csv(config.output_lo_index, index=False)
    content_items_df.to_csv(config.output_content_items, index=False)

    print(f"Wrote {config.output_lo_index} ({len(lo_index_df)} rows)")
    print(f"Wrote {config.output_content_items} ({len(content_items_df)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

