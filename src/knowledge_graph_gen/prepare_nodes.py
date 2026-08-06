"""
Prepare KG nodes from draft content CSVs ("chunks").

Input (data/raw/): concept / example / try_it draft CSVs with columns
  lo_id, raw_content, type, book, learning_objective, unit, chapter

Output (run folder):
  lo_index.csv       — LO nodes + curriculum order columns
  content_items.csv  — concept / example / try_it nodes (answer keys stripped)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Curriculum chapter order for this corpus (OpenStax Precalc 2e + Calc Vol 1).
CHAPTER_ORDER: Dict[Tuple[str, str], int] = {
    ("Precalculus 2e", "Functions"): 1,
    ("Precalculus 2e", "Trigonometric Identities and Equations"): 2,
    ("Precalculus 2e", "Further Applications of Trigonometry"): 3,
    ("Calculus Volume 1", "Functions and Graphs"): 4,
    ("Calculus Volume 1", "Limits"): 5,
}

BOOK_ORDER: Dict[str, int] = {
    "Precalculus 2e": 1,
    "Calculus Volume 1": 2,
}

_MD_IMAGE = re.compile(r"!\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^)\s]+)\)")
_OPENSTAX_CDN = re.compile(
    r"https://openstax\.org/apps/image-cdn/[^\s\)\"']+"
)


@dataclass
class PrepareConfig:
    raw_dir: str = "data/raw"
    output_dir: str = "knowledge_graph/runs"
    lo_index_name: str = "lo_index.csv"
    content_items_name: str = "content_items.csv"


def find_raw_csvs(raw_dir: str) -> List[str]:
    patterns = [
        "concept_draft_contents*.csv",
        "example_draft_contents*.csv",
        "try_it_draft_contents*.csv",
    ]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(sorted(glob(os.path.join(raw_dir, pat))))
    if not paths:
        raise FileNotFoundError(f"No draft CSVs found under {raw_dir}")
    return paths


def extract_image_urls(markdown: str) -> List[str]:
    """Collect image URLs from markdown tokens and bare OpenStax CDN links."""
    found: List[str] = []
    seen = set()
    for url in _MD_IMAGE.findall(markdown or ""):
        if url not in seen:
            seen.add(url)
            found.append(url)
    for url in _OPENSTAX_CDN.findall(markdown or ""):
        if url not in seen:
            seen.add(url)
            found.append(url)
    return found


def strip_markdown_images(markdown: str) -> str:
    return _MD_IMAGE.sub("", markdown or "")


def concept_text(obj: Dict[str, Any], csv_type: str) -> Tuple[str, List[str]]:
    content = str(obj.get("content") or "")
    urls = extract_image_urls(content)
    body = strip_markdown_images(content)
    if "type" in obj:
        text = f"{obj.get('type') or csv_type}\n\n{body}"
    else:
        text = body
    return text.strip(), urls


def example_or_tryit_text(obj: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Problem + solution step text only — quiz answers are dropped on purpose."""
    problem = str(obj.get("problem") or "").strip()
    solution = obj.get("solution") or {}
    steps = solution.get("steps") if isinstance(solution, dict) else []
    if not isinstance(steps, list):
        steps = []

    step_texts: List[str] = []
    all_urls: List[str] = list(extract_image_urls(problem))

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_text = str(step.get("step") or "").strip()
        all_urls.extend(extract_image_urls(step_text))
        clean_step = strip_markdown_images(step_text).strip()
        if clean_step:
            step_texts.append(clean_step)

    seen = set()
    uniq_urls: List[str] = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            uniq_urls.append(u)

    problem_clean = strip_markdown_images(problem).strip()
    parts = [p for p in [problem_clean, *step_texts] if p]
    return "\n\n".join(parts), uniq_urls


def build_frames(raw_paths: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    content_rows: List[Dict[str, Any]] = []
    lo_rows: Dict[int, Dict[str, Any]] = {}
    counters: Dict[Tuple[int, str], int] = {}

    for path in raw_paths:
        df = pd.read_csv(path)
        required = {"lo_id", "raw_content", "type", "book", "learning_objective", "unit", "chapter"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for r in df.itertuples(index=False):
            lo_id = int(r.lo_id)
            ctype = str(r.type).strip().lower()
            book = str(r.book)
            unit = str(r.unit)
            chapter = str(r.chapter)
            lo_text = str(r.learning_objective)

            if lo_id not in lo_rows:
                lo_rows[lo_id] = {
                    "lo_id": lo_id,
                    "learning_objective": lo_text,
                    "unit": unit,
                    "chapter": chapter,
                    "book": book,
                }

            key = (lo_id, ctype)
            counters[key] = counters.get(key, 0) + 1
            content_id = f"{lo_id}_{ctype}_{counters[key]}"

            obj = json.loads(str(r.raw_content))
            if ctype == "concept":
                text, urls = concept_text(obj, ctype)
            elif ctype in {"example", "try_it"}:
                text, urls = example_or_tryit_text(obj)
            else:
                raise ValueError(f"Unknown content type {ctype!r} in {path}")

            content_rows.append(
                {
                    "content_id": content_id,
                    "content_type": ctype,
                    "lo_id_parent": lo_id,
                    "text": text,
                    "image_urls": json.dumps(urls, ensure_ascii=False),
                    "book": book,
                    "learning_objective": lo_text,
                    "unit": unit,
                    "chapter": chapter,
                }
            )

    content_df = pd.DataFrame(content_rows)
    lo_df = pd.DataFrame(list(lo_rows.values()))

    # Explicit order columns so prereq discovery can enforce forward-only edges.
    lo_df["book_order"] = lo_df["book"].map(lambda b: BOOK_ORDER.get(str(b), 99))
    lo_df["chapter_order"] = lo_df.apply(
        lambda r: CHAPTER_ORDER.get((str(r["book"]), str(r["chapter"])), 99),
        axis=1,
    )
    lo_df = lo_df.sort_values(["book_order", "chapter_order", "lo_id"]).reset_index(drop=True)

    unit_rank: Dict[Tuple[str, str, str], int] = {}
    unit_orders: List[int] = []
    for _, row in lo_df.iterrows():
        key = (str(row["book"]), str(row["chapter"]), str(row["unit"]))
        if key not in unit_rank:
            unit_rank[key] = len([k for k in unit_rank if k[0] == key[0] and k[1] == key[1]]) + 1
        unit_orders.append(unit_rank[key])
    lo_df["unit_order"] = unit_orders
    lo_df["lo_order"] = range(1, len(lo_df) + 1)

    lo_df = lo_df[
        [
            "lo_id",
            "learning_objective",
            "unit",
            "chapter",
            "book",
            "book_order",
            "chapter_order",
            "unit_order",
            "lo_order",
        ]
    ]
    return lo_df, content_df


def write_outputs(lo_df: pd.DataFrame, content_df: pd.DataFrame, config: PrepareConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    lo_path = os.path.join(config.output_dir, config.lo_index_name)
    content_path = os.path.join(config.output_dir, config.content_items_name)
    lo_df.to_csv(lo_path, index=False)
    content_df.to_csv(content_path, index=False)
    print(f"Wrote {lo_path} ({len(lo_df)} LOs)")
    print(f"Wrote {content_path} ({len(content_df)} content items)")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare KG nodes from raw draft CSVs")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run folder (writes lo_index.csv and content_items.csv here)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Deprecated alias for --run-dir",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = args.run_dir or args.output_dir
    if not output_dir:
        raise SystemExit("Pass --run-dir (or --output-dir) pointing at a run folder")

    config = PrepareConfig(raw_dir=args.raw_dir, output_dir=output_dir)
    paths = find_raw_csvs(config.raw_dir)
    print("Reading:")
    for p in paths:
        print(f"  {p}")
    lo_df, content_df = build_frames(paths)
    write_outputs(lo_df, content_df, config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
