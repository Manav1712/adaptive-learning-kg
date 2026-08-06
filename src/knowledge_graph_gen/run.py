"""
End-to-end knowledge graph generation.

Creates a versioned run folder, then:
  1. prepare_nodes        → lo_index.csv, content_items.csv
  2. discover_prereqs     → edges_prereqs.csv
  3. discover_content_links → edges_content.csv
  4. (optional) evaluate  → intermediates/

Usage:
  python src/knowledge_graph_gen/run.py --raw-dir data/raw
  python src/knowledge_graph_gen/run.py --raw-dir data/raw --eval
  python src/knowledge_graph_gen/run.py --raw-dir data/raw --limit 5   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from . import prepare_nodes
from . import discover_prereqs
from . import discover_content_links
from . import evaluate_heuristic
from . import evaluate_llm


ARTIFACT_ROOT = "knowledge_graph/runs"


def new_run_dir(artifact_root: str = ARTIFACT_ROOT) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifact_root, run_id)
    os.makedirs(os.path.join(run_dir, "intermediates"), exist_ok=True)
    return run_dir


def _count_rows(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    return int(len(pd.read_csv(path)))


def write_manifest(run_dir: str, raw_dir: str, limit: Optional[int], did_eval: bool) -> None:
    files = {
        "lo_index": os.path.join(run_dir, "lo_index.csv"),
        "content_items": os.path.join(run_dir, "content_items.csv"),
        "edges_prereqs": os.path.join(run_dir, "edges_prereqs.csv"),
        "edges_content": os.path.join(run_dir, "edges_content.csv"),
    }
    manifest = {
        "run_id": os.path.basename(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": raw_dir,
        "limit": limit,
        "eval": did_eval,
        "thresholds": {
            "prereqs": discover_prereqs.PrereqConfig.score_threshold,
            "content": discover_content_links.DiscoveryConfig.score_threshold,
        },
        "model": discover_prereqs.PrereqConfig.model,
        "modality": discover_prereqs.PrereqConfig.modality,
        "row_counts": {name: _count_rows(path) for name, path in files.items()},
        "graph_files": [os.path.basename(p) for p in files.values()],
    }
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {path}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run full knowledge graph generation")
    parser.add_argument("--raw-dir", default="data/raw", help="Folder with draft content CSVs")
    parser.add_argument(
        "--artifact-root",
        default=ARTIFACT_ROOT,
        help="Parent folder for versioned runs",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Reuse an existing run folder (skip creating a new one)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test limit for discover steps")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Also run heuristic (+ optional LLM) evaluation into intermediates/",
    )
    parser.add_argument(
        "--eval-llm",
        action="store_true",
        help="With --eval, also run LLM edge checks (costs API calls)",
    )
    parser.add_argument(
        "--skip-discover",
        action="store_true",
        help="Only prepare nodes (useful to inspect nodes before scoring)",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir or new_run_dir(args.artifact_root)
    os.makedirs(os.path.join(run_dir, "intermediates"), exist_ok=True)
    print(f"Run folder: {run_dir}")

    # 1) Chunks → nodes
    prepare_nodes.main(["--raw-dir", args.raw_dir, "--run-dir", run_dir])

    if not args.skip_discover:
        discover_argv = ["--run-dir", run_dir, "--mode", "both"]
        if args.limit is not None:
            discover_argv.extend(["--limit", str(args.limit)])

        # 2) LO → LO prerequisites
        discover_prereqs.main(discover_argv)

        # 3) Content ↔ LO links
        discover_content_links.main(discover_argv)

    if args.eval:
        for kind in ("prereqs", "content"):
            evaluate_heuristic.main(["--run-dir", run_dir, "--edges-kind", kind])
            if args.eval_llm:
                evaluate_llm.main(["--run-dir", run_dir, "--edges-kind", kind])

    write_manifest(run_dir, args.raw_dir, args.limit, bool(args.eval))
    print(f"\nDone. Graph files are in: {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
