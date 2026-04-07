"""
CLI for the pedagogy eval harness.

Examples::

    python -m workflow_demo.pedagogy_eval --json-out results.json
    WORKFLOW_DEMO_TUTOR_MATH_GUARD=1 python -m workflow_demo.pedagogy_eval --verbose

Assertions use ``pedagogy_context`` (backend state), not snapshot-only. Snapshot in JSON
is for reporting. Scenario 6 (math guard) SKIPs when the guard env is off or SymPy is
missing — SKIPs do not fail the run.

Regression suite::

    pytest tests/workflow_demo/test_acceptance_phase9.py -m acceptance
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Dict, List

from src.workflow_demo.pedagogy_eval.harness import print_report, run_pedagogy_eval


def _write_csv(path: str, aggregate: Dict[str, Any]) -> None:
    rows = aggregate["results"]
    if not rows:
        return
    fieldnames: List[str] = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = dict(r)
            snap = flat.get("pedagogy_snapshot")
            if snap is not None:
                flat["pedagogy_snapshot"] = json.dumps(snap)
            w.writerow(flat)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compact pedagogy eval: control-loop checks via pedagogy_context. "
            "Set WORKFLOW_DEMO_TUTOR_MATH_GUARD=1 and install sympy to run scenario 6."
        )
    )
    p.add_argument(
        "--json-out",
        metavar="PATH",
        help="Write full aggregate JSON (results + summary) to PATH",
    )
    p.add_argument(
        "--csv",
        metavar="PATH",
        help="Write per-scenario CSV to PATH",
    )
    p.add_argument("--verbose", action="store_true", help="Print pc-derived fields and snapshot line")
    args = p.parse_args(argv)

    aggregate, summary = run_pedagogy_eval(verbose=args.verbose)
    print_report(aggregate, verbose=args.verbose)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)

    if args.csv:
        _write_csv(args.csv, aggregate)

    return 1 if summary.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
