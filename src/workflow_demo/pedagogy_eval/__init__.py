"""
Compact scenario-based pedagogy evaluation harness.

Deterministic control-loop checks use ``pedagogy_context`` (not snapshot-only).
Run: ``python -m workflow_demo.pedagogy_eval --help``
Regression: ``pytest tests/workflow_demo/test_acceptance_phase9.py -m acceptance``
"""

from __future__ import annotations

from src.workflow_demo.pedagogy_eval.harness import run_pedagogy_eval

__all__ = ["run_pedagogy_eval"]
