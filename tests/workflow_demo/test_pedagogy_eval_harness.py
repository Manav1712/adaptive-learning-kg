"""Drift guard for the compact pedagogy eval harness (run_all)."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy_eval.harness import run_pedagogy_eval


@pytest.mark.integration
@pytest.mark.pedagogy_eval
def test_pedagogy_eval_harness_no_failures():
    """All scenarios pass or skip; skips (e.g. math guard env) are not failures."""
    aggregate, summary = run_pedagogy_eval(verbose=False)
    assert summary.failed == 0, aggregate["results"]
    assert summary.total_scenarios == 7
    assert summary.passed + summary.skipped == 7
    assert aggregate["summary"]["failed"] == 0
