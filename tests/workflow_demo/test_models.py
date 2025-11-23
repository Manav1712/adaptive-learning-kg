"""Unit tests for workflow_demo.models dataclasses."""

from dataclasses import asdict

import pytest

from src.workflow_demo.models import PlanStep, SessionPlan, TeachingPack


@pytest.mark.unit
def test_teaching_pack_structure(sample_teaching_pack: TeachingPack):
    """Sample TeachingPack exposes populated list attributes."""
    assert isinstance(sample_teaching_pack.key_points, list)
    assert sample_teaching_pack.key_points[0].startswith("The Tangent Problem")
    assert sample_teaching_pack.examples and sample_teaching_pack.prerequisites
    assert all("content_id" in example for example in sample_teaching_pack.examples)


@pytest.mark.unit
def test_plan_step_defaults():
    """PlanStep budget defaults to 250 tokens."""
    step = PlanStep(step_id="1", step_type="explain", goal="Test goal", lo_id=1)
    assert step.budget_tokens == 250


@pytest.mark.unit
def test_session_plan_serialization(sample_session_plan: SessionPlan):
    """SessionPlan should serialize cleanly via dataclasses.asdict."""
    payload = asdict(sample_session_plan)
    assert payload["subject"] == "calculus"
    assert payload["current_plan"][0]["goal"] == "Describe the core idea."
    assert payload["teaching_pack"]["key_points"][0].startswith("The Tangent Problem")
