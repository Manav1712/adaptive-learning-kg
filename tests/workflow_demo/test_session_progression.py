"""Tests for session-local tutoring progression (MVP)."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy.session_progression import (
    apply_session_progression_update,
    build_initial_session_progression,
)
from src.workflow_demo.pedagogy.turn_progression import TurnProgressionSignals


@pytest.mark.unit
def test_build_initial_session_progression_from_current_plan():
    prog = build_initial_session_progression(
        {
            "learning_objective": "Main",
            "current_plan": [
                {"title": "Prereq LO", "is_primary": False},
                {"title": "Primary LO", "is_primary": True},
            ],
        }
    )
    assert len(prog["steps"]) == 2
    assert prog["steps"][0]["lo"] == "Prereq LO"
    assert prog["steps"][0]["kind"] == "support"
    assert prog["steps"][1]["kind"] == "primary"
    assert prog["active_step_index"] == 0
    assert prog["current_step_passed"] is False


@pytest.mark.unit
def test_substantive_answer_attempt_does_not_advance_step():
    prog = {
        "steps": [{"lo": "A", "kind": "primary"}, {"lo": "B", "kind": "primary"}],
        "active_step_index": 0,
        "current_step_passed": False,
    }
    sig = TurnProgressionSignals(
        explicit_advance_intent=False,
        adequate_check_response=False,
        current_confusion_signal=False,
        short_low_signal_ack=False,
        learner_requested_example=False,
        substantive_answer_attempt=True,
        suppress_repeat_diagnostic=True,
    )
    new_prog, event = apply_session_progression_update(prog, sig)
    assert new_prog["active_step_index"] == 0
    assert event is False


@pytest.mark.unit
def test_adequate_check_advances_step():
    prog = {
        "steps": [{"lo": "A", "kind": "primary"}, {"lo": "B", "kind": "primary"}],
        "active_step_index": 0,
        "current_step_passed": False,
    }
    sig = TurnProgressionSignals(
        explicit_advance_intent=False,
        adequate_check_response=True,
        current_confusion_signal=False,
        short_low_signal_ack=False,
        learner_requested_example=False,
        substantive_answer_attempt=False,
        suppress_repeat_diagnostic=False,
    )
    # adequate_check_response is computed from user text in real flow; here we set directly
    assert sig.adequate_check_response is True
    new_prog, event = apply_session_progression_update(prog, sig)
    assert new_prog["active_step_index"] == 1
    assert event is True


@pytest.mark.unit
def test_support_step_blocks_explicit_advance_without_adequate():
    prog = {
        "steps": [{"lo": "Support", "kind": "support"}, {"lo": "Primary", "kind": "primary"}],
        "active_step_index": 0,
        "current_step_passed": False,
    }
    sig = TurnProgressionSignals(
        explicit_advance_intent=True,
        adequate_check_response=False,
        current_confusion_signal=False,
        short_low_signal_ack=False,
        learner_requested_example=False,
        substantive_answer_attempt=False,
        suppress_repeat_diagnostic=False,
    )
    new_prog, event = apply_session_progression_update(prog, sig)
    assert new_prog["active_step_index"] == 0
    assert event is False
