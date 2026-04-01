"""Tests for tutor turn progression signals (repeat diagnostic suppression gate)."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy.constants import TeachingMoveType
from src.workflow_demo.pedagogy.turn_progression import (
    compute_turn_progression_signals,
    matches_explicit_advance_intent,
    matches_short_low_signal_ack,
)


@pytest.mark.unit
def test_gate_suppresses_when_prior_diagnostic_and_explicit_advance():
    sig = compute_turn_progression_signals(
        user_input="assume I know this, let's continue with integration",
        previous_last_selected_move_type=TeachingMoveType.DIAGNOSTIC_QUESTION.value,
    )
    assert sig.explicit_advance_intent is True
    assert sig.suppress_repeat_diagnostic is True
    assert sig.short_low_signal_ack is False


@pytest.mark.unit
def test_gate_blocked_by_short_ack():
    sig = compute_turn_progression_signals(
        user_input="ok",
        previous_last_selected_move_type=TeachingMoveType.DIAGNOSTIC_QUESTION.value,
    )
    assert sig.short_low_signal_ack is True
    assert sig.suppress_repeat_diagnostic is False


@pytest.mark.unit
def test_gate_blocked_by_confusion():
    sig = compute_turn_progression_signals(
        user_input="I'm confused and don't understand, but let's continue",
        previous_last_selected_move_type=TeachingMoveType.DIAGNOSTIC_QUESTION.value,
    )
    assert sig.current_confusion_signal is True
    assert sig.suppress_repeat_diagnostic is False


@pytest.mark.unit
def test_gate_adequate_paraphrase_without_explicit_advance():
    text = (
        "The height is the vertical distance from the top to the base, and the base "
        "runs horizontally along the bottom. That's how I picture a triangle on a graph."
    )
    sig = compute_turn_progression_signals(
        user_input=text,
        previous_last_selected_move_type=TeachingMoveType.DIAGNOSTIC_QUESTION.value,
    )
    assert sig.adequate_check_response is True
    assert sig.suppress_repeat_diagnostic is True


@pytest.mark.unit
def test_no_suppression_without_prior_diagnostic():
    sig = compute_turn_progression_signals(
        user_input="assume I know this, let's continue with integration",
        previous_last_selected_move_type=TeachingMoveType.GRADUATED_HINT.value,
    )
    assert sig.explicit_advance_intent is True
    assert sig.suppress_repeat_diagnostic is False


@pytest.mark.unit
def test_explicit_advance_phrase_detection():
    assert matches_explicit_advance_intent("let's continue with integration") is True
    assert matches_explicit_advance_intent("go back to the area problem") is True
    assert matches_short_low_signal_ack("yes") is True
