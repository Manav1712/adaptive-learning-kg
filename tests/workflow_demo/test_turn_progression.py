"""Tests for tutor turn progression signals (repeat diagnostic suppression gate)."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy.constants import TeachingMoveType
from src.workflow_demo.pedagogy.turn_progression import (
    compute_turn_progression_signals,
    matches_explicit_advance_intent,
    matches_learner_requested_example,
    matches_short_low_signal_ack,
    matches_substantive_answer_attempt,
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


@pytest.mark.unit
def test_learner_requested_example_detected():
    assert matches_learner_requested_example("give me an example problem") is True
    assert matches_learner_requested_example("Can you show me an example?") is True


@pytest.mark.unit
def test_learner_requested_example_not_triggered_by_unrelated():
    assert matches_learner_requested_example("I like examples in general") is False


@pytest.mark.unit
def test_substantive_answer_attempt_with_digit_and_cue():
    prior = TeachingMoveType.DIAGNOSTIC_QUESTION.value
    assert matches_substantive_answer_attempt("I think the answer is 5", prior) is True


@pytest.mark.unit
def test_substantive_answer_attempt_false_positives_rejected():
    prior = TeachingMoveType.DIAGNOSTIC_QUESTION.value
    assert matches_substantive_answer_attempt("I have 2 questions", prior) is False
    assert matches_substantive_answer_attempt("can we do #3?", prior) is False
    assert matches_substantive_answer_attempt("I got 5 minutes", prior) is False


@pytest.mark.unit
def test_substantive_answer_attempt_with_equation_and_expr():
    prior = TeachingMoveType.DIAGNOSTIC_QUESTION.value
    assert matches_substantive_answer_attempt("the derivative is 2x", prior) is True
    assert matches_substantive_answer_attempt("I got x^2 + 3", prior) is True
    assert matches_substantive_answer_attempt("x = 7", prior) is True


@pytest.mark.unit
def test_substantive_answer_attempt_blocked_without_prior_diagnostic():
    prior = TeachingMoveType.GRADUATED_HINT.value
    assert matches_substantive_answer_attempt("x = 5", prior) is False


@pytest.mark.unit
def test_substantive_answer_attempt_blocked_by_short_ack():
    prior = TeachingMoveType.DIAGNOSTIC_QUESTION.value
    assert matches_substantive_answer_attempt("yes", prior) is False


@pytest.mark.unit
def test_suppress_fires_on_example_request_after_diagnostic():
    sig = compute_turn_progression_signals(
        user_input="can you show me an example?",
        previous_last_selected_move_type=TeachingMoveType.DIAGNOSTIC_QUESTION.value,
    )
    assert sig.learner_requested_example is True
    assert sig.suppress_repeat_diagnostic is True


@pytest.mark.unit
def test_suppress_fires_on_substantive_attempt_after_diagnostic():
    sig = compute_turn_progression_signals(
        user_input="I got x^2 + 3",
        previous_last_selected_move_type=TeachingMoveType.DIAGNOSTIC_QUESTION.value,
    )
    assert sig.substantive_answer_attempt is True
    assert sig.suppress_repeat_diagnostic is True


@pytest.mark.unit
def test_new_advance_phrases_detected():
    assert matches_explicit_advance_intent("let's keep going") is True
    assert matches_explicit_advance_intent("can we continue") is True
    assert matches_explicit_advance_intent("i get it, let's move on") is True
