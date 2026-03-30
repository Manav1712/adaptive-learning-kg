"""Phase 3 tests for deterministic teaching move generation."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy import (
    LearnerState,
    MisconceptionDiagnosis,
    TeachingMoveGenerator,
)
from src.workflow_demo.pedagogy.constants import RetrievalIntent, TeachingMoveType


@pytest.mark.unit
def test_low_confidence_includes_diagnostic_question():
    generator = TeachingMoveGenerator()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="uncertain_or_low_signal",
        confidence=0.2,
        rationale="Low signal response.",
        prerequisite_gap_los=[],
    )
    state = LearnerState(active_session_id="s1", current_focus_lo="Derivatives")
    moves = generator.generate_candidates(diagnosis, state, "Derivatives", "ok")
    move_types = {m.move_type for m in moves}
    assert TeachingMoveType.DIAGNOSTIC_QUESTION in move_types


@pytest.mark.unit
def test_prerequisite_gap_includes_prereq_remediation():
    generator = TeachingMoveGenerator()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Integrals",
        suspected_misconception="prerequisite_gap",
        confidence=0.8,
        rationale="Needs prerequisite review.",
        prerequisite_gap_los=["Derivatives", "Functions"],
    )
    state = LearnerState(active_session_id="s1", current_focus_lo="Integrals")
    moves = generator.generate_candidates(diagnosis, state, "Integrals", "I forgot derivatives")
    move_types = {m.move_type for m in moves}
    assert TeachingMoveType.PREREQ_REMEDIATION in move_types


@pytest.mark.unit
def test_generator_returns_at_least_two_candidates():
    generator = TeachingMoveGenerator()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Limits",
        suspected_misconception="uncertain_reasoning",
        confidence=0.45,
        rationale="Uncertain answer.",
        prerequisite_gap_los=[],
    )
    state = LearnerState(active_session_id="s1")
    moves = generator.generate_candidates(diagnosis, state, "Limits", "maybe?")
    assert len(moves) >= 2


@pytest.mark.unit
def test_move_ids_are_unique_and_stable():
    generator = TeachingMoveGenerator()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="conceptual_confusion",
        confidence=0.66,
        rationale="Student says confused.",
        prerequisite_gap_los=[],
    )
    state = LearnerState(active_session_id="s1", current_focus_lo="Derivatives")
    first = generator.generate_candidates(diagnosis, state, "Derivatives", "I am confused")
    second = generator.generate_candidates(diagnosis, state, "Derivatives", "I am confused")
    assert len({m.move_id for m in first}) == len(first)
    assert [m.move_id for m in first] == [m.move_id for m in second]


@pytest.mark.unit
def test_retrieval_intent_mapping_is_correct():
    generator = TeachingMoveGenerator()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="prerequisite_gap",
        confidence=0.7,
        rationale="Needs basics.",
        prerequisite_gap_los=["Functions"],
    )
    state = LearnerState(active_session_id="s1", current_focus_lo="Derivatives")
    moves = generator.generate_candidates(diagnosis, state, "Derivatives", "I forgot basics")
    by_type = {m.move_type: m for m in moves}
    assert by_type[TeachingMoveType.PREREQ_REMEDIATION].retrieval_intent == RetrievalIntent.PREREQUISITE_REFRESH
    if TeachingMoveType.WORKED_EXAMPLE in by_type:
        assert by_type[TeachingMoveType.WORKED_EXAMPLE].retrieval_intent == RetrievalIntent.WORKED_PARALLEL


@pytest.mark.integration
def test_tutor_flow_populates_teaching_moves_in_context(monkeypatch, coach_agent):
    responses = [
        {
            "message_to_student": "Let's continue derivatives.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {"student_understanding": "needs_practice"},
        },
        {
            "message_to_student": "Great question, let's check that.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {"student_understanding": "needs_practice"},
        },
    ]

    def _fake_tutor_bot(**_kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _fake_tutor_bot)

    coach_agent.planner_result = {
        "status": "complete",
        "plan": {
            "subject": "calculus",
            "mode": "conceptual_review",
            "current_plan": [
                {
                    "lo_id": 5,
                    "title": "Derivatives",
                    "proficiency": 0.2,
                    "how_to_teach": "",
                    "why_to_teach": "",
                    "notes": "",
                    "is_primary": True,
                }
            ],
            "future_plan": [],
        },
    }

    opening = coach_agent.bot_session_manager.begin(
        bot_type="tutor",
        tool_params={"student_request": "Teach me derivatives"},
        conversation_summary="Student requested tutoring.",
    )
    assert opening

    followup = coach_agent.process_turn("I'm confused, derivative of x^2 is x?")
    assert "check" in followup.lower()

    context = coach_agent.bot_session_manager.handoff_context or {}
    pedagogy_context = context.get("pedagogy_context") or {}
    assert pedagogy_context.get("diagnosis")
    teaching_moves = pedagogy_context.get("teaching_moves") or []
    assert len(teaching_moves) >= 2
