"""Tests for Phase 2 misconception diagnoser and tutor wiring."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy import LearnerState, MisconceptionDiagnoser


@pytest.mark.unit
def test_low_signal_fallback_diagnosis():
    diagnoser = MisconceptionDiagnoser()
    state = LearnerState(active_session_id="s1", current_focus_lo="Derivatives")
    diagnosis = diagnoser.diagnose_turn(
        session_id="s1",
        user_input="ok",
        current_focus_lo="Derivatives",
        learner_state=state,
    )
    assert diagnosis.target_lo == "Derivatives"
    assert diagnosis.suspected_misconception == "uncertain_or_low_signal"
    assert diagnosis.confidence <= 0.2


@pytest.mark.unit
def test_confusion_heuristic_diagnosis():
    diagnoser = MisconceptionDiagnoser()
    state = LearnerState(active_session_id="s1", current_focus_lo="Integrals")
    diagnosis = diagnoser.diagnose_turn(
        session_id="s1",
        user_input="I'm confused and I don't understand this step.",
        current_focus_lo="Integrals",
        learner_state=state,
    )
    assert diagnosis.suspected_misconception == "conceptual_confusion"
    assert diagnosis.confidence >= 0.6


@pytest.mark.unit
def test_prerequisite_gap_diagnosis():
    diagnoser = MisconceptionDiagnoser()
    state = LearnerState(active_session_id="s1", current_focus_lo="Derivatives")
    diagnosis = diagnoser.diagnose_turn(
        session_id="s1",
        user_input="I forgot the basics, what is a function before this?",
        current_focus_lo="Derivatives",
        learner_state=state,
    )
    assert diagnosis.suspected_misconception == "prerequisite_gap"
    assert diagnosis.prerequisite_gap_los


@pytest.mark.unit
def test_diagnosis_validity_with_minimal_input():
    diagnoser = MisconceptionDiagnoser()
    diagnosis = diagnoser.diagnose_turn(
        session_id="s-min",
        user_input="",
        current_focus_lo=None,
        learner_state=LearnerState(active_session_id="s-min"),
    )
    assert diagnosis.target_lo
    assert diagnosis.suspected_misconception
    assert 0.0 <= diagnosis.confidence <= 1.0


@pytest.mark.unit
def test_diagnosis_persistence_dedupes_entries():
    from src.workflow_demo.pedagogy import LearnerStateEngine, LearnerStateStore
    from src.workflow_demo.pedagogy.models import MisconceptionDiagnosis

    store = LearnerStateStore()
    engine = LearnerStateEngine(store=store)
    engine.initialize_from_profile("s1", {"lo_mastery": {}}, current_focus_lo="Derivatives")
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="conceptual_confusion",
        confidence=0.7,
        rationale="Learner says they are confused.",
        prerequisite_gap_los=[],
    )
    state1 = engine.record_misconception("s1", diagnosis)
    state2 = engine.record_misconception("s1", diagnosis)
    assert state1.misconceptions["Derivatives"] == ["conceptual_confusion"]
    assert state2.misconceptions["Derivatives"] == ["conceptual_confusion"]


@pytest.mark.integration
def test_diagnosis_persists_into_learner_state_and_context(monkeypatch, coach_agent):
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
            "message_to_student": "Good question.",
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
    _ = coach_agent.bot_session_manager.begin(
        bot_type="tutor",
        tool_params={"student_request": "Teach me derivatives"},
        conversation_summary="Student requested tutoring.",
    )
    _ = coach_agent.process_turn("I'm confused, derivative of x^2 is x?")

    session_id = coach_agent.bot_session_manager.active_learner_session_id
    assert session_id is not None
    state = coach_agent.learner_state_store.get(session_id)
    assert state is not None
    assert state.misconceptions.get("Derivatives")

    context = coach_agent.bot_session_manager.handoff_context or {}
    pedagogy_context = context.get("pedagogy_context") or {}
    diagnosis_payload = pedagogy_context.get("diagnosis") or {}
    assert diagnosis_payload.get("target_lo") == "Derivatives"
    assert diagnosis_payload.get("suspected_misconception")
