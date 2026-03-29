"""Unit + integration tests for learner-state Phase 1 engine/store wiring."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy import LearnerStateEngine, LearnerStateStore


@pytest.mark.unit
def test_state_store_ensure_get_update():
    store = LearnerStateStore()
    ensured = store.ensure("s1")
    assert ensured.active_session_id == "s1"
    assert store.get("s1") is not None

    updated = store.update("s1", current_focus_lo="Derivatives", mastery={"Derivatives": 0.7})
    assert updated.current_focus_lo == "Derivatives"
    assert updated.mastery["Derivatives"] == 0.7

    replaced = store.set("s1", updated)
    assert replaced.active_session_id == "s1"


@pytest.mark.unit
def test_initialize_from_profile():
    events = []
    engine = LearnerStateEngine(
        store=LearnerStateStore(),
        event_emitter=lambda event_type, message, metadata: events.append(
            {"event_type": event_type, "message": message, "metadata": metadata}
        ),
    )

    state = engine.initialize_from_profile(
        session_id="runtime",
        student_profile={"lo_mastery": {"Derivatives": 0.8, "Limits": 0.4}},
        current_focus_lo="Derivatives",
    )
    assert state.active_session_id == "runtime"
    assert state.current_focus_lo == "Derivatives"
    assert state.mastery == {"Derivatives": 0.8, "Limits": 0.4}
    assert state.recent_attempts == []
    assert state.hint_history == []
    assert events[-1]["event_type"] == "pedagogy_learner_state_initialized"


@pytest.mark.unit
def test_initialize_with_missing_profile():
    engine = LearnerStateEngine(store=LearnerStateStore())
    state = engine.initialize_from_profile(session_id="runtime", student_profile=None)
    assert state.active_session_id == "runtime"
    assert state.mastery == {}
    assert state.misconceptions == {}
    assert state.recent_attempts == []
    assert state.hint_history == []


@pytest.mark.unit
def test_record_turn_appends_attempt():
    engine = LearnerStateEngine(store=LearnerStateStore())
    engine.initialize_from_profile("s1", {"lo_mastery": {"Derivatives": 0.3}})

    state = engine.record_turn(
        session_id="s1",
        turn_index=0,
        student_text="I think derivative of x^2 is x.",
        lo_id=5,
        is_correct=False,
    )
    assert len(state.recent_attempts) == 1
    attempt = state.recent_attempts[0]
    assert attempt.turn_index == 0
    assert attempt.lo_id == 5
    assert attempt.is_correct is False


@pytest.mark.unit
def test_attach_hint_event_updates_hint_history():
    engine = LearnerStateEngine(store=LearnerStateStore())
    engine.initialize_from_profile("s1", {"lo_mastery": {}})
    state = engine.attach_hint_event("s1", "Try applying the power rule.")
    assert state.hint_history[-1] == "Try applying the power rule."


@pytest.mark.integration
def test_tutor_flow_initializes_and_updates_learner_state(
    monkeypatch,
    coach_agent,
):
    responses = [
        {
            "message_to_student": "Let's start with derivatives.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": ["Derivatives"],
                "student_understanding": "needs_practice",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "",
            },
        },
        {
            "message_to_student": "Close - check the exponent rule.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": ["Derivatives"],
                "student_understanding": "needs_practice",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "",
            },
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
    assert "derivatives" in opening.lower()
    session_id = coach_agent.bot_session_manager.active_learner_session_id
    assert session_id is not None
    initialized_state = coach_agent.learner_state_store.get(session_id)
    assert initialized_state is not None
    assert initialized_state.active_session_id == session_id

    _ = coach_agent.process_turn("Derivative of x^2 is x")
    updated_state = coach_agent.learner_state_store.get(session_id)
    assert updated_state is not None
    assert len(updated_state.recent_attempts) == 1
    assert "x^2" in updated_state.recent_attempts[0].student_response_excerpt
