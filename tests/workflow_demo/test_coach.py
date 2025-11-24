"""Unit tests for the CoachAgent orchestrator."""

import json
from copy import deepcopy

import pytest

from src.workflow_demo.coach import CoachAgent


@pytest.mark.unit
def test_coach_process_turn_replays_llm_message(coach_agent: CoachAgent):
    """A 'none' directive should return the LLM-authored message."""
    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "I need a bit more detail.",
            "action": "none",
            "tool_params": {"subject": "calculus"},
            "conversation_summary": "",
        }
    )
    reply = coach_agent.process_turn("Hi")
    assert reply == "I need a bit more detail."
    assert coach_agent.collected_params["subject"] == "calculus"


@pytest.mark.unit
def test_coach_calls_tutoring_planner_when_directed(coach_agent: CoachAgent, mock_retriever):
    """Planner directives should return the plan summary and wait for confirmation."""
    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "call_tutoring_planner",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "conceptual_review",
            },
            "conversation_summary": "",
        }
    )

    reply = coach_agent.process_turn("Let's learn derivatives")
    assert "Here's what I put together" in reply
    assert coach_agent.awaiting_confirmation
    assert coach_agent.awaiting_confirmation_prompted
    assert mock_retriever.calls, "Retriever should be invoked once."
    assert mock_retriever.calls[0]["query"] == "Let's learn derivatives"


@pytest.mark.unit
def test_coach_confirmation_triggers_bot_session(
    monkeypatch, coach_agent: CoachAgent, sample_planner_response
):
    """When awaiting confirmation, a 'yes' should start the tutor session."""

    def _fake_tutor_bot(**kwargs):
        return {
            "message_to_student": "Hi from tutor",
            "end_activity": True,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": ["Derivatives"],
                "student_understanding": "good",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "Test session",
            },
        }

    monkeypatch.setattr("src.workflow_demo.coach.tutor_bot", _fake_tutor_bot)

    coach_agent.awaiting_confirmation = True
    coach_agent.pending_session_type = "tutor"
    coach_agent.collected_params = {"subject": "calculus", "learning_objective": "Derivatives", "mode": "practice"}
    coach_agent.planner_result = deepcopy(sample_planner_response)

    summary = coach_agent.process_turn("")
    assert "Here's what I put together" in summary
    assert coach_agent.awaiting_confirmation_prompted

    reply = coach_agent.process_turn("yes")
    assert reply.startswith("Hi! I'm your learning coach")
    assert coach_agent.in_bot_session is False


@pytest.mark.unit
def test_coach_presents_plan_before_handoff(coach_agent: CoachAgent, sample_planner_response):
    """Coach should summarize the plan and await confirmation before starting tutor."""
    coach_agent.awaiting_confirmation = True
    coach_agent.pending_session_type = "tutor"
    coach_agent.collected_params = sample_planner_response["plan"]
    coach_agent.planner_result = deepcopy(sample_planner_response)
    coach_agent.awaiting_confirmation_prompted = False

    reply = coach_agent.process_turn("")

    assert "Here's what I put together" in reply
    assert "Would you like to start with this plan" in reply
    assert coach_agent.awaiting_confirmation
    assert coach_agent.awaiting_confirmation_prompted


@pytest.mark.unit
def test_coach_blocks_same_turn_handoff(coach_agent: CoachAgent, mock_retriever):
    """Planner completion and tutor handoff must not happen in the same turn."""
    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "call_tutoring_planner",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "examples",
            },
            "conversation_summary": "",
        }
    )
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "start_tutor",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "examples",
                "student_request": "Help me with derivatives",
            },
            "conversation_summary": "Student confirmed derivatives tutoring.",
        }
    )

    reply = coach_agent.process_turn("Help me with derivatives")

    assert "Here's what I put together" in reply
    assert coach_agent.awaiting_confirmation
    assert not coach_agent.in_bot_session
    assert mock_retriever.calls, "Planner should still run once."
    assert len(coach_agent.llm_client._queued) == 1, "Second directive should stay queued."


@pytest.mark.unit
def test_coach_calls_faq_planner_for_syllabus(
    monkeypatch, coach_agent: CoachAgent, sample_syllabus_faq_plan
):
    """LLM directive for syllabus questions should call the FAQ planner."""

    captured = {}

    def _fake_faq_plan(params):
        captured["params"] = params
        return sample_syllabus_faq_plan

    monkeypatch.setattr(coach_agent.faq_planner, "create_plan", _fake_faq_plan)

    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "call_faq_planner",
            "tool_params": {"topic": "syllabus_topics", "student_request": "List the major concepts"},
            "conversation_summary": "",
        }
    )

    reply = coach_agent.process_turn("I want to ask a question about the syllabus")
    assert "syllabus" in reply.lower()
    assert coach_agent.awaiting_confirmation
    assert captured["params"]["topic"] == "syllabus_topics"


@pytest.mark.unit
def test_coach_forces_syllabus_plan_after_clarifications(
    monkeypatch, coach_agent: CoachAgent, sample_syllabus_faq_plan
):
    """Repeated clarifications for syllabus questions should trigger the FAQ planner fallback."""

    plan_calls = {"count": 0}

    def _fake_faq_plan(params):
        plan_calls["count"] += 1
        return sample_syllabus_faq_plan

    monkeypatch.setattr(coach_agent.faq_planner, "create_plan", _fake_faq_plan)

    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "Could you clarify which concepts you need?",
            "action": "none",
            "tool_params": {},
            "conversation_summary": "",
        }
    )

    reply1 = coach_agent.process_turn("I want to ask a question about the syllabus")
    assert "clarify" in reply1.lower()
    assert plan_calls["count"] == 0

    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "Still not sure which concepts you need.",
            "action": "none",
            "tool_params": {},
            "conversation_summary": "",
        }
    )

    reply2 = coach_agent.process_turn("Can you list out the major concepts in my syllabus?")
    assert "syllabus" in reply2.lower()
    assert coach_agent.awaiting_confirmation
    assert plan_calls["count"] == 1


@pytest.mark.unit
def test_coach_overrides_stale_student_request(coach_agent: CoachAgent, mock_retriever):
    """LLM-provided student_request should be replaced with the latest student wording."""
    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "call_tutoring_planner",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "conceptual_review",
                "student_request": "Old request wording",
            },
            "conversation_summary": "",
        }
    )

    reply = coach_agent.process_turn("Fresh conceptual review please")

    assert "Here's what I put together" in reply
    assert mock_retriever.calls, "Planner should be invoked."
    assert mock_retriever.calls[0]["query"] == "Fresh conceptual review please"
