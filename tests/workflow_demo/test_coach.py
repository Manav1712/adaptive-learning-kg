import json
from copy import deepcopy
from unittest.mock import patch, MagicMock

import pytest

from src.workflow_demo.coach_agent import CoachAgent


@pytest.fixture
def coach_agent(mock_openai_client, mock_retriever):
    # Pass a valid path to avoid loading default which might be stale
    from src.workflow_demo.coach_llm_client import CoachLLMClient
    from src.workflow_demo.coach_router import CoachRouter
    from src.workflow_demo.bot_sessions import BotSessionManager

    coach = CoachAgent(retriever=mock_retriever, session_memory_path=":memory:", llm_model="test-model")
    coach.llm_client = mock_openai_client
    coach.llm_model = "test-model"
    # Re-wire delegates with the mock LLM client
    coach.llm_client_wrapper = CoachLLMClient(mock_openai_client, "test-model")
    coach.coach_router = CoachRouter(coach, coach.llm_client_wrapper)
    coach.bot_session_manager = BotSessionManager(coach)
    return coach


@pytest.fixture
def sample_planner_response():
    return {
        "status": "complete",
        "plan": {
            "subject": "calculus",
            "learning_objective": "Derivatives",
            "mode": "practice",
            "book": "Calculus Volume 1",
            "chapter": "1: The Tangent Problem",
            "current_plan": [],
            "first_question": "What do you already know about the tangent problem?",
        },
        "message": None,
    }


@pytest.fixture
def sample_syllabus_faq_plan():
    return {
        "status": "complete",
        "plan": {
            "topic": "syllabus_topics",
            "student_request": "Can you list the major concepts?",
            "script": "Major calculus concepts include limits, derivatives, integrals, FTC.",
            "first_question": "Which concept would you like to dive into next?",
        },
        "message": None,
    }


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
def test_coach_overrides_stale_student_request(coach_agent: CoachAgent, mock_retriever):
    """LLM-provided student_request should be replaced with the latest student wording."""
    coach_agent.llm_client._queued.clear()
    # Mock planner result to stop the loop
    def _fake_plan(params):
        return {
            "status": "complete",
            "plan": {"subject": "calculus", "learning_objective": "Derivatives", "mode": "conceptual_review"}
        }
    coach_agent.tutoring_planner.create_plan = _fake_plan

    # In the new flow, the coach loops and continues if planner is complete. 
    # Let's just make it return an empty message and exit.
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "Done",
            "action": "none",
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
    assert "Old request wording" not in coach_agent.collected_params.get("student_request", "")
    assert coach_agent.collected_params.get("student_request") == "Fresh conceptual review please"


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
    # The loop will continue since status="complete", so we need to provide a follow-up response
    # to stop the loop, e.g., starting the bot session.
    
    def _fake_faq_bot(**kwargs):
        return {
            "message_to_student": "Here is the syllabus.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "session_summary": {"topics_covered": ["syllabus_topics"]},
        }
    monkeypatch.setattr("src.workflow_demo.bot_sessions.faq_bot", _fake_faq_bot)

    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "start_faq",
            "tool_params": {"topic": "syllabus_topics", "student_request": "List the major concepts"},
            "conversation_summary": "starting FAQ",
        }
    )

    reply = coach_agent.process_turn("I want to ask a question about the syllabus")
    
    assert captured["params"]["topic"] == "syllabus_topics"
    assert "Here is the syllabus." in reply


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

    def _fake_faq_bot(**kwargs):
        return {
            "message_to_student": "Here is the syllabus overview.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "session_summary": {"topics_covered": ["syllabus_topics"]},
        }
    monkeypatch.setattr("src.workflow_demo.bot_sessions.faq_bot", _fake_faq_bot)

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
    assert "Could you clarify" in reply1
    assert plan_calls["count"] == 0

    # Second try with another 'none' action pushes clarification count to 2 (limit)
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "Still not sure what you mean.",
            "action": "none",
            "tool_params": {},
            "conversation_summary": "",
        }
    )
    reply2 = coach_agent.process_turn("Just give me the syllabus")
    # Clarification limit is reached, FAQ planner should be forced, starting FAQ bot session.
    assert "Here is the syllabus overview." in reply2
    assert plan_calls["count"] == 1
