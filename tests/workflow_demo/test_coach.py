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
    """A planner directive should invoke the tutoring planner and return its follow-up message."""
    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "call_tutoring_planner",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "conceptual_review",
                "student_request": "Teach me derivatives",
            },
            "conversation_summary": "",
        }
    )
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "Plan created—ready to start when you are.",
            "action": "none",
            "tool_params": {},
            "conversation_summary": "",
        }
    )

    reply = coach_agent.process_turn("Let's learn derivatives")
    assert reply == "Plan created—ready to start when you are."
    assert mock_retriever.calls, "Retriever should be invoked once."


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

    reply = coach_agent.process_turn("yes")
    assert reply.startswith("Hi! I'm your learning coach")
    assert coach_agent.in_bot_session is False
