"""End-to-end conversational flow tests for workflow_demo."""

import pytest

from src.workflow_demo.coach import CoachAgent


@pytest.mark.e2e
def test_tutoring_flow_end_to_end(monkeypatch, coach_agent: CoachAgent, mock_retriever):
    """Simulate a full tutoring flow from intent to completion."""

    def _ending_tutor_bot(**kwargs):
        return {
            "message_to_student": "Tutor kicking off the session...",
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
                "notes": "Tutor E2E",
            },
        }

    monkeypatch.setattr("src.workflow_demo.coach.tutor_bot", _ending_tutor_bot)

    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "I can help with derivatives. Let me craft a plan.",
            "action": "call_tutoring_planner",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "conceptual_review",
                "student_request": "I want to learn derivatives",
            },
            "conversation_summary": "",
        }
    )
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "Here's the plan—ready to start?",
            "action": "none",
            "tool_params": {},
            "conversation_summary": "",
        }
    )

    reply1 = coach_agent.process_turn("I want to learn derivatives")
    assert "plan" in reply1.lower()

    reply2 = coach_agent.process_turn("yes")
    assert reply2.startswith("Hi! I'm your learning coach")
    assert any(
        entry["summary"].get("notes") == "Tutor E2E"
        for entry in coach_agent.session_memory.get_recent_sessions()
    )


@pytest.mark.e2e
def test_faq_flow_end_to_end(monkeypatch, coach_agent: CoachAgent):
    """Simulate a FAQ session including confirmation and completion."""

    def _faq_bot(**kwargs):
        return {
            "message_to_student": "Exams run next Monday.",
            "end_activity": True,
            "silent_end": False,
            "needs_topic_confirmation": False,
            "session_summary": {
                "topics_addressed": ["exam schedule"],
                "questions_answered": ["When is the exam?"],
                "switch_topic_request": None,
                "notes": "FAQ E2E",
            },
        }

    monkeypatch.setattr("src.workflow_demo.coach.faq_bot", _faq_bot)

    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "Let me look up the exam information.",
            "action": "call_faq_planner",
            "tool_params": {"topic": "exam schedule", "student_request": "When is the exam?"},
            "conversation_summary": "",
        }
    )
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "I have the exam details—shall I connect you?",
            "action": "none",
            "tool_params": {},
            "conversation_summary": "",
        }
    )

    reply1 = coach_agent.process_turn("When is the exam?")
    assert "exam" in reply1.lower()

    reply2 = coach_agent.process_turn("yes")
    assert reply2.startswith("Hi! I'm your learning coach")
    assert any(
        entry["summary"].get("notes") == "FAQ E2E"
        for entry in coach_agent.session_memory.get_recent_sessions()
        if entry["type"] == "faq"
    )
