"""End-to-end conversational flow tests for workflow_demo."""

import pytest

from src.workflow_demo.coach_agent import CoachAgent


@pytest.mark.e2e
def test_tutoring_flow_end_to_end(monkeypatch, coach_agent: CoachAgent, mock_retriever):
    """Simulate a full tutoring flow from intent to session completion in one turn."""

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

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _ending_tutor_bot)

    # New flow: planner is called first, then the loop calls the LLM again and
    # gets start_tutor — both happen in the same process_turn() call.
    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
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
            "message_to_student": "",
            "action": "start_tutor",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "conceptual_review",
            },
            "conversation_summary": "Starting derivatives tutoring session.",
        }
    )

    reply = coach_agent.process_turn("I want to learn derivatives")
    assert "Nice work on Derivatives" in reply or reply.startswith("Hi! I'm your learning coach")
    assert any(
        entry["summary"].get("notes") == "Tutor E2E"
        for entry in coach_agent.session_memory.get_recent_sessions()
    )


@pytest.mark.e2e
def test_faq_flow_end_to_end(monkeypatch, coach_agent: CoachAgent):
    """Simulate a FAQ session going straight from planner to FAQ bot in one turn."""

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

    monkeypatch.setattr("src.workflow_demo.bot_sessions.faq_bot", _faq_bot)

    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "call_faq_planner",
            "tool_params": {"topic": "exam schedule", "student_request": "When is the exam?"},
            "conversation_summary": "",
        }
    )
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "start_faq",
            "tool_params": {"topic": "exam schedule", "student_request": "When is the exam?"},
            "conversation_summary": "Starting FAQ session for exam schedule.",
        }
    )

    reply = coach_agent.process_turn("When is the exam?")
    assert "Glad I could help" in reply or reply.startswith("Hi! I'm your learning coach")
    assert any(
        entry["summary"].get("notes") == "FAQ E2E"
        for entry in coach_agent.session_memory.get_recent_sessions()
        if entry["type"] == "faq"
    )


@pytest.mark.e2e
def test_syllabus_faq_flow_end_to_end(monkeypatch, coach_agent: CoachAgent):
    """Ensure syllabus-style questions route to FAQ flow in one turn."""

    def _syllabus_faq_bot(**kwargs):
        return {
            "message_to_student": "Here are the major calculus concepts...",
            "end_activity": True,
            "silent_end": False,
            "needs_topic_confirmation": False,
            "session_summary": {
                "topics_addressed": ["syllabus_topics"],
                "questions_answered": ["Can you list out the major concepts in my syllabus?"],
                "switch_topic_request": None,
                "notes": "Syllabus FAQ",
            },
        }

    monkeypatch.setattr("src.workflow_demo.bot_sessions.faq_bot", _syllabus_faq_bot)

    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "call_faq_planner",
            "tool_params": {"topic": "syllabus_topics", "student_request": "Can you list the major concepts?"},
            "conversation_summary": "",
        }
    )
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "start_faq",
            "tool_params": {"topic": "syllabus_topics", "student_request": "Can you list the major concepts?"},
            "conversation_summary": "Starting syllabus FAQ session.",
        }
    )

    reply = coach_agent.process_turn("Can you list out the major concepts in my syllabus?")
    assert "Glad I could help" in reply or reply.startswith("Hi! I'm your learning coach")
    assert any(
        entry["summary"].get("notes") == "Syllabus FAQ"
        for entry in coach_agent.session_memory.get_recent_sessions()
        if entry["type"] == "faq"
    )
