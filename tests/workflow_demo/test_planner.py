"""Unit tests for workflow_demo planner utilities."""

from types import SimpleNamespace

import pytest

from src.workflow_demo.planner import FAQPlanner, TutoringPlanner


@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    """Prevent real OpenAI client creation during planner tests."""

    class _DummyOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=None)

    monkeypatch.setattr("src.workflow_demo.planner.OpenAI", lambda *args, **kwargs: _DummyOpenAI())


@pytest.mark.unit
def test_tutoring_planner_missing_fields(monkeypatch, mock_retriever):
    """Planner should request clarification when required fields are blank."""
    planner = TutoringPlanner(mock_retriever)
    planner.llm_enabled = False  # skip LLM call for this unit test
    response = planner.create_plan({"subject": "calculus", "learning_objective": "", "mode": ""})
    assert response["status"] == "need_info"
    assert "Missing fields" in response["message"]


@pytest.mark.unit
def test_tutoring_planner_returns_complete_plan(monkeypatch, mock_retriever, sample_session_plan):
    """Planner should call the retriever and return the SessionPlan payload."""
    planner = TutoringPlanner(mock_retriever)
    planner.llm_enabled = False
    params = {
        "subject": "calculus",
        "learning_objective": sample_session_plan.learning_objective,
        "mode": "conceptual_review",
        "student_request": "Teach me derivatives",
        "student_profile": {"lo_mastery": {}},
    }
    response = planner.create_plan(params)
    assert response["status"] == "complete"
    assert response["plan"]["learning_objective"] == sample_session_plan.learning_objective
    assert mock_retriever.calls, "Retriever should be invoked with planner params."


@pytest.mark.unit
def test_tutoring_planner_llm_need_info(monkeypatch, mock_retriever):
    """When the LLM asks for clarification, the planner should propagate need_info."""
    planner = TutoringPlanner(mock_retriever)

    def _fake_chat_json(system_prompt, user_prompt):
        return {"status": "need_info", "message": "Clarify the learning objective."}

    monkeypatch.setattr(planner, "_chat_json", _fake_chat_json)
    planner.llm_enabled = True

    response = planner.create_plan(
        {"subject": "calculus", "learning_objective": "Derivatives", "mode": "practice", "student_request": "Help"}
    )
    assert response["status"] == "need_info"
    assert response["message"] == "Clarify the learning objective."
    assert not mock_retriever.calls, "Retriever should not run when clarification is pending."


@pytest.mark.unit
def test_faq_planner_known_topic(monkeypatch):
    """FAQ planner should return the canned script for known topics."""
    planner = FAQPlanner()
    planner.llm_enabled = False
    response = planner.create_plan({"topic": "exam schedule", "student_request": "When is the exam?"})
    assert response["status"] == "complete"
    assert "exam calendar" in response["plan"]["script"]


@pytest.mark.unit
def test_faq_planner_syllabus_topics(monkeypatch):
    """FAQ planner should support the syllabus topics overview."""
    planner = FAQPlanner()
    planner.llm_enabled = False
    response = planner.create_plan({"topic": "syllabus_topics", "student_request": "List the major concepts"})
    assert response["status"] == "complete"
    assert response["plan"]["topic"] == "syllabus_topics"
    assert "Limits and continuity" in response["plan"]["script"]


@pytest.mark.unit
def test_faq_planner_unknown_topic(monkeypatch):
    """Unknown topics should prompt for clarification."""
    planner = FAQPlanner()
    planner.llm_enabled = False
    response = planner.create_plan({"topic": "", "student_request": ""})
    assert response["status"] == "need_info"
    assert "Which FAQ or syllabus topic" in response["message"]


@pytest.mark.unit
def test_faq_planner_llm_need_info(monkeypatch):
    """LLM clarification from FAQ planner should surface to the caller."""
    planner = FAQPlanner()

    def _fake_chat_json(system_prompt, user_prompt):
        return {"status": "need_info", "message": "Are you asking about grading or homework?"}

    monkeypatch.setattr(planner, "_chat_json", _fake_chat_json)
    planner.llm_enabled = True

    response = planner.create_plan({"topic": None, "student_request": "Tell me about grades"})
    assert response["status"] == "need_info"
    assert "grading" in response["message"]
