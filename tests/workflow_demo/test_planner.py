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
def test_tutoring_planner_missing_query(monkeypatch, mock_retriever):
    """Planner should request clarification when no tutoring query is provided."""
    planner = TutoringPlanner(mock_retriever)
    response = planner.create_plan({"student_request": "", "learning_objective": "", "mode": ""})
    assert response["status"] == "need_info"
    assert "What topic would you like to learn about?" in response["message"]


@pytest.mark.unit
def test_tutoring_planner_returns_complete_plan(monkeypatch, mock_retriever, sample_session_plan):
    """Planner should call candidate retrieval and return a simplified plan payload."""
    planner = TutoringPlanner(mock_retriever)
    params = {
        "student_request": "Teach me derivatives",
        "mode": "conceptual_review",
        "student_profile": {"lo_mastery": {}},
    }
    response = planner.create_plan(params)
    assert response["status"] == "complete"
    assert response["plan"]["mode"] == "conceptual_review"
    assert response["plan"]["subject"] == "calculus"
    assert mock_retriever.calls, "Retriever should be invoked with planner params."
    assert response["plan"]["current_plan"]
    assert "how_to_teach" in response["plan"]["current_plan"][0]
    assert "why_to_teach" in response["plan"]["current_plan"][0]


@pytest.mark.unit
def test_tutoring_planner_llm_need_info(monkeypatch, mock_retriever):
    """When LLM output is unusable, planner should fall back to deterministic planning."""
    planner = TutoringPlanner(mock_retriever)

    def _fake_chat_json(system_prompt, user_prompt):
        return {"status": "need_info", "message": "Clarify the learning objective."}

    monkeypatch.setattr(planner, "_chat_json", _fake_chat_json)
    planner.llm_client = object()
    planner.llm_model = "mock-model"

    response = planner.create_plan(
        {"learning_objective": "Derivatives", "mode": "practice", "student_request": "Help"}
    )
    assert response["status"] == "complete"
    assert response["plan"]["mode"] == "practice"
    assert mock_retriever.calls, "Retriever should run before LLM-assisted plan selection."


@pytest.mark.unit
def test_tutoring_planner_infers_trigonometry_from_unit_context(mock_retriever):
    """Trig-heavy unit metadata should map the tutoring subject to trigonometry."""
    planner = TutoringPlanner(mock_retriever)
    candidate = mock_retriever.retrieve_candidates("Trigonometry").merged_candidates[0]
    candidate.book = "Calculus Volume 1"
    candidate.unit = "Trigonometric Functions"
    candidate.chapter = "Functions and Graphs"
    candidate.title = "Trigonometric Identities"

    plan = planner._build_heuristic_plan(
        candidates=[candidate],
        mode="conceptual_review",
        proficiency_map={candidate.title: 0.0},
    )

    assert plan["subject"] == "trigonometry"


@pytest.mark.unit
def test_faq_planner_known_topic(monkeypatch):
    """FAQ planner should return the canned script for known topics."""
    planner = FAQPlanner()
    response = planner.create_plan({"topic": "exam schedule", "student_request": "When is the exam?"})
    assert response["status"] == "complete"
    assert "exam calendar" in response["plan"]["script"]


@pytest.mark.unit
def test_faq_planner_syllabus_topics(monkeypatch):
    """FAQ planner should support the syllabus topics overview."""
    planner = FAQPlanner()
    response = planner.create_plan({"topic": "syllabus_topics", "student_request": "List the major concepts"})
    assert response["status"] == "complete"
    assert response["plan"]["topic"] == "syllabus_topics"
    assert "Limits and continuity" in response["plan"]["script"]


@pytest.mark.unit
def test_faq_planner_unknown_topic(monkeypatch):
    """Unknown topics should prompt for clarification."""
    planner = FAQPlanner()
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
    planner.llm_client = object()
    planner.llm_model = "mock-model"

    response = planner.create_plan({"topic": None, "student_request": "Tell me about grades"})
    assert response["status"] == "need_info"
    assert "grading" in response["message"]
