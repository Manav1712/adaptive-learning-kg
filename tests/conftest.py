"""Shared pytest fixtures for the workflow_demo test suite.

These fixtures provide:
* Mock OpenAI clients with controllable responses
* Sample knowledge graph data for retriever-agnostic tests
* Canonical dataclass instances (SessionPlan, TeachingPack, etc.)
* Reusable component stubs (mock retriever, pre-wired CoachAgent)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.workflow_demo.coach import CoachAgent
from src.workflow_demo.data_loader import KnowledgeGraphData
from src.workflow_demo.models import PlanStep, SessionPlan, TeachingPack
from src.workflow_demo.session_memory import SessionMemory, create_handoff_context


class DummyLLMClient:
    """Minimal OpenAI-compatible client for deterministic unit tests."""

    def __init__(self) -> None:
        self._queued: List[str] = []
        self.calls: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    def queue_response(self, payload: Any) -> None:
        """Add a JSON (or dict) response to the outgoing queue."""
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        self._queued.append(payload)

    def _create_completion(self, **kwargs: Any) -> SimpleNamespace:
        if not self._queued:
            raise AssertionError("DummyLLMClient received a call with no queued responses.")
        content = self._queued.pop(0)
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


# ---------------------------------------------------------------------------
# Mock LLM fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_response():
    """Factory to turn a dict payload into a serialized chat completion response."""

    def _factory(payload: Dict[str, Any]) -> str:
        return json.dumps(payload)

    return _factory


@pytest.fixture
def mock_openai_client():
    """Provide a queued-response OpenAI client stand-in."""
    return DummyLLMClient()


@pytest.fixture
def sample_coach_directive() -> Dict[str, Any]:
    """Representative directive emitted by the coach brain LLM."""
    return {
        "message_to_student": "Let's gather a bit more detail first.",
        "action": "none",
        "tool_params": {"subject": "calculus"},
        "conversation_summary": "",
    }


@pytest.fixture
def sample_tutor_response() -> Dict[str, Any]:
    """Representative tutor bot JSON output."""
    return {
        "message_to_student": "Here is a focused explanation grounded in the teaching pack.",
        "end_activity": False,
        "silent_end": False,
        "needs_mode_confirmation": False,
        "needs_topic_confirmation": False,
        "requested_mode": None,
        "session_summary": {
            "topics_covered": ["The Tangent Problem and Differential Calculus"],
            "student_understanding": "good",
            "suggested_next_topic": None,
            "switch_topic_request": None,
            "switch_mode_request": None,
            "notes": "Sample summary",
        },
    }


@pytest.fixture
def sample_faq_response() -> Dict[str, Any]:
    """Representative FAQ bot JSON output."""
    return {
        "message_to_student": "The next exam window runs from May 10-12.",
        "end_activity": True,
        "silent_end": False,
        "needs_topic_confirmation": False,
        "session_summary": {
            "topics_addressed": ["exam schedule"],
            "questions_answered": ["When is the next exam?"],
            "switch_topic_request": None,
            "notes": "Sample FAQ session",
        },
    }


@pytest.fixture
def sample_syllabus_faq_plan() -> Dict[str, Any]:
    """FAQ planner output for syllabus topics."""
    return {
        "status": "complete",
        "plan": {
            "topic": "syllabus_topics",
            "script": "Major calculus concepts: limits, derivatives, applications, integrals, FTC.",
            "student_request": "Can you list the major concepts?",
            "first_question": "Which concept would you like to dive into next?",
        },
        "message": None,
    }


@pytest.fixture
def sample_planner_response() -> Dict[str, Any]:
    """Canonical planner response structure."""
    return {
        "status": "complete",
        "plan": {
            "subject": "calculus",
            "book": "Calculus Volume 1",
            "unit": "Limits",
            "chapter": "1: The Tangent Problem",
            "learning_objective": "The Tangent Problem and Differential Calculus",
            "mode": "conceptual_review",
            "first_question": "What do you already know about the tangent problem?",
            "current_plan": [],
            "future_plan": [],
            "teaching_pack": {
                "key_points": ["Key point placeholder"],
                "examples": [],
                "practice": [],
                "prerequisites": [],
                "citations": [],
            },
        },
        "message": None,
    }


# ---------------------------------------------------------------------------
# Data + dataclass fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_teaching_pack() -> TeachingPack:
    """Reusable teaching pack for tutoring-related tests."""
    return TeachingPack(
        key_points=[
            "The Tangent Problem is the anchor concept for derivatives.",
            "Related LO: Linear Functions and Slope.",
        ],
        examples=[
            {
                "content_id": "ex_1",
                "lo_title": "Identifying Basic Toolkit Functions",
                "content_type": "concept",
                "snippet": "Example snippet text.",
                "score": 0.42,
            }
        ],
        practice=[
            {
                "content_id": "pr_1",
                "lo_title": "The Tangent Problem and Differential Calculus",
                "content_type": "try_it",
                "snippet": "Practice prompt snippet.",
                "score": 0.37,
            }
        ],
        prerequisites=[
            {
                "lo_id": 101,
                "title": "Functions and Function Notation",
                "note": "Recommended refresher.",
            }
        ],
        citations=[
            {"content_id": "ex_1", "lo_title": "Identifying Basic Toolkit Functions", "score": 0.42},
        ],
    )


@pytest.fixture
def sample_session_plan(sample_teaching_pack: TeachingPack) -> SessionPlan:
    """Session plan assembled from the sample teaching pack."""
    return SessionPlan(
        subject="calculus",
        learning_objective="The Tangent Problem and Differential Calculus",
        mode="conceptual_review",
        current_plan=[
            PlanStep(
                step_id="1",
                step_type="explain",
                goal="Describe the core idea.",
                lo_id=1893,
                content_id=None,
            )
        ],
        future_plan=[
            PlanStep(
                step_id="F1",
                step_type="extension",
                goal="Extend to Linear Functions and Slope.",
                lo_id=1872,
                content_id=None,
            )
        ],
        first_question="What do you already know about the tangent problem?",
        teaching_pack=sample_teaching_pack,
        book="Calculus Volume 1",
        unit="A Preview of Calculus",
        chapter="Limits",
    )


@pytest.fixture
def sample_session_memory(sample_session_plan: SessionPlan) -> SessionMemory:
    """Pre-populated session memory for continuity tests."""
    memory = SessionMemory(max_entries=3)
    memory.add_session(
        session_type="tutor",
        params={
            "subject": sample_session_plan.subject,
            "learning_objective": sample_session_plan.learning_objective,
            "mode": sample_session_plan.mode,
        },
        summary={
            "topics_covered": [sample_session_plan.learning_objective],
            "student_understanding": "good",
        },
        conversation_exchanges=[{"speaker": "student", "text": "Hi"}, {"speaker": "assistant", "text": "Hello"}],
    )
    return memory


@pytest.fixture
def sample_handoff_context(sample_session_plan: SessionPlan, sample_session_memory: SessionMemory) -> Dict[str, Any]:
    """Handoff context mirroring the notebook-inspired structure."""
    plan_dict = {
        "subject": sample_session_plan.subject,
        "book": sample_session_plan.book,
        "unit": sample_session_plan.unit,
        "chapter": sample_session_plan.chapter,
        "learning_objective": sample_session_plan.learning_objective,
        "mode": sample_session_plan.mode,
        "current_plan": [asdict(step) for step in sample_session_plan.current_plan],
        "future_plan": [asdict(step) for step in sample_session_plan.future_plan],
        "first_question": sample_session_plan.first_question,
        "teaching_pack": asdict(sample_session_plan.teaching_pack),
    }
    return create_handoff_context(
        from_agent="coach",
        to_agent="tutor",
        session_params=plan_dict,
        conversation_summary="Student wants to revisit derivatives.",
        session_memory=sample_session_memory,
        student_state={"lo_mastery": {1893: 0.6}},
    )


@pytest.fixture
def sample_kg_data() -> KnowledgeGraphData:
    """Minimal in-memory knowledge graph for retriever/planner tests."""
    los_df = pd.DataFrame(
        [
            {
                "lo_id": 1893,
                "learning_objective": "The Tangent Problem and Differential Calculus",
                "unit": "A Preview of Calculus",
                "chapter": "Limits",
                "book": "Calculus Volume 1",
                "subject": "calculus",
            },
            {
                "lo_id": 1872,
                "learning_objective": "Linear Functions and Slope",
                "unit": "Basic Functions",
                "chapter": "Functions",
                "book": "Calculus Volume 1",
                "subject": "calculus",
            },
        ]
    )
    content_df = pd.DataFrame(
        [
            {
                "content_id": "ex_1",
                "lo_id_parent": 1893,
                "content_type": "concept",
                "learning_objective": "The Tangent Problem and Differential Calculus",
                "text": "Concept text snippet.",
            },
            {
                "content_id": "pr_1",
                "lo_id_parent": 1893,
                "content_type": "try_it",
                "learning_objective": "The Tangent Problem and Differential Calculus",
                "text": "Practice text snippet.",
            },
        ]
    )
    edges_prereqs_df = pd.DataFrame(
        [
            {"source_lo_id": 1872, "target_lo_id": 1893},
        ]
    )
    edges_content_df = pd.DataFrame(
        [
            {"source_lo_id": 1893, "target_content_id": "ex_1"},
            {"source_lo_id": 1893, "target_content_id": "pr_1"},
        ]
    )
    prereq_map = {1893: [1872]}
    content_ids_map = {1893: ["ex_1", "pr_1"]}
    lo_lookup = {
        1893: los_df.iloc[0].to_dict(),
        1872: los_df.iloc[1].to_dict(),
    }
    return KnowledgeGraphData(
        los=los_df,
        content=content_df,
        edges_prereqs=edges_prereqs_df,
        edges_content=edges_content_df,
        prereq_in_map=prereq_map,
        content_ids_map=content_ids_map,
        lo_lookup=lo_lookup,
    )


# ---------------------------------------------------------------------------
# Component fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_retriever(sample_session_plan: SessionPlan, sample_kg_data: KnowledgeGraphData):
    """Simple retriever stub that records calls and returns the sample session plan."""

    class _MockRetriever:
        def __init__(self, plan: SessionPlan, kg: KnowledgeGraphData) -> None:
            self.plan = plan
            self.kg = kg
            self.calls: List[Dict[str, Any]] = []

        def retrieve_plan(self, **kwargs: Any) -> SessionPlan:
            self.calls.append(kwargs)
            return self.plan

    return _MockRetriever(sample_session_plan, sample_kg_data)


@pytest.fixture
def coach_agent(mock_retriever, mock_openai_client, sample_coach_directive):
    """
    CoachAgent wired up with the mock retriever and mock LLM client.

    Tests enqueue directives into the mock client before invoking `process_turn`.
    """

    agent = CoachAgent(retriever=mock_retriever, llm_model="mock-model")
    agent.llm_client = mock_openai_client
    agent.llm_model = "mock-model"
    mock_openai_client.queue_response(sample_coach_directive)
    return agent
