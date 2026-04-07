"""
Minimal CoachAgent wiring for the pedagogy eval harness (no pytest dependency).
Mirrors ``tests/conftest.py`` ``coach_agent`` fixture.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd

from src.workflow_demo.bot_sessions import BotSessionManager
from src.workflow_demo.coach_agent import CoachAgent
from src.workflow_demo.coach_llm_client import CoachLLMClient
from src.workflow_demo.coach_router import CoachRouter
from src.workflow_demo.data_loader import KnowledgeGraphData
from src.workflow_demo.models import PlanStep, RetrievalCandidate, RetrievalResult, SessionPlan, TeachingPack


class DummyLLMClient:
    """Minimal OpenAI-compatible client for deterministic eval runs."""

    def __init__(self) -> None:
        self._queued: List[str] = []
        self.calls: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    def queue_response(self, payload: Any) -> None:
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        self._queued.append(payload)

    def _create_completion(self, **kwargs: Any) -> SimpleNamespace:
        if not self._queued:
            raise AssertionError("DummyLLMClient received a call with no queued responses.")
        content = self._queued.pop(0)
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


SAMPLE_COACH_DIRECTIVE: Dict[str, Any] = {
    "message_to_student": "Let's gather a bit more detail first.",
    "action": "none",
    "tool_params": {"subject": "calculus"},
    "conversation_summary": "",
}


def _sample_teaching_pack() -> TeachingPack:
    return TeachingPack(
        key_points=["The Tangent Problem is the anchor concept for derivatives."],
        examples=[
            {
                "content_id": "ex_1",
                "lo_title": "Toolkit",
                "content_type": "concept",
                "snippet": "Example snippet.",
                "score": 0.42,
            }
        ],
        practice=[],
        prerequisites=[],
        citations=[],
    )


def _sample_session_plan() -> SessionPlan:
    pack = _sample_teaching_pack()
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
                how_to_teach="Intuition first.",
                why_to_teach="Ground the learner.",
            )
        ],
        future_plan=[],
        first_question="What do you already know?",
        teaching_pack=pack,
        book="Calculus Volume 1",
        unit="Limits",
        chapter="1",
    )


def _sample_kg_data() -> KnowledgeGraphData:
    los_df = pd.DataFrame(
        [
            {
                "lo_id": 1893,
                "learning_objective": "The Tangent Problem and Differential Calculus",
                "unit": "Limits",
                "chapter": "1",
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
                "text": "Snippet.",
            },
        ]
    )
    edges_prereqs_df = pd.DataFrame(columns=["source_lo_id", "target_lo_id"])
    edges_content_df = pd.DataFrame(
        [{"source_lo_id": 1893, "target_content_id": "ex_1"}]
    )
    return KnowledgeGraphData(
        los=los_df,
        content=content_df,
        edges_prereqs=edges_prereqs_df,
        edges_content=edges_content_df,
        prereq_in_map={1893: []},
        content_ids_map={1893: ["ex_1"]},
        lo_lookup={1893: los_df.iloc[0].to_dict()},
    )


class _MockRetriever:
    def __init__(self, plan: SessionPlan, kg: KnowledgeGraphData) -> None:
        self.plan = plan
        self.kg = kg
        self.calls: List[Dict[str, Any]] = []

    def retrieve_candidates(
        self,
        text_query: str,
        image_path=None,
        top_k: int = 6,
        debug: bool = False,
    ) -> RetrievalResult:
        self.calls.append({"text_query": text_query, "image_path": image_path})
        candidate = RetrievalCandidate(
            lo_id=1,
            title=self.plan.learning_objective,
            score=0.95,
            source="text_embedding",
            book=self.plan.book,
        )
        return RetrievalResult(
            query=text_query,
            text_candidates=[candidate],
            image_candidates=[],
            merged_candidates=[candidate],
        )

    def retrieve_plan(self, **kwargs: Any) -> SessionPlan:
        self.calls.append(kwargs)
        return self.plan


def build_eval_coach() -> CoachAgent:
    """Return a CoachAgent wired like the pytest ``coach_agent`` fixture."""
    plan = _sample_session_plan()
    kg = _sample_kg_data()
    retriever = _MockRetriever(plan, kg)
    agent = CoachAgent(retriever=retriever, llm_model="mock-model")
    llm = DummyLLMClient()
    llm.queue_response(SAMPLE_COACH_DIRECTIVE)
    agent.llm_client = llm
    agent.llm_model = "mock-model"
    agent.llm_client_wrapper = CoachLLMClient(llm, "mock-model")
    agent.coach_router = CoachRouter(agent, agent.llm_client_wrapper)
    agent.bot_session_manager = BotSessionManager(agent)
    return agent


def sample_handoff_context_for_math_guard() -> Dict[str, Any]:
    """Minimal handoff for direct ``tutor_bot`` math-guard scenario (mirrors test fixture shape)."""
    from src.workflow_demo.session_memory import SessionMemory, create_handoff_context

    plan = _sample_session_plan()
    memory = SessionMemory(max_entries=3)
    plan_dict = {
        "subject": plan.subject,
        "book": plan.book,
        "unit": plan.unit,
        "chapter": plan.chapter,
        "learning_objective": plan.learning_objective,
        "mode": plan.mode,
        "current_plan": [asdict(step) for step in plan.current_plan],
        "future_plan": [asdict(step) for step in plan.future_plan],
        "first_question": plan.first_question,
        "teaching_pack": asdict(plan.teaching_pack),
    }
    return create_handoff_context(
        from_agent="coach",
        to_agent="tutor",
        session_params=plan_dict,
        conversation_summary="Eval math guard.",
        session_memory=memory,
        student_state={"lo_mastery": {1893: 0.6}},
    )
