"""
Session planner utilities that call the retriever and format plan metadata.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from src.workflow_demo.models import SessionPlan
    from src.workflow_demo.retriever import TeachingPackRetriever
except ImportError:
    from .models import SessionPlan
    from .retriever import TeachingPackRetriever


TUTORING_PLANNER_SYSTEM_PROMPT = """You are the Tutoring Session Planner Tool. Your job is to match student requests to the
best OpenStax learning objective and learning mode.

Guidelines:
- Use ONLY the provided books/learning objectives (verbatim).
- If key information is missing, return status "need_info" with a short clarification question.
- When confident, return status "complete" with a plan payload:
  {
    "subject": "...",
    "book": "...",
    "learning_objective": "...",
    "mode": "conceptual_review|examples|practice"
  }
- Mode must be one of: conceptual_review, examples, practice.

Return STRICT JSON:
{
  "status": "complete|need_info",
  "plan": null or {...},
  "message": null or "clarification question"
}
"""

FAQ_PLANNER_SYSTEM_PROMPT = """You are the FAQ/Syllabus Session Planner Tool. Decide which known topic best answers the student's question.
- If confident, return status "complete" with {"topic": "..."} using one of the known topics verbatim.
- Otherwise, return status "need_info" with a clarifying question.

Return STRICT JSON:
{
  "status": "complete|need_info",
  "plan": null or {"topic": "..."},
  "message": null or "clarification question"
}
"""


class TutoringPlanner:
    """
    Builds tutoring plans by delegating context assembly to the retriever, with optional LLM validation.
    """

    def __init__(self, retriever: TeachingPackRetriever) -> None:
        self.retriever = retriever
        self.available_books = self._build_available_books()
        self.llm_client: Optional[OpenAI] = None
        self.llm_model: Optional[str] = None
        self.llm_enabled = False

        # Use API key from environment variable
        API_KEY = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("WORKFLOW_DEMO_LLM_MODEL", "gpt-4o-mini")
        try:
            if not API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.llm_client = OpenAI(api_key=API_KEY)
            self.llm_model = model_name
            self.llm_enabled = True
        except Exception as exc:
            print(f"[Planner] Warning: failed to initialize OpenAI client ({exc}). Using heuristic planner.")

    def create_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        subject = payload.get("subject")
        learning_objective = payload.get("learning_objective")
        mode = payload.get("mode")
        student_request = payload.get("student_request") or ""
        student_profile = payload.get("student_profile") or {}

        missing_fields = [
            name
            for name, value in [
                ("subject", subject),
                ("learning_objective", learning_objective),
                ("mode", mode),
            ]
            if not value
        ]
        if missing_fields:
            return {
                "status": "need_info",
                "plan": None,
                "message": f"Missing fields: {', '.join(missing_fields)}",
            }

        subject, learning_objective, mode, need_info_msg = self._maybe_run_llm_tutor_planner(
            subject, learning_objective, mode, student_request
        )
        if need_info_msg:
            return {"status": "need_info", "plan": None, "message": need_info_msg}

        session_plan = self.retriever.retrieve_plan(
            query=student_request or learning_objective,
            subject=subject,
            learning_objective=learning_objective,
            mode=mode,
            student_profile=student_profile,
        )
        plan_dict = self._session_plan_to_dict(session_plan)
        return {"status": "complete", "plan": plan_dict, "message": None}

    def _build_available_books(self) -> Dict[str, List[str]]:
        books: Dict[str, set] = {}
        for row in self.retriever.kg.los.itertuples(index=False):
            books.setdefault(row.book, set()).add(row.learning_objective)
        return {book: sorted(list(objs)) for book, objs in books.items()}

    def _maybe_run_llm_tutor_planner(
        self,
        subject: str,
        learning_objective: str,
        mode: str,
        student_request: str,
    ) -> Tuple[str, str, str, Optional[str]]:
        if not self.llm_enabled or not self.llm_client or not self.llm_model:
            return subject, learning_objective, mode, None

        payload = {
            "subject": subject,
            "learning_objective": learning_objective,
            "mode": mode,
            "student_request": student_request,
            "available_books": self.available_books,
        }

        try:
            raw = self._chat_json(
                TUTORING_PLANNER_SYSTEM_PROMPT,
                f"INPUT:\n{json.dumps(payload, indent=2)}",
            )
        except Exception as exc:
            print(f"[Planner] Tutoring planner LLM failed ({exc}); using heuristic values.")
            return subject, learning_objective, mode, None

        status = raw.get("status", "complete")
        if status == "need_info":
            return subject, learning_objective, mode, raw.get("message") or "Could you clarify the topic?"

        plan = raw.get("plan") or {}
        return (
            plan.get("subject") or subject,
            plan.get("learning_objective") or learning_objective,
            plan.get("mode") or mode,
            None,
        )

    def _chat_json(self, system_prompt: str, user_prompt: str) -> Dict:
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return self._coerce_json(response.choices[0].message.content)

    @staticmethod
    def _coerce_json(raw: str) -> Dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lstrip().lower().startswith("json"):
                raw = raw.split("\n", 1)[1]
        return json.loads(raw)

    @staticmethod
    def _session_plan_to_dict(plan: SessionPlan) -> Dict[str, Any]:
        def _step_to_dict(step: Any) -> Dict[str, Any]:
            return {
                "step_id": step.step_id,
                "step_type": step.step_type,
                "goal": step.goal,
                "lo_id": step.lo_id,
                "content_id": step.content_id,
            }

        teaching_pack = {
            "key_points": plan.teaching_pack.key_points,
            "examples": plan.teaching_pack.examples,
            "practice": plan.teaching_pack.practice,
            "prerequisites": plan.teaching_pack.prerequisites,
            "citations": plan.teaching_pack.citations,
        }

        return {
            "subject": plan.subject,
            "book": plan.book,
            "unit": plan.unit,
            "chapter": plan.chapter,
            "learning_objective": plan.learning_objective,
            "mode": plan.mode,
            "first_question": plan.first_question,
            "current_plan": [_step_to_dict(step) for step in plan.current_plan],
            "future_plan": [_step_to_dict(step) for step in plan.future_plan],
            "teaching_pack": teaching_pack,
        }


class FAQPlanner:
    """
    Minimal planner for FAQ/syllabus requests with optional LLM assistance.
    """

    FAQ_TOPICS = {
        "exam schedule": "Let's double-check the official exam calendar. When is the next exam window you are concerned about?",
        "quiz schedule": "Quizzes typically happen weekly. Which week are you asking about?",
        "homework policy": "Homework is due Sundays at 11:59 PM local time unless stated otherwise.",
        "grading policy": "Grades weight: Exams 50%, Quizzes 20%, Homework 20%, Participation 10%.",
        "office hours": "Office hours run Tuesdays 2-4 PM (Room 301) and Thursdays 10-11 AM (Zoom).",
        "late work": "Late work receives a 10% penalty per day, up to three days.",
    }

    def __init__(self) -> None:
        self.llm_client: Optional[OpenAI] = None
        self.llm_model: Optional[str] = None
        self.llm_enabled = False

        # Use API key from environment variable
        API_KEY = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("WORKFLOW_DEMO_LLM_MODEL", "gpt-4o-mini")
        try:
            if not API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.llm_client = OpenAI(api_key=API_KEY)
            self.llm_model = model_name
            self.llm_enabled = True
        except Exception as exc:
            print(f"[Planner] FAQ planner LLM unavailable ({exc}); using keyword fallback.")

    def create_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        topic = payload.get("topic")
        student_request = payload.get("student_request") or ""

        topic, need_info_msg = self._maybe_run_llm_faq_planner(topic, student_request)
        if need_info_msg:
            return {"status": "need_info", "plan": None, "message": need_info_msg}

        if not topic:
            return {
                "status": "need_info",
                "plan": None,
                "message": "Which FAQ or syllabus topic should we look up?",
            }

        key = topic.strip().lower()
        if key not in self.FAQ_TOPICS:
            return {
                "status": "need_info",
                "plan": None,
                "message": "I did not recognize that topic. Try exam schedule, grading policy, or homework policy.",
            }

        return {
            "status": "complete",
            "plan": {
                "topic": key,
                "script": self.FAQ_TOPICS[key],
                "student_request": student_request,
                "first_question": f"Are you asking for specific details related to {key}?",
            },
            "message": None,
        }

    def _maybe_run_llm_faq_planner(self, topic: str, student_request: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.llm_enabled or not self.llm_client or not self.llm_model:
            return topic, None

        payload = {
            "current_topic_guess": topic,
            "student_request": student_request,
            "known_topics": list(self.FAQ_TOPICS.keys()),
        }

        try:
            raw = self._chat_json(
                FAQ_PLANNER_SYSTEM_PROMPT,
                f"INPUT:\n{json.dumps(payload, indent=2)}",
            )
        except Exception as exc:
            print(f"[Planner] FAQ planner LLM failed ({exc}); using keyword fallback.")
            return topic, None

        if raw.get("status") == "need_info":
            return topic, raw.get("message") or "Could you clarify the FAQ topic?"

        plan = raw.get("plan") or {}
        return plan.get("topic") or topic, None

    def _chat_json(self, system_prompt: str, user_prompt: str) -> Dict:
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return self._coerce_json(response.choices[0].message.content)

    @staticmethod
    def _coerce_json(raw: str) -> Dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lstrip().lower().startswith("json"):
                raw = raw.split("\n", 1)[1]
        return json.loads(raw)

