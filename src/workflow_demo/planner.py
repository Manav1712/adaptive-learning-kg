"""
Session planner utilities that call the retriever and format plan metadata.

New simplified plan format per meeting requirements:
- current_plan: 1 primary LO + up to 2 dependent LOs (all same mode)
- future_plan: 1 LO
- Each LO includes proficiency score and teaching notes from KG
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from .json_utils import coerce_json
from .models import RetrievalCandidate
from .retriever import TeachingPackRetriever


TUTORING_PLANNER_PROMPT = """You are the tutoring session planner for an adaptive learning system.

Your job is to convert retrieved learning-objective candidates into one focused tutoring plan.

You will receive:
- `student_request`: the student's question or goal
- `mode`: one of `conceptual_review`, `examples`, or `practice`
- `candidates`: ranked learning-objective candidates from retrieval
- `student_proficiency`: a mapping from candidate title to mastery score from `0.0` to `1.0`

Plan selection rules:
1. Use only the provided candidates. Never invent or rename ids, titles, book, unit, chapter, `how_to_teach`, or `why_to_teach`.
2. Return exactly 1 primary learning objective in `current_plan` with `is_primary: true`.
3. Add 0 to 2 supporting learning objectives to `current_plan` only if they are genuine prerequisites or close supporting concepts for the primary objective.
4. Return exactly 1 learning objective in `future_plan`. It must not duplicate any item already used in `current_plan`.
5. Keep `mode` exactly equal to the input mode.
6. Prefer the candidate that best matches the student request. Use retrieval rank as a tiebreaker.
7. Use lower proficiency scores to justify support LOs and beginner notes. Use higher proficiency scores to justify shorter or more advanced notes.
8. Copy `how_to_teach` and `why_to_teach` directly from the chosen candidates. Do not rewrite them unless the input value is empty.
9. Keep `notes` short, concrete, and student-specific.
10. Set `subject` from the selected primary candidate context. Use only `calculus`, `algebra`, or `trigonometry`.

Return only valid JSON. Do not include markdown, comments, or extra text.

Return one of these shapes:
{
  "status": "complete",
  "plan": {
    "subject": "calculus",
    "mode": "conceptual_review",
    "current_plan": [
      {
        "lo_id": 123,
        "title": "Limits and continuity",
        "proficiency": 0.42,
        "how_to_teach": "Use intuitive graphs before formal notation.",
        "why_to_teach": "This concept supports derivative reasoning.",
        "notes": "Start with intuition, then formalize.",
        "is_primary": true
      },
      {
        "lo_id": 98,
        "title": "Function notation",
        "proficiency": 0.3,
        "how_to_teach": "Briefly review input-output mapping.",
        "why_to_teach": "Needed to read limit expressions correctly.",
        "notes": "Quick prerequisite refresh.",
        "is_primary": false
      }
    ],
    "future_plan": [
      {
        "lo_id": 140,
        "title": "Derivative as rate of change",
        "proficiency": 0.0,
        "how_to_teach": "Connect slopes to motion examples.",
        "why_to_teach": "Natural next step after limits.",
        "notes": "Next session topic.",
        "is_primary": false
      }
    ],
    "book": "OpenStax Calculus Volume 1",
    "unit": "Unit 1",
    "chapter": "Chapter 2"
  },
  "message": null
}

{
  "status": "need_info",
  "plan": null,
  "message": "Short clarification question"
}
"""

FAQ_PLANNER_SYSTEM_PROMPT = """You are the FAQ and syllabus topic planner for an adaptive learning system.

Your job is to map the student's question to exactly one known FAQ topic.

You will receive:
- `current_topic_guess`: an optional existing topic guess
- `student_request`: the student's question
- `known_topics`: the only valid topic labels

Decision rules:
1. Use only a topic from `known_topics`. Never invent, rename, or generalize a topic.
2. Choose `complete` only when one known topic is clearly the best match.
3. Choose `need_info` when the request is ambiguous, multi-topic, or does not match any known topic closely enough.
4. If you return `need_info`, ask one short clarification question that helps distinguish between the known topics.

Return only valid JSON. Do not include markdown or extra text.

Return one of these shapes:
{
  "status": "complete",
  "plan": {"topic": "exact topic from known_topics"},
  "message": null
}

{
  "status": "need_info",
  "plan": null,
  "message": "Short clarification question"
}
"""


def _planner_llm_requested() -> bool:
    """
    Returns True when planners should call the LLM. Default is False so tests and
    local runs stay deterministic unless explicitly opted-in via env var.
    """
    flag = os.getenv("WORKFLOW_DEMO_ENABLE_PLANNER_LLM", "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _init_planner_llm(warning_label: str) -> Tuple[Optional[OpenAI], Optional[str]]:
    """
    Initialize the shared planner LLM client when the feature flag is enabled.

    Inputs:
        warning_label: Short label used in fallback log messages.

    Outputs:
        Tuple of `(llm_client, llm_model)`, or `(None, None)` when LLM planning
        is disabled or initialization fails.
    """
    if not _planner_llm_requested():
        return None, None

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("WORKFLOW_DEMO_LLM_MODEL", "gpt-5.4-mini")
    if not api_key:
        print(f"[Planner] {warning_label} unavailable (OPENAI_API_KEY environment variable not set).")
        return None, None

    try:
        return OpenAI(api_key=api_key), model_name
    except Exception as exc:
        print(f"[Planner] {warning_label} unavailable ({exc}).")
        return None, None


class TutoringPlanner:
    """
    Builds tutoring plans by delegating retrieval to the retriever and optional plan selection to the LLM.
    """

    def __init__(self, retriever: TeachingPackRetriever) -> None:
        self.retriever = retriever
        self.llm_client, self.llm_model = _init_planner_llm("Tutoring planner LLM")

    def create_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build one tutoring plan from the student request using the simplified flow.
        
        Inputs:
            payload: Request dictionary with tutoring fields. Primary fields are
                `student_request`, `mode`, `student_profile`, and `image_path`.
                Legacy fields like `learning_objective` are still accepted as a
                fallback query source.
        
        Outputs:
            Standard planner response dictionary:
            - `status`: "complete" or "need_info"
            - `plan`: plan payload or None
            - `message`: clarification text or None
        """
        # Normalize incoming tutoring inputs from both new and legacy payload shapes.
        request = self._normalize_tutoring_payload(payload)
        query = request["query"]
        mode = request["mode"]
        student_profile = request["student_profile"]
        image_path = request["image_path"]

        # Ask for clarification early when there is no usable tutoring query.
        if not query:
            return self._build_need_info_response("What topic would you like to learn about?")

        # Retrieve top LO candidates across text and optional image signals.
        candidates = self._retrieve_tutoring_candidates(query, image_path)
        if not candidates:
            return self._build_need_info_response(
                "I couldn't find relevant learning objectives. Could you rephrase your question?"
            )

        # Build proficiency context used by both LLM and heuristic planning.
        proficiency_map = self._build_proficiency_map(candidates, student_profile)

        # Use LLM planning when available, otherwise fall back to deterministic heuristics.
        if self.llm_client and self.llm_model:
            plan = self._generate_plan_with_llm(
                student_request=query,
                mode=mode,
                candidates=candidates,
                proficiency_map=proficiency_map,
            )
            if plan:
                return self._build_complete_response(plan)
        
        # Deterministic fallback keeps tutoring available even when LLM is disabled.
        plan = self._build_heuristic_plan(
            candidates=candidates,
            mode=mode,
            proficiency_map=proficiency_map,
        )
        return self._build_complete_response(plan)

    def _normalize_tutoring_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tutoring request data across current and legacy payload shapes."""
        student_request = (payload.get("student_request") or "").strip()
        learning_objective = (payload.get("learning_objective") or "").strip()
        query = student_request or learning_objective
        mode = (payload.get("mode") or "conceptual_review").strip() or "conceptual_review"
        return {
            "query": query,
            "mode": mode,
            "student_profile": payload.get("student_profile") or {},
            "image_path": payload.get("image_path"),
        }

    def _retrieve_tutoring_candidates(
        self,
        query: str,
        image_path: Optional[str],
    ) -> List[RetrievalCandidate]:
        """Retrieve merged tutoring candidates from text and optional image signals."""
        result = self.retriever.retrieve_candidates(
            text_query=query,
            image_path=image_path,
            top_k=6,
            debug=True,
        )
        return result.merged_candidates

    @staticmethod
    def _build_proficiency_map(
        candidates: List[RetrievalCandidate],
        student_profile: Dict[str, Any],
    ) -> Dict[str, float]:
        """Map each candidate title to the student's proficiency score."""
        lo_mastery = student_profile.get("lo_mastery", {})
        proficiency_map: Dict[str, float] = {}
        for candidate in candidates:
            proficiency_map[candidate.title] = lo_mastery.get(
                candidate.title,
                lo_mastery.get(str(candidate.lo_id), 0.0),
            )
        return proficiency_map

    @staticmethod
    def _build_need_info_response(message: str) -> Dict[str, Any]:
        """Build a standard clarification response envelope."""
        return {"status": "need_info", "plan": None, "message": message}

    @staticmethod
    def _build_complete_response(plan: Dict[str, Any]) -> Dict[str, Any]:
        """Build a standard complete response envelope."""
        return {"status": "complete", "plan": plan, "message": None}
    
    def _generate_plan_with_llm(
        self,
        student_request: str,
        mode: str,
        candidates: List[RetrievalCandidate],
        proficiency_map: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to generate simplified plan from candidates."""
        # Format candidates for LLM
        candidates_data = [
            {
                "lo_id": c.lo_id,
                "title": c.title,
                "score": round(c.score, 3),
                "book": c.book,
                "unit": c.unit,
                "chapter": c.chapter,
                "how_to_teach": c.how_to_teach,
                "why_to_teach": c.why_to_teach,
            }
            for c in candidates[:8]  # Limit to top 8 to save tokens
        ]
        
        payload = {
            "student_request": student_request,
            "mode": mode,
            "candidates": candidates_data,
            "student_proficiency": proficiency_map,
        }
        
        try:
            raw = self._chat_json(
                TUTORING_PLANNER_PROMPT,
                f"INPUT:\n{json.dumps(payload, indent=2)}",
            )
            if raw.get("status") == "complete" and raw.get("plan"):
                return raw["plan"]
        except Exception as exc:
            print(f"[Planner] LLM plan generation failed ({exc}); using heuristic.")
        
        return None
    
    def _build_heuristic_plan(
        self,
        candidates: List[RetrievalCandidate],
        mode: str,
        proficiency_map: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build simplified plan heuristically from candidates (no LLM)."""
        if not candidates:
            return {}
        
        # Primary LO = top candidate
        primary = candidates[0]
        primary_prof = proficiency_map.get(primary.title, 0.0)
        
        current_plan = [
            {
                "lo_id": primary.lo_id,
                "title": primary.title,
                "proficiency": primary_prof,
                "how_to_teach": primary.how_to_teach or "",
                "why_to_teach": primary.why_to_teach or "",
                "notes": self._generate_proficiency_note(primary_prof),
                "is_primary": True,
            }
        ]
        
        # Add up to 2 dependent LOs if student proficiency is low
        if primary_prof < 0.65:
            for c in candidates[1:3]:
                prof = proficiency_map.get(c.title, 0.0)
                current_plan.append({
                    "lo_id": c.lo_id,
                    "title": c.title,
                    "proficiency": prof,
                    "how_to_teach": c.how_to_teach or "",
                    "why_to_teach": c.why_to_teach or "",
                    "notes": self._generate_proficiency_note(prof),
                    "is_primary": False,
                })
        
        # Future plan = next candidate not in current plan
        future_plan = []
        current_lo_ids = {lo["lo_id"] for lo in current_plan}
        for c in candidates:
            if c.lo_id not in current_lo_ids:
                future_plan.append({
                    "lo_id": c.lo_id,
                    "title": c.title,
                    "proficiency": proficiency_map.get(c.title, 0.0),
                    "how_to_teach": c.how_to_teach or "",
                    "why_to_teach": c.why_to_teach or "",
                    "notes": "",
                    "is_primary": False,
                })
                break
        
        # Infer subject from the candidate context, not just the book title.
        subject = self._infer_subject(primary)
        
        return {
            "subject": subject,
            "mode": mode,
            "current_plan": current_plan,
            "future_plan": future_plan,
            "book": primary.book,
            "unit": primary.unit,
            "chapter": primary.chapter,
        }

    @staticmethod
    def _infer_subject(candidate: RetrievalCandidate) -> str:
        """Infer the tutoring subject from book and nearby curriculum metadata."""
        context = " ".join(
            part for part in [
                candidate.book or "",
                candidate.unit or "",
                candidate.chapter or "",
                candidate.title or "",
            ]
            if part
        ).lower()

        if "trig" in context or "trigon" in context:
            return "trigonometry"
        if "algebra" in context or "quadratic" in context or "polynomial" in context:
            return "algebra"
        return "calculus"

    def _generate_proficiency_note(self, proficiency: float) -> str:
        """Generate a short note based on proficiency score."""
        if proficiency >= 0.85:
            return "Student has mastered this - can skip basics."
        elif proficiency >= 0.65:
            return "Student understands well - focus on applications."
        elif proficiency >= 0.4:
            return "Student needs practice - include examples."
        else:
            return "Student is new to this - start from fundamentals."

    def _chat_json(self, system_prompt: str, user_prompt: str) -> Dict:
        assert self.llm_client is not None and self.llm_model is not None
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return coerce_json(response.choices[0].message.content)

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
        "syllabus_topics": "\nHere are the major concept buckets for this calculus course:\n1. Limits and continuity\n2. Derivatives and rates of change\n3. Applications of derivatives (optimization, related rates)\n4. Integrals and area-under-the-curve reasoning\n5. Fundamental Theorem of Calculus and connecting derivatives to integrals",
    }

    def __init__(self) -> None:
        self.llm_client, self.llm_model = _init_planner_llm("FAQ planner LLM")

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
        if not self.llm_client or not self.llm_model:
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
        assert self.llm_client is not None and self.llm_model is not None
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return coerce_json(response.choices[0].message.content)

