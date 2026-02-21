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

try:
    from src.workflow_demo.models import (
        SessionPlan, SimplifiedPlan, LearningObjectiveEntry,
        RetrievalCandidate, RetrievalResult,
    )
    from src.workflow_demo.retriever import TeachingPackRetriever
except ImportError:
    from .models import (
        SessionPlan, SimplifiedPlan, LearningObjectiveEntry,
        RetrievalCandidate, RetrievalResult,
    )
from .json_utils import coerce_json
from .retriever import TeachingPackRetriever


# New simplified planner prompt
SIMPLIFIED_PLANNER_PROMPT = """You are the Tutoring Session Planner. Given retrieval candidates from the knowledge graph,
create a focused tutoring plan.

INPUT: You receive:
- student_request: What the student asked for
- mode: The tutoring mode (conceptual_review, examples, or practice)
- candidates: Top LO candidates from retrieval, each with:
  - lo_id, title, score, book, unit, chapter
  - how_to_teach: Instructional approach
  - why_to_teach: Pedagogical rationale
- student_proficiency: Dict mapping LO titles to proficiency scores (0.0-1.0)

OUTPUT: Return a simplified plan as strict JSON:
{
  "status": "complete",
  "plan": {
    "subject": "calculus|algebra|trigonometry",
    "mode": "conceptual_review|examples|practice",
    "current_plan": [
      {
        "lo_id": 123,
        "title": "Primary LO Title",
        "proficiency": 0.65,
        "how_to_teach": "...",
        "why_to_teach": "...",
        "notes": "Short note about student's state",
        "is_primary": true
      },
      // Up to 2 more dependent/prerequisite LOs (is_primary: false)
    ],
    "future_plan": [
      {
        "lo_id": 456,
        "title": "Next LO Title",
        "proficiency": 0.0,
        "how_to_teach": "...",
        "why_to_teach": "...",
        "notes": "",
        "is_primary": false
      }
    ],
    "book": "...",
    "unit": "...",
    "chapter": "..."
  }
}

Rules:
1. Select 1 PRIMARY LO that best matches the student's request (is_primary: true)
2. Add up to 2 DEPENDENT LOs if the student's proficiency is low (<0.65) on prerequisites
3. Add 1 LO to future_plan for the next session
4. All LOs in current_plan must use the SAME mode
5. Use proficiency scores to write helpful notes (e.g., "Student has mastered X, skip basics")
6. Keep how_to_teach and why_to_teach from the candidates - these are from the knowledge graph
"""

# Legacy prompt (kept for backward compatibility)
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


def _planner_llm_requested() -> bool:
    """
    Returns True when planners should call the LLM. Default is False so tests and
    local runs stay deterministic unless explicitly opted-in via env var.
    """
    flag = os.getenv("WORKFLOW_DEMO_ENABLE_PLANNER_LLM", "")
    return flag.lower() in {"1", "true", "yes", "on"}


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

        if _planner_llm_requested():
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

    def create_simplified_plan(
        self,
        student_request: str,
        mode: str,
        student_profile: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a simplified tutoring plan using the new retrieval + LLM flow.
        
        This is the new interface that:
        1. Calls retriever.retrieve_candidates() to get candidates from text + image
        2. Passes candidates to LLM with proficiency scores
        3. Returns simplified plan (1 primary + 2 dependent LOs, 1 future LO)
        
        Inputs:
            student_request: What the student asked for (text, may include OCR)
            mode: Tutoring mode (conceptual_review, examples, practice)
            student_profile: Dict with lo_mastery scores
            image_path: Optional path to image for CLIP retrieval
        
        Outputs:
            Dict with status, plan (SimplifiedPlan as dict), and optional message
        """
        mode = (mode or "conceptual_review").strip() or "conceptual_review"
        lo_mastery = (student_profile or {}).get("lo_mastery", {})
        
        # Step 1: Retrieve candidates (text + image in parallel)
        retrieval_result = self.retriever.retrieve_candidates(
            text_query=student_request,
            image_path=image_path,
            top_k=6,
            debug=True,  # Print top retrieved LOs
        )
        
        if not retrieval_result.merged_candidates:
            return {
                "status": "need_info",
                "plan": None,
                "message": "I couldn't find relevant learning objectives. Could you rephrase your question?",
            }
        
        # Step 2: Build proficiency map for candidates
        proficiency_map = {}
        for c in retrieval_result.merged_candidates:
            # Check lo_mastery by title (main key) and lo_id (fallback)
            prof = lo_mastery.get(c.title, lo_mastery.get(str(c.lo_id), 0.0))
            proficiency_map[c.title] = prof
        
        # Step 3: Call LLM to generate simplified plan (if enabled)
        if self.llm_enabled and self.llm_client:
            plan = self._generate_plan_with_llm(
                student_request=student_request,
                mode=mode,
                candidates=retrieval_result.merged_candidates,
                proficiency_map=proficiency_map,
            )
            if plan:
                return {"status": "complete", "plan": plan, "message": None}
        
        # Step 4: Fallback - build plan heuristically from top candidates
        plan = self._build_heuristic_plan(
            candidates=retrieval_result.merged_candidates,
            mode=mode,
            proficiency_map=proficiency_map,
        )
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
                SIMPLIFIED_PLANNER_PROMPT,
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
        
        # Infer subject from book name
        subject = "calculus"
        if primary.book:
            book_lower = primary.book.lower()
            if "algebra" in book_lower:
                subject = "algebra"
            elif "trig" in book_lower:
                subject = "trigonometry"
        
        return {
            "subject": subject,
            "mode": mode,
            "current_plan": current_plan,
            "future_plan": future_plan,
            "book": primary.book,
            "unit": primary.unit,
            "chapter": primary.chapter,
        }
    
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
        if not self.llm_enabled:
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
        return coerce_json(response.choices[0].message.content)

    @staticmethod
    def _session_plan_to_dict(plan: SessionPlan) -> Dict[str, Any]:
        def _step_to_dict(step: Any) -> Dict[str, Any]:
            return {
                "step_id": step.step_id,
                "step_type": step.step_type,
                "goal": step.goal,
                "lo_id": step.lo_id,
                "content_id": step.content_id,
                "how_to_teach": step.how_to_teach,
                "why_to_teach": step.why_to_teach,
            }

        teaching_pack = {
            "key_points": plan.teaching_pack.key_points,
            "examples": plan.teaching_pack.examples,
            "practice": plan.teaching_pack.practice,
            "prerequisites": plan.teaching_pack.prerequisites,
            "citations": plan.teaching_pack.citations,
            "images": getattr(plan.teaching_pack, "images", []),
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
            "session_guidance": plan.session_guidance,
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
        "syllabus_topics": "\nHere are the major concept buckets for this calculus course:\n1. Limits and continuity\n2. Derivatives and rates of change\n3. Applications of derivatives (optimization, related rates)\n4. Integrals and area-under-the-curve reasoning\n5. Fundamental Theorem of Calculus and connecting derivatives to integrals",
    }

    def __init__(self) -> None:
        self.llm_client: Optional[OpenAI] = None
        self.llm_model: Optional[str] = None
        self.llm_enabled = False

        if _planner_llm_requested():
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
        if not self.llm_enabled:
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
        return coerce_json(response.choices[0].message.content)

