"""
Session planner utilities that call the retriever and format plan metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    from src.workflow_demo.models import SessionPlan
    from src.workflow_demo.retriever import TeachingPackRetriever
except ImportError:
    from .models import SessionPlan
    from .retriever import TeachingPackRetriever


@dataclass
class PlannerResult:
    """
    Lightweight wrapper so the Coach can reason on both tutoring and FAQ flows.
    """

    status: str
    session_plan: Optional[SessionPlan] = None
    message: Optional[str] = None
    metadata: Optional[Dict] = None


class TutoringPlanner:
    """
    Builds tutoring plans by delegating context assembly to the retriever.
    """

    def __init__(self, retriever: TeachingPackRetriever) -> None:
        """
        Store retriever dependency used to assemble teaching packs.

        Inputs:
            retriever: TeachingPackRetriever instance.

        Outputs:
            None.
        """

        self.retriever = retriever

    def create_plan(
        self,
        subject: str,
        learning_objective: str,
        mode: str,
        student_request: str,
        student_profile: Optional[Dict] = None,
    ) -> PlannerResult:
        """
        Build a SessionPlan for the supplied tutoring parameters.

        Inputs:
            subject: Subject domain (calculus, algebra, etc.).
            learning_objective: LO anchor for the session.
            mode: Delivery preference (conceptual_review, examples, practice).
            student_request: Original user utterance to keep nuance.
            student_profile: Optional proficiency metadata.

        Outputs:
            PlannerResult with status=complete and the populated SessionPlan.
        """

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
            return PlannerResult(
                status="need_info",
                message=f"Missing fields: {', '.join(missing_fields)}",
            )

        session_plan = self.retriever.retrieve_plan(
            query=student_request or learning_objective,
            subject=subject,
            learning_objective=learning_objective,
            mode=mode,
            student_profile=student_profile or {},
        )
        return PlannerResult(status="complete", session_plan=session_plan)


class FAQPlanner:
    """
    Minimal planner for FAQ/syllabus requests with predefined topics.
    """

    FAQ_TOPICS = {
        "exam schedule": "Let's double-check the official exam calendar. When is the next exam window you are concerned about?",
        "quiz schedule": "Quizzes typically happen weekly. Which week are you asking about?",
        "homework policy": "Homework is due Sundays at 11:59 PM local time unless stated otherwise.",
        "grading policy": "Grades weight: Exams 50%, Quizzes 20%, Homework 20%, Participation 10%.",
        "office hours": "Office hours run Tuesdays 2-4 PM (Room 301) and Thursdays 10-11 AM (Zoom).",
        "late work": "Late work receives a 10% penalty per day, up to three days.",
    }

    def create_plan(self, topic: str, student_request: str) -> PlannerResult:
        """
        Build a FAQ response scaffold for a recognized topic.

        Inputs:
            topic: Requested FAQ topic (case insensitive).
            student_request: Original utterance for reference.

        Outputs:
            PlannerResult with canned context or a request for clarification.
        """

        if not topic:
            return PlannerResult(
                status="need_info",
                message="Which FAQ or syllabus topic should we look up?",
            )

        key = topic.strip().lower()
        if key not in self.FAQ_TOPICS:
            return PlannerResult(
                status="need_info",
                message="I did not recognize that topic. Try exam schedule, grading policy, or homework policy.",
            )

        return PlannerResult(
            status="complete",
            metadata={
                "topic": key,
                "script": self.FAQ_TOPICS[key],
                "student_request": student_request,
                "first_question": f"Are you asking for specific dates related to {key}?",
            },
        )

