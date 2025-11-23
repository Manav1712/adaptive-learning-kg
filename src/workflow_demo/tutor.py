"""
Tutor and FAQ execution stubs that simulate downstream tool behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .models import PlanStep, SessionPlan, TeachingPack


@dataclass
class TutorTurnResult:
    """
    Result payload returned after running a tutoring step.
    """

    message: str
    new_pointer: int
    completed: bool
    updated_profile: Dict


class TutorAgent:
    """
    Simple tutor stub that consumes SessionPlan steps sequentially.
    """

    def deliver_step(
        self,
        plan: SessionPlan,
        pointer: int,
        student_profile: Optional[Dict] = None,
    ) -> TutorTurnResult:
        """
        Execute the current plan step and craft a natural language response.

        Inputs:
            plan: Active SessionPlan to follow.
            pointer: Index of the step in plan.current_plan to execute.
            student_profile: Mutable profile dict tracking proficiency signals.

        Outputs:
            TutorTurnResult with assistant message, new pointer, completion flag,
            and updated student profile.
        """

        profile = student_profile or {"lo_mastery": {}}
        steps = plan.current_plan
        if pointer >= len(steps):
            return TutorTurnResult(
                message="We have already completed the current plan. Ready for the next learning goal!",
                new_pointer=pointer,
                completed=True,
                updated_profile=profile,
            )

        step = steps[pointer]
        response = self._format_step_response(step, plan.teaching_pack)

        mastery = profile.setdefault("lo_mastery", {})
        lo_key = plan.learning_objective
        mastery[lo_key] = min(0.95, mastery.get(lo_key, 0.4) + 0.12)

        new_pointer = pointer + 1
        completed = new_pointer >= len(steps)

        if completed and plan.future_plan:
            response += (
                "\n\nNext up, we can explore: "
                + ", ".join(step.goal for step in plan.future_plan[:2])
                + "."
            )

        return TutorTurnResult(
            message=response,
            new_pointer=new_pointer,
            completed=completed,
            updated_profile=profile,
        )

    def _format_step_response(self, step: PlanStep, pack: TeachingPack) -> str:
        """
        Map PlanStep types to conversational snippets that reference the teaching pack.
        """

        if step.step_type == "prereq_review" and pack.prerequisites:
            target = pack.prerequisites[0]
            return (
                f"Let's briefly revisit {target['title']}—"
                f"{target['note']} Do you recall how it connects to our main topic?"
            )

        if step.step_type == "explain":
            bullets = pack.key_points[:2] if pack.key_points else []
            bullet_text = "\n- ".join(bullets) if bullets else "We'll clarify the core idea together."
            return (
                f"Here's the high-level intuition for {step.goal}:\n- {bullet_text}"
                "\nWhat stands out or feels fuzzy so far?"
            )

        if step.step_type == "example" and pack.examples:
            example = pack.examples[0]
            return (
                f"Let's walk a concrete example ({example['content_id']} - {example['content_type']}):\n"
                f"{example['snippet']}\n\nWhat would be your next move?"
            )

        if step.step_type == "practice" and pack.practice:
            practice = pack.practice[0]
            return (
                f"Your turn: try this practice check ({practice['content_id']}):\n"
                f"{practice['snippet']}\n\nTalk me through your reasoning and we'll compare."
            )

        return f"Let's keep building: {step.goal}"


class FAQAgent:
    """
    Stub that returns canned FAQ answers plus a nudge back to the Coach.
    """

    def answer(self, topic_script: str, first_question: str) -> str:
        """
        Format a short FAQ response.

        Inputs:
            topic_script: Canonical answer for the topic.
            first_question: Clarifying follow-up question.

        Outputs:
            Friendly FAQ string.
        """

        return f"{topic_script}\n\n{first_question}"

