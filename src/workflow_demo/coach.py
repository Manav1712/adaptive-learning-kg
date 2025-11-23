"""
Coach agent tailored for the workflow demo that orchestrates tutoring and FAQ flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .models import SessionPlan
from .planner import FAQPlanner, PlannerResult, TutoringPlanner
from .retriever import TeachingPackRetriever
from .tutor import FAQAgent, TutorAgent


@dataclass
class CoachState:
    """
    Holds the mutable fields the Coach relies on between turns.
    """

    current_intent: Optional[str] = None
    subject: Optional[str] = None
    learning_objective: Optional[str] = None
    mode: Optional[str] = None
    faq_topic: Optional[str] = None
    active_plan: Optional[SessionPlan] = None
    plan_pointer: int = 0
    awaiting_confirmation: bool = False
    student_profile: Dict = field(default_factory=lambda: {"lo_mastery": {}})


class CoachAgent:
    """
    Conversational coordinator that mirrors the 9 Oct coach notebook behavior.
    """

    def __init__(self, retriever: Optional[TeachingPackRetriever] = None) -> None:
        """
        Wire planner + tutor stubs together for the demo pipeline.

        Inputs:
            retriever: Optional pre-built TeachingPackRetriever.

        Outputs:
            None. Instantiates planners, tutor stubs, and fresh state.
        """

        self.retriever = retriever or TeachingPackRetriever()
        self.tutoring_planner = TutoringPlanner(self.retriever)
        self.faq_planner = FAQPlanner()
        self.tutor = TutorAgent()
        self.faq_agent = FAQAgent()
        self.state = CoachState()

    def process_turn(self, user_input: str) -> str:
        """
        Primary entrypoint that routes between gathering info, planning, and execution.

        Inputs:
            user_input: Raw student utterance for this turn.

        Outputs:
            Coach response string (may include tutor output when executing a plan).
        """

        normalized = user_input.lower().strip()

        if self._should_continue_plan(normalized):
            return self._advance_plan()

        if self.state.awaiting_confirmation:
            if self._is_positive_confirmation(normalized):
                self.state.awaiting_confirmation = False
                return self._advance_plan(prelude=self.state.active_plan.first_question)
            if self._is_negative_confirmation(normalized):
                self.state.active_plan = None
                return "No problem. Tell me what you'd like to work on instead."
            return "Whenever you're ready, just say the word and we'll kick off the plan."

        intent = self._classify_intent(normalized)
        if intent == "faq":
            self.state.current_intent = "faq"
            topic = self._extract_faq_topic(normalized)
            self.state.faq_topic = topic or self.state.faq_topic
            plan_result = self.faq_planner.create_plan(self.state.faq_topic, user_input)
            if plan_result.status != "complete":
                return plan_result.message
            answer = self.faq_agent.answer(
                plan_result.metadata["script"],
                plan_result.metadata["first_question"],
            )
            self._reset_state(keep_profile=True)
            return answer

        self.state.current_intent = "tutor"
        subject = self._infer_subject(normalized) or self.state.subject
        learning_objective = self._extract_learning_objective(user_input) or self.state.learning_objective
        mode = self._extract_mode(normalized) or self.state.mode
        self.state.subject = subject
        self.state.learning_objective = learning_objective
        self.state.mode = mode

        missing = [
            label
            for label, value in [
                ("subject", subject),
                ("topic", learning_objective),
                ("mode", mode),
            ]
            if not value
        ]
        if missing:
            return self._prompt_for_missing(missing[0])

        plan_result = self.tutoring_planner.create_plan(
            subject=subject,
            learning_objective=learning_objective,
            mode=mode,
            student_request=user_input,
            student_profile=self.state.student_profile,
        )

        if plan_result.status != "complete":
            return plan_result.message

        self.state.active_plan = plan_result.session_plan
        self.state.plan_pointer = 0
        self.state.awaiting_confirmation = True
        summary = self._summarize_plan(plan_result.session_plan)
        return summary

    def _advance_plan(self, prelude: Optional[str] = None) -> str:
        """
        Move the tutoring session forward by a single step.

        Inputs:
            prelude: Optional string (e.g., first question) prepended to tutor output.

        Outputs:
            Full assistant message for this turn.
        """

        plan = self.state.active_plan
        if not plan:
            return "We don't have an active plan yet. What would you like to learn?"

        tutor_result = self.tutor.deliver_step(
            plan,
            self.state.plan_pointer,
            student_profile=self.state.student_profile,
        )
        self.state.plan_pointer = tutor_result.new_pointer
        self.state.student_profile = tutor_result.updated_profile

        if tutor_result.completed:
            self.state.active_plan = None
            self.state.plan_pointer = 0

        if prelude:
            return f"{prelude}\n\n{tutor_result.message}"
        return tutor_result.message

    def _should_continue_plan(self, normalized_input: str) -> bool:
        """
        Decide whether the current utterance should advance the active plan.
        """

        if not self.state.active_plan:
            return False
        if self.state.awaiting_confirmation:
            return False
        keywords = {"continue", "next", "yes", "sure", "go on", "ready"}
        return any(keyword in normalized_input for keyword in keywords)

    def _reset_state(self, keep_profile: bool = False) -> None:
        """
        Clear contextual state while optionally keeping student proficiency.
        """

        profile = self.state.student_profile if keep_profile else {"lo_mastery": {}}
        self.state = CoachState(student_profile=profile)

    @staticmethod
    def _is_positive_confirmation(normalized_input: str) -> bool:
        """
        Detect if the student agreed to start the plan.
        """

        positives = {"yes", "yep", "sure", "let's go", "start", "ready"}
        return any(token in normalized_input for token in positives)

    @staticmethod
    def _is_negative_confirmation(normalized_input: str) -> bool:
        """
        Detect if the student declined the plan.
        """

        negatives = {"no", "not now", "later", "another"}
        return any(token in normalized_input for token in negatives)

    @staticmethod
    def _classify_intent(normalized_input: str) -> str:
        """
        Classify user intent between tutoring and FAQ requests.
        """

        faq_terms = {"exam", "quiz", "policy", "schedule", "syllabus", "office hours"}
        if any(term in normalized_input for term in faq_terms):
            return "faq"
        tutoring_terms = {"teach", "learn", "practice", "explain", "help", "review"}
        if any(term in normalized_input for term in tutoring_terms):
            return "tutor"
        return "tutor"

    @staticmethod
    def _extract_mode(normalized_input: str) -> Optional[str]:
        """
        Extract tutoring mode from user utterance.
        """

        if any(word in normalized_input for word in ["practice", "problems", "exercise"]):
            return "practice"
        if "example" in normalized_input or "walk through" in normalized_input:
            return "examples"
        if "concept" in normalized_input or "review" in normalized_input:
            return "conceptual_review"
        return None

    @staticmethod
    def _extract_learning_objective(user_input: str) -> Optional[str]:
        """
        Use a lightweight heuristic to select the learning objective string.
        """

        keywords = ["chain rule", "derivative", "integral", "differential", "trigonometric", "limits"]
        lowered = user_input.lower()
        for phrase in keywords:
            if phrase in lowered:
                return phrase.title()
        return user_input.strip() if len(user_input.split()) <= 6 else None

    @staticmethod
    def _infer_subject(normalized_input: str) -> Optional[str]:
        """
        Infer subject label from the utterance.
        """

        subject_map = {
            "calculus": "calculus",
            "derivative": "calculus",
            "integral": "calculus",
            "algebra": "algebra",
            "trig": "trigonometry",
            "geometry": "geometry",
        }
        for token, subject in subject_map.items():
            if token in normalized_input:
                return subject
        return None

    @staticmethod
    def _extract_faq_topic(normalized_input: str) -> Optional[str]:
        """
        Map FAQ utterances to canonical topics.
        """

        for topic in FAQPlanner.FAQ_TOPICS.keys():
            if topic in normalized_input:
                return topic
        if "exam" in normalized_input:
            return "exam schedule"
        if "quiz" in normalized_input:
            return "quiz schedule"
        if "grade" in normalized_input:
            return "grading policy"
        if "homework" in normalized_input:
            return "homework policy"
        return None

    def _prompt_for_missing(self, field: str) -> str:
        """
        Ask the student for whichever field is still needed.
        """

        prompts = {
            "subject": "Which subject should we focus on? (e.g., calculus, algebra, trigonometry)",
            "topic": "Great! What specific topic or learning objective should we target?",
            "mode": "How would you like to learn—practice problems, conceptual review, or examples?",
        }
        return prompts.get(field, "Could you clarify a bit more?")

    @staticmethod
    def _summarize_plan(plan: SessionPlan) -> str:
        """
        Produce a student-friendly summary of the generated plan.
        """

        steps = "\n".join(f"- {step.step_type.title()}: {step.goal}" for step in plan.current_plan)
        future = (
            ", ".join(step.goal for step in plan.future_plan[:2]) if plan.future_plan else "additional review as needed"
        )
        return (
            f"Here's the plan for {plan.learning_objective} via {plan.mode.replace('_', ' ')}:\n"
            f"{steps}\n"
            f"Up next we can explore: {future}.\n"
            f"Say 'yes' when you're ready and we'll begin with: {plan.first_question}"
        )

