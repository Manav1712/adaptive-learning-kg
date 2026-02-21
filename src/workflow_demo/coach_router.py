"""Coach router and policy loop for handling coach-mode turns."""

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .coach_llm_client import CoachLLMClient
from .planner import FAQPlanner

if TYPE_CHECKING:
    from .coach_agent import CoachAgent


# Policy routing constants
SYLLABUS_FAQ_TOPIC = "syllabus_topics"
SYLLABUS_CLARIFICATION_LIMIT = 2
SYLLABUS_KEYWORDS = {
    "syllabus",
    "course outline",
    "course-outline",
    "topics in this course",
    "topics for this course",
    "major concepts",
    "course topics",
    "syllabus topics",
}

COACH_GREETING = (
    "Hi! I'm your learning coach. I can help you start a tutoring session or answer "
    "questions about FAQs and syllabus. What would you like to work on?"
)

# Planner-input keys where a value mismatch warrants re-planning.
# These are the keys the planner directly honors from its input params.
# Derived fields (learning_objective, teaching_pack, etc.) are not
# controllable via re-plan input, so conflicts on those are ignored.
REPLANNABLE_KEYS = frozenset({"mode", "subject", "topic"})


class CoachRouter:
    """Routes and processes user inputs in coach mode (pre-planner, planner coordination, policy interception)."""

    def __init__(self, agent: "CoachAgent", llm_client: CoachLLMClient) -> None:
        """Initialize the coach router.

        Args:
            agent: Reference to the CoachAgent (owns state).
            llm_client: CoachLLMClient for getting directives from the LLM.
        """
        self.agent = agent
        self.llm_client = llm_client

    def handle_turn(self, user_input: str, synthetic: bool = False) -> str:
        """Process a coach-mode turn: classify, loop calling brain, dispatch to planners/bots.

        Args:
            user_input: The student's text input.
            synthetic: If True, this turn was generated internally (e.g. topic switch).

        Returns:
            Response to send back to the student.
        """
        if user_input.strip():
            self.agent._record_message("student", user_input)

        early_reply = self._classify_input(user_input)
        if early_reply is not None:
            return early_reply

        latest_request = user_input.strip()

        # Track actions to prevent infinite loops
        MAX_ITERATIONS = 5
        attempted_actions = []

        for iteration in range(MAX_ITERATIONS):
            coach_directive = self._get_directive()
            action = coach_directive.get("action", "none")
            message = (coach_directive.get("message_to_student") or "").strip()
            tool_params = coach_directive.get("tool_params") or {}

            # Track this action attempt
            attempted_actions.append(action)

            # ensure student_request reflects latest utterance unless planner override provided
            if latest_request:
                tool_params["student_request"] = latest_request
            elif "student_request" not in tool_params:
                last_message = self.agent._last_student_message()
                if last_message:
                    tool_params["student_request"] = last_message

            self._update_collected_params(tool_params)

            if action == "call_tutoring_planner":
                self.agent.planner_result = self.agent._call_tutoring_planner(tool_params)
                status = self.agent.planner_result.get("status")
                if status == "need_info":
                    msg = self.agent.planner_result.get("message") or "Could you clarify that a bit more?"
                    return self._coach_reply(msg)
                if status == "complete":
                    continue
                # Unexpected status - log and break to prevent infinite loop
                print(
                    f"[Coach] Warning: Tutoring planner returned unexpected status '{status}'. "
                    f"Attempted actions: {attempted_actions}"
                )
                return self._coach_reply("I'm having trouble processing that request. Could you try rephrasing?")

            if action == "call_faq_planner":
                self.agent.planner_result = self.agent._call_faq_planner(tool_params)
                status = self.agent.planner_result.get("status")
                if status == "need_info":
                    msg = self.agent.planner_result.get("message") or "Which FAQ topic should we cover?"
                    return self._coach_reply(msg)
                if status == "complete":
                    continue
                # Unexpected status - log and break to prevent infinite loop
                print(
                    f"[Coach] Warning: FAQ planner returned unexpected status '{status}'. "
                    f"Attempted actions: {attempted_actions}"
                )
                return self._coach_reply("I'm having trouble processing that request. Could you try rephrasing?")

            if action == "start_tutor":
                conflicts = self._detect_plan_conflicts(tool_params)
                if conflicts:
                    self.agent.planner_result = self.agent._call_tutoring_planner(tool_params)
                    continue
                return self.agent.bot_session_manager.begin(
                    bot_type="tutor",
                    tool_params=tool_params,
                    conversation_summary=coach_directive.get("conversation_summary")
                    or "Student requested tutoring.",
                )

            if action == "start_faq":
                conflicts = self._detect_plan_conflicts(tool_params)
                if conflicts:
                    self.agent.planner_result = self.agent._call_faq_planner(tool_params)
                    continue
                return self.agent.bot_session_manager.begin(
                    bot_type="faq",
                    tool_params=tool_params,
                    conversation_summary=coach_directive.get("conversation_summary") or "Student requested FAQ help.",
                )

            if action == "show_proficiency":
                return self._coach_reply(self.agent.bot_session_manager.format_proficiency_report())

            # action == "none"
            forced_summary = self._maybe_force_syllabus_plan(latest_request)
            if forced_summary:
                return forced_summary
            if message:
                return self._coach_reply(message)
            # No action needed and no message - terminate loop
            return ""

        # Loop exhausted - log and return error message
        print(f"[Coach] Error: Maximum iterations ({MAX_ITERATIONS}) reached. Attempted actions: {attempted_actions}")
        return self._coach_reply(
            "I'm having trouble processing your request. Could you try rephrasing or asking something else?"
        )

    def _get_directive(self) -> Dict[str, Any]:
        """Build the LLM payload and get a directive from the LLM."""
        payload = {
            "conversation_history": self.agent.conversation_history[-10:],
            "recent_sessions": self.agent.session_memory.get_recent_sessions(),
            "last_tutoring_session": self.agent.session_memory.last_tutoring_session(),
            "planner_result": self.agent.planner_result,
            "collected_params": self.agent.collected_params,
            "returning_from_session": self.agent.returning_from_session,
        }
        return self.llm_client.get_directive(payload)

    def _classify_input(self, user_input: str) -> Optional[str]:
        """Pre-process user input before the action loop.

        Restores session params when returning from a bot session, flags
        syllabus and FAQ intents, and intercepts session-history questions.

        Returns:
            A reply string if input was fully handled, or None to continue.
        """
        if self.agent.returning_from_session and not self.agent.collected_params:
            last_session = self.agent.session_memory.last_tutoring_session()
            if last_session:
                params = last_session.get("params") or {}
                for key in ("subject", "learning_objective", "mode"):
                    value = params.get(key)
                    if value:
                        self.agent.collected_params.setdefault(key, value)
            self.agent.returning_from_session = False

        normalized = user_input.strip().lower()

        if self._contains_syllabus_keyword(normalized):
            if not self.agent.syllabus_request_active:
                self.agent.syllabus_clarification_count = 0
            self.agent.syllabus_request_active = True

        faq_topic = self._detect_faq_topic(normalized)
        if faq_topic:
            self.agent.collected_params.setdefault("topic", faq_topic)
            self.agent.collected_params["student_request"] = user_input

        if self._is_session_history_question(normalized):
            return self._handle_session_history_question()

        return None

    def _coach_reply(self, message: str) -> str:
        """Record a message and return it."""
        self.agent._record_message("assistant", message)
        return message

    def _detect_plan_conflicts(self, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        """Return tool_params entries that conflict with existing planner decisions.

        Only checks REPLANNABLE_KEYS -- keys the planner directly honors from
        its input. Derived fields (learning_objective, teaching_pack, etc.)
        are ignored because re-planning with a different value wouldn't help.

        Args:
            tool_params: Parameters from the coach directive.

        Returns:
            Dict of conflicting key-value pairs from tool_params.
        """
        plan = (self.agent.planner_result or {}).get("plan") or {}
        return {
            k: v for k, v in tool_params.items()
            if k in REPLANNABLE_KEYS and v and plan.get(k) and v != plan[k]
        }

    def _update_collected_params(self, tool_params: Dict[str, Any]) -> None:
        """Merge non-empty tool params into collected params."""
        if not tool_params:
            return
        for key, value in tool_params.items():
            if value:
                self.agent.collected_params[key] = value

    def _is_session_history_question(self, normalized_input: str) -> bool:
        """Detect if the student is asking about past sessions.

        Args:
            normalized_input: Lowercase, stripped user input.

        Returns:
            True if asking about session history.
        """
        patterns = [
            r"\bwhat\s+did\s+we\s+cover\b",
            r"\bwhat\s+did\s+we\s+do\b",
            r"\bwhat\s+did\s+we\s+learn\b",
            r"\blast\s+session\b",
            r"\bprevious\s+session\b",
            r"\bpast\s+session\b",
            r"\bwhat\s+was\s+the\s+last\b",
            r"\bwhat\s+were\s+we\s+working\s+on\b",
            r"\bwhere\s+did\s+we\s+leave\s+off\b",
            r"\bwhat\s+did\s+we\s+study\b",
        ]

        return any(re.search(pattern, normalized_input, re.IGNORECASE) for pattern in patterns)

    def _handle_session_history_question(self) -> str:
        """Handle questions about session history.

        Returns:
            Formatted message with last session info and continuation prompt.
        """
        last_session = self.agent.session_memory.last_tutoring_session()
        recent_sessions = self.agent.session_memory.get_recent_sessions()

        if not last_session and not recent_sessions:
            return self._coach_reply(
                "You haven't completed any sessions yet. Would you like to start a new tutoring session?"
            )

        if last_session:
            params = last_session.get("params", {})
            summary = last_session.get("summary", {})

            lo = params.get("learning_objective", "a topic")
            mode = params.get("mode", "").replace("_", " ")
            subject = params.get("subject", "")
            understanding = summary.get("student_understanding", "")

            parts = [f"Last time we worked on {lo}"]
            if subject:
                parts.append(f"in {subject}")
            if mode:
                parts.append(f"using {mode} mode")

            description = " ".join(parts) + "."

            understanding_note = ""
            if understanding:
                understanding_map = {
                    "excellent": "You showed excellent understanding",
                    "good": "You showed good understanding",
                    "satisfactory": "You showed satisfactory understanding",
                    "needs_practice": "You needed more practice",
                    "struggling": "You were struggling with it",
                }
                understanding_msg = understanding_map.get(understanding.lower(), "")
                if understanding_msg:
                    understanding_note = f" {understanding_msg.lower()}."

            response = f"{description}{understanding_note}\n\nWould you like to pick up where we left off, or start something new?"

            return self._coach_reply(response)

        # Fallback: just mention recent sessions exist
        return self._coach_reply(
            f"You have {len(recent_sessions)} recent session(s). Would you like to continue with one of them, or start something new?"
        )

    @staticmethod
    def _detect_faq_topic(normalized_input: str) -> Optional[str]:
        """Detect FAQ topic from keywords.

        Args:
            normalized_input: Lowercase, stripped user input.

        Returns:
            FAQ topic string or None.
        """
        keyword_map = {
            "exam": "exam schedule",
            "quiz": "quiz schedule",
            "homework": "homework policy",
            "grading": "grading policy",
            "grade": "grading policy",
            "office hours": "office hours",
            "faq": "exam schedule",
            "syllabus": SYLLABUS_FAQ_TOPIC,
            "course outline": SYLLABUS_FAQ_TOPIC,
            "course-outline": SYLLABUS_FAQ_TOPIC,
            "major concepts": SYLLABUS_FAQ_TOPIC,
            "course topics": SYLLABUS_FAQ_TOPIC,
        }
        for key, topic in keyword_map.items():
            if key in normalized_input:
                return topic
        for topic in FAQPlanner.FAQ_TOPICS.keys():
            if topic in normalized_input:
                return topic
        return None

    def _contains_syllabus_keyword(self, normalized_input: str) -> bool:
        """Check if input contains a syllabus keyword."""
        return any(keyword in normalized_input for keyword in SYLLABUS_KEYWORDS)

    def _reset_syllabus_escalation(self) -> None:
        """Clear syllabus escalation flags."""
        self.agent.syllabus_request_active = False
        self.agent.syllabus_clarification_count = 0

    def _maybe_force_syllabus_plan(self, latest_request: str) -> Optional[str]:
        """If syllabus flag is active and clarification limit reached, force-call FAQ planner.

        Args:
            latest_request: The latest student request text.

        Returns:
            A bot session response or None.
        """
        if not self.agent.syllabus_request_active:
            return None
        self.agent.syllabus_clarification_count += 1
        if self.agent.syllabus_clarification_count < SYLLABUS_CLARIFICATION_LIMIT:
            return None

        params = {
            "topic": SYLLABUS_FAQ_TOPIC,
            "student_request": latest_request
            or self.agent._last_student_message()
            or "Student asked about the syllabus topics.",
        }
        self.agent.planner_result = self.agent._call_faq_planner(params)
        self._reset_syllabus_escalation()
        status = self.agent.planner_result.get("status")
        if status == "need_info":
            return self._coach_reply(
                self.agent.planner_result.get("message") or "Could you clarify which part of the syllabus you need?"
            )

        return self.agent.bot_session_manager.begin(
            bot_type="faq",
            tool_params=params,
            conversation_summary="Starting syllabus FAQ session due to repeated syllabus queries.",
        )
