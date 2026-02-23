"""Coach router and policy loop for handling coach-mode turns.

Sits between the student's message and the planners/bots. Runs a loop
that asks the LLM for a directive, validates it against policy rules
(plan conflict detection, syllabus escalation), and dispatches to the
appropriate planner or bot session. Pre-classifies simple inputs
(FAQ keywords, session history questions) to avoid unnecessary LLM calls.
"""
import re
from typing import TYPE_CHECKING, Any, Dict, Optional
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

# If the LLM disagrees with the planner on any of these keys,
# force a re-plan instead of starting the session.
REPLANNABLE_KEYS = frozenset({"mode", "subject", "topic"})

_MAX_LOOP_ITERATIONS = 5

# Single compiled regex for session-history questions.
# Maps understanding labels to human-friendly descriptions.
_UNDERSTANDING_MAP = {
    "excellent": "You showed excellent understanding",
    "good": "You showed good understanding",
    "satisfactory": "You showed satisfactory understanding",
    "needs_practice": "You needed more practice",
    "struggling": "You were struggling with it",
}

# Keyword-to-FAQ-topic mapping for pre-classification.
_FAQ_KEYWORD_MAP = {
    "exam": "exam schedule",
    "quiz": "quiz schedule",
    "homework": "homework policy",
    "grading": "grading policy",
    "grade": "grading policy",
    "office hours": "office hours",
    "faq": "exam schedule",
}

_SESSION_HISTORY_RE = re.compile(
    r"\bwhat\s+did\s+we\s+cover\b"
    r"|\bwhat\s+did\s+we\s+do\b"
    r"|\bwhat\s+did\s+we\s+learn\b"
    r"|\blast\s+session\b"
    r"|\bprevious\s+session\b"
    r"|\bpast\s+session\b"
    r"|\bwhat\s+was\s+the\s+last\b"
    r"|\bwhat\s+were\s+we\s+working\s+on\b"
    r"|\bwhere\s+did\s+we\s+leave\s+off\b"
    r"|\bwhat\s+did\s+we\s+study\b",
    re.IGNORECASE,
)

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

    def handle_turn(
        self, user_input: str, synthetic: bool = False
    ) -> str:
        """Process a coach-mode turn: classify, loop
        calling brain, dispatch to planners/bots.

        Args:
            user_input: The student's text input.
            synthetic: If True, this turn was generated
                internally (e.g. topic switch).

        Returns:
            Response to send back to the student.
        """
        if user_input.strip():
            self.agent._record_message("student", user_input)

        # Quick checks (FAQ keywords, session history,
        # syllabus flags). Skip the LLM loop if handled.
        early_reply = self._classify_input(user_input)
        if early_reply is not None:
            return early_reply

        latest_request = user_input.strip()
        attempted_actions: list[str] = []

        for _ in range(_MAX_LOOP_ITERATIONS):
            # Ask the LLM what to do next.
            directive = self._get_directive()
            action = directive.get("action", "none")
            message = (
                directive.get("message_to_student") or ""
            ).strip()
            tool_params = directive.get("tool_params") or {}
            attempted_actions.append(action)

            # Ensure student_request is always present.
            tool_params["student_request"] = (
                latest_request
                or tool_params.get("student_request")
                or self.agent._last_student_message()
                or ""
            )
            self._update_collected_params(tool_params)

            # -- Planner calls --
            if action == "call_tutoring_planner":
                result = self._handle_planner_call(
                    self.agent._call_tutoring_planner,
                    tool_params,
                    "Could you clarify that a bit more?",
                    attempted_actions,
                )
                if result is None:
                    continue
                return result

            if action == "call_faq_planner":
                result = self._handle_planner_call(
                    self.agent._call_faq_planner,
                    tool_params,
                    "Which FAQ topic should we cover?",
                    attempted_actions,
                )
                if result is None:
                    continue
                return result

            # -- Session starts --
            if action == "start_tutor":
                result = self._start_session(
                    "tutor",
                    tool_params,
                    self.agent._call_tutoring_planner,
                    directive,
                    "Student requested tutoring.",
                )
                if result is None:
                    continue
                return result

            if action == "start_faq":
                result = self._start_session(
                    "faq",
                    tool_params,
                    self.agent._call_faq_planner,
                    directive,
                    "Student requested FAQ help.",
                )
                if result is None:
                    continue
                return result

            # -- Proficiency report --
            if action == "show_proficiency":
                report = (
                    self.agent.bot_session_manager
                    .format_proficiency_report()
                )
                return self._coach_reply(report)

            # -- action == "none": LLM just wants to talk. --
            forced = self._maybe_force_syllabus_plan(
                latest_request
            )
            if forced:
                return forced
            if message:
                return self._coach_reply(message)
            return ""

        # Safety net: loop exhausted.
        print(
            f"[Coach] Error: Max iterations "
            f"({_MAX_LOOP_ITERATIONS}) reached. "
            f"Actions: {attempted_actions}"
        )
        return self._coach_reply(
            "I'm having trouble processing your request. "
            "Could you try rephrasing or asking "
            "something else?"
        )

    def _get_directive(self) -> Dict[str, Any]:
        """Build the LLM payload and get a directive.

        Payload includes recent conversation history, past
        session summaries, current planner result, collected
        params, and a flag for returning from a bot session.
        """
        payload = {
            "conversation_history": (
                self.agent.conversation_history[-10:]
            ),
            "recent_sessions": (
                self.agent.session_memory.get_recent_sessions()
            ),
            "last_tutoring_session": (
                self.agent.session_memory.last_tutoring_session()
            ),
            "planner_result": self.agent.planner_result,
            "collected_params": self.agent.collected_params,
            "returning_from_session": (
                self.agent.returning_from_session
            ),
        }
        return self.llm_client.get_directive(payload)

    def _handle_planner_call(
        self,
        planner_fn,
        tool_params: Dict[str, Any],
        fallback_msg: str,
        attempted_actions: list,
    ) -> Optional[str]:
        """Call a planner and handle its status.

        Returns a reply string if the planner needs info or
        hit an unexpected status. Returns None if the plan is
        complete (caller should continue the loop).
        """
        self.agent.planner_result = planner_fn(tool_params)
        status = self.agent.planner_result.get("status")

        if status == "need_info":
            msg = (
                self.agent.planner_result.get("message")
                or fallback_msg
            )
            return self._coach_reply(msg)

        if status == "complete":
            return None

        # Unexpected status -- log and bail out.
        print(
            f"[Coach] Warning: Planner returned "
            f"unexpected status '{status}'. "
            f"Actions: {attempted_actions}"
        )
        return self._coach_reply(
            "I'm having trouble processing that request. "
            "Could you try rephrasing?"
        )

    def _start_session(
        self,
        bot_type: str,
        tool_params: Dict[str, Any],
        planner_fn,
        directive: Dict[str, Any],
        default_summary: str,
    ) -> Optional[str]:
        """Check for plan conflicts and start a bot session.

        If tool_params conflict with the current plan, forces
        a re-plan and returns None (caller should continue the
        loop). Otherwise starts the session and returns the
        bot's first message.
        """
        # Conflict found -- re-plan instead of starting.
        if self._detect_plan_conflicts(tool_params):
            self.agent.planner_result = planner_fn(tool_params)
            return None

        # No conflict -- start the bot session.
        summary = (
            directive.get("conversation_summary")
            or default_summary
        )
        return self.agent.bot_session_manager.begin(
            bot_type=bot_type,
            tool_params=tool_params,
            conversation_summary=summary,
        )

    def _classify_input(self, user_input: str) -> Optional[str]:
        """Pre-process user input before the action loop.

        Restores session params when returning from a bot session, flags
        syllabus and FAQ intents, and intercepts session-history questions.

        Returns:
            A reply string if input was fully handled, or None to continue.
        """
        # Restore last session's params if returning from a
        # bot session with no params carried over.
        if (
            self.agent.returning_from_session
            and not self.agent.collected_params
        ):
            last_session = (
                self.agent.session_memory
                .last_tutoring_session()
            )
            if last_session:
                params = last_session.get("params") or {}
                for key in (
                    "subject", "learning_objective", "mode"
                ):
                    value = params.get(key)
                    if value:
                        self.agent.collected_params.setdefault(
                            key, value
                        )
            self.agent.returning_from_session = False

        normalized = user_input.strip().lower()

        # Flag syllabus intent for escalation tracking.
        if self._contains_syllabus_keyword(normalized):
            if not self.agent.syllabus_request_active:
                self.agent.syllabus_clarification_count = 0
            self.agent.syllabus_request_active = True

        # Detect FAQ topic from keywords and store it.
        faq_topic = self._detect_faq_topic(normalized)
        if faq_topic:
            self.agent.collected_params.setdefault(
                "topic", faq_topic
            )
            self.agent.collected_params[
                "student_request"
            ] = user_input

        # Intercept "what did we do last time?" questions.
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
        # Compare tool_params against the current plan on
        # replannable keys only.
        plan = (
            (self.agent.planner_result or {}).get("plan")
            or {}
        )
        return {
            k: v
            for k, v in tool_params.items()
            if k in REPLANNABLE_KEYS
            and v
            and plan.get(k)
            and v != plan[k]
        }

    def _update_collected_params(
        self, tool_params: Dict[str, Any]
    ) -> None:
        """Merge non-empty tool params into collected params."""
        if not tool_params:
            return
        # Only overwrite with truthy values.
        for key, value in tool_params.items():
            if value:
                self.agent.collected_params[key] = value

    @staticmethod
    def _is_session_history_question(
        normalized_input: str,
    ) -> bool:
        """Detect if the student is asking about past
        sessions using the precompiled regex."""
        return bool(
            _SESSION_HISTORY_RE.search(normalized_input)
        )

    def _handle_session_history_question(self) -> str:
        """Handle questions about session history.

        Returns:
            Formatted message with last session info and continuation prompt.
        """
        last_session = (
            self.agent.session_memory.last_tutoring_session()
        )
        recent_sessions = (
            self.agent.session_memory.get_recent_sessions()
        )

        # No sessions at all -- prompt to start one.
        if not last_session and not recent_sessions:
            return self._coach_reply(
                "You haven't completed any sessions yet. "
                "Would you like to start a new tutoring "
                "session?"
            )

        if last_session:
            # Extract session details for the summary.
            params = last_session.get("params", {})
            summary = last_session.get("summary", {})
            lo = params.get(
                "learning_objective", "a topic"
            )
            mode = (
                params.get("mode", "").replace("_", " ")
            )
            subject = params.get("subject", "")
            understanding = summary.get(
                "student_understanding", ""
            )

            # Build a human-readable description.
            parts = [f"Last time we worked on {lo}"]
            if subject:
                parts.append(f"in {subject}")
            if mode:
                parts.append(f"using {mode} mode")
            description = " ".join(parts) + "."

            # Map understanding label to a friendly note.
            understanding_note = ""
            if understanding:
                msg = _UNDERSTANDING_MAP.get(
                    understanding.lower(), ""
                )
                if msg:
                    understanding_note = (
                        f" {msg.lower()}."
                    )

            response = (
                f"{description}{understanding_note}"
                "\n\nWould you like to pick up where we "
                "left off, or start something new?"
            )
            return self._coach_reply(response)

        # Fallback: just mention recent sessions exist.
        return self._coach_reply(
            f"You have {len(recent_sessions)} recent "
            "session(s). Would you like to continue "
            "with one of them, or start something new?"
        )

    @staticmethod
    def _detect_faq_topic(
        normalized_input: str,
    ) -> Optional[str]:
        """Detect FAQ topic from keywords.

        Checks the module-level FAQ keyword map first,
        then syllabus keywords, then planner-defined topics.
        """
        # Check non-syllabus FAQ keywords.
        for key, topic in _FAQ_KEYWORD_MAP.items():
            if key in normalized_input:
                return topic
        # Check syllabus keywords (derived from the
        # single SYLLABUS_KEYWORDS set).
        for kw in SYLLABUS_KEYWORDS:
            if kw in normalized_input:
                return SYLLABUS_FAQ_TOPIC
        # Fall back to planner-defined FAQ topics.
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
        # Not in a syllabus escalation -- nothing to do.
        if not self.agent.syllabus_request_active:
            return None

        # Increment count; bail if under the limit.
        self.agent.syllabus_clarification_count += 1
        if (
            self.agent.syllabus_clarification_count
            < SYLLABUS_CLARIFICATION_LIMIT
        ):
            return None

        # Limit reached -- bypass LLM and force FAQ plan.
        params = {
            "topic": SYLLABUS_FAQ_TOPIC,
            "student_request": (
                latest_request
                or self.agent._last_student_message()
                or "Student asked about the syllabus "
                "topics."
            ),
        }
        self.agent.planner_result = (
            self.agent._call_faq_planner(params)
        )
        self._reset_syllabus_escalation()

        # Planner still needs info -- relay the question.
        status = self.agent.planner_result.get("status")
        if status == "need_info":
            return self._coach_reply(
                self.agent.planner_result.get("message")
                or "Could you clarify which part of the "
                "syllabus you need?"
            )

        # Plan ready -- start the FAQ session.
        return self.agent.bot_session_manager.begin(
            bot_type="faq",
            tool_params=params,
            conversation_summary=(
                "Starting syllabus FAQ session due to "
                "repeated syllabus queries."
            ),
        )
