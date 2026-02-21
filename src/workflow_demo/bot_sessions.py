"""Bot session lifecycle management for coach agent."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .session_memory import create_handoff_context
from .tutor import tutor_bot, faq_bot

# Import CoachAgent only for type checkers to avoid circular import at runtime.
if TYPE_CHECKING:
    from .coach_agent import CoachAgent


UNDERSTANDING_TO_MASTERY = {
    "excellent": 0.9,
    "good": 0.7,
    "satisfactory": 0.6,
    "needs_practice": 0.4,
    "struggling": 0.3,
}


class BotSessionManager:
    """Manages the lifecycle of tutor/FAQ bot sessions.

    Owns all bot-session state (active flag, type, handoff context,
    conversation history). The CoachAgent delegates to this manager
    rather than tracking session fields itself.
    """

    def __init__(self, agent: "CoachAgent") -> None:
        """Initialize the bot session manager.

        Args:
            agent: Reference to the CoachAgent instance (owns shared state).
        """
        self.agent = agent
        self.is_active = False
        self.bot_type: Optional[str] = None
        self.handoff_context: Optional[Dict[str, Any]] = None
        self.conversation_history: List[Dict[str, str]] = []

    def handle_turn(self, user_input: str) -> str:
        """Process a turn while in a bot session.

        Appends the user input to conversation history and invokes the bot.

        Args:
            user_input: The student's text input.

        Returns:
            The bot's response message.
        """
        self.conversation_history.append({"speaker": "student", "text": user_input})
        return self._invoke_bot()

    def begin(
        self,
        bot_type: str,
        tool_params: Dict[str, Any],
        conversation_summary: str,
    ) -> str:
        """Begin a new bot session (tutor or FAQ).

        Builds session params from planner result + tool params, creates a
        handoff context, resets all transient state, and invokes the bot to
        get its opening message.

        Args:
            bot_type: "tutor" or "faq".
            tool_params: Tool parameters from coach directive.
            conversation_summary: Summary of why the session is starting.

        Returns:
            The bot's opening message.
        """
        session_params = self._build_session_params(tool_params)
        session_image = self.agent.current_image

        context = create_handoff_context(
            from_agent="coach",
            to_agent=bot_type,
            session_params=session_params,
            conversation_summary=conversation_summary,
            session_memory=self.agent.session_memory,
            student_state=self.agent.student_profile,
            image=session_image,
        )

        self._reset()
        self.is_active = True
        self.bot_type = bot_type
        self.handoff_context = context
        # Restore image for the duration of this bot session.
        self.agent.current_image = session_image

        return self._invoke_bot(initial=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_session_params(self, tool_params: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble session parameters from planner output, supplemented by tool params.

        Plan values take precedence for keys the planner already decided.
        tool_params only supplies values for keys the plan does not cover
        (e.g. student_request, image_query). Adds student_request and
        image_query fallbacks.

        Args:
            tool_params: Parameters from the coach directive.

        Returns:
            Merged session parameters dict.
        """
        # Start with everything the planner decided.
        plan = (self.agent.planner_result or {}).get("plan") or {}
        session_params = dict(plan)

        # Fill in gaps from tool_params (only keys the plan didn't cover).
        for k, v in tool_params.items():
            if v and not session_params.get(k):
                session_params[k] = v

        # Ensure student_request is always present.
        session_params["student_request"] = (
            session_params.get("student_request") or self.agent._last_student_message()
        )

        # Attach image analysis query if the student submitted an image.
        if self.agent.current_image_query and "image_query" not in session_params:
            session_params["image_query"] = self.agent.current_image_query

        return session_params

    def _reset(self) -> None:
        """Clear all bot session state and associated agent transient state."""
        self.is_active = False
        self.bot_type = None
        self.handoff_context = None
        self.conversation_history = []
        self.agent.collected_params = {}
        self.agent.planner_result = None
        self.agent._reset_syllabus_escalation()
        self.agent.current_image = None
        self.agent.current_image_query = None

    def _invoke_bot(self, initial: bool = False) -> str:
        """Invoke the bot (tutor or FAQ) with the current conversation history.

        Args:
            initial: If True, pass empty history (first message from bot).

        Returns:
            The bot's response message or a final greeting if session ended.
        """
        # Guard: if session state is missing, bail to coach greeting.
        if not self.bot_type or not self.handoff_context:
            self._reset()
            return self.agent.initial_greeting()

        # First call gets empty history so the bot generates its opener.
        history = [] if initial else self.conversation_history

        # Safety net (should never fire -- manager is only created with LLM).
        if not self.agent.llm_client or not self.agent.llm_model:
            raise RuntimeError("LLM client is required for tutor/FAQ bots.")

        # Dispatch to the appropriate bot.
        if self.bot_type == "tutor":
            bot_response = tutor_bot(
                llm_client=self.agent.llm_client,
                llm_model=self.agent.llm_model,
                handoff_context=self.handoff_context,
                conversation_history=history,
                image=self.agent.current_image,
            )
        else:
            bot_response = faq_bot(
                llm_client=self.agent.llm_client,
                llm_model=self.agent.llm_model,
                handoff_context=self.handoff_context,
                conversation_history=history,
            )

        # Record the bot's reply in conversation history.
        message = (bot_response.get("message_to_student") or "").strip()
        if message:
            self.conversation_history.append({"speaker": "assistant", "text": message})

        # If the bot ended the session, finalize (save memory, update mastery).
        if bot_response.get("end_activity"):
            return self._finalize_bot_session(bot_response)

        return message or ""

    def _finalize_bot_session(self, bot_response: Dict[str, Any]) -> str:
        """Save the session to memory, update proficiency, and handle topic/mode switches.

        Args:
            bot_response: The final response from the bot.

        Returns:
            A return greeting or switch request routed back to coach.
        """
        # Capture session data before reset clears it.
        summary = bot_response.get("session_summary") or {}
        session_type = self.bot_type or "tutor"
        params = (self.handoff_context or {}).get("session_params", {})
        exchanges = list(self.conversation_history)

        # Persist the completed session record.
        self.agent.session_memory.add_session(session_type, params, summary, exchanges)

        # Update mastery score and flush to disk (tutor sessions only).
        if session_type == "tutor":
            self._update_lo_mastery(params, summary)
            self.agent.session_memory.save()

        # Extract switch requests before reset wipes them.
        switch_topic = summary.get("switch_topic_request")
        switch_mode = summary.get("switch_mode_request")

        self._reset()

        # If the bot requested a topic or mode switch, route back to coach.
        if switch_topic:
            self.agent.returning_from_session = True
            return self.agent._handle_coach_turn(switch_topic, synthetic=True)

        if switch_mode:
            self.agent.returning_from_session = True
            return self.agent._handle_coach_turn(switch_mode, synthetic=True)

        # Normal end: return a continuity-aware greeting.
        greeting = self._build_return_greeting(params, session_type)
        self.agent._record_message("assistant", greeting)
        return greeting

    def _update_lo_mastery(self, params: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """Map the tutor's student_understanding assessment to a numeric mastery score.

        Args:
            params: Session parameters containing the learning objective.
            summary: Session summary containing student_understanding assessment.
        """
        # Resolve the LO identifier (title or numeric id).
        lo_key = params.get("learning_objective") or params.get("lo_id")
        if not lo_key:
            return

        # Map the tutor's qualitative label to a 0-1 score (default 0.4).
        understanding = summary.get("student_understanding") or ""
        score = UNDERSTANDING_TO_MASTERY.get(understanding.lower(), 0.4)
        self.agent.student_profile.setdefault("lo_mastery", {})[lo_key] = score

    def format_proficiency_report(self) -> str:
        """Format lo_mastery scores into a readable summary for the student.

        Returns:
            Formatted string showing proficiency levels for each learning objective.
        """
        lo_mastery = self.agent.student_profile.get("lo_mastery", {})

        if not lo_mastery:
            return (
                "You haven't completed any tutoring sessions yet, so I don't have proficiency data. "
                "Start a tutoring session and I'll track your progress!"
            )

        def score_to_label(score: float) -> str:
            if score >= 0.85:
                return "Excellent"
            elif score >= 0.65:
                return "Good"
            elif score >= 0.5:
                return "Satisfactory"
            elif score >= 0.35:
                return "Needs Practice"
            else:
                return "Struggling"

        sorted_items = sorted(lo_mastery.items(), key=lambda x: -x[1])

        strong = [(lo, score) for lo, score in sorted_items if score >= 0.65]
        needs_work = [(lo, score) for lo, score in sorted_items if score < 0.65]

        lines = ["Your Learning Progress:"]
        lines.append("")

        if strong:
            lines.append("Strong areas:")
            for lo, score in strong:
                pct = int(score * 100)
                label = score_to_label(score)
                lines.append(f"  - {lo}: {pct}% ({label})")

        if needs_work:
            if strong:
                lines.append("")
            lines.append("Areas to focus on:")
            for lo, score in needs_work:
                pct = int(score * 100)
                label = score_to_label(score)
                lines.append(f"  - {lo}: {pct}% ({label})")

        if needs_work:
            lowest_lo = needs_work[-1][0]
            lines.append("")
            lines.append(f"Tip: Consider practicing '{lowest_lo}' next to strengthen your foundation.")

        return "\n".join(lines)

    def _build_return_greeting(self, params: Dict[str, Any], session_type: str) -> str:
        """Build a continuity-aware greeting after a completed session.

        Args:
            params: Session parameters.
            session_type: "tutor" or "faq".

        Returns:
            A greeting message referencing the completed session.
        """
        lo = params.get("learning_objective")
        mode = params.get("mode")
        if session_type == "tutor" and lo:
            mode_str = (mode or "").replace("_", " ")
            if mode_str:
                return f"Nice work on {lo} in {mode_str} mode! What would you like to work on next?"
            return f"Nice work on {lo}! What would you like to work on next?"
        if session_type == "faq":
            topic = params.get("topic")
            if topic:
                return f"Glad I could help with {topic}. What else would you like to explore?"
        return "Hi! I'm your learning coach. I can help you start a tutoring session or answer questions about FAQs and syllabus. What would you like to work on?"
