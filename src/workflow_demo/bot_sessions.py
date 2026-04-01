"""Bot session lifecycle management for coach agent."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Tutor-only REPL-style debug (must not reach the tutor LLM or pedagogy pipeline).
_TUTOR_DEBUG_COMMANDS = frozenset({"!retrieval", "!policy", "!diagnosis", "!state"})

from .pedagogy import (
    PedagogyRuntimeEvent,
    PedagogicalRetrievalPolicy,
    PolicyScorer,
    TeachingMoveGenerator,
    derive_instruction_lo,
    parse_prior_snapshot,
)
from .pedagogy.tutor_pedagogy_snapshot import build_tutor_pedagogy_snapshot
from .pedagogy.turn_progression import compute_turn_progression_signals
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
        self.active_learner_session_id: Optional[str] = None
        self.teaching_move_generator = TeachingMoveGenerator()
        self.policy_scorer = PolicyScorer()
        self._last_pedagogy_snapshot: Optional[Dict[str, Any]] = None

    def handle_turn(self, user_input: str) -> str:
        """Process a turn while in a bot session.

        Appends the user input to conversation history and invokes the bot.

        Args:
            user_input: The student's text input.

        Returns:
            The bot's response message.
        """
        self.conversation_history.append({"speaker": "student", "text": user_input})
        return self._invoke_bot(student_input=user_input)

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
        active_learner_session_id: Optional[str] = None
        if bot_type == "tutor":
            active_learner_session_id = self._build_tutor_state_session_id(context)
            context["pedagogy_context"] = self.agent.ensure_tutor_learner_context(
                session_id=active_learner_session_id,
                session_params=session_params,
            )

        self._reset()
        self.is_active = True
        self.bot_type = bot_type
        self.handoff_context = context
        self.active_learner_session_id = active_learner_session_id
        # Restore image for the duration of this bot session.
        self.agent.current_image = session_image
        self.agent.emit_event(
            "bot_session_started",
            f"Started {bot_type} session.",
            phase="session",
            bot_type=bot_type,
            subject=session_params.get("subject"),
            mode=session_params.get("mode"),
            topic=session_params.get("topic"),
            current_plan_titles=[
                item.get("title")
                for item in session_params.get("current_plan", [])
                if item.get("title")
            ],
            future_plan_titles=[
                item.get("title")
                for item in session_params.get("future_plan", [])
                if item.get("title")
            ],
        )

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
        self.active_learner_session_id = None
        self.agent.collected_params = {}
        self.agent.planner_result = None
        self.agent._reset_syllabus_escalation()
        self.agent.current_image = None
        self.agent.current_image_query = None
        self._last_pedagogy_snapshot = None

    def _refresh_pedagogy_snapshot(self) -> None:
        """Update cached API/UI snapshot from current handoff (tutor only)."""
        if self.bot_type != "tutor":
            self._last_pedagogy_snapshot = None
            return
        self._last_pedagogy_snapshot = build_tutor_pedagogy_snapshot(
            handoff_context=self.handoff_context,
            bot_type=self.bot_type,
            active_learner_session_id=self.active_learner_session_id,
            learner_state_engine=self.agent.learner_state_engine,
        )

    def _emit_math_guard_runtime_events(self, oc: Dict[str, Any]) -> None:
        """Emit Phase 8 math guard events from guard outcome dict."""
        sid = self.active_learner_session_id
        self.agent.emit_event(
            PedagogyRuntimeEvent.MATH_GUARD_CHECKED.value,
            "Math example guard evaluated.",
            phase="pedagogy",
            session_id=sid,
            candidate_type=oc.get("candidate_type"),
            verified=oc.get("verified"),
            repaired=oc.get("repaired"),
            reason=str(oc.get("reason") or ""),
        )
        if oc.get("repaired") is True:
            self.agent.emit_event(
                PedagogyRuntimeEvent.MATH_GUARD_REPAIRED.value,
                "Math example guard repaired claim.",
                phase="pedagogy",
                session_id=sid,
                candidate_type=oc.get("candidate_type"),
                reason=str(oc.get("reason") or ""),
            )

    def _invoke_bot(self, initial: bool = False, student_input: str = "") -> str:
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
            dbg_cmd = self._parse_tutor_debug_command(student_input)
            if student_input and dbg_cmd:
                dbg = self._format_tutor_debug_command(dbg_cmd)
                if dbg:
                    self.conversation_history.append({"speaker": "assistant", "text": dbg})
                self._refresh_pedagogy_snapshot()
                return dbg or ""
            if student_input and self.active_learner_session_id:
                self._run_misconception_diagnosis(student_input, history)
            bot_response = tutor_bot(
                llm_client=self.agent.llm_client,
                llm_model=self.agent.llm_model,
                handoff_context=self.handoff_context,
                conversation_history=history,
                image=self.agent.current_image,
                on_math_guard_outcome=self._emit_math_guard_runtime_events,
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

        if self.bot_type == "tutor" and student_input and self.active_learner_session_id:
            if not self._parse_tutor_debug_command(student_input):
                self._record_learner_turn(student_input)

        self._refresh_pedagogy_snapshot()

        # If the bot ended the session, finalize (save memory, update mastery).
        if bot_response.get("end_activity"):
            return self._finalize_bot_session(bot_response)

        return message or ""

    @staticmethod
    def _parse_tutor_debug_command(student_input: str) -> Optional[str]:
        """Return debug command token (e.g. '!policy') or None."""
        s = (student_input or "").strip()
        if s in _TUTOR_DEBUG_COMMANDS:
            return s
        return None

    def _format_tutor_debug_command(self, cmd: str) -> str:
        """Format one tutor-only debug reply from the canonical snapshot."""
        snap = build_tutor_pedagogy_snapshot(
            handoff_context=self.handoff_context,
            bot_type=self.bot_type,
            active_learner_session_id=self.active_learner_session_id,
            learner_state_engine=self.agent.learner_state_engine,
        ) or {}
        if cmd == "!retrieval":
            return self._format_tutor_retrieval_debug_from_snapshot(snap)
        if cmd == "!policy":
            return self._format_tutor_policy_debug_from_snapshot(snap)
        if cmd == "!diagnosis":
            return self._format_tutor_diagnosis_debug_from_snapshot(snap)
        if cmd == "!state":
            return self._format_tutor_state_debug_from_snapshot(snap)
        return ""

    def _format_tutor_retrieval_debug_from_snapshot(self, snap: Dict[str, Any]) -> str:
        lines = [
            "[DEBUG] Tutor retrieval / pedagogy state",
            "(This is an internal debug dump, not a tutoring reply.)",
            "",
            f"target_lo: {snap.get('target_lo')!r}",
            f"instruction_lo: {snap.get('instruction_lo')!r}",
            f"retrieval_intent: {snap.get('retrieval_intent')!r}",
            f"retrieval_action: {snap.get('retrieval_action')!r}",
            f"retrieval_execution_mode: {snap.get('retrieval_execution_mode')!r}",
            f"pack_focus_lo: {snap.get('pack_focus_lo')!r}",
            f"pack_revision: {snap.get('pack_revision')!r}",
            f"last_diagnosis_fingerprint: {snap.get('last_diagnosis_fingerprint')!r}",
            f"last_selected_move_type: {snap.get('last_selected_move_type')!r}",
            f"policy_reason: {snap.get('policy_reason')!r}",
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _format_tutor_policy_debug_from_snapshot(snap: Dict[str, Any]) -> str:
        lines = [
            "[DEBUG] Tutor policy",
            "(Internal debug — not a tutoring reply.)",
            "",
            f"selected_move_type: {snap.get('selected_move_type')!r}",
            f"policy_reason: {snap.get('policy_reason')!r}",
            f"candidate_move_types: {snap.get('candidate_move_types')!r}",
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _format_tutor_diagnosis_debug_from_snapshot(snap: Dict[str, Any]) -> str:
        lines = [
            "[DEBUG] Tutor diagnosis",
            "(Internal debug — not a tutoring reply.)",
            "",
            f"diagnosis_target_lo: {snap.get('diagnosis_target_lo')!r}",
            f"suspected_misconception: {snap.get('suspected_misconception')!r}",
            f"diagnosis_confidence: {snap.get('diagnosis_confidence')!r}",
            f"prerequisite_gap_los: {snap.get('prerequisite_gap_los')!r}",
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _format_tutor_state_debug_from_snapshot(snap: Dict[str, Any]) -> str:
        learner = snap.get("learner") or {}
        lines = [
            "[DEBUG] Learner snapshot",
            "(Internal debug — not a tutoring reply.)",
            "",
            f"recent_attempt_count: {learner.get('recent_attempt_count')!r}",
            f"hint_count: {learner.get('hint_count')!r}",
            f"top_mastery: {learner.get('top_mastery')!r}",
            f"recent_misconceptions: {learner.get('recent_misconceptions')!r}",
        ]
        return "\n".join(lines).strip()

    @property
    def last_pedagogy_snapshot(self) -> Optional[Dict[str, Any]]:
        """Last built tutor pedagogy snapshot for API/UI (tutor sessions only)."""
        return self._last_pedagogy_snapshot

    def _finalize_bot_session(self, bot_response: Dict[str, Any]) -> str:
        """Save session to memory, update proficiency, handle switches.

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
        self.agent.emit_event(
            "session_saved",
            "Saved completed session.",
            phase="session",
            session_type=session_type,
            topic=params.get("topic"),
            subject=params.get("subject"),
            mode=params.get("mode"),
        )

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

    def _update_lo_mastery(
        self, params: Dict[str, Any], summary: Dict[str, Any],
    ) -> None:
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

    def _record_learner_turn(self, student_input: str) -> None:
        """Record one learner attempt in the centralized learner state store."""
        if not self.active_learner_session_id:
            return
        student_turns = [
            item for item in self.conversation_history
            if item.get("speaker") == "student"
        ]
        session_params = (self.handoff_context or {}).get("session_params", {})
        lo_value = session_params.get("lo_id")
        if lo_value is None:
            current_plan = session_params.get("current_plan") or []
            if current_plan and isinstance(current_plan[0], dict):
                lo_value = current_plan[0].get("lo_id")
        lo_id = lo_value if isinstance(lo_value, int) else None
        self.agent.learner_state_engine.record_turn(
            session_id=self.active_learner_session_id,
            turn_index=max(len(student_turns) - 1, 0),
            student_text=student_input,
            lo_id=lo_id,
        )

    def _run_misconception_diagnosis(
        self,
        student_input: str,
        history: List[Dict[str, str]],
    ) -> None:
        """Diagnose misconceptions for tutor turns and attach into pedagogy_context."""
        if not self.active_learner_session_id or not self.handoff_context:
            return
        session_params = self.handoff_context.get("session_params", {})
        pedagogy_context: Dict[str, Any] = dict(self.handoff_context.get("pedagogy_context") or {})
        learner_payload = pedagogy_context.get("learner_state") or {}
        learner_state = self.agent.learner_state_store.ensure(self.active_learner_session_id)
        if learner_payload:
            learner_state = self.agent.learner_state_store.update(
                self.active_learner_session_id,
                **learner_payload,
            )

        plan_focus = (
            session_params.get("learning_objective")
            or next(
                (
                    str(item.get("title"))
                    for item in session_params.get("current_plan", [])
                    if item.get("title")
                ),
                None,
            )
            or learner_state.current_focus_lo
        )
        session_target_lo = (pedagogy_context.get("target_lo") or plan_focus or "unknown").strip()
        prior_instruction_lo = pedagogy_context.get("instruction_lo")
        prior_session_target_lo = pedagogy_context.get("target_lo")
        prior_instruction_str = (
            prior_instruction_lo.strip()
            if isinstance(prior_instruction_lo, str) and prior_instruction_lo.strip()
            else None
        )
        diagnoser_focus = prior_instruction_str or session_target_lo

        prior_rs = pedagogy_context.get("retrieval_session")
        previous_last_selected_move_type = None
        if isinstance(prior_rs, dict):
            previous_last_selected_move_type = prior_rs.get("last_selected_move_type")

        progression_signals = compute_turn_progression_signals(
            user_input=student_input,
            previous_last_selected_move_type=previous_last_selected_move_type,
        )

        diagnosis = self.agent.misconception_diagnoser.diagnose_turn(
            session_id=self.active_learner_session_id,
            user_input=student_input,
            current_focus_lo=diagnoser_focus,
            learner_state=learner_state,
            recent_messages=history[-6:],
        )
        updated_state = self.agent.learner_state_engine.record_misconception(
            session_id=self.active_learner_session_id,
            diagnosis=diagnosis,
        )
        teaching_moves = self.teaching_move_generator.generate_candidates(
            diagnosis=diagnosis,
            learner_state=updated_state,
            current_focus_lo=diagnoser_focus or "unknown",
            user_input=student_input,
        )
        pedagogy_context["diagnosis"] = diagnosis.model_dump(mode="json")
        pedagogy_context["learner_state"] = updated_state.model_dump(mode="json")
        pedagogy_context["teaching_moves"] = [
            candidate.model_dump(mode="json")
            for candidate in teaching_moves
        ]
        policy_decision = self.policy_scorer.select_best_move(
            diagnosis=diagnosis,
            learner_state=updated_state,
            teaching_moves=teaching_moves,
            current_focus_lo=diagnoser_focus or "unknown",
            user_input=student_input,
            progression_signals=progression_signals,
        )
        pedagogy_context["policy_decision"] = policy_decision.model_dump(mode="json")
        pedagogy_context["turn_progression_signals"] = progression_signals.to_json_dict()

        instruction_lo = derive_instruction_lo(
            session_target_lo=session_target_lo,
            diagnosis=diagnosis,
            selected_move_type=policy_decision.selected_move.move_type,
        )
        prior_snapshot = parse_prior_snapshot(pedagogy_context.get("retrieval_session"))

        retrieval = PedagogicalRetrievalPolicy(self.agent.retriever)
        r_out = retrieval.run(
            session_target_lo=session_target_lo,
            instruction_lo=instruction_lo,
            prior_session_target_lo=prior_session_target_lo,
            prior_instruction_lo=prior_instruction_lo,
            student_input=student_input,
            diagnosis=diagnosis,
            policy_decision=policy_decision,
            learner_state=updated_state,
            session_params=session_params,
            prior_snapshot=prior_snapshot,
            image_path=self.agent.current_image,
            student_profile=self.agent.student_profile,
        )
        if r_out.teaching_pack is not None:
            session_params["teaching_pack"] = r_out.teaching_pack
            self.handoff_context["session_params"] = session_params

        pedagogy_context["target_lo"] = session_target_lo
        pedagogy_context["instruction_lo"] = instruction_lo
        pedagogy_context["retrieval_intent"] = r_out.pedagogical_retrieval_intent.value
        pedagogy_context["retrieval_action"] = r_out.action.value
        pedagogy_context["retrieval_execution_mode"] = r_out.retrieval_execution_mode.value
        pedagogy_context["retrieval_session"] = r_out.state.model_dump(mode="json")
        tutor_instruction_directives = {
            "session_target_lo": session_target_lo,
            "instruction_lo": instruction_lo,
            "selected_move_type": policy_decision.selected_move.move_type.value,
            "retrieval_intent": r_out.pedagogical_retrieval_intent.value,
            "retrieval_action": r_out.action.value,
            "policy_reason": (policy_decision.decision_reason or "")[:500],
        }
        pedagogy_context["tutor_instruction_directives"] = tutor_instruction_directives
        # Legacy alias: same six fields (retrieval_execution_mode lives on pedagogy_context only).
        pedagogy_context["tutor_directives"] = dict(tutor_instruction_directives)

        self.handoff_context["pedagogy_context"] = pedagogy_context
        self.agent.emit_event(
            PedagogyRuntimeEvent.MISCONCEPTION_DIAGNOSED.value,
            "Diagnosed misconception signal for tutor turn.",
            phase="pedagogy",
            session_id=self.active_learner_session_id,
            target_lo=diagnosis.target_lo,
            suspected_misconception=diagnosis.suspected_misconception,
            confidence=round(diagnosis.confidence, 3),
            prerequisite_gap_los=diagnosis.prerequisite_gap_los,
        )
        self.agent.emit_event(
            PedagogyRuntimeEvent.TEACHING_MOVES_GENERATED.value,
            "Generated candidate teaching moves.",
            phase="pedagogy",
            session_id=self.active_learner_session_id,
            target_lo=diagnoser_focus,
            move_count=len(teaching_moves),
            move_types=[move.move_type.value for move in teaching_moves],
        )
        self.agent.emit_event(
            PedagogyRuntimeEvent.POLICY_DECISION_MADE.value,
            "Scored candidate moves and selected policy decision.",
            phase="pedagogy",
            session_id=self.active_learner_session_id,
            target_lo=diagnoser_focus,
            selected_move_type=policy_decision.selected_move.move_type.value,
            selected_move_id=policy_decision.selected_move.move_id,
            candidate_count=len(teaching_moves),
        )
        rs_dump = r_out.state.model_dump(mode="json")
        self.agent.emit_event(
            PedagogyRuntimeEvent.RETRIEVAL_POLICY_DECIDED.value,
            "Retrieval policy decided for tutor turn.",
            phase="pedagogy",
            session_id=self.active_learner_session_id,
            pedagogical_retrieval_intent=r_out.pedagogical_retrieval_intent.value,
            retrieval_action=r_out.action.value,
            retrieval_execution_mode=r_out.retrieval_execution_mode.value,
            reason_codes=r_out.reason_codes,
        )
        self.agent.emit_event(
            PedagogyRuntimeEvent.RETRIEVAL_EXECUTED.value,
            "Retrieval execution applied for tutor turn.",
            phase="pedagogy",
            session_id=self.active_learner_session_id,
            pack_revision=rs_dump.get("pack_revision"),
            teaching_pack_updated=r_out.teaching_pack is not None,
            material_triggers=r_out.material_triggers,
        )

    @staticmethod
    def _build_tutor_state_session_id(context: Dict[str, Any]) -> str:
        """Build a learner-state session id for one tutor handoff."""
        metadata = context.get("handoff_metadata", {})
        timestamp = str(metadata.get("timestamp") or "")
        if timestamp:
            return f"tutor:{timestamp}"
        return "tutor:unknown"

    def format_proficiency_report(self) -> str:
        """Format lo_mastery scores into a readable summary for the student.

        Returns:
            Formatted string showing proficiency levels for each learning objective.
        """
        lo_mastery = self.agent.student_profile.get("lo_mastery", {})

        if not lo_mastery:
            return (
                "You haven't completed any tutoring sessions yet, "
                "so I don't have proficiency data. "
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

        # Split LOs into strong (>=65%) and weak (<65%), sorted best-first.
        sorted_items = sorted(lo_mastery.items(), key=lambda x: -x[1])
        strong = [(lo, score) for lo, score in sorted_items if score >= 0.65]
        needs_work = [(lo, score) for lo, score in sorted_items if score < 0.65]

        lines = ["Your Learning Progress:", ""]

        # List strong areas.
        if strong:
            lines.append("Strong areas:")
            for lo, score in strong:
                lines.append(f"  - {lo}: {int(score * 100)}% ({score_to_label(score)})")

        # List weak areas, with a blank separator if both sections exist.
        if needs_work:
            if strong:
                lines.append("")
            lines.append("Areas to focus on:")
            for lo, score in needs_work:
                lines.append(f"  - {lo}: {int(score * 100)}% ({score_to_label(score)})")

            # Suggest the weakest LO (last in descending-sorted list).
            weakest = needs_work[-1][0]
            lines.append("")
            lines.append(
                f"Tip: Consider practicing '{weakest}' "
                "next to strengthen your foundation."
            )

        return "\n".join(lines)

    def _build_return_greeting(self, params: Dict[str, Any], session_type: str) -> str:
        """Build a greeting after a completed session.

        Args:
            params: Session parameters.
            session_type: "tutor" or "faq".

        Returns:
            A greeting message referencing the completed session.
        """
        lo = params.get("learning_objective")
        mode = params.get("mode")

        # Tutor: mention the LO (and mode if present).
        if session_type == "tutor" and lo:
            mode_str = (mode or "").replace("_", " ")
            if mode_str:
                return (
                    f"Nice work on {lo} in {mode_str} mode! "
                    "What would you like to work on next?"
                )
            return (
                f"Nice work on {lo}! "
                "What would you like to work on next?"
            )

        # FAQ: mention the topic.
        if session_type == "faq":
            topic = params.get("topic")
            if topic:
                return (
                    f"Glad I could help with {topic}. "
                    "What else would you like to explore?"
                )

        # Fallback: generic coach greeting.
        return (
            "Hi! I'm your learning coach. I can help you start a "
            "tutoring session or answer questions about FAQs and "
            "syllabus. What would you like to work on?"
        )
