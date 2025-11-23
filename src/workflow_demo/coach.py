"""
Notebook-style multi-agent coach that routes between tutoring and FAQ sessions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from src.workflow_demo.planner import FAQPlanner, TutoringPlanner
    from src.workflow_demo.retriever import TeachingPackRetriever
    from src.workflow_demo.session_memory import SessionMemory, create_handoff_context
    from src.workflow_demo.tutor import faq_bot, tutor_bot
except ImportError:
    from .planner import FAQPlanner, TutoringPlanner
    from .retriever import TeachingPackRetriever
    from .session_memory import SessionMemory, create_handoff_context
    from .tutor import faq_bot, tutor_bot


COACH_GREETING = (
    "Hi! I'm your learning coach. I can help you start a tutoring session or answer "
    "questions about FAQs and syllabus. What would you like to work on?"
)

COACH_SYSTEM_PROMPT = """
You are the orchestrator of a learning assistant. The student only talks to you,
but behind the scenes you can call tools (tutoring planner, FAQ planner, tutor bot, FAQ bot).

INPUT: A JSON payload with:
- conversation_history: last 10 messages between coach and student.
- recent_sessions: up to 5 session summaries (tutoring or FAQ) with params and summaries.
- last_tutoring_session: shortcut to the most recent tutoring session (or null).
- planner_result: most recent planner call result, if any.
- collected_params: currently known subject/topic/mode/faq topic metadata.
- awaiting_confirmation: bool flag if the student has already seen a plan and we are waiting on approval.
- returning_from_session: true when the student just came back from a tutor/FAQ session via a switch request.

YOU MUST RETURN STRICT JSON:
{
  "message_to_student": "plain text or empty string when handing off",
  "action": "none|call_tutoring_planner|call_faq_planner|start_tutor|start_faq",
  "tool_params": {
    "subject": "...",
    "learning_objective": "...",
    "mode": "conceptual_review|examples|practice",
    "topic": "...",
    "student_request": "original wording"
  },
  "conversation_summary": "short summary used when handing off (optional)"
}

Guidelines:
1. Intent detection:
   - Tutoring intent: wants to learn/practice/review a topic. Gather subject + learning_objective + mode.
   - FAQ intent: wants logistics/policy info. Gather topic only.
   - Topic switch or "back to" phrases after a session should jump straight into planning with the new topic. If the student says
     "back to the previous topic", reuse last_tutoring_session.params (subject + learning_objective) as defaults.
   - Mode switch ("switch to practice/examples/conceptual review") should reuse last_tutoring_session.params for subject + learning_objective,
     override the mode, and immediately call the tutoring planner.
2. Planner usage:
   - When required info is missing, ask one clarifying question (message_to_student) and set action="none".
   - Once all tutoring params are ready, call tutoring planner via action="call_tutoring_planner" with the params.
   - Same for FAQ planner with action="call_faq_planner".
   - When planner returns status need_info, relay the planner's message verbatim and continue collecting info.
3. Confirmation + handoff:
   - After receiving a complete plan, summarize it to the student unless they already confirmed.
   - When the student agrees (positive confirmations like "yes", "let's go"), set action="start_tutor" (for tutoring) or "start_faq" (for FAQ)
     and include conversation_summary describing why we are starting the session.
   - Leave message_to_student empty when handing off so the tutor/FAQ bot can speak first.
4. Returning from sessions:
   - If returning_from_session is true, immediately honor the switch request contained in the latest message.
   - Do NOT greet; instead analyze the request and call the appropriate planner or start a new plan, using last_tutoring_session when helpful.
5. Always keep tool_params specific to the current intent. Do not mix FAQ params with tutoring params.
"""


def _coerce_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lstrip().lower().startswith("json"):
            raw = raw.split("\n", 1)[1]
    return json.loads(raw)


class CoachAgent:
    """
    State machine that mirrors the multi-agent flow from the 9 Oct notebook.
    """

    def __init__(
        self,
        retriever: Optional[TeachingPackRetriever] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.retriever = retriever or TeachingPackRetriever()
        self.session_memory = SessionMemory()
        self.conversation_history: List[Dict[str, str]] = []
        self.collected_params: Dict[str, Any] = {}
        self.planner_result: Optional[Dict[str, Any]] = None
        self.awaiting_confirmation = False
        self.returning_from_session = False
        self.in_bot_session = False
        self.bot_type: Optional[str] = None
        self.bot_handoff_context: Optional[Dict[str, Any]] = None
        self.bot_conversation_history: List[Dict[str, str]] = []
        self.student_profile: Dict[str, Any] = {"lo_mastery": {}}
        self.pending_session_type: Optional[str] = None

        self.llm_client: Optional[OpenAI] = None
        self.llm_model: Optional[str] = None
        self._init_llm(llm_model)

        self.tutoring_planner = TutoringPlanner(self.retriever)
        self.faq_planner = FAQPlanner()

    def _init_llm(self, llm_model: Optional[str]) -> None:
        API_KEY = os.getenv("OPENAI_API_KEY")
        model_name = llm_model or os.getenv("WORKFLOW_DEMO_LLM_MODEL", "gpt-4o-mini")
        try:
            if not API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.llm_client = OpenAI(api_key=API_KEY)
            self.llm_model = model_name
        except Exception as exc:
            self.llm_client = None
            self.llm_model = None
            print(f"[Coach] Warning: failed to initialize OpenAI client ({exc}).")

    def initial_greeting(self) -> str:
        return COACH_GREETING

    def process_turn(self, user_input: str) -> str:
        if self.in_bot_session:
            return self._handle_bot_turn(user_input)
        return self._handle_coach_turn(user_input)

    # ------------------------------------------------------------------
    # Coach loop
    # ------------------------------------------------------------------

    def _handle_coach_turn(self, user_input: str, synthetic: bool = False) -> str:
        if user_input.strip():
            self._record_message("student", user_input)

        if self.returning_from_session and not self.collected_params:
            last_session = self.session_memory.last_tutoring_session()
            if last_session:
                params = last_session.get("params") or {}
                for key in ("subject", "learning_objective", "mode"):
                    value = params.get(key)
                    if value:
                        self.collected_params.setdefault(key, value)

        normalized = user_input.strip().lower()
        if self.awaiting_confirmation:
            if self._is_positive_confirmation(normalized):
                summary = (
                    "Student confirmed the tutoring plan."
                    if self.pending_session_type == "tutor"
                    else "Student confirmed the FAQ plan."
                )
                bot_type = self.pending_session_type or "tutor"
                return self._begin_bot_session(
                    bot_type=bot_type,
                    tool_params=dict(self.collected_params),
                    conversation_summary=summary,
                )
            if self._is_negative_confirmation(normalized):
                self.awaiting_confirmation = False
                self.pending_session_type = None
                self.planner_result = None
                self.collected_params = {}
                return self._coach_reply("No problem. What would you like to work on instead?")

        if not self.awaiting_confirmation:
            faq_topic = self._detect_faq_topic(normalized)
            if faq_topic:
                self.collected_params.setdefault("topic", faq_topic)
                self.collected_params["student_request"] = user_input

        loop_guard = 0
        while True:
            loop_guard += 1
            if loop_guard > 4:
                return "Let's come back to this in a moment."

            coach_directive = self._run_coach_brain()
            action = coach_directive.get("action", "none")
            message = (coach_directive.get("message_to_student") or "").strip()
            tool_params = coach_directive.get("tool_params") or {}
            self._update_collected_params(tool_params)

            if action == "call_tutoring_planner":
                self.planner_result = self._call_tutoring_planner(tool_params)
                if self.planner_result.get("status") == "need_info":
                    msg = self.planner_result.get("message") or "Could you clarify that a bit more?"
                    return self._coach_reply(msg)
                continue

            if action == "call_faq_planner":
                self.planner_result = self._call_faq_planner(tool_params)
                if self.planner_result.get("status") == "need_info":
                    msg = self.planner_result.get("message") or "Which FAQ topic should we cover?"
                    return self._coach_reply(msg)
                continue

            if action == "start_tutor":
                return self._begin_bot_session(
                    bot_type="tutor",
                    tool_params=tool_params,
                    conversation_summary=coach_directive.get("conversation_summary") or "Student requested tutoring.",
                )

            if action == "start_faq":
                return self._begin_bot_session(
                    bot_type="faq",
                    tool_params=tool_params,
                    conversation_summary=coach_directive.get("conversation_summary") or "Student requested FAQ help.",
                )

            # action == "none"
            if message:
                return self._coach_reply(message)
            return ""

    def _run_coach_brain(self) -> Dict[str, Any]:
        payload = {
            "conversation_history": self.conversation_history[-10:],
            "recent_sessions": self.session_memory.get_recent_sessions(),
            "last_tutoring_session": self.session_memory.last_tutoring_session(),
            "planner_result": self.planner_result,
            "collected_params": self.collected_params,
            "awaiting_confirmation": self.awaiting_confirmation,
            "returning_from_session": self.returning_from_session,
        }
        self.returning_from_session = False
        if not self.llm_client or not self.llm_model:
            raise RuntimeError("LLM client is not available; coach flow requires it.")

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": COACH_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
        )
        return _coerce_json(response.choices[0].message.content)

    def _coach_reply(self, message: str) -> str:
        self._record_message("assistant", message)
        return message

    def _update_collected_params(self, tool_params: Dict[str, Any]) -> None:
        if not tool_params:
            return
        for key, value in tool_params.items():
            if value:
                self.collected_params[key] = value

    # ------------------------------------------------------------------
    # Planner calls
    # ------------------------------------------------------------------

    def _call_tutoring_planner(self, params: Dict[str, Any]) -> Dict[str, Any]:
        print("\n🔧 Tutoring Session Planner")
        print(json.dumps(params, indent=2))
        result = self.tutoring_planner.create_plan(
            {
                **params,
                "student_profile": self.student_profile,
            }
        )
        print("📩 Planner response:")
        print(json.dumps(result, indent=2))
        if result.get("status") == "complete":
            self.awaiting_confirmation = True
            self.pending_session_type = "tutor"
        return result

    def _call_faq_planner(self, params: Dict[str, Any]) -> Dict[str, Any]:
        print("\n🔧 FAQ/Syllabus Planner")
        print(json.dumps(params, indent=2))
        result = self.faq_planner.create_plan(params)
        print("📩 Planner response:")
        print(json.dumps(result, indent=2))
        if result.get("status") == "complete":
            self.awaiting_confirmation = True
            self.pending_session_type = "faq"
        return result

    # ------------------------------------------------------------------
    # Bot orchestration
    # ------------------------------------------------------------------

    def _begin_bot_session(
        self,
        bot_type: str,
        tool_params: Dict[str, Any],
        conversation_summary: str,
    ) -> str:
        plan = (self.planner_result or {}).get("plan") or {}
        session_params = {
            **plan,
            **{k: v for k, v in tool_params.items() if v},
        }
        plan_teaching_pack = (self.planner_result or {}).get("plan", {}).get("teaching_pack")
        if plan_teaching_pack and "teaching_pack" not in session_params:
            session_params["teaching_pack"] = plan_teaching_pack
        student_request = session_params.get("student_request") or self._last_student_message()
        session_params["student_request"] = student_request

        context = create_handoff_context(
            from_agent="coach",
            to_agent=bot_type,
            session_params=session_params,
            conversation_summary=conversation_summary,
            session_memory=self.session_memory,
            student_state=self.student_profile,
        )

        self.in_bot_session = True
        self.bot_type = bot_type
        self.bot_handoff_context = context
        self.bot_conversation_history = []
        self.awaiting_confirmation = False
        self.collected_params = {}
        self.planner_result = None
        self.pending_session_type = None

        return self._invoke_bot(initial=True)

    def _handle_bot_turn(self, user_input: str) -> str:
        self.bot_conversation_history.append({"speaker": "student", "text": user_input})
        return self._invoke_bot()

    def _invoke_bot(self, initial: bool = False) -> str:
        if not self.bot_type or not self.bot_handoff_context:
            self._reset_after_session()
            return self.initial_greeting()

        history = self.bot_conversation_history
        if initial:
            history = []

        if not self.llm_client or not self.llm_model:
            raise RuntimeError("LLM client is required for tutor/FAQ bots.")

        if self.bot_type == "tutor":
            bot_response = tutor_bot(
                llm_client=self.llm_client,
                llm_model=self.llm_model,
                handoff_context=self.bot_handoff_context,
                conversation_history=history,
            )
        else:
            bot_response = faq_bot(
                llm_client=self.llm_client,
                llm_model=self.llm_model,
                handoff_context=self.bot_handoff_context,
                conversation_history=history,
            )

        message = (bot_response.get("message_to_student") or "").strip()
        if message:
            self.bot_conversation_history.append({"speaker": "assistant", "text": message})

        if bot_response.get("end_activity"):
            return self._finalize_bot_session(bot_response)

        return message or ""

    def _finalize_bot_session(self, bot_response: Dict[str, Any]) -> str:
        summary = bot_response.get("session_summary") or {}
        session_type = self.bot_type or "tutor"
        params = (self.bot_handoff_context or {}).get("session_params", {})
        exchanges = list(self.bot_conversation_history)
        self.session_memory.add_session(session_type, params, summary, exchanges)

        switch_topic = summary.get("switch_topic_request")
        switch_mode = summary.get("switch_mode_request")

        self._reset_after_session()

        if switch_topic:
            self.returning_from_session = True
            return self._handle_coach_turn(switch_topic, synthetic=True)

        if switch_mode:
            self.returning_from_session = True
            return self._handle_coach_turn(switch_mode, synthetic=True)

        greeting = self.initial_greeting()
        self._record_message("assistant", greeting)
        return greeting

    def _reset_after_session(self) -> None:
        self.in_bot_session = False
        self.bot_type = None
        self.bot_handoff_context = None
        self.bot_conversation_history = []
        self.collected_params = {}
        self.planner_result = None
        self.awaiting_confirmation = False

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    def _record_message(self, speaker: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.conversation_history.append({"speaker": speaker, "text": text})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _last_student_message(self) -> str:
        for message in reversed(self.conversation_history):
            if message["speaker"] == "student":
                return message["text"]
        return ""

    @staticmethod
    def _is_positive_confirmation(normalized_input: str) -> bool:
        positives = {
            "yes",
            "yep",
            "sure",
            "let's go",
            "ready",
            "start",
            "absolutely",
            "yeah",
            "ok let's do it",
        }
        return any(token in normalized_input for token in positives)

    @staticmethod
    def _is_negative_confirmation(normalized_input: str) -> bool:
        negatives = {
            "no",
            "not now",
            "later",
            "another time",
            "change",
            "switch",
            "stop",
        }
        return any(token in normalized_input for token in negatives)

    @staticmethod
    def _detect_faq_topic(normalized_input: str) -> Optional[str]:
        keyword_map = {
            "exam": "exam schedule",
            "quiz": "quiz schedule",
            "homework": "homework policy",
            "grading": "grading policy",
            "grade": "grading policy",
            "office hours": "office hours",
            "syllabus": "exam schedule",
            "faq": "exam schedule",
        }
        for key, topic in keyword_map.items():
            if key in normalized_input:
                return topic
        for topic in FAQPlanner.FAQ_TOPICS.keys():
            if topic in normalized_input:
                return topic
        return None

