"""
Multi-agent coach that routes between tutoring and FAQ sessions.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError, OpenAI

from .image_preprocessor import ImagePreprocessor
from .json_utils import coerce_json
from .planner import FAQPlanner, TutoringPlanner
from .retriever import TeachingPackRetriever
from .session_memory import SessionMemory, create_handoff_context
from .tutor import faq_bot, tutor_bot


SUBJECT_KEYWORDS = {
    "calculus": {
        "calculus",
        "derivative",
        "derivatives",
        "differential",
        "differentials",
        "integral",
        "integrals",
        "limit",
        "limits",
        "tangent",
        "rate of change",
    },
    "algebra": {
        "algebra",
        "quadratic",
        "quadratics",
        "polynomial",
        "polynomials",
        "equation",
        "equations",
        "linear",
        "system of equations",
    },
    "trigonometry": {
        "trigonometry",
        "trig",
        "sine",
        "cosine",
        "tangent function",
        "angle",
        "angles",
    },
}

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
  "action": "none|call_tutoring_planner|call_faq_planner|start_tutor|start_faq|show_proficiency",
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
   - Proficiency intent: wants to see their learning progress. Phrases like "show my proficiency", "how am I doing",
     "what's my progress", "show my scores", "my learning progress" → set action="show_proficiency".
   - Syllabus / course-outline / "major concepts" questions should go to FAQ mode using topic "syllabus_topics" and student_request containing the learner's wording.
   - Topic switch phrases ("teach me X instead", "let's do Y") after a session should jump straight into planning with the NEW topic mentioned.
   - "Back to" or "continue" phrases should reuse last_tutoring_session.params (subject + learning_objective) as defaults.
   - IMPORTANT: If the student does NOT name a topic/subject (e.g., "start a tutoring session", "I want to learn"), you MUST ASK:
     "What specific topic or learning objective would you like to focus on?" and set action="none".
   - Do NOT infer subject or learning_objective from recent_sessions or last_tutoring_session unless the student explicitly says to continue/return (e.g., "continue where I left off", "back to previous topic").
   - Mode switch ("switch to practice/examples/conceptual review") should reuse last_tutoring_session.params for subject + learning_objective,
     override the mode, and immediately call the tutoring planner.
2. Planner usage:
   - When required info is missing, ask one clarifying question (message_to_student) and set action="none".
   - Once all tutoring params are ready, call tutoring planner via action="call_tutoring_planner" with the params.
   - Same for FAQ planner with action="call_faq_planner".
   - When planner returns status need_info, relay the planner's message verbatim and continue collecting info.
3. Confirmation + handoff:
   - After receiving a complete plan, summarize it back to the student (subject, objective, mode, book) and explicitly ask if they want to begin.
   - Planner calls and tutor/FAQ handoffs must never happen in the same student turn. Always wait for the next student reply after presenting a plan before emitting a start_* action.
   - While awaiting_confirmation == true, NEVER emit action="start_tutor" or "start_faq". Respond with action="none" and either (a) repeat the plan summary or (b) capture any requested adjustments until the student clearly says yes/no.
   - When the student agrees, set action="start_tutor" (for tutoring) or "start_faq" (for FAQ) and include conversation_summary describing why we are starting the session, then leave message_to_student empty so the bot can speak first.
   - If the student declines, clear awaiting_confirmation and gather new requirements before calling planners again.
4. Returning from sessions:
   - If returning_from_session is true, immediately honor the switch request contained in the latest message.
   - Do NOT greet; instead analyze the request and call the appropriate planner or start a new plan, using last_tutoring_session when helpful.
5. Always keep tool_params specific to the current intent. Do not mix FAQ params with tutoring params.
"""


class CoachAgent:


    UNDERSTANDING_TO_MASTERY: Dict[str, float] = {
        "excellent": 0.9,
        "good": 0.7,
        "satisfactory": 0.6,
        "needs_practice": 0.4,
        "struggling": 0.3,
    }

    def __init__(
        self,
        retriever: Optional[TeachingPackRetriever] = None,
        llm_model: Optional[str] = None,
        session_memory_path: Optional[str] = None,
    ) -> None:
        self.retriever = retriever or TeachingPackRetriever()
        self.session_memory = SessionMemory(persistence_path=session_memory_path)
        self.conversation_history: List[Dict[str, str]] = []
        self.collected_params: Dict[str, Any] = {}
        self.planner_result: Optional[Dict[str, Any]] = None
        self.awaiting_confirmation = False
        self.returning_from_session = False
        self.in_bot_session = False
        self.bot_type: Optional[str] = None
        self.bot_handoff_context: Optional[Dict[str, Any]] = None
        self.bot_conversation_history: List[Dict[str, str]] = []
        self.student_profile: Dict[str, Any] = self.session_memory.student_profile
        self.pending_session_type: Optional[str] = None
        self.awaiting_confirmation_prompted = False
        self.plan_confirmation_summary: Optional[str] = None
        self.syllabus_request_active = False
        self.syllabus_clarification_count = 0
        self.current_image: Optional[str] = None
        self.current_image_query: Optional[str] = None

        self.llm_client: Optional[OpenAI] = None
        self.llm_model: Optional[str] = None
        self._init_llm(llm_model)

        self.tutoring_planner = TutoringPlanner(self.retriever)
        self.faq_planner = FAQPlanner()
        self.image_preprocessor: Optional[ImagePreprocessor] = None
        try:
            self.image_preprocessor = ImagePreprocessor()
        except Exception as exc:
            print(f"[Coach] Warning: failed to initialize image preprocessor ({exc}).")

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
        """
        Handle a text-only turn.
        
        If an image was previously submitted (self.current_image is set),
        it will persist and be passed to the tutor for follow-up questions.
        """
        if self.in_bot_session:
            return self._handle_bot_turn(user_input)
        return self._handle_coach_turn(user_input)

    def process_multimodal_turn(self, text: str, image: str) -> str:
        """
        Handle a turn that includes an image (file path or URL).
        
        The image is stored in self.current_image and persists across follow-up turns
        within the same session, allowing the tutor to reference it in subsequent
        responses. The image is passed directly to GPT-4o for native multimodal understanding.
        """
        self.current_image = image
        self.current_image_query = None

        text = text or ""
        query_text = text
        query_parts: List[str] = [text] if text else []

        print(f"\n🖼️  Image detected: {image}")
        
        if self.image_preprocessor:
            try:
                print("🔍 Analyzing image")
                result = self.image_preprocessor.process_image(image, text or None)
                self.current_image_query = result.query
                
                # Debug output
                print(f"📋 Image Analysis Results:")
                print(f"   Type: {result.detected_type}")
                print(f"   Confidence: {result.confidence:.0%}")
                if result.likely_topic:
                    print(f"   Topic: {result.likely_topic}")
                if result.latex_content:
                    print(f"   LaTeX: {result.latex_content[:2]}...")  # First 2
                if result.key_features:
                    print(f"   Features: {result.key_features[:3]}")
                print(f"   Generated Query: {result.query[:100]}...")
                print()
                
                if result.latex_content:
                    query_parts.append(" ".join(result.latex_content))
                if result.likely_topic:
                    query_parts.append(result.likely_topic)
                if result.key_features:
                    query_parts.append(" ".join(result.key_features[:3]))
                if result.query:
                    query_parts.append(result.query)
                combined = "\n".join(part for part in query_parts if part).strip()
                query_text = combined or result.query or text
            except Exception as exc:
                print(f"[Coach] Warning: image preprocessing failed ({exc}); proceeding with text only.")

        if self.in_bot_session:
            return self._handle_bot_turn(query_text)
        return self._handle_coach_turn(query_text)

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
            # Reset flag immediately after using it to prevent race condition
            self.returning_from_session = False

        normalized = user_input.strip().lower()
        latest_request = user_input.strip()
        if self._contains_syllabus_keyword(normalized):
            if not self.syllabus_request_active:
                self.syllabus_clarification_count = 0
            self.syllabus_request_active = True
            self.collected_params.setdefault("topic", SYLLABUS_FAQ_TOPIC)
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
                self.awaiting_confirmation_prompted = False
                self.plan_confirmation_summary = None
                self._reset_syllabus_escalation()
                return self._coach_reply("No problem. What would you like to work on instead?")

            if not self.awaiting_confirmation_prompted or not self.plan_confirmation_summary:
                summary_message = self._present_plan_for_confirmation()
            else:
                summary_message = self.plan_confirmation_summary
            return self._coach_reply(
                summary_message
                or "I have a plan ready - say 'yes' to start or 'no' if you'd like to adjust it."
            )

        if not self.awaiting_confirmation:
            faq_topic = self._detect_faq_topic(normalized)
            if faq_topic:
                self.collected_params.setdefault("topic", faq_topic)
                self.collected_params["student_request"] = user_input

            # Check for session history questions
            if self._is_session_history_question(normalized):
                return self._handle_session_history_question()

        # Track actions to prevent infinite loops
        MAX_ITERATIONS = 5
        attempted_actions = []
        
        for iteration in range(MAX_ITERATIONS):
            coach_directive = self._run_coach_brain()
            action = coach_directive.get("action", "none")
            message = (coach_directive.get("message_to_student") or "").strip()
            tool_params = coach_directive.get("tool_params") or {}
            
            # Track this action attempt
            attempted_actions.append(action)
            
            # ensure student_request reflects latest utterance unless planner override provided
            if latest_request:
                tool_params["student_request"] = latest_request
            elif "student_request" not in tool_params:
                last_message = self._last_student_message()
                if last_message:
                    tool_params["student_request"] = last_message

            self._update_collected_params(tool_params)

            if self.awaiting_confirmation:
                summary_message = self.plan_confirmation_summary or self._present_plan_for_confirmation()
                fallback = "I have a plan ready - say 'yes' to start or 'no' if you'd like to adjust it."
                return self._coach_reply(summary_message or fallback)

            if action == "call_tutoring_planner":
                self.planner_result = self._call_tutoring_planner(tool_params)
                status = self.planner_result.get("status")
                if status == "need_info":
                    msg = self.planner_result.get("message") or "Could you clarify that a bit more?"
                    return self._coach_reply(msg)
                if status == "complete":
                    # Check if using legacy flow (with confirmation)
                    use_legacy = os.getenv("USE_LEGACY_PLANNER", "").lower() in {"1", "true", "yes"}
                    if not use_legacy:
                        # Go straight to tutor - no confirmation needed
                        return self._begin_bot_session(
                            bot_type="tutor",
                            tool_params=tool_params,
                            conversation_summary="Starting tutoring session.",
                        )
                    # Legacy flow: present plan for confirmation
                    summary_message = self._present_plan_for_confirmation()
                    return self._coach_reply(
                        summary_message
                        or "I have a plan ready - say 'yes' to start or 'no' if you'd like to adjust it."
                    )
                # Unexpected status - log and break to prevent infinite loop
                print(f"[Coach] Warning: Tutoring planner returned unexpected status '{status}'. Attempted actions: {attempted_actions}")
                return self._coach_reply("I'm having trouble processing that request. Could you try rephrasing?")

            if action == "call_faq_planner":
                self.planner_result = self._call_faq_planner(tool_params)
                status = self.planner_result.get("status")
                if status == "need_info":
                    msg = self.planner_result.get("message") or "Which FAQ topic should we cover?"
                    return self._coach_reply(msg)
                if status == "complete":
                    summary_message = self._present_plan_for_confirmation()
                    return self._coach_reply(
                        summary_message
                        or "I have a plan ready - say 'yes' to start or 'no' if you'd like to adjust it."
                    )
                # Unexpected status - log and break to prevent infinite loop
                print(f"[Coach] Warning: FAQ planner returned unexpected status '{status}'. Attempted actions: {attempted_actions}")
                return self._coach_reply("I'm having trouble processing that request. Could you try rephrasing?")

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

            if action == "show_proficiency":
                return self._coach_reply(self._format_proficiency_report())

            # action == "none"
            forced_summary = self._maybe_force_syllabus_plan(latest_request)
            if forced_summary:
                return self._coach_reply(forced_summary)
            if message:
                return self._coach_reply(message)
            # No action needed and no message - terminate loop
            return ""
        
        # Loop exhausted - log and return error message
        print(f"[Coach] Error: Maximum iterations ({MAX_ITERATIONS}) reached. Attempted actions: {attempted_actions}")
        return self._coach_reply("I'm having trouble processing your request. Could you try rephrasing or asking something else?")

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
        # Note: returning_from_session flag is reset in _handle_coach_turn() after use
        # to prevent race condition when this method is called multiple times in a loop
        if not self.llm_client or not self.llm_model:
            raise RuntimeError("LLM client is not available; coach flow requires it.")

        # Call LLM with retry logic and error handling
        MAX_RETRIES = 3
        RETRY_DELAY_BASE = 1.0  # Base delay in seconds for exponential backoff
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": COACH_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, indent=2)},
                    ],
                    timeout=30.0,  # 30 second timeout
                )
                break  # Success, exit retry loop
                
            except RateLimitError as e:
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff: wait longer for each retry
                    wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"[Coach] Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt failed
                    print(f"[Coach] Rate limit error after {MAX_RETRIES} attempts: {e}")
                    return {
                        "action": "none",
                        "message_to_student": "I'm a bit busy right now. Could you try again in a moment?",
                        "tool_params": {},
                        "conversation_summary": None,
                    }
                    
            except APIConnectionError as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"[Coach] Connection error (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[Coach] Connection error after {MAX_RETRIES} attempts: {e}")
                    return {
                        "action": "none",
                        "message_to_student": "I'm having connection issues. Please try again.",
                        "tool_params": {},
                        "conversation_summary": None,
                    }
                    
            except APITimeoutError as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"[Coach] Timeout error (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[Coach] Timeout error after {MAX_RETRIES} attempts: {e}")
                    return {
                        "action": "none",
                        "message_to_student": "The request took too long. Could you try again?",
                        "tool_params": {},
                        "conversation_summary": None,
                    }
                    
            except APIError as e:
                # Other API errors (500, 503, etc.) - retry for transient errors
                status_code = getattr(e, "status_code", None)
                if status_code and status_code >= 500 and attempt < MAX_RETRIES - 1:
                    # Server errors (5xx) are retryable
                    wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"[Coach] API error {status_code} (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Client errors (4xx) or final attempt - don't retry
                    print(f"[Coach] API error: {e}")
                    return {
                        "action": "none",
                        "message_to_student": "I encountered an error. Could you rephrase your request?",
                        "tool_params": {},
                        "conversation_summary": None,
                    }
                    
            except Exception as e:
                # Catch-all for unexpected errors
                print(f"[Coach] Unexpected error during LLM call: {e}")
                return {
                    "action": "none",
                    "message_to_student": "Something went wrong. Please try again.",
                    "tool_params": {},
                    "conversation_summary": None,
                }
        else:
            # This should never happen, but safety net
            print(f"[Coach] Failed to get LLM response after {MAX_RETRIES} attempts")
            return {
                "action": "none",
                "message_to_student": "I'm having trouble connecting right now. Could you try again?",
                "tool_params": {},
                "conversation_summary": None,
            }
        
        # Safely parse JSON response with fallback
        try:
            content = response.choices[0].message.content
            if not content:
                print("[Coach] Warning: LLM returned empty response")
                return {"action": "none", "message_to_student": "I didn't receive a response. Could you try again?", "tool_params": {}}
            return coerce_json(content)
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            print(f"[Coach] Error: Failed to parse LLM response: {e}")
            # Return safe fallback directive
            return {
                "action": "none",
                "message_to_student": "I'm having trouble processing that. Could you rephrase your request?",
                "tool_params": {},
                "conversation_summary": None,
            }

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
        """
        Call tutoring planner - uses simplified flow by default.
        Set USE_LEGACY_PLANNER=1 to use the old planner with confirmation.
        """
        # Check if we should use legacy planner (with confirmation)
        use_legacy = os.getenv("USE_LEGACY_PLANNER", "").lower() in {"1", "true", "yes"}
        
        if not use_legacy:
            return self._call_simplified_tutoring_planner(params)
        
        # Legacy flow with confirmation
        print("\n🔧 Tutoring Session Planner (legacy)")
        planner_params = dict(params)
        inferred_subject = self._infer_subject(planner_params)
        subject = planner_params.get("subject")
        if inferred_subject and (not subject or subject.lower() != inferred_subject):
            planner_params["subject"] = inferred_subject

        if not planner_params.get("student_request"):
            planner_params["student_request"] = self._last_student_message() or "Student requested tutoring."

        print(json.dumps(planner_params, indent=2))
        result = self.tutoring_planner.create_plan(
            {
                **planner_params,
                "student_profile": self.student_profile,
            }
        )
        print("📩 Planner response:")
        print(json.dumps(result, indent=2))
        if result.get("status") == "complete":
            self.awaiting_confirmation = True
            self.pending_session_type = "tutor"
            self.awaiting_confirmation_prompted = False
            self.plan_confirmation_summary = None
        return result
    
    def _call_simplified_tutoring_planner(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        New simplified planner flow:
        1. Calls retriever with forked image pipeline (OCR text + CLIP image in parallel)
        2. Generates simplified plan (1 primary + 2 dependent LOs, 1 future)
        3. NO confirmation - goes straight to tutor
        
        This is the new flow per meeting requirements.
        """
        print("\n🔧 Tutoring Session Planner (simplified - no confirmation)")
        
        student_request = params.get("student_request") or self._last_student_message() or ""
        mode = params.get("mode") or "conceptual_review"
        
        # Build combined query from text + OCR
        query_parts = [student_request]
        if self.current_image_query:
            query_parts.append(self.current_image_query)
        combined_query = " ".join(p for p in query_parts if p).strip()
        
        if not combined_query:
            return {
                "status": "need_info",
                "plan": None,
                "message": "What topic would you like to learn about?",
            }
        
        print(f"  Query: {combined_query[:100]}...")
        if self.current_image:
            print(f"  Image: {self.current_image}")
        
        # Call simplified planner (does forked retrieval internally)
        result = self.tutoring_planner.create_simplified_plan(
            student_request=combined_query,
            mode=mode,
            student_profile=self.student_profile,
            image_path=self.current_image,  # For CLIP image retrieval
        )
        
        print("📩 Simplified Planner response:")
        print(json.dumps(result, indent=2, default=str))
        
        # No confirmation needed - mark ready for immediate handoff
        if result.get("status") == "complete":
            self.pending_session_type = "tutor"
            # Skip confirmation - coach will start tutor immediately
            self.awaiting_confirmation = False
        
        return result

    def _call_faq_planner(self, params: Dict[str, Any]) -> Dict[str, Any]:
        print("\n🔧 FAQ/Syllabus Planner")
        planner_params = dict(params)
        if not planner_params.get("student_request"):
            planner_params["student_request"] = self._last_student_message() or "Student requested FAQ help."
        print(json.dumps(planner_params, indent=2))
        result = self.faq_planner.create_plan(planner_params)
        print("📩 Planner response:")
        print(json.dumps(result, indent=2))
        if result.get("status") == "complete":
            self.awaiting_confirmation = True
            self.pending_session_type = "faq"
            self.awaiting_confirmation_prompted = False
            self.plan_confirmation_summary = None
            self._reset_syllabus_escalation()
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
        if self.current_image_query and "image_query" not in session_params:
            session_params["image_query"] = self.current_image_query

        context = create_handoff_context(
            from_agent="coach",
            to_agent=bot_type,
            session_params=session_params,
            conversation_summary=conversation_summary,
            session_memory=self.session_memory,
            student_state=self.student_profile,
            image=self.current_image,
        )

        self.in_bot_session = True
        self.bot_type = bot_type
        self.bot_handoff_context = context
        self.bot_conversation_history = []
        self.awaiting_confirmation = False
        self.collected_params = {}
        self.planner_result = None
        self.pending_session_type = None
        self.awaiting_confirmation_prompted = False
        self.plan_confirmation_summary = None
        self._reset_syllabus_escalation()

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
            # Pass image to tutor for native multimodal understanding (GPT-4o vision)
            # Image persists across all turns in the session until session ends
            bot_response = tutor_bot(
                llm_client=self.llm_client,
                llm_model=self.llm_model,
                handoff_context=self.bot_handoff_context,
                conversation_history=history,
                image=self.current_image,
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

        # Update per-LO proficiency from tutor session summary
        if session_type == "tutor":
            self._update_lo_mastery(params, summary)
            self.session_memory.save()

        switch_topic = summary.get("switch_topic_request")
        switch_mode = summary.get("switch_mode_request")

        self._reset_after_session()

        if switch_topic:
            self.returning_from_session = True
            return self._handle_coach_turn(switch_topic, synthetic=True)

        if switch_mode:
            self.returning_from_session = True
            return self._handle_coach_turn(switch_mode, synthetic=True)

        greeting = self._build_return_greeting(params, session_type)
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
        self.awaiting_confirmation_prompted = False
        self.plan_confirmation_summary = None
        self._reset_syllabus_escalation()
        self.current_image = None
        self.current_image_query = None

    def _update_lo_mastery(self, params: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """
        Map the tutor's student_understanding assessment to a numeric mastery score and
        store it in student_profile["lo_mastery"] keyed by learning_objective.
        """
        understanding = summary.get("student_understanding") or ""
        lo_key = params.get("learning_objective") or params.get("lo_id")
        if not lo_key:
            return
        score = self.UNDERSTANDING_TO_MASTERY.get(understanding.lower(), 0.4)
        self.student_profile.setdefault("lo_mastery", {})[lo_key] = score

    def _format_proficiency_report(self) -> str:
        """
        Format lo_mastery scores into a readable summary for the student.
        
        Returns:
            Formatted string showing proficiency levels for each learning objective.
        """
        lo_mastery = self.student_profile.get("lo_mastery", {})
        
        if not lo_mastery:
            return (
                "You haven't completed any tutoring sessions yet, so I don't have proficiency data. "
                "Start a tutoring session and I'll track your progress!"
            )
        
        # Score to label mapping (reverse of UNDERSTANDING_TO_MASTERY)
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
        
        # Sort by score descending
        sorted_items = sorted(lo_mastery.items(), key=lambda x: -x[1])
        
        # Separate into strong and needs work
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
        
        # Add suggestion for lowest score
        if needs_work:
            lowest_lo = needs_work[-1][0]
            lines.append("")
            lines.append(f"Tip: Consider practicing '{lowest_lo}' next to strengthen your foundation.")
        
        return "\n".join(lines)

    def _build_return_greeting(self, params: Dict[str, Any], session_type: str) -> str:
        """
        Produce a continuity-aware greeting referencing the just-completed session.
        Falls back to generic greeting if required info is missing.
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
        return COACH_GREETING

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

    def _infer_subject(self, params: Dict[str, Any]) -> Optional[str]:
        phrases = " ".join(
            filter(
                None,
                [
                    params.get("subject"),
                    params.get("learning_objective"),
                    params.get("student_request"),
                    self._last_student_message(),
                ],
            )
        ).lower()
        for subject, keywords in SUBJECT_KEYWORDS.items():
            if any(keyword in phrases for keyword in keywords):
                return subject
        return None

    def _present_plan_for_confirmation(self) -> Optional[str]:
        plan = (self.planner_result or {}).get("plan")
        if not plan:
            return None
        topic = plan.get("topic")
        if topic:
            summary = [
                "Here's what I found:",
                f"- Topic: {topic}",
            ]
            script = plan.get("script")
            if script:
                summary.append(f"- Plan details: {script}")
            if topic == SYLLABUS_FAQ_TOPIC:
                summary.append("- We'll recap the major course concepts from your syllabus.")
            summary.append("Would you like me to walk through this info now?")
        else:
            subject = plan.get("subject", "this subject")
            learning_objective = plan.get("learning_objective", "this topic")
            mode = plan.get("mode", "").replace("_", " ")
            book = plan.get("book")
            session_guidance = plan.get("session_guidance")
            summary = [
                "Here's what I put together:",
                f"- Subject: {subject}",
                f"- Learning objective: {learning_objective}",
            ]
            if book:
                summary.append(f"- Book/unit: {book}")
            if mode:
                summary.append(f"- Mode: {mode}")
            if session_guidance:
                summary.append(f"- Approach: {session_guidance}")
            summary.append("Would you like to start with this plan?")
        summary_text = "\n".join(summary)
        self.awaiting_confirmation_prompted = True
        self.plan_confirmation_summary = summary_text
        return summary_text

    def _is_session_history_question(self, normalized_input: str) -> bool:
        """
        Detect if the student is asking about past sessions using regex with word boundaries.
        Uses word boundaries to avoid false positives from substring matching.
        
        Args:
            normalized_input: Lowercase, stripped user input.
            
        Returns:
            True if the input appears to be asking about session history.
        """
        patterns = [
            r'\bwhat\s+did\s+we\s+cover\b',
            r'\bwhat\s+did\s+we\s+do\b',
            r'\bwhat\s+did\s+we\s+learn\b',
            r'\blast\s+session\b',
            r'\bprevious\s+session\b',
            r'\bpast\s+session\b',
            r'\bwhat\s+was\s+the\s+last\b',
            r'\bwhat\s+were\s+we\s+working\s+on\b',
            r'\bwhere\s+did\s+we\s+leave\s+off\b',
            r'\bwhat\s+did\s+we\s+study\b',
        ]
        
        return any(re.search(pattern, normalized_input, re.IGNORECASE) for pattern in patterns)
    
    def _handle_session_history_question(self) -> str:
        """
        Handle questions about session history by looking up the last session
        and asking if the student wants to continue.
        
        Returns:
            Formatted message with last session info and continuation prompt.
        """
        last_session = self.session_memory.last_tutoring_session()
        recent_sessions = self.session_memory.get_recent_sessions()
        
        if not last_session and not recent_sessions:
            return self._coach_reply(
                "You haven't completed any sessions yet. Would you like to start a new tutoring session?"
            )
        
        # Build response about last session
        if last_session:
            params = last_session.get("params", {})
            summary = last_session.get("summary", {})
            
            lo = params.get("learning_objective", "a topic")
            mode = params.get("mode", "").replace("_", " ")
            subject = params.get("subject", "")
            understanding = summary.get("student_understanding", "")
            
            # Build natural language description
            parts = [f"Last time we worked on {lo}"]
            if subject:
                parts.append(f"in {subject}")
            if mode:
                parts.append(f"using {mode} mode")
            
            description = " ".join(parts) + "."
            
            # Add understanding assessment if available
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
        return any(keyword in normalized_input for keyword in SYLLABUS_KEYWORDS)

    def _reset_syllabus_escalation(self) -> None:
        self.syllabus_request_active = False
        self.syllabus_clarification_count = 0

    def _maybe_force_syllabus_plan(self, latest_request: str) -> Optional[str]:
        if not self.syllabus_request_active:
            return None
        self.syllabus_clarification_count += 1
        if self.syllabus_clarification_count < SYLLABUS_CLARIFICATION_LIMIT:
            return None

        params = {
            "topic": SYLLABUS_FAQ_TOPIC,
            "student_request": latest_request
            or self._last_student_message()
            or "Student asked about the syllabus topics.",
        }
        self.planner_result = self._call_faq_planner(params)
        self._reset_syllabus_escalation()
        status = self.planner_result.get("status")
        if status == "need_info":
            return self.planner_result.get("message") or "Could you clarify which part of the syllabus you need?"
        summary_message = self._present_plan_for_confirmation()
        return summary_message or "I put together a syllabus overview—would you like to go through it?"

