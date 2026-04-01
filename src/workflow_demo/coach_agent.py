"""
Multi-agent coach that routes between tutoring and FAQ sessions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .bot_sessions import BotSessionManager
from .coach_router import COACH_GREETING, CoachRouter
from .image_preprocessor import ImagePreprocessor
from .coach_llm_client import CoachLLMClient
from .planner import FAQPlanner, TutoringPlanner
from .pedagogy import (
    LearnerStateEngine,
    LearnerStateStore,
    MisconceptionDiagnoser,
    PedagogicalContext,
    RetrievalSessionSnapshot,
)
from .retriever import TeachingPackRetriever
from .runtime_events import RuntimeEventCallback, emit_runtime_event
from .session_memory import SessionMemory


class CoachAgent:
    """Orchestrator that routes between tutoring, FAQ, and proficiency tracking."""

    def __init__(
        self,
        retriever: Optional[TeachingPackRetriever] = None,
        llm_model: Optional[str] = None,
        session_memory_path: Optional[str] = None,
        event_callback: Optional[RuntimeEventCallback] = None,
    ) -> None:
        """Initialize the coach agent and wire up all dependencies.

        Args:
            retriever: Optional custom retriever (defaults to TeachingPackRetriever).
            llm_model: Optional LLM model name (defaults to gpt-4o-mini).
            session_memory_path: Optional path to persist session memory.
            event_callback: Optional runtime event sink used by the web UI.
        """
        self.retriever = retriever or TeachingPackRetriever()
        self.session_memory = SessionMemory(persistence_path=session_memory_path)
        self.event_callback = event_callback
        self.learner_state_store = LearnerStateStore()
        self.learner_state_engine = LearnerStateEngine(
            store=self.learner_state_store,
            event_emitter=self._emit_pedagogy_event,
        )

        # Coach-level state
        self.conversation_history: List[Dict[str, str]] = []
        self.collected_params: Dict[str, Any] = {}
        self.planner_result: Optional[Dict[str, Any]] = None
        self.returning_from_session = False

        # Student state
        self.student_profile: Dict[str, Any] = self.session_memory.student_profile

        # Syllabus escalation tracking
        self.syllabus_request_active = False
        self.syllabus_clarification_count = 0

        # Image state
        self.current_image: Optional[str] = None
        self.current_image_query: Optional[str] = None

        # LLM setup
        self.llm_client: Optional[OpenAI] = None
        self.llm_model: Optional[str] = None
        self._init_llm(llm_model)
        self.misconception_diagnoser = MisconceptionDiagnoser(
            llm_client=self.llm_client,
            llm_model=self.llm_model,
        )

        # Planners
        self.tutoring_planner = TutoringPlanner(self.retriever)
        self.faq_planner = FAQPlanner()

        # Image preprocessing
        self.image_preprocessor: Optional[ImagePreprocessor] = None
        try:
            self.image_preprocessor = ImagePreprocessor()
        except Exception as exc:
            print(f"[Coach] Warning: failed to initialize image preprocessor ({exc}).")

        # Wire up delegates
        self.llm_client_wrapper: Optional[CoachLLMClient] = None
        self.coach_router: Optional[CoachRouter] = None
        self.bot_session_manager: Optional[BotSessionManager] = None
        if self.llm_client and self.llm_model:
            self.llm_client_wrapper = CoachLLMClient(self.llm_client, self.llm_model)
            self.coach_router = CoachRouter(self, self.llm_client_wrapper)
            self.bot_session_manager = BotSessionManager(self)

    def _init_llm(self, llm_model: Optional[str]) -> None:
        """Check for API key in environment variables and set up LLM client and model.

        Args:
            llm_model: Optional model name override.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = OpenAI(api_key=api_key)
        self.llm_model = llm_model or os.getenv("WORKFLOW_DEMO_LLM_MODEL", "gpt-4o-mini")

    def initial_greeting(self) -> str:
        """Return the initial greeting."""
        return COACH_GREETING

    def get_pedagogy_snapshot_for_api(self) -> Optional[Dict[str, Any]]:
        """Latest tutor pedagogy snapshot for web UI (None if no active tutor session)."""
        mgr = self.bot_session_manager
        if not mgr or not mgr.is_active:
            return None
        return mgr.last_pedagogy_snapshot

    def tutor_session_active_for_api(self) -> bool:
        """True when a tutor bot session is active (for UI panel visibility)."""
        mgr = self.bot_session_manager
        return bool(mgr and mgr.is_active and mgr.bot_type == "tutor")

    def emit_event(
        self,
        event_type: str,
        message: str,
        phase: str = "system",
        **metadata: Any,
    ) -> Dict[str, Any]:
        """Emit one structured runtime event when a web sink is attached."""
        return emit_runtime_event(
            self.event_callback,
            event_type,
            message,
            phase=phase,
            **metadata,
        )

    def _emit_pedagogy_event(
        self,
        event_type: str,
        message: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Bridge pedagogy-engine events to the existing runtime event sink."""
        self.emit_event(
            event_type,
            message,
            phase="pedagogy",
            **metadata,
        )

    def initialize_learner_state_from_profile(
        self,
        session_id: str = "runtime",
        current_focus_lo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initialize centralized learner state from the current student profile.

        Returns:
            JSON-serializable learner-state payload.
        """
        state = self.learner_state_engine.initialize_from_profile(
            session_id=session_id,
            student_profile=self.student_profile,
            current_focus_lo=current_focus_lo,
        )
        return state.model_dump(mode="json")

    def ensure_tutor_learner_context(
        self,
        session_id: str,
        session_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Ensure learner state exists for one tutor session and build context payload.
        """
        current_focus_lo = (
            session_params.get("learning_objective")
            or session_params.get("title")
            or next(
                (
                    str(item.get("title"))
                    for item in session_params.get("current_plan", [])
                    if item.get("title")
                ),
                None,
            )
        )
        learner_state = self.learner_state_engine.initialize_from_profile(
            session_id=session_id,
            student_profile=self.student_profile,
            current_focus_lo=current_focus_lo,
        )
        seed_lo = (current_focus_lo or "").strip() or None
        pedagogy_context = PedagogicalContext(
            learner_state=learner_state,
            target_lo=seed_lo,
            instruction_lo=seed_lo,
            retrieval_session=RetrievalSessionSnapshot(
                pack_focus_lo=(seed_lo or ""),
            ),
        )
        return pedagogy_context.model_dump(mode="json")

    def process_turn(self, user_input: str) -> str:
        """Handle a text-only turn.

        If an image was previously submitted (self.current_image is set),
        it will persist and be passed to the tutor for follow-up questions.

        Args:
            user_input: The student's text input.

        Returns:
            The agent's response.
        """
        if self.bot_session_manager and self.bot_session_manager.is_active:
            return self._handle_bot_turn(user_input)
        return self._handle_coach_turn(user_input)

    def process_multimodal_turn(self, text: str, image: str) -> str:
        """Handle a turn that includes an image.

        Stores the image, builds a combined text query from the image analysis,
        then delegates to process_turn.

        Args:
            text: Optional text input.
            image: Path or identifier for the image.

        Returns:
            The agent's response.
        """
        query_text = self._build_image_query(text or "", image)
        return self.process_turn(query_text)

    def _build_image_query(self, text: str, image: str) -> str:
        """Preprocess an image and return a combined text+image query string.

        Sets self.current_image and self.current_image_query as side-effects
        so the planner and tutor can access them downstream.

        Args:
            text: Optional user text.
            image: Path or identifier for the image.

        Returns:
            Combined query string.
        """
        self.current_image = image
        self.current_image_query = None
        self.emit_event(
            "image_preprocess_started",
            "Analyzing image input.",
            phase="input",
            image=image,
        )

        print(f"\n🖼️  Image detected: {image}")

        extra_parts: List[str] = []
        if self.image_preprocessor:
            try:
                result = self.image_preprocessor.process_image(image, text or None)
                self.current_image_query = result.query
                self._log_image_result(result)
                self.emit_event(
                    "image_preprocess_completed",
                    "Image context extracted.",
                    phase="input",
                    image=image,
                    detected_type=result.detected_type,
                    likely_topic=result.likely_topic,
                    confidence=round(result.confidence, 3),
                )

                extra_parts = [
                    " ".join(result.latex_content) if result.latex_content else "",
                    result.likely_topic or "",
                    " ".join(result.key_features[:3]) if result.key_features else "",
                    result.query or "",
                ]
            except Exception as exc:
                print(f"[Coach] Warning: image preprocessing failed ({exc}); proceeding with text only.")
                self.emit_event(
                    "image_preprocess_completed",
                    "Image preprocessing failed; continuing with text only.",
                    phase="input",
                    image=image,
                    success=False,
                    error=str(exc),
                )

        parts = [p for p in [text] + extra_parts if p]
        return "\n".join(parts).strip() or text

    @staticmethod
    def _log_image_result(result: Any) -> None:
        """Print image analysis details for debugging.

        Args:
            result: Image analysis result from preprocessor.
        """
        print("📋 Image Analysis Results:")
        print(f"   Type: {result.detected_type}")
        print(f"   Confidence: {result.confidence:.0%}")
        if result.likely_topic:
            print(f"   Topic: {result.likely_topic}")
        if result.latex_content:
            print(f"   LaTeX: {result.latex_content[:2]}...")
        if result.key_features:
            print(f"   Features: {result.key_features[:3]}")
        print(f"   Generated Query: {result.query[:100]}...")
        print()

    # -----------------------------------------------
    # Coach mode delegation (state machines below)
    # -----------------------------------------------

    def _handle_coach_turn(self, user_input: str, synthetic: bool = False) -> str:
        """Route to coach router."""
        if not self.coach_router:
            return "Coach system not initialized."
        return self.coach_router.handle_turn(user_input, synthetic)

    def _handle_bot_turn(self, user_input: str) -> str:
        """Route to bot session manager."""
        if not self.bot_session_manager:
            return "Bot system not initialized."
        return self.bot_session_manager.handle_turn(user_input)

    # -----------------------------------------------
    # Conversation helpers (used by both router + bot manager)
    # -----------------------------------------------

    def _record_message(self, speaker: str, text: str) -> None:
        """Record a message in the conversation history.

        Args:
            speaker: "student" or "assistant".
            text: The message text.
        """
        text = (text or "").strip()
        if not text:
            return
        self.conversation_history.append({"speaker": speaker, "text": text})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _last_student_message(self) -> str:
        """Return the most recent student message from conversation history.

        Returns:
            The message text or empty string if no student message found.
        """
        for message in reversed(self.conversation_history):
            if message["speaker"] == "student":
                return message["text"]
        return ""

    def _reset_syllabus_escalation(self) -> None:
        """Clear syllabus escalation flags."""
        self.syllabus_request_active = False
        self.syllabus_clarification_count = 0

    # -----------------------------------------------
    # Planner calls (delegated from coach_router)
    # -----------------------------------------------

    def _call_tutoring_planner(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call the tutoring planner with the given parameters.

        Args:
            params: Planner parameters.

        Returns:
            Planner result dict.
        """
        import json

        print("\n🔧 Tutoring Session Planner")

        student_request = params.get("student_request") or self._last_student_message() or ""
        mode = params.get("mode") or "conceptual_review"
        self.emit_event(
            "planning_started",
            "Building tutoring plan.",
            phase="planning",
            planner="tutoring",
            mode=mode,
        )

        # Build combined query from text + OCR
        query_parts = [student_request]
        if self.current_image_query:
            query_parts.append(self.current_image_query)
        combined_query = " ".join(p for p in query_parts if p).strip()

        if not combined_query:
            result = {
                "status": "need_info",
                "plan": None,
                "message": "What topic would you like to learn about?",
            }
            self.emit_event(
                "planning_completed",
                "Tutoring plan needs more information.",
                phase="planning",
                planner="tutoring",
                status="need_info",
                mode=mode,
            )
            return result

        print(f"  Query: {combined_query[:100]}...")
        if self.current_image:
            print(f"  Image: {self.current_image}")

        result = self.tutoring_planner.create_plan(
            {
                "student_request": combined_query,
                "mode": mode,
                "student_profile": self.student_profile,
                "image_path": self.current_image,
            }
        )

        print("📩 Planner response:")
        print(json.dumps(result, indent=2, default=str))
        plan = result.get("plan") or {}
        self.emit_event(
            "planning_completed",
            "Tutoring plan finished.",
            phase="planning",
            planner="tutoring",
            status=result.get("status"),
            mode=plan.get("mode") or mode,
            subject=plan.get("subject"),
            current_plan_titles=[
                item.get("title")
                for item in plan.get("current_plan", [])
                if item.get("title")
            ],
            future_plan_titles=[
                item.get("title")
                for item in plan.get("future_plan", [])
                if item.get("title")
            ],
        )
        return result

    def _call_faq_planner(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call the FAQ planner with the given parameters.

        Args:
            params: Planner parameters.

        Returns:
            Planner result dict.
        """
        import json

        print("\n🔧 FAQ/Syllabus Planner")
        self.emit_event(
            "planning_started",
            "Routing FAQ request.",
            phase="planning",
            planner="faq",
            topic=params.get("topic"),
        )
        planner_params = dict(params)
        if not planner_params.get("student_request"):
            planner_params["student_request"] = self._last_student_message() or "Student requested FAQ help."
        print(json.dumps(planner_params, indent=2))
        result = self.faq_planner.create_plan(planner_params)
        print("📩 Planner response:")
        print(json.dumps(result, indent=2))
        plan = result.get("plan") or {}
        self.emit_event(
            "planning_completed",
            "FAQ planner finished.",
            phase="planning",
            planner="faq",
            status=result.get("status"),
            topic=plan.get("topic") or planner_params.get("topic"),
        )
        if result.get("status") == "complete":
            self._reset_syllabus_escalation()
        return result
