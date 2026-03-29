"""
Learner-state engine for tutor-session scoped updates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .events import PedagogyRuntimeEvent
from .models import AttemptRecord, LearnerState, MisconceptionDiagnosis
from .state_store import LearnerStateStore


PedagogyEventEmitter = Callable[[str, str, Dict[str, Any]], None]


class LearnerStateEngine:
    """Initializes and updates learner state during tutoring sessions."""

    def __init__(
        self,
        store: LearnerStateStore,
        event_emitter: Optional[PedagogyEventEmitter] = None,
    ) -> None:
        self.store = store
        self._emit_event = event_emitter

    def initialize_from_profile(
        self,
        session_id: str,
        student_profile: Optional[Dict[str, Any]],
        current_focus_lo: Optional[str] = None,
    ) -> LearnerState:
        """
        Initialize or refresh learner state from a profile payload.

        The profile is expected to optionally include `lo_mastery`.
        """
        profile = student_profile or {}
        lo_mastery = profile.get("lo_mastery") if isinstance(profile, dict) else {}
        mastery: Dict[str, float] = {}
        if isinstance(lo_mastery, dict):
            for raw_key, raw_value in lo_mastery.items():
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if 0.0 <= value <= 1.0:
                    mastery[str(raw_key)] = value

        current = self.store.get(session_id)
        state = LearnerState(
            active_session_id=session_id,
            current_focus_lo=current_focus_lo or (current.current_focus_lo if current else None),
            mastery=mastery,
            misconceptions=dict((current.misconceptions if current else {})),
            recent_attempts=list((current.recent_attempts if current else [])),
            hint_history=list((current.hint_history if current else [])),
        )
        stored = self.store.set(session_id, state)
        self._emit(
            PedagogyRuntimeEvent.LEARNER_STATE_INITIALIZED.value,
            "Initialized learner state.",
            {
                "session_id": session_id,
                "mastery_count": len(stored.mastery),
                "current_focus_lo": stored.current_focus_lo,
            },
        )
        return stored

    def record_turn(
        self,
        session_id: str,
        turn_index: int,
        student_text: str,
        lo_id: Optional[int] = None,
        is_correct: Optional[bool] = None,
        latency_ms: Optional[int] = None,
    ) -> LearnerState:
        """Append one attempt record and persist the updated learner state."""
        state = self.store.ensure(session_id)
        attempt = AttemptRecord(
            attempt_id=f"{session_id}:turn:{turn_index}:attempt:{len(state.recent_attempts) + 1}",
            turn_index=turn_index,
            lo_id=lo_id,
            is_correct=is_correct,
            student_response_excerpt=(student_text or "")[:4000],
            latency_ms=latency_ms,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
        )
        attempts = [*state.recent_attempts, attempt][-32:]
        updated = self.store.update(
            session_id,
            recent_attempts=attempts,
            current_focus_lo=state.current_focus_lo or (str(lo_id) if lo_id is not None else None),
        )
        self._emit(
            PedagogyRuntimeEvent.LEARNER_STATE_UPDATED.value,
            "Recorded learner attempt.",
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "attempt_count": len(updated.recent_attempts),
                "current_focus_lo": updated.current_focus_lo,
            },
        )
        return updated

    def record_misconception(
        self,
        session_id: str,
        diagnosis: MisconceptionDiagnosis,
    ) -> LearnerState:
        """
        Persist one misconception evidence label under state.misconceptions[target_lo].

        Repeated misconception labels for the same target LO are deduplicated.
        """
        state = self.store.ensure(session_id)
        target_lo = (diagnosis.target_lo or state.current_focus_lo or "unknown").strip() or "unknown"
        misconceptions = dict(state.misconceptions)
        existing = list(misconceptions.get(target_lo, []))
        entry = diagnosis.suspected_misconception.strip()
        if entry and entry not in existing:
            existing.append(entry)
        misconceptions[target_lo] = existing
        updated = self.store.update(
            session_id,
            misconceptions=misconceptions,
            current_focus_lo=state.current_focus_lo or target_lo,
        )
        self._emit(
            PedagogyRuntimeEvent.LEARNER_STATE_UPDATED.value,
            "Persisted misconception evidence.",
            {
                "session_id": session_id,
                "target_lo": target_lo,
                "misconception_count_for_lo": len(updated.misconceptions.get(target_lo, [])),
            },
        )
        return updated

    def attach_hint_event(
        self,
        session_id: str,
        hint_text: str,
    ) -> LearnerState:
        """Append one hint event to learner state history."""
        state = self.store.ensure(session_id)
        hint = (hint_text or "").strip()
        if not hint:
            return state
        hint_history = [*state.hint_history, hint][-64:]
        updated = self.store.update(session_id, hint_history=hint_history)
        self._emit(
            PedagogyRuntimeEvent.LEARNER_STATE_UPDATED.value,
            "Attached hint event.",
            {
                "session_id": session_id,
                "hint_history_count": len(updated.hint_history),
            },
        )
        return updated

    def summarize_for_prompt(self, session_id: str) -> str:
        """Return a compact learner-state summary string for optional prompts."""
        state = self.store.get(session_id)
        if not state:
            return "Learner state unavailable."
        focus = state.current_focus_lo or "none"
        attempts = len(state.recent_attempts)
        mastery_items = sorted(state.mastery.items(), key=lambda kv: -kv[1])[:3]
        mastery_text = ", ".join(f"{key}:{value:.2f}" for key, value in mastery_items) or "none"
        return (
            f"focus={focus}; attempts={attempts}; "
            f"mastery_top={mastery_text}; hints={len(state.hint_history)}"
        )

    def _emit(self, event_type: str, message: str, metadata: Dict[str, Any]) -> None:
        if self._emit_event is None:
            return
        self._emit_event(event_type, message, metadata)
