from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar

from pydantic import BaseModel, Field

from .events import PedagogyRuntimeEvent
from .models import AttemptRecord, HintEvent, LearnerState, MisconceptionDiagnosis
from .state_store import LearnerStateStore


PedagogyEventEmitter = Callable[[str, str, Dict[str, Any]], None]

MAX_RECENT_ATTEMPTS = 32
MAX_HINT_EVENTS = 64
MAX_RESPONSE_EXCERPT_CHARS = 1000
DEFAULT_CONFIDENCE = 0.5

_T = TypeVar("_T")


class LearnerStateSnapshot(BaseModel):
    current_focus_lo: Optional[str] = None
    recent_attempt_count: int = 0
    hint_count: int = 0
    top_mastery: list[tuple[str, float]] = Field(default_factory=list)
    recent_misconceptions: dict[str, list[str]] = Field(default_factory=dict)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_lo_ref(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bounded_float_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for raw_key, raw_value in raw.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= value <= 1.0:
            key = _normalize_lo_ref(raw_key)
            if key:
                out[key] = value
    return out


def _clip_text(text: Optional[str], limit: int) -> str:
    return (text or "").strip()[:limit]


def _append_bounded(items: list[_T], item: _T, limit: int) -> list[_T]:
    return [*items, item][-limit:]


class LearnerStateEngine:
    """
    Phase-4 learner-state service.

    Responsibilities:
    - seed session state from durable student_profile
    - persist turn observations
    - persist misconception evidence
    - persist structured hint events
    - expose a structured snapshot for downstream prompt/policy code

    Non-responsibilities:
    - prompt string formatting
    - KT/BKT updates
    - policy scoring
    - retrieval routing
    """

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
        student_profile: Optional[Mapping[str, Any]],
        current_focus_lo: Optional[str] = None,
    ) -> LearnerState:
        """
        Seed learner state from a durable profile.

        Important:
        - Creates fresh state if missing.
        - If state already exists, preserves live session evidence and only fills
          missing seeded fields.
        """
        existing = self.store.get(session_id)
        profile = student_profile or {}

        seed_mastery = _coerce_bounded_float_map(profile.get("lo_mastery"))
        seed_confidence = _coerce_bounded_float_map(profile.get("confidence_seed"))

        if existing is None:
            state = LearnerState(
                active_session_id=session_id,
                current_focus_lo=_normalize_lo_ref(current_focus_lo),
                mastery=seed_mastery,
                confidence=seed_confidence,
                misconceptions={},
                recent_attempts=[],
                hint_events=[],
            )
            stored = self.store.set(session_id, state)
        else:
            merged_mastery = {**seed_mastery, **existing.mastery}
            merged_confidence = {
                **{key: DEFAULT_CONFIDENCE for key in merged_mastery},
                **seed_confidence,
                **existing.confidence,
            }
            stored = self.store.update(
                session_id,
                active_session_id=session_id,
                current_focus_lo=_normalize_lo_ref(current_focus_lo) or existing.current_focus_lo,
                mastery=merged_mastery,
                confidence=merged_confidence,
            )

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
        state = self.store.ensure(session_id)

        attempt = AttemptRecord(
            attempt_id=f"{session_id}:turn:{turn_index}:attempt:{len(state.recent_attempts) + 1}",
            turn_index=turn_index,
            lo_id=lo_id,
            is_correct=is_correct,
            student_response_excerpt=_clip_text(student_text, MAX_RESPONSE_EXCERPT_CHARS),
            latency_ms=latency_ms,
            created_at_iso=_utc_now_iso(),
        )

        updated = self.store.update(
            session_id,
            recent_attempts=_append_bounded(state.recent_attempts, attempt, MAX_RECENT_ATTEMPTS),
            current_focus_lo=state.current_focus_lo
            or (str(lo_id) if lo_id is not None else None),
        )

        self._emit(
            PedagogyRuntimeEvent.LEARNER_STATE_UPDATED.value,
            "Recorded learner attempt.",
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "attempt_count": len(updated.recent_attempts),
                "current_focus_lo": updated.current_focus_lo,
                "update_kind": "attempt",
            },
        )
        return updated

    def record_misconception(
        self,
        session_id: str,
        diagnosis: MisconceptionDiagnosis,
    ) -> LearnerState:
        state = self.store.ensure(session_id)

        target_lo = _normalize_lo_ref(diagnosis.target_lo) or state.current_focus_lo or "unknown"
        label = (diagnosis.suspected_misconception or "").strip()
        if not label:
            return state

        misconceptions = dict(state.misconceptions)
        existing = list(misconceptions.get(target_lo, []))
        if label not in existing:
            existing.append(label)
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
                "update_kind": "misconception",
            },
        )
        return updated

    def attach_hint_event(
        self,
        session_id: str,
        hint_text: str,
        *,
        hint_type: str = "other",
        target_lo: Optional[str] = None,
        turn_index: Optional[int] = None,
    ) -> LearnerState:
        state = self.store.ensure(session_id)
        excerpt = _clip_text(hint_text, 500)
        if not excerpt:
            return state

        event = HintEvent(
            hint_type=hint_type,
            target_lo=_normalize_lo_ref(target_lo) or state.current_focus_lo,
            text_excerpt=excerpt,
            turn_index=turn_index,
            created_at_iso=_utc_now_iso(),
        )

        updated = self.store.update(
            session_id,
            hint_events=_append_bounded(state.hint_events, event, MAX_HINT_EVENTS),
        )

        self._emit(
            PedagogyRuntimeEvent.LEARNER_STATE_UPDATED.value,
            "Attached hint event.",
            {
                "session_id": session_id,
                "hint_events_count": len(updated.hint_events),
                "update_kind": "hint",
            },
        )
        return updated

    def build_snapshot(self, session_id: str) -> LearnerStateSnapshot:
        state = self.store.get(session_id)
        if state is None:
            return LearnerStateSnapshot()

        top_mastery = sorted(
            state.mastery.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:3]

        recent_misconceptions = {
            lo: labels[-2:]
            for lo, labels in state.misconceptions.items()
            if labels
        }

        return LearnerStateSnapshot(
            current_focus_lo=state.current_focus_lo,
            recent_attempt_count=len(state.recent_attempts),
            hint_count=len(state.hint_events),
            top_mastery=top_mastery,
            recent_misconceptions=recent_misconceptions,
        )

    def _emit(self, event_type: str, message: str, metadata: Dict[str, Any]) -> None:
        if self._emit_event is None:
            return
        self._emit_event(event_type, message, metadata)