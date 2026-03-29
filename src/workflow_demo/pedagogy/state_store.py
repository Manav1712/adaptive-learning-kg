"""
In-memory learner state storage for pedagogy layer sessions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import LearnerState


class LearnerStateStore:
    """Simple in-memory state registry keyed by tutoring session_id."""

    def __init__(self) -> None:
        self._states: Dict[str, LearnerState] = {}

    @staticmethod
    def _require_session_id(session_id: str) -> str:
        key = (session_id or "").strip()
        if not key:
            raise ValueError("session_id is required")
        return key

    def get(self, session_id: str) -> Optional[LearnerState]:
        """Return learner state for session_id, if present."""
        return self._states.get(self._require_session_id(session_id))

    def set(self, session_id: str, state: LearnerState) -> LearnerState:
        """Set learner state for session_id and return a stored deep copy."""
        key = self._require_session_id(session_id)
        normalized = state.model_copy(deep=True)
        if not normalized.active_session_id:
            normalized.active_session_id = key
        self._states[key] = normalized
        return normalized

    def ensure(self, session_id: str) -> LearnerState:
        """
        Ensure learner state exists for session_id and return it.

        Initializes a blank state when one does not exist yet.
        """
        key = self._require_session_id(session_id)
        existing = self._states.get(key)
        if existing is not None:
            return existing
        created = LearnerState(active_session_id=key)
        self._states[key] = created
        return created

    def update(self, session_id: str, **patch: object) -> LearnerState:
        """
        Update fields on a learner state and return the validated result.

        Creates a default state first when needed.
        """
        current = self.ensure(session_id)
        merged = current.model_dump()
        merged.update(patch)
        updated = LearnerState.model_validate(merged)
        return self.set(session_id, updated)

    def dump_all_json(self) -> Dict[str, Any]:
        """JSON-serializable snapshot of all stored learner states (debug/CLI)."""
        return {k: v.model_dump(mode="json") for k, v in self._states.items()}
