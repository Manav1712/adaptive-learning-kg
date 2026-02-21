"""
Session memory utility used by the notebook-style multi-agent flow.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SessionEntry:
    """
    Canonical record of a completed tutoring or FAQ session.
    """

    timestamp: str
    type: str  # "tutor" or "faq"
    params: Dict[str, Any]
    summary: Dict[str, Any]
    conversation_exchanges: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.type,
            "params": self.params,
            "summary": self.summary,
            "conversation_exchanges": self.conversation_exchanges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionEntry":
        return cls(
            timestamp=data.get("timestamp", ""),
            type=data.get("type", "tutor"),
            params=data.get("params", {}),
            summary=data.get("summary", {}),
            conversation_exchanges=data.get("conversation_exchanges", []),
        )


class SessionMemory:
    """
    Maintains a bounded list of recent sessions for continuity.
    Optionally persists to a JSON file for cross-run continuity.
    """

    def __init__(self, max_entries: int = 5, persistence_path: Optional[str] = None) -> None:
        self.max_entries = max_entries
        self.persistence_path = persistence_path
        self._entries: List[SessionEntry] = []
        self.student_profile: Dict[str, Any] = {"lo_mastery": {}}
        if persistence_path:
            self._load()

    def _load(self) -> None:
        """Load session entries from JSON file if it exists."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                sessions = data
                profile = {"lo_mastery": {}}
            elif isinstance(data, dict):
                sessions = data.get("sessions", [])
                profile = data.get("student_profile") or {"lo_mastery": {}}
                if "lo_mastery" not in profile or not isinstance(profile["lo_mastery"], dict):
                    profile["lo_mastery"] = {}
            else:
                sessions = []
                profile = {"lo_mastery": {}}

            self._entries = [SessionEntry.from_dict(e) for e in sessions[-self.max_entries:]]
            self.student_profile = profile
        except (json.JSONDecodeError, IOError) as exc:
            print(f"[SessionMemory] Warning: failed to load from {self.persistence_path}: {exc}")
            self._entries = []
            self.student_profile = {"lo_mastery": {}}

    def _save(self) -> None:
        """Persist session entries to JSON file."""
        if not self.persistence_path:
            return
        payload = {
            "sessions": [e.to_dict() for e in self._entries],
            "student_profile": self.student_profile,
        }
        try:
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except IOError as exc:
            print(f"[SessionMemory] Warning: failed to save to {self.persistence_path}: {exc}")

    def add_session(
        self,
        session_type: str,
        params: Dict[str, Any],
        summary: Dict[str, Any],
        conversation_exchanges: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        entry = SessionEntry(
            timestamp=datetime.now().isoformat(),
            type=session_type,
            params=params,
            summary=summary,
            conversation_exchanges=conversation_exchanges or [],
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        self._save()

    def get_recent_sessions(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries]

    def last_tutoring_session(self) -> Optional[Dict[str, Any]]:
        for entry in reversed(self._entries):
            if entry.type == "tutor":
                return entry.to_dict()
        return None

    def save(self) -> None:
        """Public helper for persisting the current state."""
        self._save()


def create_handoff_context(
    from_agent: str,
    to_agent: str,
    session_params: Dict[str, Any],
    conversation_summary: str,
    session_memory: SessionMemory,
    student_state: Optional[Dict[str, Any]] = None,
    image: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bundle the context passed between coach and downstream tutor/FAQ agents.
    """

    return {
        "handoff_metadata": {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "timestamp": datetime.now().isoformat(),
        },
        "session_params": session_params,
        "conversation_summary": conversation_summary,
        "recent_sessions": session_memory.get_recent_sessions(),
        "student_state": student_state or {},
        "image": image,
    }

