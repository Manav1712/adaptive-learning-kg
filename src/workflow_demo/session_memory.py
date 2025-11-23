"""
Session memory utility used by the notebook-style multi-agent flow.
"""

from __future__ import annotations

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


class SessionMemory:
    """
    Maintains a bounded list of recent sessions for continuity.
    """

    def __init__(self, max_entries: int = 5) -> None:
        self.max_entries = max_entries
        self._entries: List[SessionEntry] = []

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
            self._entries = self._entries[-self.max_entries :]

    def get_recent_sessions(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries]

    def last_tutoring_session(self) -> Optional[Dict[str, Any]]:
        for entry in reversed(self._entries):
            if entry.type == "tutor":
                return entry.to_dict()
        return None


def create_handoff_context(
    from_agent: str,
    to_agent: str,
    session_params: Dict[str, Any],
    conversation_summary: str,
    session_memory: SessionMemory,
    student_state: Optional[Dict[str, Any]] = None,
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
    }

