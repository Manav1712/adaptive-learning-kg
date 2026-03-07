"""
Runtime event helpers used by the demo runtime and web bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4


RuntimeEventCallback = Callable[[Dict[str, Any]], None]


def _utc_now_iso() -> str:
    """Return a stable UTC timestamp string for event ordering."""
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert common Python objects into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass
class RuntimeEvent:
    """
    Structured runtime event surfaced to the UI or logs.

    Inputs:
        event_type: Stable event identifier.
        message: Human-readable status text.
        phase: Broad lifecycle phase.
        metadata: Optional JSON-safe structured context.

    Outputs:
        Serialized event payload via `to_dict()`.
    """

    event_type: str
    message: str
    phase: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event for transport or storage."""
        return {
            "id": self.event_id,
            "type": self.event_type,
            "phase": self.phase,
            "message": self.message,
            "created_at": self.created_at,
            "metadata": _json_safe(self.metadata),
        }


def emit_runtime_event(
    callback: Optional[RuntimeEventCallback],
    event_type: str,
    message: str,
    phase: str = "system",
    **metadata: Any,
) -> Dict[str, Any]:
    """
    Create and emit one runtime event through the given callback.

    Inputs:
        callback: Optional consumer for the serialized event.
        event_type: Stable event identifier.
        message: Human-readable event message.
        phase: Broad runtime phase.
        metadata: Extra structured context.

    Outputs:
        Serialized event dictionary.
    """

    event = RuntimeEvent(
        event_type=event_type,
        message=message,
        phase=phase,
        metadata=_json_safe(metadata),
    ).to_dict()
    if callback:
        callback(event)
    return event


class RuntimeEventCollector:
    """
    Thread-safe collector for runtime events emitted during a turn.

    Inputs:
        on_emit: Optional callback to fan out each event.
        max_events: Maximum retained event count.

    Outputs:
        Event snapshots via `snapshot()`.
    """

    def __init__(
        self,
        on_emit: Optional[RuntimeEventCallback] = None,
        max_events: int = 100,
    ) -> None:
        self._on_emit = on_emit
        self._max_events = max_events
        self._events: List[Dict[str, Any]] = []
        self._lock = Lock()

    def callback(self, event: Dict[str, Any]) -> None:
        """Record one serialized event and forward it if configured."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]
        if self._on_emit:
            self._on_emit(event)

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a copy of the collected events."""
        with self._lock:
            return list(self._events)
