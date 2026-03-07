"""
Shared runtime factory for CLI and web demo entrypoints.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

from .coach_agent import CoachAgent
from .demo_profiles import get_active_profile, get_profile_name
from .retriever import TeachingPackRetriever
from .runtime_events import RuntimeEventCallback, emit_runtime_event


_SHARED_RETRIEVER: Optional[TeachingPackRetriever] = None
_SHARED_RETRIEVER_LOCK = Lock()


def get_shared_retriever(
    event_callback: Optional[RuntimeEventCallback] = None,
) -> TeachingPackRetriever:
    """
    Return a process-wide retriever so resets stay cheap.

    Inputs:
        event_callback: Optional runtime event sink used during initialization.

    Outputs:
        Shared `TeachingPackRetriever` instance.
    """

    global _SHARED_RETRIEVER

    with _SHARED_RETRIEVER_LOCK:
        if _SHARED_RETRIEVER is not None:
            emit_runtime_event(
                event_callback,
                "retriever_init_completed",
                "Using cached retriever.",
                phase="startup",
                cached=True,
            )
            return _SHARED_RETRIEVER

        emit_runtime_event(
            event_callback,
            "retriever_init_started",
            "Initializing retriever.",
            phase="startup",
        )
        try:
            _SHARED_RETRIEVER = TeachingPackRetriever()
        except Exception as exc:
            emit_runtime_event(
                event_callback,
                "turn_failed",
                "Retriever initialization failed.",
                phase="startup",
                error=str(exc),
            )
            raise

        emit_runtime_event(
            event_callback,
            "retriever_init_completed",
            "Retriever initialized.",
            phase="startup",
            cached=False,
        )
        return _SHARED_RETRIEVER


def is_shared_retriever_ready() -> bool:
    """
    Report whether the shared retriever has already been initialized.

    Inputs:
        None.

    Outputs:
        `True` when the shared retriever is ready.
    """

    return _SHARED_RETRIEVER is not None


def build_coach_runtime(
    session_memory_path: Optional[str] = None,
    event_callback: Optional[RuntimeEventCallback] = None,
    retriever: Optional[TeachingPackRetriever] = None,
) -> CoachAgent:
    """
    Build one configured `CoachAgent` for the CLI or web UI.

    Inputs:
        session_memory_path: Optional persistence path for session memory.
        event_callback: Optional runtime event sink.
        retriever: Optional prebuilt retriever. When omitted, the shared retriever
            is reused across sessions.

    Outputs:
        Ready-to-use `CoachAgent` instance.
    """

    emit_runtime_event(
        event_callback,
        "app_boot_started",
        "Starting coach runtime.",
        phase="startup",
    )

    coach = CoachAgent(
        retriever=retriever or get_shared_retriever(event_callback),
        session_memory_path=session_memory_path,
        event_callback=event_callback,
    )

    profile = get_active_profile()
    coach.student_profile.update(profile)
    coach.session_memory.student_profile.update(profile)
    coach.session_memory.save()

    emit_runtime_event(
        event_callback,
        "app_boot_completed",
        "Coach runtime ready.",
        phase="startup",
        profile_name=get_profile_name(),
    )
    return coach
