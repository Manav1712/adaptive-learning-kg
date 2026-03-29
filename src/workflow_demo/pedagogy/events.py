"""
Canonical runtime event type strings for pedagogy-layer instrumentation.

Emit through CoachAgent.emit_event / emit_runtime_event using these values
as event_type when the layer is wired in later phases.
"""

from __future__ import annotations

from enum import Enum


class PedagogyRuntimeEvent(str, Enum):
    """Stable identifiers for pedagogy-related UI/log events."""

    LEARNER_STATE_INITIALIZED = "pedagogy_learner_state_initialized"
    LEARNER_STATE_UPDATED = "pedagogy_learner_state_updated"
    MISCONCEPTION_DIAGNOSED = "pedagogy_misconception_diagnosed"
    DIAGNOSIS_COMPLETED = "pedagogy_diagnosis_completed"
    MOVES_GENERATED = "pedagogy_moves_generated"
    POLICY_SCORED = "pedagogy_policy_scored"
    RETRIEVAL_WRAPPER_INVOKED = "pedagogy_retrieval_wrapper_invoked"
    CRITIC_COMPLETED = "pedagogy_critic_completed"
    CONTEXT_SNAPSHOT = "pedagogy_context_snapshot"
