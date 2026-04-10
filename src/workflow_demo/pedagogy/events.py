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
    TEACHING_MOVES_GENERATED = "pedagogy_teaching_moves_generated"
    POLICY_DECISION_MADE = "pedagogy_policy_decision_made"
    RETRIEVAL_POLICY_DECIDED = "pedagogy_retrieval_policy_decided"
    RETRIEVAL_EXECUTED = "pedagogy_retrieval_executed"
    MATH_GUARD_CHECKED = "pedagogy_math_guard_checked"
    MATH_GUARD_REPAIRED = "pedagogy_math_guard_repaired"
    # Practice-loop / sequencing events (Round 2+)
    PRACTICE_PROBLEM_STARTED = "practice_problem_started"
    PRACTICE_PROBLEM_COMPLETED = "practice_problem_completed"
    SEQUENCER_DIFFICULTY_CHOSEN = "sequencer_difficulty_chosen"
    SEQUENCER_STATE_UPDATED = "sequencer_state_updated"

    # Legacy / reserved (not emitted by current runtime paths)
    DIAGNOSIS_COMPLETED = "pedagogy_diagnosis_completed"
    MOVES_GENERATED = "pedagogy_moves_generated"
    POLICY_SCORED = "pedagogy_policy_scored"
    RETRIEVAL_WRAPPER_INVOKED = "pedagogy_retrieval_wrapper_invoked"
    CRITIC_COMPLETED = "pedagogy_critic_completed"
    CONTEXT_SNAPSHOT = "pedagogy_context_snapshot"
