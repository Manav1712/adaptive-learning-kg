"""Unit tests for workflow_demo.pedagogy Pydantic models (Phase 0)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.workflow_demo.pedagogy.constants import RetrievalIntent, TeachingMoveType
from src.workflow_demo.pedagogy.models import (
    AttemptRecord,
    CriticVerdict,
    LearnerState,
    MisconceptionDiagnosis,
    PedagogicalContext,
    PolicyDecision,
    TeachingMoveCandidate,
)


@pytest.mark.unit
def test_attempt_record_json_round_trip():
    rec = AttemptRecord(
        attempt_id="a1",
        turn_index=0,
        lo_id=42,
        is_correct=False,
        student_response_excerpt="x + x = x^2",
        latency_ms=1200,
        created_at_iso="2026-03-27T12:00:00+00:00",
    )
    data = rec.model_dump(mode="json")
    restored = AttemptRecord.model_validate(data)
    assert restored == rec


@pytest.mark.unit
def test_attempt_record_validation_bounds():
    with pytest.raises(ValidationError):
        AttemptRecord(attempt_id="", turn_index=0)
    with pytest.raises(ValidationError):
        AttemptRecord(attempt_id="x", turn_index=-1)
    with pytest.raises(ValidationError):
        AttemptRecord(attempt_id="x", turn_index=0, lo_id=-1)
    with pytest.raises(ValidationError):
        AttemptRecord(attempt_id="x", turn_index=0, latency_ms=-1)
    with pytest.raises(ValidationError):
        AttemptRecord(attempt_id="x", turn_index=0, latency_ms=3_600_001)


@pytest.mark.unit
def test_learner_state_mastery_bounds():
    LearnerState(lo_mastery_proxy={"Limits": 0.0, "Derivatives": 1.0})
    with pytest.raises(ValidationError):
        LearnerState(lo_mastery_proxy={"Bad": 1.01})
    with pytest.raises(ValidationError):
        LearnerState(lo_mastery_proxy={"Bad": -0.01})


@pytest.mark.unit
def test_misconception_related_lo_ids_non_negative():
    # Legacy fields should still coerce into the canonical model shape.
    diag = MisconceptionDiagnosis(
        code="M1",
        label="Linearization error",
        confidence=0.7,
        related_lo_ids=[0, 10],
    )
    assert diag.suspected_misconception == "Linearization error"
    assert diag.prerequisite_gap_los == ["0", "10"]

    with pytest.raises(ValidationError):
        MisconceptionDiagnosis(
            target_lo="Derivatives",
            suspected_misconception="x",
            confidence=1.2,
            rationale="bad confidence",
        )


@pytest.mark.unit
def test_teaching_move_invalid_enum():
    with pytest.raises(ValidationError):
        TeachingMoveCandidate.model_validate(
            {
                "move_type": "not_a_canonical_move",
                "priority_score": 0.5,
            }
        )


@pytest.mark.unit
def test_retrieval_intent_invalid_in_list():
    with pytest.raises(ValidationError):
        TeachingMoveCandidate.model_validate(
            {
                "move_type": "explain_concept",
                "priority_score": 0.9,
                "retrieval_intents": ["not_an_intent"],
            }
        )


@pytest.mark.unit
def test_pedagogical_context_nested_round_trip():
    attempt = AttemptRecord(attempt_id="t0", turn_index=1, is_correct=True)
    learner = LearnerState(
        session_id="sess-1",
        active_lo_id=7,
        lo_mastery_proxy={"LO7": 0.55},
        recent_attempts=[attempt],
    )
    diagnosis = MisconceptionDiagnosis(
        target_lo="LO7",
        suspected_misconception="Dropped negative when squaring",
        confidence=0.62,
        rationale="Student sign handling appears inconsistent.",
        evidence_quotes=['Student wrote "(−x)^2 = −x^2"'],
        prerequisite_gap_los=["Sign rules"],
    )
    chosen = TeachingMoveCandidate(
        move_type=TeachingMoveType.CORRECTIVE_FEEDBACK,
        priority_score=0.88,
        rationale="Address sign error before proceeding.",
        retrieval_intents=[
            RetrievalIntent.COUNTER_EXAMPLE,
            RetrievalIntent.DEFINITION_SNIPPET,
        ],
        metadata={"focus": "sign_rules"},
    )
    alt = TeachingMoveCandidate(
        move_type=TeachingMoveType.WORKED_EXAMPLE,
        priority_score=0.4,
    )
    policy = PolicyDecision(
        chosen=chosen,
        alternatives=[alt],
        policy_version="0",
        trace_notes="test-trace",
    )
    critic = CriticVerdict(
        approved=True,
        severity="none",
        violations=[],
        revision_hint=None,
        confidence=0.95,
    )
    ctx = PedagogicalContext(
        layer_version="0",
        learner_state=learner,
        diagnosis=diagnosis,
        policy=policy,
        last_critic=critic,
        active_move=chosen,
        extensions={"note": "integration test payload"},
    )

    raw_json = json.dumps(ctx.model_dump(mode="json"))
    loaded = PedagogicalContext.model_validate_json(raw_json)
    assert loaded == ctx


@pytest.mark.unit
def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        AttemptRecord.model_validate(
            {
                "attempt_id": "x",
                "turn_index": 0,
                "unknown_field": 123,
            }
        )


@pytest.mark.unit
def test_critic_verdict_bounds():
    with pytest.raises(ValidationError):
        CriticVerdict(approved=True, confidence=1.5)
