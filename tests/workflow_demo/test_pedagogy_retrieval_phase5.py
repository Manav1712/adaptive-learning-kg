"""Phase 5 pedagogical retrieval policy (intent, action, execution_mode, material triggers)."""

import pytest

from src.workflow_demo.pedagogy.constants import (
    PedagogicalRetrievalIntent,
    RetrievalExecutionMode,
    TeachingMoveType,
)
from src.workflow_demo.pedagogy.models import (
    MisconceptionDiagnosis,
    PolicyDecision,
    RetrievalSessionSnapshot,
    TeachingMoveCandidate,
)
from src.workflow_demo.pedagogy.retrieval_policy import (
    RetrievalPolicyAction,
    decide_pedagogical_retrieval_intent,
    decide_retrieval_action,
    diagnosis_fingerprint,
    map_action_to_execution_mode,
)


@pytest.mark.unit
def test_diagnosis_fingerprint_ignores_prereq_order():
    a = MisconceptionDiagnosis(
        target_lo="X",
        suspected_misconception="m",
        confidence=0.5,
        prerequisite_gap_los=["b", "a"],
    )
    b = MisconceptionDiagnosis(
        target_lo="X",
        suspected_misconception="m",
        confidence=0.9,
        prerequisite_gap_los=["a", "b"],
    )
    assert diagnosis_fingerprint(a) == diagnosis_fingerprint(b)


@pytest.mark.unit
def test_map_action_to_execution_mode_v1():
    assert map_action_to_execution_mode(RetrievalPolicyAction.REUSE_PACK) == RetrievalExecutionMode.NO_IO
    assert map_action_to_execution_mode(
        RetrievalPolicyAction.AUGMENT_PACK
    ) == RetrievalExecutionMode.CONSTRAINED_REFRESH
    assert map_action_to_execution_mode(
        RetrievalPolicyAction.REFRESH_PACK
    ) == RetrievalExecutionMode.FULL_REFRESH


@pytest.mark.unit
def test_decide_pedagogical_retrieval_intent_worked_example():
    d = MisconceptionDiagnosis(
        target_lo="T",
        suspected_misconception="x",
        confidence=0.5,
    )
    intent = decide_pedagogical_retrieval_intent(
        session_target_lo="T",
        instruction_lo="T",
        diagnosis=d,
        move_type=TeachingMoveType.WORKED_EXAMPLE,
    )
    assert intent == PedagogicalRetrievalIntent.RETRIEVE_WORKED_EXAMPLE


@pytest.mark.unit
def test_decide_retrieval_action_reuse_when_no_material_triggers():
    pack = {"key_points": ["k"], "examples": [], "practice": []}
    d = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="ok",
        confidence=0.6,
    )
    prior = RetrievalSessionSnapshot(
        pack_focus_lo="Derivatives",
        last_diagnosis_fingerprint=diagnosis_fingerprint(d),
    )
    intent = PedagogicalRetrievalIntent.TEACH_CURRENT_CONCEPT
    action, triggers, _ = decide_retrieval_action(
        session_target_lo="Derivatives",
        instruction_lo="Derivatives",
        prior_session_target_lo="Derivatives",
        prior_instruction_lo="Derivatives",
        pedagogical_intent=intent,
        diagnosis=d,
        prior_snapshot=prior,
        teaching_pack=pack,
    )
    assert action == RetrievalPolicyAction.REUSE_PACK
    assert not triggers


@pytest.mark.unit
def test_decision_augment_when_fingerprint_changes_and_pack_nonempty():
    d1 = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="a",
        confidence=0.6,
    )
    d2 = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="b",
        confidence=0.6,
    )
    pack = {"key_points": ["k"], "examples": [], "practice": []}
    prior = RetrievalSessionSnapshot(
        pack_focus_lo="Derivatives",
        last_diagnosis_fingerprint=diagnosis_fingerprint(d1),
    )
    intent = PedagogicalRetrievalIntent.TEACH_CURRENT_CONCEPT
    action, triggers, _ = decide_retrieval_action(
        session_target_lo="Derivatives",
        instruction_lo="Derivatives",
        prior_session_target_lo="Derivatives",
        prior_instruction_lo="Derivatives",
        pedagogical_intent=intent,
        diagnosis=d2,
        prior_snapshot=prior,
        teaching_pack=pack,
    )
    assert "t4_fingerprint_changed" in triggers
    assert action == RetrievalPolicyAction.AUGMENT_PACK


@pytest.mark.unit
def test_pedagogical_retrieval_policy_augment_sets_constrained_refresh():
    from unittest.mock import MagicMock

    from src.workflow_demo.models import SessionPlan, TeachingPack
    from src.workflow_demo.pedagogy.retrieval_policy import PedagogicalRetrievalPolicy

    retriever = MagicMock()
    retriever.retrieve_plan.return_value = SessionPlan(
        subject="calculus",
        learning_objective="L",
        mode="conceptual_review",
        current_plan=[],
        future_plan=[],
        first_question="q",
        teaching_pack=TeachingPack(key_points=["refreshed"]),
    )

    d1 = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="a",
        confidence=0.6,
    )
    d2 = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="b",
        confidence=0.6,
    )
    pack = {"key_points": ["k"], "examples": [], "practice": []}
    prior = RetrievalSessionSnapshot(
        pack_focus_lo="Derivatives",
        last_diagnosis_fingerprint=diagnosis_fingerprint(d1),
    )
    move = TeachingMoveCandidate(move_type=TeachingMoveType.GRADUATED_HINT, move_id="m1")
    policy = PolicyDecision(
        selected_move=move,
        rejected_moves=[],
        decision_reason="test",
        scores={"m1": 1.0},
    )
    from src.workflow_demo.pedagogy.models import LearnerState

    out = PedagogicalRetrievalPolicy(retriever).run(
        session_target_lo="Derivatives",
        instruction_lo="Derivatives",
        prior_session_target_lo="Derivatives",
        prior_instruction_lo="Derivatives",
        student_input="ok",
        diagnosis=d2,
        policy_decision=policy,
        learner_state=LearnerState(active_session_id="s"),
        session_params={"teaching_pack": pack, "subject": "calculus", "mode": "conceptual_review"},
        prior_snapshot=prior,
        student_profile={},
    )
    assert out.action == RetrievalPolicyAction.AUGMENT_PACK
    assert out.retrieval_execution_mode == RetrievalExecutionMode.CONSTRAINED_REFRESH
    retriever.retrieve_plan.assert_called_once()
    call_kw = retriever.retrieve_plan.call_args.kwargs
    assert call_kw.get("top_los") == 4
    assert call_kw.get("top_content") == 4


@pytest.mark.unit
def test_pedagogical_retrieval_policy_reuse_skips_retriever():
    from unittest.mock import MagicMock

    from src.workflow_demo.pedagogy.models import LearnerState
    from src.workflow_demo.pedagogy.retrieval_policy import PedagogicalRetrievalPolicy

    retriever = MagicMock()
    d = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="stable",
        confidence=0.6,
    )
    pack = {"key_points": ["k"], "examples": [{"snippet": "e"}], "practice": []}
    prior = RetrievalSessionSnapshot(
        pack_focus_lo="Derivatives",
        last_diagnosis_fingerprint=diagnosis_fingerprint(d),
    )
    move = TeachingMoveCandidate(move_type=TeachingMoveType.GRADUATED_HINT, move_id="m1")
    policy = PolicyDecision(
        selected_move=move,
        rejected_moves=[],
        decision_reason="test",
        scores={"m1": 1.0},
    )
    out = PedagogicalRetrievalPolicy(retriever).run(
        session_target_lo="Derivatives",
        instruction_lo="Derivatives",
        prior_session_target_lo="Derivatives",
        prior_instruction_lo="Derivatives",
        student_input="ok",
        diagnosis=d,
        policy_decision=policy,
        learner_state=LearnerState(active_session_id="s"),
        session_params={"teaching_pack": pack, "subject": "calculus"},
        prior_snapshot=prior,
        student_profile={},
    )
    assert out.action == RetrievalPolicyAction.REUSE_PACK
    assert out.retrieval_execution_mode == RetrievalExecutionMode.NO_IO
    retriever.retrieve_plan.assert_not_called()
    retriever.retrieve_candidates.assert_not_called()
