"""Phase 4 tests for deterministic policy scoring and move selection."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy import (
    LearnerState,
    MisconceptionDiagnosis,
    PolicyScorer,
    TeachingMoveCandidate,
)
from src.workflow_demo.pedagogy.constants import RetrievalIntent, TeachingMoveType


def _candidate(
    move_id: str,
    move_type: TeachingMoveType,
    *,
    target_lo: str = "Derivatives",
    expected_learning_gain: float = 0.6,
    leakage_risk: float = 0.2,
    priority_score: float = 0.6,
) -> TeachingMoveCandidate:
    return TeachingMoveCandidate(
        move_id=move_id,
        move_type=move_type,
        target_lo=target_lo,
        reason=f"Reason for {move_type.value}",
        retrieval_intent={
            TeachingMoveType.DIAGNOSTIC_QUESTION: RetrievalIntent.MISCONCEPTION_REPAIR,
            TeachingMoveType.GRADUATED_HINT: RetrievalIntent.PRACTICE_ITEM,
            TeachingMoveType.WORKED_EXAMPLE: RetrievalIntent.WORKED_PARALLEL,
            TeachingMoveType.PREREQ_REMEDIATION: RetrievalIntent.PREREQUISITE_REFRESH,
        }[move_type],
        expected_learning_gain=expected_learning_gain,
        leakage_risk=leakage_risk,
        priority_score=priority_score,
    )


@pytest.mark.unit
def test_low_confidence_selects_diagnostic_question():
    scorer = PolicyScorer()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="uncertain_or_low_signal",
        confidence=0.2,
        rationale="Low confidence diagnosis.",
        prerequisite_gap_los=[],
    )
    moves = [
        _candidate("m_diag", TeachingMoveType.DIAGNOSTIC_QUESTION),
        _candidate("m_hint", TeachingMoveType.GRADUATED_HINT),
        _candidate("m_example", TeachingMoveType.WORKED_EXAMPLE),
    ]
    decision = scorer.select_best_move(
        diagnosis=diagnosis,
        learner_state=LearnerState(active_session_id="s1", mastery={"Derivatives": 0.5}),
        teaching_moves=moves,
        current_focus_lo="Derivatives",
        user_input="ok",
    )
    assert decision.selected_move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION


@pytest.mark.unit
def test_prerequisite_gap_selects_prereq_remediation():
    scorer = PolicyScorer()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Integrals",
        suspected_misconception="prerequisite_gap",
        confidence=0.85,
        rationale="Needs prerequisite refreshers.",
        prerequisite_gap_los=["Derivatives"],
    )
    moves = [
        _candidate("m_prereq", TeachingMoveType.PREREQ_REMEDIATION, target_lo="Integrals"),
        _candidate("m_hint", TeachingMoveType.GRADUATED_HINT, target_lo="Integrals"),
        _candidate("m_example", TeachingMoveType.WORKED_EXAMPLE, target_lo="Integrals"),
    ]
    decision = scorer.select_best_move(
        diagnosis=diagnosis,
        learner_state=LearnerState(active_session_id="s1", mastery={"Integrals": 0.6}),
        teaching_moves=moves,
        current_focus_lo="Integrals",
        user_input="I forgot derivatives",
    )
    assert decision.selected_move.move_type == TeachingMoveType.PREREQ_REMEDIATION


@pytest.mark.unit
def test_low_mastery_stuck_learner_selects_graduated_hint():
    scorer = PolicyScorer()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="conceptual_confusion",
        confidence=0.7,
        rationale="Student appears stuck.",
        prerequisite_gap_los=[],
    )
    learner = LearnerState(
        active_session_id="s1",
        mastery={"Derivatives": 0.2},
    )
    moves = [
        _candidate("m_hint", TeachingMoveType.GRADUATED_HINT),
        _candidate("m_example", TeachingMoveType.WORKED_EXAMPLE),
    ]
    decision = scorer.select_best_move(
        diagnosis=diagnosis,
        learner_state=learner,
        teaching_moves=moves,
        current_focus_lo="Derivatives",
        user_input="I don't understand",
    )
    assert decision.selected_move.move_type == TeachingMoveType.GRADUATED_HINT


@pytest.mark.unit
def test_scores_include_all_candidates():
    scorer = PolicyScorer()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Limits",
        suspected_misconception="uncertain_reasoning",
        confidence=0.55,
        rationale="Uncertain answer.",
        prerequisite_gap_los=[],
    )
    moves = [
        _candidate("m1", TeachingMoveType.DIAGNOSTIC_QUESTION, target_lo="Limits"),
        _candidate("m2", TeachingMoveType.WORKED_EXAMPLE, target_lo="Limits"),
    ]
    decision = scorer.select_best_move(
        diagnosis=diagnosis,
        learner_state=LearnerState(active_session_id="s1", mastery={"Limits": 0.5}),
        teaching_moves=moves,
        current_focus_lo="Limits",
        user_input="maybe?",
    )
    assert set(decision.scores.keys()) == {"m1", "m2"}


@pytest.mark.unit
def test_deterministic_tie_break():
    scorer = PolicyScorer()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="uncertain_or_low_signal",
        confidence=0.1,
        rationale="Low confidence.",
        prerequisite_gap_los=[],
    )
    # Same metadata to force tie-ish scores.
    moves = [
        _candidate("diag", TeachingMoveType.DIAGNOSTIC_QUESTION, expected_learning_gain=0.5, leakage_risk=0.2, priority_score=0.5),
        _candidate("prereq", TeachingMoveType.PREREQ_REMEDIATION, expected_learning_gain=0.5, leakage_risk=0.2, priority_score=0.5),
        _candidate("hint", TeachingMoveType.GRADUATED_HINT, expected_learning_gain=0.5, leakage_risk=0.2, priority_score=0.5),
        _candidate("example", TeachingMoveType.WORKED_EXAMPLE, expected_learning_gain=0.5, leakage_risk=0.2, priority_score=0.5),
    ]
    selections = []
    for _ in range(5):
        decision = scorer.select_best_move(
            diagnosis=diagnosis,
            learner_state=LearnerState(active_session_id="s1"),
            teaching_moves=moves,
            current_focus_lo="Derivatives",
            user_input="ok",
        )
        selections.append(decision.selected_move.move_id)
    assert len(set(selections)) == 1
    assert selections[0] == "diag"


@pytest.mark.unit
def test_selected_move_is_from_candidates():
    scorer = PolicyScorer()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="conceptual_confusion",
        confidence=0.7,
        rationale="Confusion detected.",
        prerequisite_gap_los=[],
    )
    moves = [
        _candidate("m_hint", TeachingMoveType.GRADUATED_HINT),
        _candidate("m_example", TeachingMoveType.WORKED_EXAMPLE),
    ]
    decision = scorer.select_best_move(
        diagnosis=diagnosis,
        learner_state=LearnerState(active_session_id="s1"),
        teaching_moves=moves,
        current_focus_lo="Derivatives",
        user_input="I'm confused",
    )
    assert any(decision.selected_move.move_id == m.move_id for m in moves)


@pytest.mark.unit
def test_rejected_moves_are_the_remaining_candidates():
    scorer = PolicyScorer()
    diagnosis = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="conceptual_confusion",
        confidence=0.7,
        rationale="Confusion detected.",
        prerequisite_gap_los=[],
    )
    moves = [
        _candidate("m1", TeachingMoveType.GRADUATED_HINT),
        _candidate("m2", TeachingMoveType.WORKED_EXAMPLE),
        _candidate("m3", TeachingMoveType.DIAGNOSTIC_QUESTION),
    ]
    decision = scorer.select_best_move(
        diagnosis=diagnosis,
        learner_state=LearnerState(active_session_id="s1"),
        teaching_moves=moves,
        current_focus_lo="Derivatives",
        user_input="I don't understand this",
    )
    all_ids = {m.move_id for m in moves}
    rejected_ids = {m.move_id for m in decision.rejected_moves}
    assert decision.selected_move.move_id in all_ids
    assert rejected_ids == (all_ids - {decision.selected_move.move_id})


@pytest.mark.integration
def test_tutor_flow_populates_policy_decision_and_faq_untouched(monkeypatch, coach_agent):
    tutor_responses = [
        {
            "message_to_student": "Let's continue derivatives.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {"student_understanding": "needs_practice"},
        },
        {
            "message_to_student": "Great question, let's inspect that.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {"student_understanding": "needs_practice"},
        },
    ]

    def _fake_tutor_bot(**_kwargs):
        return tutor_responses.pop(0)

    def _fake_faq_bot(**_kwargs):
        return {
            "message_to_student": "Exam is next week.",
            "end_activity": False,
            "silent_end": False,
            "needs_topic_confirmation": False,
            "session_summary": {"topics_addressed": ["exam schedule"], "questions_answered": []},
        }

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _fake_tutor_bot)
    monkeypatch.setattr("src.workflow_demo.bot_sessions.faq_bot", _fake_faq_bot)

    coach_agent.planner_result = {
        "status": "complete",
        "plan": {
            "subject": "calculus",
            "mode": "conceptual_review",
            "current_plan": [
                {
                    "lo_id": 5,
                    "title": "Derivatives",
                    "proficiency": 0.2,
                    "how_to_teach": "",
                    "why_to_teach": "",
                    "notes": "",
                    "is_primary": True,
                }
            ],
            "future_plan": [],
        },
    }
    _ = coach_agent.bot_session_manager.begin(
        bot_type="tutor",
        tool_params={"student_request": "Teach me derivatives"},
        conversation_summary="Student requested tutoring.",
    )
    _ = coach_agent.process_turn("I'm confused, derivative of x^2 is x?")
    tutor_context = coach_agent.bot_session_manager.handoff_context or {}
    pedagogy_context = tutor_context.get("pedagogy_context") or {}

    assert pedagogy_context.get("diagnosis")
    assert pedagogy_context.get("teaching_moves")
    policy = pedagogy_context.get("policy_decision") or {}
    assert policy.get("selected_move")
    assert policy.get("rejected_moves") is not None
    assert policy.get("scores")

    tid = pedagogy_context.get("tutor_instruction_directives") or {}
    assert tid.get("session_target_lo")
    assert tid.get("instruction_lo") is not None
    assert tid.get("selected_move_type")
    assert "retrieval_intent" in tid
    assert "retrieval_action" in tid
    assert "policy_reason" in tid
    assert "retrieval_execution_mode" not in tid

    # FAQ path should not create pedagogy_context.
    coach_agent.planner_result = {
        "status": "complete",
        "plan": {"topic": "exam schedule", "script": "Exam details", "first_question": "Need more details?"},
    }
    _ = coach_agent.bot_session_manager.begin(
        bot_type="faq",
        tool_params={"topic": "exam schedule"},
        conversation_summary="Student asked FAQ.",
    )
    faq_context = coach_agent.bot_session_manager.handoff_context or {}
    assert "pedagogy_context" not in faq_context
