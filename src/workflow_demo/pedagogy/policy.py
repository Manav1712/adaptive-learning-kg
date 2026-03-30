"""
Deterministic policy scorer that selects one teaching move candidate.
"""

from __future__ import annotations

from typing import Dict, List

from .constants import TeachingMoveType
from .models import LearnerState, MisconceptionDiagnosis, PolicyDecision, TeachingMoveCandidate

# Lower rank wins when total scores tie (see PolicyScorer.select_best_move sort key).
MOVE_TYPE_TIEBREAK_PRIORITY: Dict[TeachingMoveType, int] = {
    TeachingMoveType.DIAGNOSTIC_QUESTION: 0,
    TeachingMoveType.PREREQ_REMEDIATION: 1,
    TeachingMoveType.GRADUATED_HINT: 2,
    TeachingMoveType.WORKED_EXAMPLE: 3,
}


class PolicyScorer:
    """Scores candidates with interpretable rules and selects exactly one move."""

    def select_best_move(
        self,
        *,
        diagnosis: MisconceptionDiagnosis,
        learner_state: LearnerState,
        teaching_moves: List[TeachingMoveCandidate],
        current_focus_lo: str,
        user_input: str,
    ) -> PolicyDecision:
        if not teaching_moves:
            raise ValueError("teaching_moves must not be empty")

        focus_lo = (current_focus_lo or diagnosis.target_lo or learner_state.current_focus_lo or "unknown").strip() or "unknown"
        scores: Dict[str, float] = {}
        for move in teaching_moves:
            scores[move.move_id] = round(
                self._score_move(
                    move=move,
                    diagnosis=diagnosis,
                    learner_state=learner_state,
                    focus_lo=focus_lo,
                    user_input=user_input or "",
                ),
                6,
            )

        ranked = sorted(
            teaching_moves,
            key=lambda move: (
                -scores.get(move.move_id, float("-inf")),
                self._tie_break_rank(move.move_type),
                move.move_id,
            ),
        )
        selected = ranked[0]
        rejected = ranked[1:]
        reason = self._build_decision_reason(
            selected=selected,
            score=scores[selected.move_id],
            diagnosis=diagnosis,
        )
        return PolicyDecision(
            selected_move=selected,
            rejected_moves=rejected,
            decision_reason=reason,
            scores=scores,
            policy_version="1",
        )

    def _score_move(
        self,
        *,
        move: TeachingMoveCandidate,
        diagnosis: MisconceptionDiagnosis,
        learner_state: LearnerState,
        focus_lo: str,
        user_input: str,
    ) -> float:
        score = 0.0

        # Baseline from candidate metadata.
        score += (move.expected_learning_gain or 0.5) * 0.40
        score += move.priority_score * 0.20
        score -= (move.leakage_risk or 0.0) * 0.35

        low_confidence = diagnosis.confidence < 0.5
        prereq_gap = bool(diagnosis.prerequisite_gap_los) or diagnosis.suspected_misconception == "prerequisite_gap"
        mastery = learner_state.mastery.get(focus_lo, learner_state.mastery.get(str(focus_lo), 0.0))
        repeated_struggle = len(learner_state.recent_attempts) >= 2
        stuck = (
            diagnosis.suspected_misconception in {
                "conceptual_confusion",
                "uncertain_reasoning",
                "uncertain_or_low_signal",
                "power_rule_exponent_misapplied",
            }
            or "confused" in user_input.lower()
            or "don't understand" in user_input.lower()
            or "do not understand" in user_input.lower()
        )
        needs_grounding = ("?" in user_input) or diagnosis.suspected_misconception in {
            "power_rule_exponent_misapplied",
            "conceptual_confusion",
        }

        if move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION and low_confidence:
            score += 1.40
        if move.move_type == TeachingMoveType.PREREQ_REMEDIATION and prereq_gap:
            score += 1.30
        if move.move_type == TeachingMoveType.GRADUATED_HINT and (mastery < 0.45 or repeated_struggle or stuck):
            score += 1.10
        if move.move_type == TeachingMoveType.WORKED_EXAMPLE and needs_grounding:
            score += 0.95

        # Small penalties for weak fit to keep selection interpretable.
        if low_confidence and move.move_type != TeachingMoveType.DIAGNOSTIC_QUESTION:
            score -= 0.15
        if prereq_gap and move.move_type == TeachingMoveType.WORKED_EXAMPLE:
            score -= 0.10

        return score

    @staticmethod
    def _tie_break_rank(move_type: TeachingMoveType) -> int:
        return MOVE_TYPE_TIEBREAK_PRIORITY.get(move_type, 999)

    @staticmethod
    def _build_decision_reason(
        *,
        selected: TeachingMoveCandidate,
        score: float,
        diagnosis: MisconceptionDiagnosis,
    ) -> str:
        return (
            f"Selected {selected.move_type.value} (score={score:.3f}) "
            f"for diagnosis '{diagnosis.suspected_misconception}' "
            f"with confidence {diagnosis.confidence:.2f}."
        )
