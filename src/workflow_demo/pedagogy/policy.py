"""
Deterministic policy scorer that selects one teaching move candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from .constants import TeachingMoveType
from .models import LearnerState, MisconceptionDiagnosis, PolicyDecision, TeachingMoveCandidate

if TYPE_CHECKING:
    from .turn_progression import TurnProgressionSignals

# Strong penalty so diagnostic_question loses to other moves when repeat gate fires (see turn_progression).
_REPEAT_DIAGNOSTIC_SCORE_PENALTY = 2.5
_EXPLICIT_ADVANCE_DIAGNOSTIC_NUDGE = 0.35
# Session progression: step advanced or final step passed — avoid another broad check.
_PROGRESSION_STRONG_DIAGNOSTIC_PENALTY = 2.0
_PROGRESSION_CONCRETE_MOVE_BOOST = 0.55
# Same-step substantive engagement: wrong/detailed attempt should change move, not advance step.
_SAME_STEP_SUBSTANTIVE_DIAGNOSTIC_PENALTY = 1.2
# Explicit move-on without prior-diagnostic suppress gate still must not default to another broad check.
_EXPLICIT_ADVANCE_STRONG_DIAGNOSTIC_PENALTY = 1.5
_EXPLICIT_ADVANCE_CONCRETE_MOVE_BOOST = 0.45
_SAME_STEP_CONCRETE_MOVE_BOOST = 0.35
# Clear understanding signal: force the next move away from another broad check.
_ADEQUATE_UNDERSTANDING_DIAGNOSTIC_PENALTY = 3.0
_ADEQUATE_UNDERSTANDING_CONCRETE_BOOST = 0.5
# Same-step focus already covered or satisfied: avoid another broad diagnostic on this LO.
_STEP_FOCUS_DIAGNOSTIC_PENALTY = 2.2
_STEP_FOCUS_CONCRETE_BOOST = 0.4

# Example request: boost worked_example; nudge diagnostic down; prefer graduated_hint if no worked_example candidate.
_EXAMPLE_REQUEST_WORKED_BOOST = 1.45
_EXAMPLE_REQUEST_DIAGNOSTIC_NUDGE = 0.95
_EXAMPLE_REQUEST_GRADUATED_HINT_FALLBACK = 1.25

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
        progression_signals: Optional["TurnProgressionSignals"] = None,
        progression_just_advanced: bool = False,
        progression_step_passed: bool = False,
        step_focus_state: Optional[str] = None,
    ) -> PolicyDecision:
        if not teaching_moves:
            raise ValueError("teaching_moves must not be empty")

        focus_lo = (current_focus_lo or diagnosis.target_lo or learner_state.current_focus_lo or "unknown").strip() or "unknown"
        has_worked_example_candidate = any(
            m.move_type == TeachingMoveType.WORKED_EXAMPLE for m in teaching_moves
        )
        scores: Dict[str, float] = {}
        for move in teaching_moves:
            scores[move.move_id] = round(
                self._score_move(
                    move=move,
                    diagnosis=diagnosis,
                    learner_state=learner_state,
                    focus_lo=focus_lo,
                    user_input=user_input or "",
                    progression_signals=progression_signals,
                    has_worked_example_candidate=has_worked_example_candidate,
                    progression_just_advanced=progression_just_advanced,
                    progression_step_passed=progression_step_passed,
                    step_focus_state=step_focus_state,
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
        progression_signals: Optional["TurnProgressionSignals"] = None,
        has_worked_example_candidate: bool = True,
        progression_just_advanced: bool = False,
        progression_step_passed: bool = False,
        step_focus_state: Optional[str] = None,
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

        if progression_signals is not None:
            if (
                progression_signals.suppress_repeat_diagnostic
                and move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION
            ):
                score -= _REPEAT_DIAGNOSTIC_SCORE_PENALTY
            elif (
                progression_signals.explicit_advance_intent
                and not progression_signals.suppress_repeat_diagnostic
                and move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION
            ):
                score -= _EXPLICIT_ADVANCE_DIAGNOSTIC_NUDGE

            if progression_just_advanced or progression_step_passed:
                if move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
                    score -= _PROGRESSION_STRONG_DIAGNOSTIC_PENALTY
                if not prereq_gap and move.move_type == TeachingMoveType.WORKED_EXAMPLE:
                    score += _PROGRESSION_CONCRETE_MOVE_BOOST
                if not prereq_gap and move.move_type == TeachingMoveType.GRADUATED_HINT:
                    score += _PROGRESSION_CONCRETE_MOVE_BOOST * 0.7

            same_step_concrete = (
                not progression_signals.current_confusion_signal
                and not progression_signals.short_low_signal_ack
                and (
                    progression_signals.substantive_answer_attempt
                    or progression_signals.learner_requested_example
                )
            )
            if same_step_concrete:
                if move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
                    score -= _SAME_STEP_SUBSTANTIVE_DIAGNOSTIC_PENALTY
                if move.move_type == TeachingMoveType.WORKED_EXAMPLE:
                    score += _SAME_STEP_CONCRETE_MOVE_BOOST
                if move.move_type == TeachingMoveType.GRADUATED_HINT:
                    score += _SAME_STEP_CONCRETE_MOVE_BOOST * 0.7

            if (
                progression_signals.explicit_advance_intent
                and not progression_signals.current_confusion_signal
                and not progression_signals.short_low_signal_ack
            ):
                if move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
                    score -= _EXPLICIT_ADVANCE_STRONG_DIAGNOSTIC_PENALTY
                if not prereq_gap and move.move_type == TeachingMoveType.WORKED_EXAMPLE:
                    score += _EXPLICIT_ADVANCE_CONCRETE_MOVE_BOOST
                if not prereq_gap and move.move_type == TeachingMoveType.GRADUATED_HINT:
                    score += _EXPLICIT_ADVANCE_CONCRETE_MOVE_BOOST * 0.7

            if (
                progression_signals.adequate_check_response
                and not progression_signals.current_confusion_signal
                and not progression_signals.short_low_signal_ack
            ):
                if move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
                    score -= _ADEQUATE_UNDERSTANDING_DIAGNOSTIC_PENALTY
                elif move.move_type in {
                    TeachingMoveType.GRADUATED_HINT,
                    TeachingMoveType.WORKED_EXAMPLE,
                }:
                    score += _ADEQUATE_UNDERSTANDING_CONCRETE_BOOST

            if progression_signals.learner_requested_example:
                if move.move_type == TeachingMoveType.WORKED_EXAMPLE:
                    score += _EXAMPLE_REQUEST_WORKED_BOOST
                if move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
                    score -= _EXAMPLE_REQUEST_DIAGNOSTIC_NUDGE
                if (
                    not has_worked_example_candidate
                    and move.move_type == TeachingMoveType.GRADUATED_HINT
                ):
                    score += _EXAMPLE_REQUEST_GRADUATED_HINT_FALLBACK

        focus = (step_focus_state or "").strip().lower()
        if focus in {"covered", "satisfied"} and not prereq_gap:
            if move.move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
                score -= _STEP_FOCUS_DIAGNOSTIC_PENALTY
            if move.move_type == TeachingMoveType.WORKED_EXAMPLE:
                score += _STEP_FOCUS_CONCRETE_BOOST
            if move.move_type == TeachingMoveType.GRADUATED_HINT:
                score += _STEP_FOCUS_CONCRETE_BOOST * 0.75

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
