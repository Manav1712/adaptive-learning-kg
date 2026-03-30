"""
Deterministic teaching-move candidate generation for tutor turns.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from .constants import RetrievalIntent, TeachingMoveType
from .models import LearnerState, MisconceptionDiagnosis, TeachingMoveCandidate


_MOVE_RETRIEVAL_INTENT: Dict[TeachingMoveType, RetrievalIntent] = {
    TeachingMoveType.DIAGNOSTIC_QUESTION: RetrievalIntent.MISCONCEPTION_REPAIR,
    TeachingMoveType.GRADUATED_HINT: RetrievalIntent.PRACTICE_ITEM,
    TeachingMoveType.WORKED_EXAMPLE: RetrievalIntent.WORKED_PARALLEL,
    TeachingMoveType.PREREQ_REMEDIATION: RetrievalIntent.PREREQUISITE_REFRESH,
}

_MOVE_DEFAULT_GAIN: Dict[TeachingMoveType, float] = {
    TeachingMoveType.DIAGNOSTIC_QUESTION: 0.45,
    TeachingMoveType.GRADUATED_HINT: 0.62,
    TeachingMoveType.WORKED_EXAMPLE: 0.58,
    TeachingMoveType.PREREQ_REMEDIATION: 0.66,
}

_MOVE_DEFAULT_RISK: Dict[TeachingMoveType, float] = {
    TeachingMoveType.DIAGNOSTIC_QUESTION: 0.12,
    TeachingMoveType.GRADUATED_HINT: 0.28,
    TeachingMoveType.WORKED_EXAMPLE: 0.22,
    TeachingMoveType.PREREQ_REMEDIATION: 0.18,
}


class TeachingMoveGenerator:
    """Generate 2-4 deterministic teaching move candidates from diagnosis/state."""

    MAX_CANDIDATES = 4
    MIN_CANDIDATES = 2

    def generate_candidates(
        self,
        diagnosis: MisconceptionDiagnosis,
        learner_state: LearnerState,
        current_focus_lo: str,
        user_input: str,
    ) -> List[TeachingMoveCandidate]:
        focus_lo = (current_focus_lo or diagnosis.target_lo or learner_state.current_focus_lo or "unknown").strip() or "unknown"
        text = (user_input or "").lower()
        kinds: List[TeachingMoveType] = []

        low_confidence = diagnosis.confidence < 0.5
        uncertain = "uncertain" in diagnosis.suspected_misconception or "unknown" in diagnosis.suspected_misconception
        prereq_gap = (
            diagnosis.suspected_misconception == "prerequisite_gap"
            or bool(diagnosis.prerequisite_gap_los)
        )
        stuck = (
            diagnosis.suspected_misconception in {
                "conceptual_confusion",
                "uncertain_reasoning",
                "uncertain_or_low_signal",
                "power_rule_exponent_misapplied",
            }
            or "confused" in text
            or "don't understand" in text
            or "do not understand" in text
        )
        needs_grounding = (
            "?" in text
            or diagnosis.suspected_misconception in {
                "power_rule_exponent_misapplied",
                "conceptual_confusion",
            }
        )

        if low_confidence or uncertain:
            kinds.append(TeachingMoveType.DIAGNOSTIC_QUESTION)
        if prereq_gap:
            kinds.append(TeachingMoveType.PREREQ_REMEDIATION)
        if stuck:
            kinds.append(TeachingMoveType.GRADUATED_HINT)
        if needs_grounding:
            kinds.append(TeachingMoveType.WORKED_EXAMPLE)

        # Guarantee 2-4 stable candidates.
        fallback_order = [
            TeachingMoveType.GRADUATED_HINT,
            TeachingMoveType.WORKED_EXAMPLE,
            TeachingMoveType.DIAGNOSTIC_QUESTION,
            TeachingMoveType.PREREQ_REMEDIATION,
        ]
        for item in fallback_order:
            if item not in kinds and len(kinds) < self.MIN_CANDIDATES:
                kinds.append(item)
        kinds = kinds[: self.MAX_CANDIDATES]

        candidates: List[TeachingMoveCandidate] = []
        for idx, move_type in enumerate(kinds):
            intent = _MOVE_RETRIEVAL_INTENT[move_type]
            reason = self._build_reason(move_type, diagnosis, focus_lo)
            candidates.append(
                TeachingMoveCandidate(
                    move_id=self._build_move_id(move_type, focus_lo, diagnosis, idx),
                    move_type=move_type,
                    target_lo=focus_lo,
                    reason=reason,
                    retrieval_intent=intent,
                    expected_learning_gain=_MOVE_DEFAULT_GAIN[move_type],
                    leakage_risk=_MOVE_DEFAULT_RISK[move_type],
                    priority_score=round(1.0 - (idx * 0.12), 3),
                    metadata={"diagnosis": diagnosis.suspected_misconception, "target_lo": focus_lo},
                )
            )
        return candidates

    @staticmethod
    def _build_reason(
        move_type: TeachingMoveType,
        diagnosis: MisconceptionDiagnosis,
        focus_lo: str,
    ) -> str:
        if move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
            return (
                f"Diagnosis confidence is {diagnosis.confidence:.2f}; ask a targeted check question to disambiguate "
                f"understanding in {focus_lo}."
            )
        if move_type == TeachingMoveType.PREREQ_REMEDIATION:
            prereqs = ", ".join(diagnosis.prerequisite_gap_los[:3]) or "foundational prerequisites"
            return f"Diagnosis indicates prerequisite gaps; remediate prerequisites first ({prereqs})."
        if move_type == TeachingMoveType.GRADUATED_HINT:
            return "Provide a stepwise hint progression to reduce cognitive load while preserving student agency."
        return "Ground abstract reasoning with a concrete worked example anchored to the current focus."

    @staticmethod
    def _build_move_id(
        move_type: TeachingMoveType,
        focus_lo: str,
        diagnosis: MisconceptionDiagnosis,
        index: int,
    ) -> str:
        basis = "|".join(
            [
                move_type.value,
                focus_lo,
                diagnosis.suspected_misconception,
                f"{diagnosis.confidence:.3f}",
                str(index),
            ]
        )
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
        return f"move_{move_type.value}_{digest}"
