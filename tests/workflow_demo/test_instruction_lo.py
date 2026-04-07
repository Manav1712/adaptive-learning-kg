"""Tests for derive_instruction_lo (Phase 5)."""

import pytest

from src.workflow_demo.pedagogy.constants import TeachingMoveType
from src.workflow_demo.pedagogy.instruction_lo import derive_instruction_lo
from src.workflow_demo.pedagogy.models import MisconceptionDiagnosis


@pytest.mark.unit
def test_prereq_remediation_uses_first_prerequisite_gap():
    d = MisconceptionDiagnosis(
        target_lo="FTC",
        suspected_misconception="prerequisite_gap",
        confidence=0.8,
        prerequisite_gap_los=["Area under curve", "Limits"],
    )
    lo = derive_instruction_lo(
        session_target_lo="Fundamental Theorem",
        diagnosis=d,
        selected_move_type=TeachingMoveType.PREREQ_REMEDIATION,
    )
    assert lo == "Area under curve"


@pytest.mark.unit
def test_non_prereq_uses_diagnosis_target_when_concrete():
    d = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="uncertain_or_low_signal",
        confidence=0.4,
    )
    lo = derive_instruction_lo(
        session_target_lo="Integrals",
        diagnosis=d,
        selected_move_type=TeachingMoveType.DIAGNOSTIC_QUESTION,
    )
    assert lo == "Derivatives"


@pytest.mark.unit
def test_fallback_to_session_target_when_target_unknown():
    d = MisconceptionDiagnosis(
        target_lo="unknown",
        suspected_misconception="uncertain_or_low_signal",
        confidence=0.3,
    )
    lo = derive_instruction_lo(
        session_target_lo="Limits",
        diagnosis=d,
        selected_move_type=TeachingMoveType.GRADUATED_HINT,
    )
    assert lo == "Limits"


@pytest.mark.unit
def test_active_progression_lo_used_when_not_prereq_remediation():
    d = MisconceptionDiagnosis(
        target_lo="Derivatives",
        suspected_misconception="uncertain_or_low_signal",
        confidence=0.4,
    )
    lo = derive_instruction_lo(
        session_target_lo="Integrals",
        diagnosis=d,
        selected_move_type=TeachingMoveType.DIAGNOSTIC_QUESTION,
        active_progression_lo="Area Problem",
    )
    assert lo == "Area Problem"


@pytest.mark.unit
def test_prereq_remediation_still_overrides_active_progression():
    d = MisconceptionDiagnosis(
        target_lo="FTC",
        suspected_misconception="prerequisite_gap",
        confidence=0.8,
        prerequisite_gap_los=["Limits review"],
    )
    lo = derive_instruction_lo(
        session_target_lo="Fundamental Theorem",
        diagnosis=d,
        selected_move_type=TeachingMoveType.PREREQ_REMEDIATION,
        active_progression_lo="Should not win",
    )
    assert lo == "Limits review"
