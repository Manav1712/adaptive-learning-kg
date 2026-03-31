"""
Single precedence rule for deriving instruction_lo per tutor turn (Phase 5).

v1: one diagnoser pass; instruction_lo may differ from diagnosis.target_lo (documented on models).
"""

from __future__ import annotations

from .constants import TeachingMoveType
from .models import MisconceptionDiagnosis


_UNKNOWN_SENTINELS = frozenset({"", "unknown"})


def derive_instruction_lo(
    *,
    session_target_lo: str,
    diagnosis: MisconceptionDiagnosis,
    selected_move_type: TeachingMoveType,
) -> str:
    """
    Active instructional focus for this turn (first match wins).

    1. prereq_remediation + non-empty prerequisite_gap_los -> first gap entry
    2. Else concrete diagnosis.target_lo (not unknown) -> that value
    3. Else session_target_lo
    """
    st = (session_target_lo or "").strip()

    if selected_move_type == TeachingMoveType.PREREQ_REMEDIATION and diagnosis.prerequisite_gap_los:
        first = (diagnosis.prerequisite_gap_los[0] or "").strip()
        if first:
            return first

    dt = (diagnosis.target_lo or "").strip()
    if dt and dt.lower() not in _UNKNOWN_SENTINELS:
        return dt

    return st or "unknown"
