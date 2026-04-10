"""
Compact tutor-session pedagogy snapshot for API, UI, and ! debug commands (Phase 8).

Assembled from current handoff + learner state — not from runtime event replay.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..practice.session import PracticeSessionManager
from .session_progression import get_active_progression_lo


def _clip(s: Optional[str], max_len: int) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    return t[:max_len] if len(t) > max_len else t


def build_tutor_pedagogy_snapshot(
    *,
    handoff_context: Optional[Dict[str, Any]],
    bot_type: Optional[str],
    active_learner_session_id: Optional[str],
    learner_state_engine: Any,
) -> Optional[Dict[str, Any]]:
    """
    Backend-owned snapshot for the active tutor session. Returns None if not in tutor mode.

    Caps per Phase 8 default snapshot rules.
    """
    if bot_type != "tutor" or not isinstance(handoff_context, dict):
        return None

    pc = handoff_context.get("pedagogy_context")
    if not isinstance(pc, dict):
        pc = {}

    tid = pc.get("tutor_instruction_directives")
    if not isinstance(tid, dict):
        tid = {}
    rs = pc.get("retrieval_session")
    if not isinstance(rs, dict):
        rs = {}
    diag = pc.get("diagnosis")
    if not isinstance(diag, dict):
        diag = {}
    pol = pc.get("policy_decision")
    if not isinstance(pol, dict):
        pol = {}
    moves = pc.get("teaching_moves")
    if not isinstance(moves, list):
        moves = []

    candidate_types: List[str] = []
    for m in moves[:12]:
        if isinstance(m, dict) and m.get("move_type"):
            candidate_types.append(str(m["move_type"]))
        elif hasattr(m, "move_type"):
            candidate_types.append(str(m.move_type))
    candidate_move_types = list(dict.fromkeys(candidate_types))[:4]

    prereq = diag.get("prerequisite_gap_los") or []
    if isinstance(prereq, list):
        prereq_out = [str(x) for x in prereq[:3]]
    else:
        prereq_out = []

    policy_reason = _clip(
        tid.get("policy_reason") or pol.get("decision_reason"),
        500,
    )

    session_progression: Optional[Dict[str, Any]] = None
    extensions = pc.get("extensions")
    if isinstance(extensions, dict):
        raw_prog = extensions.get("progression")
        if isinstance(raw_prog, dict) and raw_prog.get("steps"):
            steps = raw_prog.get("steps") or []
            session_progression = {
                "active_step_index": raw_prog.get("active_step_index"),
                "current_step_passed": raw_prog.get("current_step_passed"),
                "step_count": len(steps) if isinstance(steps, list) else 0,
                "active_step_lo": _clip(get_active_progression_lo(raw_prog), 256),
            }

    # Practice-loop / sequencing snapshot (Round 2+, safe when absent).
    practice_snapshot = None
    sequencing_snapshot = None
    if isinstance(extensions, dict):
        _ps_snap = PracticeSessionManager.build_snapshot(extensions)
        if _ps_snap is not None:
            practice_snapshot = _ps_snap.get("practice_session")
            sequencing_snapshot = _ps_snap.get("sequencing")

    snap: Dict[str, Any] = {
        "session_id": active_learner_session_id,
        "bot_type": "tutor",
        "target_lo": _clip(pc.get("target_lo"), 256),
        "instruction_lo": _clip(pc.get("instruction_lo"), 256),
        "diagnosis_target_lo": _clip(diag.get("target_lo"), 256),
        "suspected_misconception": _clip(diag.get("suspected_misconception"), 256),
        "diagnosis_confidence": diag.get("confidence"),
        "prerequisite_gap_los": prereq_out,
        "candidate_move_types": candidate_move_types,
        "selected_move_type": _clip(tid.get("selected_move_type"), 64),
        "policy_reason": policy_reason,
        "retrieval_intent": _clip(pc.get("retrieval_intent"), 128),
        "retrieval_action": _clip(pc.get("retrieval_action"), 128),
        "retrieval_execution_mode": _clip(pc.get("retrieval_execution_mode"), 64),
        "pack_focus_lo": _clip(rs.get("pack_focus_lo"), 256),
        "pack_revision": rs.get("pack_revision"),
        "last_diagnosis_fingerprint": _clip(rs.get("last_diagnosis_fingerprint"), 256),
        "last_selected_move_type": _clip(rs.get("last_selected_move_type"), 64),
        "last_guard_result": pc.get("last_guard_result"),
        "session_progression": session_progression,
        "practice_session": practice_snapshot,
        "sequencing": sequencing_snapshot,
    }

    if active_learner_session_id and learner_state_engine is not None:
        try:
            ls = learner_state_engine.build_snapshot(active_learner_session_id)
            raw_top = list(ls.top_mastery or [])[:3]
            top_m: List[Any] = [[str(a), float(b)] for a, b in raw_top]
            recent_mis: Dict[str, List[str]] = {}
            raw_m = getattr(ls, "recent_misconceptions", None) or {}
            if isinstance(raw_m, dict):
                for lo, labels in raw_m.items():
                    if isinstance(labels, list):
                        recent_mis[str(lo)] = [str(x) for x in labels[-2:]]
                    elif isinstance(labels, str):
                        recent_mis[str(lo)] = [labels]
            snap["learner"] = {
                "recent_attempt_count": ls.recent_attempt_count,
                "hint_count": ls.hint_count,
                "top_mastery": top_m,
                "recent_misconceptions": recent_mis,
            }
        except Exception:
            snap["learner"] = {
                "recent_attempt_count": 0,
                "hint_count": 0,
                "top_mastery": [],
                "recent_misconceptions": {},
            }
    else:
        snap["learner"] = {
            "recent_attempt_count": 0,
            "hint_count": 0,
            "top_mastery": [],
            "recent_misconceptions": {},
        }

    return snap
