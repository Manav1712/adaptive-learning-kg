"""
Session-local tutoring progression (MVP).

Stores a tiny step list under pedagogy_context["extensions"]["progression"].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .turn_progression import TurnProgressionSignals


def build_initial_session_progression(session_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build progression.steps from current_plan (order preserved, titles de-duplicated).

    If current_plan is empty, fall back to learning_objective / title as a single step.
    """
    current_plan = session_params.get("current_plan") or []
    steps: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in current_plan:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        is_primary = bool(item.get("is_primary"))
        kind = "primary" if is_primary else "support"
        steps.append({"lo": title, "kind": kind})

    seed = (
        str(session_params.get("learning_objective") or session_params.get("title") or "")
        .strip()
    )
    if not steps and seed:
        steps = [{"lo": seed, "kind": "fallback"}]

    return {
        "steps": steps,
        "active_step_index": 0,
        "current_step_passed": False,
        # Same-LO sub-step: fresh -> covered (engaged after teaching) -> satisfied (move-on / adequate).
        "active_step_focus_state": "fresh",
    }


def get_active_progression_lo(progression: Dict[str, Any]) -> Optional[str]:
    """Return the LO title for the active step, or None if no steps."""
    steps = progression.get("steps") or []
    if not steps:
        return None
    idx = int(progression.get("active_step_index", 0) or 0)
    if idx < 0:
        idx = 0
    if idx >= len(steps):
        idx = len(steps) - 1
    step = steps[idx]
    if not isinstance(step, dict):
        return None
    lo = str(step.get("lo") or "").strip()
    return lo or None


def apply_session_progression_update(
    progression: Dict[str, Any],
    signals: TurnProgressionSignals,
) -> Tuple[Dict[str, Any], bool]:
    """
    Update progression after computing turn signals.

    Step index advances only on:
    - adequate_check_response (safe), or
    - explicit_advance_intent when not blocked (support steps require adequate evidence).

    substantive_answer_attempt and learner_requested_example do NOT advance the index.

    Returns:
        (updated_progression, progression_event_occurred)
        progression_event_occurred is True if the index changed or current_step_passed
        flipped False -> True this turn.
    """
    steps: List[Any] = list(progression.get("steps") or [])
    idx = int(progression.get("active_step_index", 0) or 0)
    passed = bool(progression.get("current_step_passed"))

    if not steps or passed:
        return progression, False

    if idx < 0:
        idx = 0
    if idx >= len(steps):
        idx = len(steps) - 1

    blocked = signals.current_confusion_signal or signals.short_low_signal_ack
    if blocked:
        return progression, False

    current = steps[idx] if isinstance(steps[idx], dict) else {}
    kind = str(current.get("kind") or "").lower()
    is_support = kind == "support"

    can_advance = False
    if signals.adequate_check_response:
        can_advance = True
    elif signals.explicit_advance_intent:
        if is_support and not signals.adequate_check_response:
            can_advance = False
        else:
            can_advance = True

    if not can_advance:
        return progression, False

    new_prog = dict(progression)
    if idx + 1 < len(steps):
        new_prog["active_step_index"] = idx + 1
        new_prog["current_step_passed"] = False
        return new_prog, True

    new_prog["current_step_passed"] = True
    return new_prog, True


def update_same_step_focus_state(
    progression: Dict[str, Any],
    signals: TurnProgressionSignals,
    step_index_changed: bool,
) -> Dict[str, Any]:
    """
    Track whether the current active LO step has already been checked or bridged,
    so policy can avoid another broad concept-check on the same subidea.

    Resets to "fresh" when active_step_index changes. Does not advance the index.
    """
    prog = dict(progression)
    if step_index_changed:
        prog["active_step_focus_state"] = "fresh"
        return prog

    if signals.current_confusion_signal or signals.short_low_signal_ack:
        return prog

    state = str(prog.get("active_step_focus_state") or "fresh").lower()
    if state not in {"fresh", "covered", "satisfied"}:
        state = "fresh"

    if signals.adequate_check_response or signals.explicit_advance_intent:
        prog["active_step_focus_state"] = "satisfied"
        return prog

    if signals.suppress_repeat_diagnostic:
        if state == "fresh":
            prog["active_step_focus_state"] = "covered"
        return prog

    return prog
