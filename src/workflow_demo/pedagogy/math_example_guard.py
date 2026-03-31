"""
Narrow, feature-gated checks on tutor worked-example output.

Enable with WORKFLOW_DEMO_TUTOR_MATH_GUARD=1. On any error, returns the input unchanged.
"""

from __future__ import annotations

import re
from typing import Any, Dict


def maybe_apply_math_example_guard(
    response: Dict[str, Any],
    handoff_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a minimal integral sanity check only for worked_example + matching prose."""
    out = dict(response)
    try:
        pc = handoff_context.get("pedagogy_context") if isinstance(handoff_context, dict) else None
        if not isinstance(pc, dict):
            return out
        directives = pc.get("tutor_instruction_directives")
        if not isinstance(directives, dict) or not directives:
            directives = pc.get("tutor_directives")
        if not isinstance(directives, dict):
            return out
        if directives.get("selected_move_type") != "worked_example":
            return out

        msg = out.get("message_to_student") or ""
        if not msg or len(msg) > 12_000:
            return out

        # ∫_a^b x dx or "integral from a to b of x" style (integers only, single x)
        m = re.search(
            r"(?:∫_\s*\{?(-?\d+)\}?\^\s*\{?(-?\d+)\}?|integral\s+from\s+(-?\d+)\s+to\s+(-?\d+))\s+"
            r"(?:of\s+)?x\s*(?:d\s*x|dx)",
            msg,
            re.IGNORECASE,
        )
        if not m:
            return out

        try:
            import sympy as sp  # type: ignore[import-untyped]
        except ImportError:
            return out

        if m.group(1) is not None:
            a, b = int(m.group(1)), int(m.group(2))
        else:
            a, b = int(m.group(3)), int(m.group(4))

        x = sp.Symbol("x", real=True)
        expected = sp.integrate(x, (x, a, b))
        expected_val = float(expected.evalf())

        eq = re.search(
            r"=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*(?:\.|$|\s)",
            msg[m.end() : m.end() + 80],
        )
        if not eq:
            return out
        rhs = eq.group(1)
        if "/" in rhs:
            num, den = rhs.split("/", 1)
            claimed = float(num) / float(den)
        else:
            claimed = float(rhs)

        if abs(claimed - expected_val) > 1e-4:
            out["message_to_student"] = (
                f"{msg}\n\n[Note: double-check the numeric result in the worked example above.]"
            )
        return out
    except Exception:
        return dict(response)
