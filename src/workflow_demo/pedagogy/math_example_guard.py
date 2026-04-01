"""
Narrow, feature-gated checks on tutor worked-example output.

Enable with WORKFLOW_DEMO_TUTOR_MATH_GUARD=1. On any error, returns the input unchanged.
Sympy is optional: if unavailable, the guard no-ops.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

_MATH_GUARD_ENV = "WORKFLOW_DEMO_TUTOR_MATH_GUARD"
_MAX_MSG = 12_000
_MAX_LIMIT = 100
_MAX_DEGREE = 5
_MAX_TERMS = 6
_TOL = 1e-4
_NOTE_PREFIX = (
    "\n\n[Note: the computed result in the worked example above may be incorrect; "
    "the correct value is "
)
_NOTE_SUFFIX = ".]"


def _debug(msg: str) -> None:
    if os.getenv(_MATH_GUARD_ENV, "").lower() not in ("1", "true", "yes"):
        return
    print(f"[MathGuard] {msg}", flush=True)


def _normalize_math_text(s: str) -> str:
    return (
        s.replace("\u2212", "-")
        .replace("\u00d7", "*")
        .replace("\u00b7", "*")
        .replace("\u2062", "")
    )


@dataclass
class _IntegralMatch:
    start: int
    end: int
    a: int
    b: int
    integrand: str
    claimed_raw: str


@dataclass
class _DerivativeMatch:
    start: int
    end: int
    poly_text: str
    claimed_raw: str


def _extract_move_type(handoff_context: Dict[str, Any]) -> Optional[str]:
    pc = handoff_context.get("pedagogy_context") if isinstance(handoff_context, dict) else None
    if not isinstance(pc, dict):
        return None
    directives = pc.get("tutor_instruction_directives")
    if not isinstance(directives, dict) or not directives:
        directives = pc.get("tutor_directives")
    if not isinstance(directives, dict):
        return None
    mt = directives.get("selected_move_type")
    return str(mt) if mt is not None else None


def _lazy_sympy() -> Optional[Tuple[Any, Callable[..., Any]]]:
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transformations = standard_transformations + (
            convert_xor,
            implicit_multiplication_application,
        )

        def _parse(s: str, local_dict: Dict[str, Any]) -> Any:
            return parse_expr(s, local_dict=local_dict, transformations=transformations)

        return (sp, _parse)
    except ImportError:
        return None


def _poly_in_scope(sp: Any, expr: Any, x: Any) -> Optional[Any]:
    from sympy import Poly

    try:
        poly = Poly(expr, x, domain="ZZ")
    except Exception:
        return None
    if poly.degree() > _MAX_DEGREE:
        return None
    if len(poly.terms()) > _MAX_TERMS:
        return None
    for _, coeff in poly.terms():
        if not coeff.is_Integer:
            return None
    return poly


def _parse_polynomial(sp: Any, parse: Callable[..., Any], text: str, x: Any) -> Optional[Any]:
    t = text.strip()
    if not t:
        return None
    try:
        expr = parse(t, {"x": x})
    except Exception:
        return None
    return _poly_in_scope(sp, expr, x)


def _parse_claimed_number(raw: str) -> Optional[float]:
    s = raw.strip()
    if not s:
        return None
    if "/" in s:
        parts = s.split("/", 1)
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]) / float(parts[1])
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _format_expected_value(sp: Any, val: Any) -> str:
    if hasattr(val, "is_Rational") and val.is_Rational and val.q != 1:
        return f"{val.p}/{val.q}"
    try:
        f = float(val.evalf())
    except Exception:
        return str(val)
    if abs(f - round(f)) < _TOL:
        return str(int(round(f)))
    return f"{f:.10g}".rstrip("0").rstrip(".")


def _find_integral_candidates(msg: str) -> List[_IntegralMatch]:
    out: List[_IntegralMatch] = []
    # ∫_a^b ... dx = R
    pat1 = re.compile(
        r"∫_\s*\{?(-?\d+)\}?\^\s*\{?(-?\d+)\}?\s*(.+?)\s+d\s*x\s*=\s*"
        r"([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pat1.finditer(msg):
        a, b = int(m.group(1)), int(m.group(2))
        if abs(a) > _MAX_LIMIT or abs(b) > _MAX_LIMIT:
            continue
        out.append(
            _IntegralMatch(
                start=m.start(),
                end=m.end(),
                a=a,
                b=b,
                integrand=m.group(3).strip(),
                claimed_raw=m.group(4),
            )
        )
    pat2 = re.compile(
        r"integral\s+from\s+(-?\d+)\s+to\s+(-?\d+)\s+of\s+(.+?)\s+d\s*x\s*=\s*"
        r"([-+]?\d+(?:\.\d+)?(?:/\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pat2.finditer(msg):
        a, b = int(m.group(1)), int(m.group(2))
        if abs(a) > _MAX_LIMIT or abs(b) > _MAX_LIMIT:
            continue
        out.append(
            _IntegralMatch(
                start=m.start(),
                end=m.end(),
                a=a,
                b=b,
                integrand=m.group(3).strip(),
                claimed_raw=m.group(4),
            )
        )
    return out


def _find_derivative_candidates(msg: str) -> List[_DerivativeMatch]:
    out: List[_DerivativeMatch] = []
    pat = re.compile(
        r"d\s*/\s*d\s*x\s*\(?\s*(.+?)\s*\)?\s*=\s*([^\n]+)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pat.finditer(msg):
        poly_t = m.group(1).strip()
        claimed = m.group(2).strip().rstrip(".")
        if not poly_t or not claimed:
            continue
        out.append(
            _DerivativeMatch(
                start=m.start(),
                end=m.end(),
                poly_text=poly_t,
                claimed_raw=claimed,
            )
        )
    return out


def _unique_replace_integral(
    msg: str,
    span: Tuple[int, int],
    claimed_raw: str,
    replacement_display: str,
) -> Optional[str]:
    """Replace '= claimed' once inside span if unique in full message; else None."""
    lo, hi = span
    segment = msg[lo:hi]
    for old_fragment in (f"={claimed_raw}", f"= {claimed_raw}"):
        pos = segment.find(old_fragment)
        if pos == -1:
            continue
        abs_start = lo + pos
        abs_end = abs_start + len(old_fragment)
        before = msg[:abs_start]
        after = msg[abs_end:]
        candidate = f"{before}={replacement_display}{after}"
        return candidate
    return None


def _verify_and_repair_integral(
    sp: Any,
    parse: Callable[..., Any],
    msg: str,
    cand: _IntegralMatch,
    x: Any,
) -> Tuple[str, Dict[str, Any]]:
    poly = _parse_polynomial(sp, parse, cand.integrand, x)
    outcome: Dict[str, Any] = {
        "guard_ran": True,
        "candidate_type": "integral",
        "verified": None,
        "repaired": False,
        "reason": "",
    }
    if poly is None:
        outcome["reason"] = "scope_or_parse_fail"
        _debug(f"no-op: {outcome['reason']}")
        return msg, outcome

    a, b = cand.a, cand.b
    expr = poly.as_expr()
    try:
        val = sp.integrate(expr, (x, a, b))
        expected_f = float(val.evalf())
    except Exception:
        outcome["reason"] = "integrate_fail"
        _debug(f"no-op: {outcome['reason']}")
        return msg, outcome

    claimed = _parse_claimed_number(cand.claimed_raw)
    if claimed is None:
        outcome["reason"] = "claimed_parse_fail"
        _debug(f"no-op: {outcome['reason']}")
        return msg, outcome

    if abs(claimed - expected_f) <= _TOL:
        outcome["verified"] = True
        outcome["reason"] = "ok"
        _debug("candidate_type=integral verified=True repaired=False reason=ok")
        return msg, outcome

    outcome["verified"] = False
    expected_display = _format_expected_value(sp, val)
    outcome["expected_value"] = expected_display
    outcome["claimed_value"] = cand.claimed_raw

    eq_pat = re.compile(r"=\s*" + re.escape(cand.claimed_raw.strip()))
    ambiguous = len(list(eq_pat.finditer(msg))) > 1
    if ambiguous:
        new_msg = f"{msg}{_NOTE_PREFIX}{expected_display}{_NOTE_SUFFIX}"
        outcome["repaired"] = True
        outcome["reason"] = "neutralize_ambiguous"
        _debug("candidate_type=integral verified=False repaired=True reason=neutralize_ambiguous")
        return new_msg, outcome

    replaced = _unique_replace_integral(msg, (cand.start, cand.end), cand.claimed_raw, expected_display)
    if replaced is not None:
        outcome["repaired"] = True
        outcome["reason"] = "exact_replace"
        _debug("candidate_type=integral verified=False repaired=True reason=exact_replace")
        return replaced, outcome

    new_msg = f"{msg}{_NOTE_PREFIX}{expected_display}{_NOTE_SUFFIX}"
    outcome["repaired"] = True
    outcome["reason"] = "neutralize_fallback"
    _debug("candidate_type=integral verified=False repaired=True reason=neutralize_fallback")
    return new_msg, outcome


def _verify_and_repair_derivative(
    sp: Any,
    parse: Callable[..., Any],
    msg: str,
    cand: _DerivativeMatch,
    x: Any,
) -> Tuple[str, Dict[str, Any]]:
    outcome: Dict[str, Any] = {
        "guard_ran": True,
        "candidate_type": "derivative",
        "verified": None,
        "repaired": False,
        "reason": "",
    }
    poly = _parse_polynomial(sp, parse, cand.poly_text, x)
    claimed_poly = _parse_polynomial(sp, parse, cand.claimed_raw, x)
    if poly is None or claimed_poly is None:
        outcome["reason"] = "scope_or_parse_fail"
        _debug(f"no-op: {outcome['reason']}")
        return msg, outcome

    try:
        expected = sp.diff(poly.as_expr(), x)
        diff = sp.simplify(expected - claimed_poly.as_expr())
    except Exception:
        outcome["reason"] = "diff_fail"
        _debug(f"no-op: {outcome['reason']}")
        return msg, outcome

    if diff == 0:
        outcome["verified"] = True
        outcome["reason"] = "ok"
        _debug("candidate_type=derivative verified=True repaired=False reason=ok")
        return msg, outcome

    outcome["verified"] = False
    expected_str = str(sp.simplify(expected))
    outcome["expected_value"] = expected_str
    outcome["claimed_value"] = cand.claimed_raw
    new_msg = f"{msg}{_NOTE_PREFIX}{expected_str}{_NOTE_SUFFIX}"
    outcome["repaired"] = True
    outcome["reason"] = "neutralize_derivative"
    _debug("candidate_type=derivative verified=False repaired=True reason=neutralize_derivative")
    return new_msg, outcome


def _emit_guard_outcome(
    on_outcome: Optional[Callable[[Dict[str, Any]], None]],
    payload: Dict[str, Any],
) -> None:
    if on_outcome is not None:
        on_outcome(dict(payload))


def maybe_apply_math_example_guard(
    response: Dict[str, Any],
    handoff_context: Dict[str, Any],
    *,
    on_outcome: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Verify simple polynomial integral/derivative claims for worked_example moves.

    If ``on_outcome`` is set, it receives a compact outcome dict whenever the guard
    evaluates a worked_example message (including sympy-missing and no-candidate exits).
    """
    out = dict(response)
    try:
        if _extract_move_type(handoff_context) != "worked_example":
            return out

        msg_raw = out.get("message_to_student") or ""
        if not msg_raw or len(msg_raw) > _MAX_MSG:
            return out

        msg = _normalize_math_text(msg_raw)

        loaded = _lazy_sympy()
        if loaded is None:
            _debug("no-op: sympy_unavailable")
            _emit_guard_outcome(
                on_outcome,
                {
                    "guard_ran": True,
                    "candidate_type": None,
                    "verified": None,
                    "repaired": False,
                    "reason": "sympy_unavailable",
                },
            )
            return out
        sp, parse = loaded
        x = sp.Symbol("x", real=True)

        ints = _find_integral_candidates(msg)
        derivs = _find_derivative_candidates(msg)
        total = len(ints) + len(derivs)
        if total == 0:
            _debug("no-op: no candidate found")
            _emit_guard_outcome(
                on_outcome,
                {
                    "guard_ran": True,
                    "candidate_type": None,
                    "verified": None,
                    "repaired": False,
                    "reason": "no_candidate",
                },
            )
            return out
        if total > 1:
            _debug("no-op: multi_candidate")
            _emit_guard_outcome(
                on_outcome,
                {
                    "guard_ran": True,
                    "candidate_type": None,
                    "verified": None,
                    "repaired": False,
                    "reason": "multi_candidate",
                },
            )
            return out

        if ints:
            new_msg, oc = _verify_and_repair_integral(sp, parse, msg, ints[0], x)
        else:
            new_msg, oc = _verify_and_repair_derivative(sp, parse, msg, derivs[0], x)

        _emit_guard_outcome(on_outcome, oc)

        if new_msg != msg:
            out["message_to_student"] = new_msg
        return out
    except Exception:
        return dict(response)
