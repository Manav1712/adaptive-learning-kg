"""Feature-gated math example guard (Phase 7)."""

import pytest

from src.workflow_demo.pedagogy import math_example_guard as meg
from src.workflow_demo.pedagogy.math_example_guard import maybe_apply_math_example_guard


def _worked_example_hc(*, legacy: bool = False) -> dict:
    key = "tutor_directives" if legacy else "tutor_instruction_directives"
    return {
        "pedagogy_context": {
            key: {
                "session_target_lo": "X",
                "instruction_lo": "Y",
                "selected_move_type": "worked_example",
                "retrieval_intent": "teach",
                "retrieval_action": "reuse_pack",
                "policy_reason": "test",
            }
        }
    }


@pytest.mark.unit
def test_math_guard_returns_unchanged_when_directives_missing():
    resp = {"message_to_student": "∫_0^1 x dx = 0.99"}
    out = maybe_apply_math_example_guard(resp, {"pedagogy_context": {}})
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_math_guard_skips_when_not_worked_example():
    resp = {"message_to_student": "∫_0^1 x dx = 0.99"}
    hc = {
        "pedagogy_context": {
            "tutor_directives": {"selected_move_type": "graduated_hint"},
        }
    }
    out = maybe_apply_math_example_guard(resp, hc)
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_math_guard_reads_tutor_instruction_directives_for_move_type():
    resp = {"message_to_student": "∫_0^1 x dx = 0.99"}
    hc = {
        "pedagogy_context": {
            "tutor_instruction_directives": {"selected_move_type": "graduated_hint"},
        }
    }
    out = maybe_apply_math_example_guard(resp, hc)
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_correct_integral_unchanged():
    resp = {"message_to_student": "Thus ∫_0^1 x dx = 1/2."}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_incorrect_integral_repaired(monkeypatch):
    monkeypatch.delenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", raising=False)
    resp = {"message_to_student": "We get ∫_0^1 x dx = 0.99."}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert "0.99" not in out["message_to_student"] or "=1/2" in out["message_to_student"] or "=0.5" in out["message_to_student"]
    assert "0.5" in out["message_to_student"] or "1/2" in out["message_to_student"]


@pytest.mark.unit
def test_correct_polynomial_integral_unchanged():
    resp = {"message_to_student": "So ∫_0^2 (3*x^2 + 1) dx = 10."}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_incorrect_polynomial_integral_repaired():
    resp = {"message_to_student": "So ∫_0^2 (3*x^2 + 1) dx = 12."}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert "=10" in out["message_to_student"].replace(" ", "") or "= 10" in out["message_to_student"]


@pytest.mark.unit
def test_correct_derivative_unchanged():
    resp = {"message_to_student": "Then d/dx (x^3) = 3*x^2."}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_incorrect_derivative_neutralized():
    resp = {"message_to_student": "Then d/dx (x^3) = 2*x^2."}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert "[Note:" in out["message_to_student"]
    assert "3*x**2" in out["message_to_student"] or "3x^2" in out["message_to_student"]


@pytest.mark.unit
def test_trig_integral_no_op():
    resp = {"message_to_student": "∫_0^1 sin(x) dx = 0.46"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_malformed_integral_no_op():
    resp = {"message_to_student": "∫_0^1 x^??? dx = 5"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_sympy_missing_no_op(monkeypatch):
    monkeypatch.setattr(meg, "_lazy_sympy", lambda: None)
    resp = {"message_to_student": "∫_0^1 x dx = 0.99"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_graduated_hint_skipped():
    hc = {
        "pedagogy_context": {
            "tutor_instruction_directives": {
                "session_target_lo": "X",
                "instruction_lo": "Y",
                "selected_move_type": "graduated_hint",
                "retrieval_intent": "teach",
                "retrieval_action": "reuse_pack",
                "policy_reason": "test",
            }
        }
    }
    resp = {"message_to_student": "∫_0^1 x dx = 0.99"}
    out = maybe_apply_math_example_guard(resp, hc)
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_two_integrals_multi_candidate_no_op():
    resp = {
        "message_to_student": "First ∫_0^1 x dx = 1. Second ∫_0^1 x dx = 1/2."
    }
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_ambiguous_duplicate_equals_neutralizes():
    resp = {
        "message_to_student": (
            "We claim ∫_0^2 x dx = 12. "
            "Also elsewhere in this sentence: = 12 means nothing."
        )
    }
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert "[Note:" in out["message_to_student"]


@pytest.mark.unit
def test_empty_message():
    resp = {"message_to_student": ""}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == ""


@pytest.mark.unit
def test_long_message_skipped():
    resp = {"message_to_student": "x" * 12_001}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert len(out["message_to_student"]) == 12_001


@pytest.mark.unit
def test_degree_too_high_no_op():
    resp = {"message_to_student": "∫_0^1 x^7 dx = 1"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_too_many_terms_no_op():
    poly = " + ".join([f"x^{i}" for i in range(7)])
    resp = {"message_to_student": f"∫_0^1 ({poly}) dx = 0"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_legacy_tutor_directives_repair_wrong_integral():
    resp = {"message_to_student": "∫_0^1 x dx = 0.99"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc(legacy=True))
    assert "0.5" in out["message_to_student"] or "1/2" in out["message_to_student"]


@pytest.mark.unit
def test_correct_integral_decimal_form():
    resp = {"message_to_student": "∫_0^1 x dx = 0.5"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_negative_coefficients_correct():
    resp = {"message_to_student": "∫_0^1 (-2*x + 3) dx = 2"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_fractional_coefficient_integrand_no_op():
    resp = {"message_to_student": "∫_0^1 (1/2)*x^2 dx = 0"}
    out = maybe_apply_math_example_guard(resp, _worked_example_hc())
    assert out["message_to_student"] == resp["message_to_student"]


@pytest.mark.unit
def test_idempotent_second_run_no_double_patch():
    resp = {"message_to_student": "∫_0^1 x dx = 0.99"}
    once = maybe_apply_math_example_guard(resp, _worked_example_hc())
    twice = maybe_apply_math_example_guard(dict(once), _worked_example_hc())
    assert twice["message_to_student"] == once["message_to_student"]


@pytest.mark.unit
def test_diagnostic_question_unchanged():
    hc = {
        "pedagogy_context": {
            "tutor_instruction_directives": {
                "session_target_lo": "X",
                "instruction_lo": "Y",
                "selected_move_type": "diagnostic_question",
                "retrieval_intent": "teach",
                "retrieval_action": "reuse_pack",
                "policy_reason": "test",
            }
        }
    }
    resp = {"message_to_student": "∫_0^1 x dx = 0.99"}
    out = maybe_apply_math_example_guard(resp, hc)
    assert out["message_to_student"] == resp["message_to_student"]
