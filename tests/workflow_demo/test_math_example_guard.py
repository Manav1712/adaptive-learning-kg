"""Feature-gated math example guard."""

import pytest

from src.workflow_demo.pedagogy.math_example_guard import maybe_apply_math_example_guard


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
