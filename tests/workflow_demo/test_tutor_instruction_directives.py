"""Phase 6: tutor_instruction_directives prompt and extraction contracts."""

import json

import pytest

from src.workflow_demo.tutor import (
    TUTOR_INSTRUCTION_DIRECTIVE_KEYS,
    TUTOR_SYSTEM_PROMPT,
    extract_tutor_instruction_directives,
    tutor_bot,
)


@pytest.mark.unit
def test_extract_tutor_instruction_directives_prefers_canonical_key():
    pc = {
        "tutor_instruction_directives": {"session_target_lo": "X", "instruction_lo": "Y"},
        "tutor_directives": {"session_target_lo": "OLD"},
    }
    out = extract_tutor_instruction_directives(pc)
    assert out["session_target_lo"] == "X"
    assert out["instruction_lo"] == "Y"
    for k in TUTOR_INSTRUCTION_DIRECTIVE_KEYS:
        assert k in out


@pytest.mark.unit
def test_tutor_system_prompt_contains_move_contracts():
    """Lock move-conditioning prose so it is not accidentally deleted."""
    assert "Move conditioning (mandatory when tutor_instruction_directives is non-empty)" in TUTOR_SYSTEM_PROMPT
    assert "Precedence:" in TUTOR_SYSTEM_PROMPT and "Teaching Flow" in TUTOR_SYSTEM_PROMPT
    assert "When selected_move_type is diagnostic_question" in TUTOR_SYSTEM_PROMPT
    assert "When selected_move_type is graduated_hint" in TUTOR_SYSTEM_PROMPT
    assert "When selected_move_type is worked_example" in TUTOR_SYSTEM_PROMPT
    assert "When selected_move_type is prereq_remediation" in TUTOR_SYSTEM_PROMPT
    assert "prerequisite" in TUTOR_SYSTEM_PROMPT.lower()
    assert "reconnect" in TUTOR_SYSTEM_PROMPT.lower()
    assert "For any other selected_move_type" in TUTOR_SYSTEM_PROMPT


@pytest.mark.unit
@pytest.mark.parametrize(
    "move_type,needle",
    [
        ("diagnostic_question", "diagnostic_question"),
        ("graduated_hint", "graduated_hint"),
        ("worked_example", "worked_example"),
        ("prereq_remediation", "prereq_remediation"),
    ],
)
def test_tutor_payload_echoes_selected_move_type(
    mock_openai_client, sample_tutor_response, sample_handoff_context, move_type, needle
):
    handoff = dict(sample_handoff_context)
    handoff["pedagogy_context"] = {
        "tutor_instruction_directives": {
            "session_target_lo": "S",
            "instruction_lo": "I",
            "selected_move_type": move_type,
            "retrieval_intent": "teach_current_concept",
            "retrieval_action": "reuse_pack",
            "policy_reason": "unit",
        }
    }
    mock_openai_client.queue_response(sample_tutor_response)
    tutor_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=handoff,
        conversation_history=[],
    )
    user_text = mock_openai_client.calls[0]["messages"][1]["content"]
    payload = json.loads(user_text)
    assert payload["tutor_instruction_directives"]["selected_move_type"] == needle
    assert mock_openai_client.calls[0]["messages"][0]["content"] == TUTOR_SYSTEM_PROMPT
    assert needle in TUTOR_SYSTEM_PROMPT
