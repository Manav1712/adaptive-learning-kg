"""Unit tests for tutor_bot and faq_bot helpers."""

import json

import pytest

from src.workflow_demo.tutor import faq_bot, tutor_bot


@pytest.mark.unit
def test_tutor_bot_returns_llm_message(
    mock_openai_client, sample_tutor_response, sample_handoff_context
):
    """Tutor bot should relay the LLM JSON response."""
    mock_openai_client.queue_response(sample_tutor_response)
    result = tutor_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=sample_handoff_context,
        conversation_history=[],
    )
    assert result["message_to_student"] == sample_tutor_response["message_to_student"]
    payload = json.loads(mock_openai_client.calls[0]["messages"][1]["content"])
    assert payload["handoff_context"]["session_params"]["subject"] == "calculus"


@pytest.mark.unit
def test_tutor_bot_handles_latex_braces(mock_openai_client, sample_handoff_context):
    """Malformed math escape sequences should still produce valid JSON."""
    malformed = (
        '{"message_to_student": "Consider \\(f(x)\\) for this example.",'
        '"end_activity": true, "silent_end": false, "needs_mode_confirmation": false,'
        '"needs_topic_confirmation": false, "requested_mode": null,'
        '"session_summary": {"topics_covered": ["Derivatives"],'
        '"student_understanding": "good", "suggested_next_topic": null,'
        '"switch_topic_request": null, "switch_mode_request": null, "notes": ""}}'
    )
    mock_openai_client.queue_response(malformed)
    result = tutor_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=sample_handoff_context,
        conversation_history=[],
    )
    assert result["end_activity"] is True
    assert result["session_summary"]["topics_covered"] == ["Derivatives"]


@pytest.mark.unit
def test_faq_bot_returns_script(mock_openai_client, sample_faq_response, sample_handoff_context):
    """FAQ bot should emit the canned answer payload."""
    mock_openai_client.queue_response(sample_faq_response)
    result = faq_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=sample_handoff_context,
        conversation_history=[],
    )
    assert result["message_to_student"].startswith("The next exam window")
    payload = json.loads(mock_openai_client.calls[0]["messages"][1]["content"])
    assert payload["handoff_context"]["session_params"]["learning_objective"] == \
        sample_handoff_context["session_params"]["learning_objective"]
