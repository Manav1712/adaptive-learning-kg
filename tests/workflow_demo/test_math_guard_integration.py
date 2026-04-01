"""Integration tests: tutor/FAQ vs math_example_guard."""

import json

import pytest

from src.workflow_demo.tutor import faq_bot, tutor_bot


def _tutor_payload(message: str) -> str:
    return json.dumps(
        {
            "message_to_student": message,
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": ["Integrals"],
                "student_understanding": "good",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "integration",
            },
        }
    )


def _worked_example_handoff(sample_handoff_context):
    h = dict(sample_handoff_context)
    h["pedagogy_context"] = {
        "tutor_instruction_directives": {
            "session_target_lo": "FTC",
            "instruction_lo": "Integrals",
            "selected_move_type": "worked_example",
            "retrieval_intent": "teach_current_concept",
            "retrieval_action": "reuse_pack",
            "policy_reason": "integration_test",
        }
    }
    return h


@pytest.mark.integration
def test_tutor_wrong_integral_guard_fixes_message(
    monkeypatch, mock_openai_client, sample_handoff_context
):
    monkeypatch.setenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", "1")
    raw = "Thus ∫_0^1 x dx = 0.99."
    mock_openai_client.queue_response(_tutor_payload(raw))
    out = tutor_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=_worked_example_handoff(sample_handoff_context),
        conversation_history=[],
    )
    assert out["message_to_student"] != raw
    assert "0.5" in out["message_to_student"] or "1/2" in out["message_to_student"]


@pytest.mark.integration
def test_tutor_correct_integral_unchanged_with_guard(
    monkeypatch, mock_openai_client, sample_handoff_context
):
    monkeypatch.setenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", "1")
    raw = "Thus ∫_0^1 x dx = 1/2."
    mock_openai_client.queue_response(_tutor_payload(raw))
    out = tutor_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=_worked_example_handoff(sample_handoff_context),
        conversation_history=[],
    )
    assert out["message_to_student"] == raw


@pytest.mark.integration
def test_tutor_diagnostic_question_guard_skipped(
    monkeypatch, mock_openai_client, sample_handoff_context
):
    monkeypatch.setenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", "1")
    raw = "∫_0^1 x dx = 0.99"
    mock_openai_client.queue_response(_tutor_payload(raw))
    h = dict(sample_handoff_context)
    h["pedagogy_context"] = {
        "tutor_instruction_directives": {
            "session_target_lo": "FTC",
            "instruction_lo": "Integrals",
            "selected_move_type": "diagnostic_question",
            "retrieval_intent": "teach_current_concept",
            "retrieval_action": "reuse_pack",
            "policy_reason": "integration_test",
        }
    }
    out = tutor_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=h,
        conversation_history=[],
    )
    assert out["message_to_student"] == raw


@pytest.mark.integration
def test_tutor_conceptual_prose_unchanged_with_guard(
    monkeypatch, mock_openai_client, sample_handoff_context
):
    monkeypatch.setenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", "1")
    raw = "Here is the conceptual idea without any integral claim."
    mock_openai_client.queue_response(_tutor_payload(raw))
    out = tutor_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=_worked_example_handoff(sample_handoff_context),
        conversation_history=[],
    )
    assert out["message_to_student"] == raw


@pytest.mark.integration
def test_faq_bot_never_invokes_math_guard(monkeypatch, mock_openai_client, sample_handoff_context):
    monkeypatch.setenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", "1")

    def _should_not_run(*_a, **_k):
        raise AssertionError("math guard must not run for FAQ")

    monkeypatch.setattr(
        "src.workflow_demo.tutor.maybe_apply_math_example_guard",
        _should_not_run,
    )
    mock_openai_client.queue_response(
        json.dumps(
            {
                "message_to_student": "The next exam window runs from May 10-12.",
                "end_activity": True,
                "silent_end": False,
                "needs_topic_confirmation": False,
                "session_summary": {
                    "topics_addressed": ["exam schedule"],
                    "questions_answered": ["When is the next exam?"],
                    "switch_topic_request": None,
                    "notes": "FAQ integration",
                },
            }
        )
    )
    out = faq_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=sample_handoff_context,
        conversation_history=[],
    )
    assert "exam" in out["message_to_student"].lower()
