"""
Phase 9 acceptance harness: end-to-end tutoring invariants with stubbed tutor LLM.

Scenarios A–F map to the Phase 9 plan. Scenario D is explicitly bounded: it checks
suppress_repeat_diagnostic + non-diagnostic selection for a long, substantive-looking
wrong-answer message (adequate_check_response heuristic), not full numeric-correctness.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from src.workflow_demo.pedagogy.models import MisconceptionDiagnosis
from src.workflow_demo.pedagogy.tutor_pedagogy_snapshot import build_tutor_pedagogy_snapshot


def _tutor_message_payload(msg: str = "ok") -> Dict[str, Any]:
    return {
        "message_to_student": msg,
        "end_activity": False,
        "silent_end": False,
        "needs_mode_confirmation": False,
        "needs_topic_confirmation": False,
        "requested_mode": None,
        "session_summary": {
            "topics_covered": [],
            "student_understanding": "good",
            "suggested_next_topic": None,
            "switch_topic_request": None,
            "switch_mode_request": None,
            "notes": "",
        },
    }


def _patch_tutor_bot(monkeypatch, responses: List[Dict[str, Any]]) -> None:
    def _fake(**_kwargs: Any) -> Dict[str, Any]:
        if not responses:
            raise AssertionError("tutor_bot called more times than expected")
        return responses.pop(0)

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _fake)


def _standard_planner_plan() -> Dict[str, Any]:
    return {
        "status": "complete",
        "plan": {
            "subject": "calculus",
            "mode": "practice",
            "current_plan": [
                {
                    "lo_id": 1,
                    "title": "Derivatives",
                    "proficiency": 0.3,
                    "how_to_teach": "",
                    "why_to_teach": "",
                    "notes": "",
                    "is_primary": True,
                }
            ],
            "future_plan": [],
            "teaching_pack": {
                "key_points": ["k"],
                "examples": [],
                "practice": [],
            },
        },
    }


@pytest.mark.acceptance
@pytest.mark.integration
def test_scenario_a_low_signal_learner_pipeline(coach_agent, monkeypatch):
    """Scenario A: low-confidence diagnosis yields moves + policy + coherent directives."""
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload(), _tutor_message_payload()])

    def _fake_diagnose(self, session_id, user_input, current_focus_lo, learner_state, recent_messages=None):
        return MisconceptionDiagnosis(
            target_lo="Derivatives",
            suspected_misconception="uncertain_or_low_signal",
            confidence=0.35,
            rationale="low signal",
            prerequisite_gap_los=[],
        )

    monkeypatch.setattr(
        "src.workflow_demo.pedagogy.diagnosis.MisconceptionDiagnoser.diagnose_turn",
        _fake_diagnose,
    )

    coach_agent.planner_result = _standard_planner_plan()
    coach_agent.bot_session_manager.begin(
        bot_type="tutor",
        tool_params={"student_request": "help"},
        conversation_summary="A",
    )
    coach_agent.process_turn("idk")

    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert pc.get("diagnosis")
    assert pc.get("teaching_moves")
    pol = pc.get("policy_decision") or {}
    assert pol.get("selected_move", {}).get("move_type") == "diagnostic_question"
    tid = pc.get("tutor_instruction_directives") or {}
    assert tid.get("session_target_lo") == "Derivatives"
    assert tid.get("instruction_lo") == "Derivatives"
    assert tid.get("selected_move_type") == "diagnostic_question"
    assert pc.get("retrieval_intent")
    assert pc.get("retrieval_action")
    assert pc.get("retrieval_execution_mode")


@pytest.mark.acceptance
@pytest.mark.integration
def test_scenario_b_prereq_remediation_target_stable_instruction_shifts(coach_agent, monkeypatch):
    """Scenario B: target_lo stable; instruction_lo shifts to first prerequisite gap."""

    def _fake_diagnose(self, session_id, user_input, current_focus_lo, learner_state, recent_messages=None):
        return MisconceptionDiagnosis(
            target_lo="Derivatives",
            suspected_misconception="prerequisite_gap",
            confidence=0.62,
            rationale="gap",
            prerequisite_gap_los=["Function Notation"],
        )

    monkeypatch.setattr(
        "src.workflow_demo.pedagogy.diagnosis.MisconceptionDiagnoser.diagnose_turn",
        _fake_diagnose,
    )
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload(), _tutor_message_payload()])

    coach_agent.planner_result = _standard_planner_plan()
    coach_agent.bot_session_manager.begin(
        bot_type="tutor",
        tool_params={"student_request": "help"},
        conversation_summary="B",
    )
    coach_agent.process_turn("I forgot function notation before derivatives.")

    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert pc.get("target_lo") == "Derivatives"
    assert pc.get("instruction_lo") == "Function Notation"
    pol = pc.get("policy_decision") or {}
    assert pol.get("selected_move", {}).get("move_type") == "prereq_remediation"
    tid = pc.get("tutor_instruction_directives") or {}
    assert tid.get("session_target_lo") == "Derivatives"
    assert tid.get("instruction_lo") == "Function Notation"


@pytest.mark.acceptance
@pytest.mark.integration
def test_scenario_c_suppress_repeat_diagnostic_after_advance(monkeypatch, coach_agent):
    """Scenario C: after prior diagnostic, advance phrasing suppresses another diagnostic_question."""
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload(), _tutor_message_payload()])

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-acc-c"
    coach_agent.bot_session_manager.handoff_context = {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "integration",
            "mode": "practice",
            "current_plan": [{"title": "integration", "lo_id": 1}],
        },
        "pedagogy_context": {
            "learner_state": {"active_session_id": "sess-acc-c"},
            "target_lo": "integration",
            "retrieval_session": {
                "pack_focus_lo": "integration",
                "pack_revision": 2,
                "last_selected_move_type": "diagnostic_question",
            },
        },
    }
    coach_agent.bot_session_manager.conversation_history = []

    coach_agent.process_turn(
        "assume I know this, let's continue with integration. "
        "The height is vertical and the base is horizontal on the graph, like we said for triangles."
    )
    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert pc.get("turn_progression_signals", {}).get("suppress_repeat_diagnostic") is True
    pol = pc.get("policy_decision") or {}
    assert pol.get("selected_move", {}).get("move_type") != "diagnostic_question"


@pytest.mark.acceptance
@pytest.mark.integration
def test_scenario_d_substantive_wrong_answer_bounded_by_adequate_heuristic(monkeypatch, coach_agent):
    """
    Scenario D (bounded): long wrong-answer text triggers adequate_check_response heuristics,
    not numeric grading. Suppresses repeat diagnostic when prior move was diagnostic_question.
    """
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload(), _tutor_message_payload()])

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-acc-d"
    coach_agent.bot_session_manager.handoff_context = {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "integration",
            "mode": "practice",
            "current_plan": [{"title": "integration", "lo_id": 1}],
        },
        "pedagogy_context": {
            "learner_state": {"active_session_id": "sess-acc-d"},
            "target_lo": "integration",
            "retrieval_session": {
                "pack_focus_lo": "integration",
                "pack_revision": 1,
                "last_selected_move_type": "diagnostic_question",
            },
        },
    }
    coach_agent.bot_session_manager.conversation_history = []

    coach_agent.process_turn(
        "I computed the integral from 0 to 1 of x dx and I got 99/100 because I treated the "
        "triangle area wrong; the base times height should relate to one half x squared evaluated."
    )
    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert pc.get("turn_progression_signals", {}).get("adequate_check_response") is True
    assert pc.get("turn_progression_signals", {}).get("suppress_repeat_diagnostic") is True
    pol = pc.get("policy_decision") or {}
    assert pol.get("selected_move", {}).get("move_type") != "diagnostic_question"


def _handoff_with_prior_diagnostic() -> Dict[str, Any]:
    return {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "integration",
            "mode": "practice",
            "current_plan": [{"title": "integration", "lo_id": 1}],
        },
        "pedagogy_context": {
            "learner_state": {"active_session_id": "sess-diag-loop"},
            "target_lo": "integration",
            "retrieval_session": {
                "pack_focus_lo": "integration",
                "pack_revision": 2,
                "last_selected_move_type": "diagnostic_question",
            },
        },
    }


@pytest.mark.acceptance
@pytest.mark.integration
def test_substantive_math_attempt_suppresses_diagnostic_and_reply_moves_forward(monkeypatch, coach_agent):
    """After diagnostic + substantive attempt: suppress repeat diagnostic; stub reply uses example framing."""
    forward_msg = (
        "For example, let's work through it: if f(x)=x^2 then f'(x)=2x. Consider the limit definition briefly."
    )
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload(forward_msg)])

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-substantive"
    coach_agent.bot_session_manager.handoff_context = _handoff_with_prior_diagnostic()
    coach_agent.bot_session_manager.conversation_history = []

    reply = coach_agent.process_turn("I think it equals 2x")
    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    tps = pc.get("turn_progression_signals") or {}
    assert tps.get("substantive_answer_attempt") is True
    assert tps.get("suppress_repeat_diagnostic") is True
    pol = pc.get("policy_decision") or {}
    assert pol.get("selected_move", {}).get("move_type") != "diagnostic_question"
    lower = reply.lower()
    assert "for example" in lower or "let's work" in lower or "consider" in lower


@pytest.mark.acceptance
@pytest.mark.integration
def test_example_request_selects_worked_example(monkeypatch, coach_agent):
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload("Here is a worked example.")])

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-ex"
    coach_agent.bot_session_manager.handoff_context = _handoff_with_prior_diagnostic()
    coach_agent.bot_session_manager.conversation_history = []

    coach_agent.process_turn("can you give me an example problem instead?")
    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert (pc.get("turn_progression_signals") or {}).get("learner_requested_example") is True
    pol = pc.get("policy_decision") or {}
    assert pol.get("selected_move", {}).get("move_type") == "worked_example"


@pytest.mark.acceptance
@pytest.mark.integration
def test_new_advance_phrases_suppress_repeat_diagnostic(monkeypatch, coach_agent):
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload("ok")])

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-adv"
    coach_agent.bot_session_manager.handoff_context = _handoff_with_prior_diagnostic()
    coach_agent.bot_session_manager.conversation_history = []

    coach_agent.process_turn("let's keep going, I get it")
    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert (pc.get("turn_progression_signals") or {}).get("suppress_repeat_diagnostic") is True


@pytest.mark.acceptance
@pytest.mark.integration
def test_short_ack_after_diagnostic_does_not_suppress(monkeypatch, coach_agent):
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload("ok")])

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-ack"
    coach_agent.bot_session_manager.handoff_context = _handoff_with_prior_diagnostic()
    coach_agent.bot_session_manager.conversation_history = []

    coach_agent.process_turn("yeah")
    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert (pc.get("turn_progression_signals") or {}).get("suppress_repeat_diagnostic") is False


@pytest.mark.acceptance
@pytest.mark.integration
def test_confusion_overrides_example_request_suppress(monkeypatch, coach_agent):
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload("ok")])

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-conf"
    coach_agent.bot_session_manager.handoff_context = _handoff_with_prior_diagnostic()
    coach_agent.bot_session_manager.conversation_history = []

    coach_agent.process_turn("I'm confused but give me an example")
    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert (pc.get("turn_progression_signals") or {}).get("suppress_repeat_diagnostic") is False


@pytest.mark.acceptance
@pytest.mark.integration
def test_scenario_e_math_guard_wrong_integral_repaired(monkeypatch, mock_openai_client, sample_handoff_context):
    """Scenario E: env-gated math guard repairs wrong numeric claim for worked_example."""
    monkeypatch.setenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", "1")
    from src.workflow_demo.tutor import tutor_bot

    h = dict(sample_handoff_context)
    h["pedagogy_context"] = {
        "tutor_instruction_directives": {
            "session_target_lo": "FTC",
            "instruction_lo": "Integrals",
            "selected_move_type": "worked_example",
            "retrieval_intent": "teach_current_concept",
            "retrieval_action": "reuse_pack",
            "policy_reason": "acceptance",
        }
    }
    raw = json.dumps(
        {
            **_tutor_message_payload("Thus ∫_0^1 x dx = 0.99."),
            "message_to_student": "Thus ∫_0^1 x dx = 0.99.",
        }
    )
    mock_openai_client.queue_response(raw)
    out = tutor_bot(
        llm_client=mock_openai_client,
        llm_model="mock-model",
        handoff_context=h,
        conversation_history=[],
    )
    assert "0.5" in out["message_to_student"] or "1/2" in out["message_to_student"]
    assert out["message_to_student"] != "Thus ∫_0^1 x dx = 0.99."


@pytest.mark.acceptance
@pytest.mark.integration
def test_scenario_f_faq_has_no_pedagogy_context(coach_agent, monkeypatch):
    """Scenario F: FAQ handoff does not attach pedagogy_context; API snapshot is absent."""
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload("FAQ line.")])

    coach_agent.planner_result = {
        "status": "complete",
        "plan": {"topic": "exam schedule", "script": "Exams soon.", "first_question": "More?"},
    }
    coach_agent.bot_session_manager.begin(
        bot_type="faq",
        tool_params={"topic": "exam schedule"},
        conversation_summary="FAQ acceptance.",
    )
    hc = coach_agent.bot_session_manager.handoff_context or {}
    assert "pedagogy_context" not in hc
    assert coach_agent.get_pedagogy_snapshot_for_api() is None
    assert coach_agent.tutor_session_active_for_api() is False


@pytest.mark.acceptance
@pytest.mark.integration
def test_sparse_first_tutor_opener_skips_diagnosis_pipeline(coach_agent, monkeypatch):
    """Invariant: initial tutor invoke does not run _run_misconception_diagnosis."""
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload("Welcome.")])

    coach_agent.planner_result = _standard_planner_plan()
    coach_agent.bot_session_manager.begin(
        bot_type="tutor",
        tool_params={"student_request": "start"},
        conversation_summary="sparse",
    )
    pc = coach_agent.bot_session_manager.handoff_context.get("pedagogy_context") or {}
    assert pc.get("policy_decision") is None
    assert pc.get("diagnosis") is None


@pytest.mark.acceptance
@pytest.mark.integration
def test_pedagogy_snapshot_matches_builder_not_events(coach_agent, monkeypatch):
    """Snapshot coherence: last_pedagogy_snapshot == build_tutor_pedagogy_snapshot(...)."""
    _patch_tutor_bot(monkeypatch, [_tutor_message_payload(), _tutor_message_payload()])

    coach_agent.planner_result = _standard_planner_plan()
    coach_agent.bot_session_manager.begin(
        bot_type="tutor",
        tool_params={"student_request": "x"},
        conversation_summary="snap",
    )
    coach_agent.process_turn("What is the derivative of x^2?")

    mgr = coach_agent.bot_session_manager
    built = build_tutor_pedagogy_snapshot(
        handoff_context=mgr.handoff_context,
        bot_type=mgr.bot_type,
        active_learner_session_id=mgr.active_learner_session_id,
        learner_state_engine=coach_agent.learner_state_engine,
    )
    assert built is not None
    assert mgr.last_pedagogy_snapshot == built
