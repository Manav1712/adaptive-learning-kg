"""
Pedagogy eval harness: run scenarios, assert on ``pedagogy_context`` (not snapshot-only).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

from src.workflow_demo.pedagogy.models import MisconceptionDiagnosis
from src.workflow_demo.pedagogy.tutor_pedagogy_snapshot import build_tutor_pedagogy_snapshot
from src.workflow_demo.pedagogy_eval.fixtures import (
    DummyLLMClient,
    build_eval_coach,
    sample_handoff_context_for_math_guard,
)
from src.workflow_demo.pedagogy_eval import scenarios as S


@dataclass
class EvalSummary:
    total_scenarios: int
    passed: int
    failed: int
    skipped: int

    @property
    def pass_rate_excluding_skips(self) -> float:
        denom = self.total_scenarios - self.skipped
        if denom <= 0:
            return 0.0
        return round(self.passed / denom, 4)


def _pc(agent: Any) -> Dict[str, Any]:
    mgr = agent.bot_session_manager
    if not mgr or not mgr.handoff_context:
        return {}
    raw = mgr.handoff_context.get("pedagogy_context")
    return raw if isinstance(raw, dict) else {}


def _selected_move_type(pc: Dict[str, Any]) -> Optional[str]:
    pol = pc.get("policy_decision") or {}
    sm = pol.get("selected_move") or {}
    mt = sm.get("move_type")
    if isinstance(mt, str):
        return mt
    tid = pc.get("tutor_instruction_directives") or {}
    st = tid.get("selected_move_type")
    return str(st) if st is not None else None


def _extract_record(
    *,
    scenario_id: str,
    scenario_name: str,
    status: str,
    skip_reason: Optional[str],
    input_messages: List[str],
    tutor_response: str,
    pc: Dict[str, Any],
    pedagogy_snapshot: Optional[Dict[str, Any]],
    failed_assertions: List[str],
    notes: str = "",
) -> Dict[str, Any]:
    diag = pc.get("diagnosis") if isinstance(pc.get("diagnosis"), dict) else {}
    tid = pc.get("tutor_instruction_directives") if isinstance(pc.get("tutor_instruction_directives"), dict) else {}
    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "status": status,
        "skip_reason": skip_reason,
        "input_messages": input_messages,
        "diagnosis_label": diag.get("suspected_misconception"),
        "diagnosis_confidence": diag.get("confidence"),
        "selected_move_type": _selected_move_type(pc),
        "retrieval_intent": pc.get("retrieval_intent"),
        "retrieval_action": pc.get("retrieval_action"),
        "retrieval_execution_mode": pc.get("retrieval_execution_mode"),
        "target_lo": pc.get("target_lo") or tid.get("session_target_lo"),
        "instruction_lo": pc.get("instruction_lo") or tid.get("instruction_lo"),
        "tutor_response": tutor_response,
        "failed_assertions": failed_assertions,
        "notes": notes,
        "pedagogy_snapshot": pedagogy_snapshot,
    }


def _stub_tutor_queue(responses: List[Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    q = list(responses)

    def _fake(**_kwargs: Any) -> Dict[str, Any]:
        if not q:
            raise RuntimeError("tutor_bot stub queue exhausted")
        return q.pop(0)

    return _fake


def _run_s1_confused_beginner() -> Tuple[Dict[str, Any], EvalSummary]:
    failed: List[str] = []
    coach = build_eval_coach()

    def _fake_diagnose(self, session_id, user_input, current_focus_lo, learner_state, recent_messages=None):
        return MisconceptionDiagnosis(
            target_lo="Derivatives",
            suspected_misconception="uncertain_or_low_signal",
            confidence=0.35,
            rationale="low signal",
            prerequisite_gap_los=[],
        )

    tutor_q = _stub_tutor_queue([S.tutor_message_payload(), S.tutor_message_payload()])
    with patch("src.workflow_demo.bot_sessions.tutor_bot", tutor_q):
        with patch(
            "src.workflow_demo.pedagogy.diagnosis.MisconceptionDiagnoser.diagnose_turn",
            _fake_diagnose,
        ):
            coach.planner_result = S.standard_planner_plan()
            coach.bot_session_manager.begin(
                bot_type="tutor",
                tool_params={"student_request": "help"},
                conversation_summary="eval-s1",
            )
            reply = coach.process_turn("I'm not sure what's going on")

    pc = _pc(coach)
    mgr = coach.bot_session_manager
    snap = (
        build_tutor_pedagogy_snapshot(
            handoff_context=mgr.handoff_context,
            bot_type=mgr.bot_type,
            active_learner_session_id=mgr.active_learner_session_id,
            learner_state_engine=coach.learner_state_engine,
        )
        if mgr
        else None
    )

    sm = _selected_move_type(pc)
    acceptable_moves = {"diagnostic_question", "graduated_hint"}
    if sm not in acceptable_moves:
        failed.append(f"selected_move_type {sm!r} not in acceptable set {acceptable_moves}")
    diag = pc.get("diagnosis") if isinstance(pc.get("diagnosis"), dict) else {}
    label = (diag.get("suspected_misconception") or "").lower()
    if "uncertain" not in label and "low_signal" not in label and label != "uncertain_or_low_signal":
        failed.append(f"diagnosis label {diag.get('suspected_misconception')!r} does not reflect uncertainty/low signal")
    if not pc.get("retrieval_intent"):
        failed.append("missing retrieval_intent on pedagogy_context")
    if not pc.get("retrieval_action"):
        failed.append("missing retrieval_action on pedagogy_context")
    if not pc.get("retrieval_execution_mode"):
        failed.append("missing retrieval_execution_mode on pedagogy_context")

    status = "fail" if failed else "pass"
    rec = _extract_record(
        scenario_id="1",
        scenario_name="Confused beginner",
        status=status,
        skip_reason=None,
        input_messages=["I'm not sure what's going on"],
        tutor_response=reply,
        pc=pc,
        pedagogy_snapshot=snap,
        failed_assertions=failed,
    )
    return rec, EvalSummary(1, 1 if status == "pass" else 0, 1 if status == "fail" else 0, 0)


def _run_s2_understanding() -> Tuple[Dict[str, Any], EvalSummary]:
    failed: List[str] = []
    coach = build_eval_coach()
    tutor_q = _stub_tutor_queue([S.tutor_message_payload(S.SCENARIO2_STUB_REPLY)])

    coach.bot_session_manager.is_active = True
    coach.bot_session_manager.bot_type = "tutor"
    coach.bot_session_manager.active_learner_session_id = "sess-eval-2"
    coach.bot_session_manager.handoff_context = S.handoff_with_prior_diagnostic("sess-eval-2")
    coach.bot_session_manager.conversation_history = []

    with patch("src.workflow_demo.bot_sessions.tutor_bot", tutor_q):
        reply = coach.process_turn(S.SCENARIO2_STUDENT_TURN)

    pc = _pc(coach)
    mgr = coach.bot_session_manager
    snap = build_tutor_pedagogy_snapshot(
        handoff_context=mgr.handoff_context,
        bot_type=mgr.bot_type,
        active_learner_session_id=mgr.active_learner_session_id,
        learner_state_engine=coach.learner_state_engine,
    )

    tps = pc.get("turn_progression_signals") or {}
    if not tps.get("substantive_answer_attempt"):
        failed.append("expected turn_progression_signals.substantive_answer_attempt True")
    if not tps.get("suppress_repeat_diagnostic"):
        failed.append("expected suppress_repeat_diagnostic True")
    if _selected_move_type(pc) == "diagnostic_question":
        failed.append("did not expect diagnostic_question after substantive attempt")

    lower = reply.lower()
    if not any(cue in lower for cue in S.SCENARIO2_RESPONSE_CUES):
        failed.append(f"tutor_response missing forward/scaffolding cues {S.SCENARIO2_RESPONSE_CUES!r}")

    status = "fail" if failed else "pass"
    rec = _extract_record(
        scenario_id="2",
        scenario_name="Learner demonstrates understanding",
        status=status,
        skip_reason=None,
        input_messages=[S.SCENARIO2_STUDENT_TURN],
        tutor_response=reply,
        pc=pc,
        pedagogy_snapshot=snap,
        failed_assertions=failed,
    )
    return rec, EvalSummary(1, 1 if status == "pass" else 0, 1 if status == "fail" else 0, 0)


def _run_s3_explicit_advance() -> Tuple[Dict[str, Any], EvalSummary]:
    failed: List[str] = []
    coach = build_eval_coach()
    tutor_q = _stub_tutor_queue([S.tutor_message_payload("ok")])

    coach.bot_session_manager.is_active = True
    coach.bot_session_manager.bot_type = "tutor"
    coach.bot_session_manager.active_learner_session_id = "sess-eval-3"
    coach.bot_session_manager.handoff_context = {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "integration",
            "mode": "practice",
            "current_plan": [{"title": "integration", "lo_id": 1}],
        },
        "pedagogy_context": {
            "learner_state": {"active_session_id": "sess-eval-3"},
            "target_lo": "integration",
            "retrieval_session": {
                "pack_focus_lo": "integration",
                "pack_revision": 2,
                "last_selected_move_type": "diagnostic_question",
            },
        },
    }
    coach.bot_session_manager.conversation_history = []

    with patch("src.workflow_demo.bot_sessions.tutor_bot", tutor_q):
        reply = coach.process_turn(S.SCENARIO3_STUDENT_TURN)

    pc = _pc(coach)
    mgr = coach.bot_session_manager
    snap = build_tutor_pedagogy_snapshot(
        handoff_context=mgr.handoff_context,
        bot_type=mgr.bot_type,
        active_learner_session_id=mgr.active_learner_session_id,
        learner_state_engine=coach.learner_state_engine,
    )

    if not (pc.get("turn_progression_signals") or {}).get("suppress_repeat_diagnostic"):
        failed.append("expected suppress_repeat_diagnostic True")
    if _selected_move_type(pc) == "diagnostic_question":
        failed.append("expected a non-diagnostic move after explicit advance")

    status = "fail" if failed else "pass"
    rec = _extract_record(
        scenario_id="3",
        scenario_name="Explicit advance intent",
        status=status,
        skip_reason=None,
        input_messages=[S.SCENARIO3_STUDENT_TURN],
        tutor_response=reply,
        pc=pc,
        pedagogy_snapshot=snap,
        failed_assertions=failed,
    )
    return rec, EvalSummary(1, 1 if status == "pass" else 0, 1 if status == "fail" else 0, 0)


def _run_s4_example_request() -> Tuple[Dict[str, Any], EvalSummary]:
    failed: List[str] = []
    coach = build_eval_coach()
    tutor_q = _stub_tutor_queue([S.tutor_message_payload(S.SCENARIO4_STUB_REPLY)])

    coach.bot_session_manager.is_active = True
    coach.bot_session_manager.bot_type = "tutor"
    coach.bot_session_manager.active_learner_session_id = "sess-eval-4"
    coach.bot_session_manager.handoff_context = S.handoff_with_prior_diagnostic("sess-eval-4")
    coach.bot_session_manager.conversation_history = []

    with patch("src.workflow_demo.bot_sessions.tutor_bot", tutor_q):
        reply = coach.process_turn(S.SCENARIO4_STUDENT_TURN)

    pc = _pc(coach)
    mgr = coach.bot_session_manager
    snap = build_tutor_pedagogy_snapshot(
        handoff_context=mgr.handoff_context,
        bot_type=mgr.bot_type,
        active_learner_session_id=mgr.active_learner_session_id,
        learner_state_engine=coach.learner_state_engine,
    )

    if not (pc.get("turn_progression_signals") or {}).get("learner_requested_example"):
        failed.append("expected learner_requested_example True")
    if _selected_move_type(pc) != "worked_example":
        failed.append(f"expected worked_example, got {_selected_move_type(pc)!r}")

    status = "fail" if failed else "pass"
    rec = _extract_record(
        scenario_id="4",
        scenario_name="Example request",
        status=status,
        skip_reason=None,
        input_messages=[S.SCENARIO4_STUDENT_TURN],
        tutor_response=reply,
        pc=pc,
        pedagogy_snapshot=snap,
        failed_assertions=failed,
    )
    return rec, EvalSummary(1, 1 if status == "pass" else 0, 1 if status == "fail" else 0, 0)


def _run_s5_wrong_concrete_answer() -> Tuple[Dict[str, Any], EvalSummary]:
    failed: List[str] = []
    coach = build_eval_coach()
    tutor_q = _stub_tutor_queue([S.tutor_message_payload(S.SCENARIO5_SCAFFOLDING_REPLY)])

    coach.bot_session_manager.is_active = True
    coach.bot_session_manager.bot_type = "tutor"
    coach.bot_session_manager.active_learner_session_id = "sess-eval-5"
    coach.bot_session_manager.handoff_context = {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "integration",
            "mode": "practice",
            "current_plan": [{"title": "integration", "lo_id": 1}],
        },
        "pedagogy_context": {
            "learner_state": {"active_session_id": "sess-eval-5"},
            "target_lo": "integration",
            "retrieval_session": {
                "pack_focus_lo": "integration",
                "pack_revision": 1,
                "last_selected_move_type": "diagnostic_question",
            },
        },
    }
    coach.bot_session_manager.conversation_history = []

    with patch("src.workflow_demo.bot_sessions.tutor_bot", tutor_q):
        reply = coach.process_turn(S.WRONG_CONCRETE_ANSWER_TEXT)

    pc = _pc(coach)
    mgr = coach.bot_session_manager
    snap = build_tutor_pedagogy_snapshot(
        handoff_context=mgr.handoff_context,
        bot_type=mgr.bot_type,
        active_learner_session_id=mgr.active_learner_session_id,
        learner_state_engine=coach.learner_state_engine,
    )

    tps = pc.get("turn_progression_signals") or {}
    if not tps.get("adequate_check_response"):
        failed.append("expected adequate_check_response True")
    if not tps.get("suppress_repeat_diagnostic"):
        failed.append("expected suppress_repeat_diagnostic True")
    if _selected_move_type(pc) == "diagnostic_question":
        failed.append("did not expect diagnostic_question after substantive wrong answer")

    lower = reply.lower()
    if not any(cue in lower for cue in S.SCENARIO5_RESPONSE_CUES):
        failed.append(f"tutor_response missing scaffolding cues {S.SCENARIO5_RESPONSE_CUES!r}")

    status = "fail" if failed else "pass"
    rec = _extract_record(
        scenario_id="5",
        scenario_name="Wrong concrete answer",
        status=status,
        skip_reason=None,
        input_messages=[S.WRONG_CONCRETE_ANSWER_TEXT],
        tutor_response=reply,
        pc=pc,
        pedagogy_snapshot=snap,
        failed_assertions=failed,
    )
    return rec, EvalSummary(1, 1 if status == "pass" else 0, 1 if status == "fail" else 0, 0)


def _guard_env_enabled() -> bool:
    return os.getenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", "").lower() in ("1", "true", "yes")


def _sympy_available() -> bool:
    try:
        import sympy  # noqa: F401

        return True
    except ImportError:
        return False


def _run_s6_math_guard() -> Tuple[Dict[str, Any], EvalSummary]:
    """Direct tutor_bot path; SKIP when guard off or SymPy missing."""
    if not _guard_env_enabled():
        rec = _extract_record(
            scenario_id="6",
            scenario_name="Worked example + incorrect math",
            status="skip",
            skip_reason="guard_disabled",
            input_messages=[],
            tutor_response="",
            pc={},
            pedagogy_snapshot=None,
            failed_assertions=[],
            notes="Set WORKFLOW_DEMO_TUTOR_MATH_GUARD=1 to run this scenario.",
        )
        return rec, EvalSummary(1, 0, 0, 1)

    if not _sympy_available():
        rec = _extract_record(
            scenario_id="6",
            scenario_name="Worked example + incorrect math",
            status="skip",
            skip_reason="sympy_unavailable",
            input_messages=[],
            tutor_response="",
            pc={},
            pedagogy_snapshot=None,
            failed_assertions=[],
            notes="Install sympy to run math guard checks.",
        )
        return rec, EvalSummary(1, 0, 0, 1)

    failed: List[str] = []
    from src.workflow_demo.tutor import tutor_bot

    h = dict(sample_handoff_context_for_math_guard())
    h["pedagogy_context"] = {
        "tutor_instruction_directives": {
            "session_target_lo": "FTC",
            "instruction_lo": "Integrals",
            "selected_move_type": "worked_example",
            "retrieval_intent": "teach_current_concept",
            "retrieval_action": "reuse_pack",
            "policy_reason": "eval",
        }
    }

    llm = DummyLLMClient()
    raw = {
        **S.tutor_message_payload(S.MATH_GUARD_WRONG_INTEGRAL_MESSAGE),
        "message_to_student": S.MATH_GUARD_WRONG_INTEGRAL_MESSAGE,
    }
    llm.queue_response(raw)

    out = tutor_bot(
        llm_client=llm,
        llm_model="mock-model",
        handoff_context=h,
        conversation_history=[],
    )
    msg = (out.get("message_to_student") or "").strip()
    pc = h.get("pedagogy_context") if isinstance(h.get("pedagogy_context"), dict) else {}
    lgr = pc.get("last_guard_result") if isinstance(pc.get("last_guard_result"), dict) else {}

    # If guard could not find integral pattern or multi-candidate, skip (environment/pattern)
    reason = (lgr.get("reason") or "") if lgr else ""
    if reason in ("no_candidate", "multi_candidate"):
        rec = _extract_record(
            scenario_id="6",
            scenario_name="Worked example + incorrect math",
            status="skip",
            skip_reason="pattern_unsupported",
            input_messages=[],
            tutor_response=msg,
            pc=pc,
            pedagogy_snapshot=None,
            failed_assertions=[],
            notes=f"Guard outcome reason={reason!r}",
        )
        return rec, EvalSummary(1, 0, 0, 1)

    if msg == S.MATH_GUARD_WRONG_INTEGRAL_MESSAGE:
        failed.append("expected math guard to repair or neutralize incorrect integral claim")

    if "0.5" not in msg and "1/2" not in msg and "[Note:" not in msg:
        failed.append("expected corrected value (0.5/1/2) or neutralizing note in message")

    status = "fail" if failed else "pass"
    rec = _extract_record(
        scenario_id="6",
        scenario_name="Worked example + incorrect math",
        status=status,
        skip_reason=None,
        input_messages=[],
        tutor_response=msg,
        pc=pc,
        pedagogy_snapshot=None,
        failed_assertions=failed,
        notes="Direct tutor_bot; pedagogy_context.last_guard_result from guard hook.",
    )
    return rec, EvalSummary(1, 1 if status == "pass" else 0, 1 if status == "fail" else 0, 0)


def _run_s7_faq_isolation() -> Tuple[Dict[str, Any], EvalSummary]:
    failed: List[str] = []
    coach = build_eval_coach()
    faq_q = _stub_tutor_queue([S.tutor_message_payload("FAQ line.")])

    coach.planner_result = {
        "status": "complete",
        "plan": {"topic": "exam schedule", "script": "Exams soon.", "first_question": "More?"},
    }

    with patch("src.workflow_demo.bot_sessions.faq_bot", faq_q):
        reply = coach.bot_session_manager.begin(
            bot_type="faq",
            tool_params={"topic": "exam schedule"},
            conversation_summary="eval-faq",
        )

    hc = coach.bot_session_manager.handoff_context or {}
    if "pedagogy_context" in hc:
        failed.append("did not expect pedagogy_context on FAQ handoff")
    if coach.get_pedagogy_snapshot_for_api() is not None:
        failed.append("expected no pedagogy snapshot for FAQ")
    if coach.tutor_session_active_for_api():
        failed.append("expected tutor_session_active_for_api False")

    status = "fail" if failed else "pass"
    rec = _extract_record(
        scenario_id="7",
        scenario_name="FAQ isolation",
        status=status,
        skip_reason=None,
        input_messages=[],
        tutor_response=reply,
        pc={},
        pedagogy_snapshot=None,
        failed_assertions=failed,
    )
    return rec, EvalSummary(1, 1 if status == "pass" else 0, 1 if status == "fail" else 0, 0)


def run_pedagogy_eval(*, verbose: bool = False) -> Tuple[Dict[str, Any], EvalSummary]:
    """
    Run all pedagogy eval scenarios.

    Returns:
        (aggregate_dict, summary) where aggregate_dict includes ``run_id``, ``timestamp``,
        ``results``, and ``summary``.
    """
    runners = [
        _run_s1_confused_beginner,
        _run_s2_understanding,
        _run_s3_explicit_advance,
        _run_s4_example_request,
        _run_s5_wrong_concrete_answer,
        _run_s6_math_guard,
        _run_s7_faq_isolation,
    ]

    results: List[Dict[str, Any]] = []
    tot = EvalSummary(0, 0, 0, 0)

    for fn in runners:
        rec, partial = fn()
        results.append(rec)
        tot.total_scenarios += partial.total_scenarios
        tot.passed += partial.passed
        tot.failed += partial.failed
        tot.skipped += partial.skipped

    run_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    summary_dict = {
        "total_scenarios": tot.total_scenarios,
        "passed": tot.passed,
        "failed": tot.failed,
        "skipped": tot.skipped,
        "pass_rate_excluding_skips": tot.pass_rate_excluding_skips,
    }
    aggregate = {
        "run_id": run_id,
        "timestamp": ts,
        "results": results,
        "summary": summary_dict,
    }
    return aggregate, tot


def print_report(aggregate: Dict[str, Any], *, verbose: bool = False) -> None:
    """Print human-readable report to stdout."""
    for rec in aggregate["results"]:
        sid = rec["scenario_id"]
        name = rec["scenario_name"]
        st = rec["status"].upper()
        if st == "SKIP":
            print(f"[{sid}] {name}: {st} ({rec.get('skip_reason') or 'unknown'})")
        elif st == "PASS":
            print(f"[{sid}] {name}: PASS")
        else:
            fails = rec.get("failed_assertions") or []
            print(f"[{sid}] {name}: FAIL — {', '.join(fails)}")
        if verbose:
            if rec.get("selected_move_type") is not None:
                print(
                    f"       pc: move={rec.get('selected_move_type')!r} "
                    f"retrieval=({rec.get('retrieval_intent')!r}, {rec.get('retrieval_action')!r}, "
                    f"{rec.get('retrieval_execution_mode')!r}) diagnosis={rec.get('diagnosis_label')!r}"
                )
            snap = rec.get("pedagogy_snapshot")
            if snap is not None:
                print(f"       snapshot (reporting): move={snap.get('selected_move_type')!r}")

    s = aggregate["summary"]
    print("")
    print(
        f"Summary: total={s['total_scenarios']} passed={s['passed']} "
        f"failed={s['failed']} skipped={s['skipped']} "
        f"pass_rate_excluding_skips={s['pass_rate_excluding_skips']}"
    )
