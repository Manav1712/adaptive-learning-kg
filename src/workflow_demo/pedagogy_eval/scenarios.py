"""
Scenario bank: constants and shared payloads for the pedagogy eval harness.
Assertion logic lives in ``harness.py`` (pedagogy_context is the source of truth).
"""

from __future__ import annotations

from typing import Any, Dict, List


def tutor_message_payload(msg: str = "ok") -> Dict[str, Any]:
    """Minimal valid tutor_bot JSON (matches Phase 9 acceptance tests)."""
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


def standard_planner_plan() -> Dict[str, Any]:
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


def handoff_with_prior_diagnostic(session_id: str = "sess-diag-loop") -> Dict[str, Any]:
    return {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "integration",
            "mode": "practice",
            "current_plan": [{"title": "integration", "lo_id": 1}],
        },
        "pedagogy_context": {
            "learner_state": {"active_session_id": session_id},
            "target_lo": "integration",
            "retrieval_session": {
                "pack_focus_lo": "integration",
                "pack_revision": 2,
                "last_selected_move_type": "diagnostic_question",
            },
        },
    }


# Scenario 5: substantive wrong answer (Phase 9 scenario D body)
WRONG_CONCRETE_ANSWER_TEXT = (
    "I computed the integral from 0 to 1 of x dx and I got 99/100 because I treated the "
    "triangle area wrong; the base times height should relate to one half x squared evaluated."
)

# Stubbed tutor reply must acknowledge the attempt and offer corrective scaffolding (not generic "ok").
SCENARIO5_SCAFFOLDING_REPLY = (
    "Not quite — your integral setup is on the right track, but the numeric result is off. "
    "Let's fix that: for ∫_0^1 x dx, the antiderivative is (1/2)x^2, so the value is 1/2, not 99/100."
)

# Substrings for scenario 5 response check (any match passes)
SCENARIO5_RESPONSE_CUES = ("not quite", "let's fix", "your")

# Scenario 2: learner demonstrates understanding (after prior diagnostic)
SCENARIO2_STUDENT_TURN = "I think it equals 2x"
SCENARIO2_STUB_REPLY = (
    "For example, let's work through it: if f(x)=x^2 then f'(x)=2x. Consider the limit definition briefly."
)
SCENARIO2_RESPONSE_CUES = ("for example", "let's work", "consider")

# Scenario 3: explicit advance
SCENARIO3_STUDENT_TURN = (
    "assume I know this, let's continue with integration. "
    "The height is vertical and the base is horizontal on the graph, like we said for triangles."
)

# Scenario 4: example request
SCENARIO4_STUDENT_TURN = "can you give me an example problem instead?"
SCENARIO4_STUB_REPLY = "Here is a worked example."

# Math guard wrong integral tutor output (before repair)
MATH_GUARD_WRONG_INTEGRAL_MESSAGE = "Thus ∫_0^1 x dx = 0.99."
