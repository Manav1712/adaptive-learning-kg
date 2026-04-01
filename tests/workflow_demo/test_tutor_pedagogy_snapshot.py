"""Phase 8: compact tutor pedagogy snapshot assembly."""

from __future__ import annotations

from src.workflow_demo.pedagogy.tutor_pedagogy_snapshot import build_tutor_pedagogy_snapshot


class _FakeLearnerEngine:
    def build_snapshot(self, session_id):
        return {
            "recent_attempt_count": 0,
            "hint_count": 0,
            "top_mastery": [],
            "recent_misconceptions": {},
        }


def test_snapshot_none_when_not_tutor():
    assert (
        build_tutor_pedagogy_snapshot(
            handoff_context={"pedagogy_context": {}},
            bot_type="faq",
            active_learner_session_id="s1",
            learner_state_engine=_FakeLearnerEngine(),
        )
        is None
    )


def test_snapshot_sparse_first_turn_no_diagnosis():
    snap = build_tutor_pedagogy_snapshot(
        handoff_context={
            "session_params": {"subject": "x"},
            "pedagogy_context": {
                "target_lo": "LO1",
            },
        },
        bot_type="tutor",
        active_learner_session_id="sess-a",
        learner_state_engine=_FakeLearnerEngine(),
    )
    assert snap is not None
    assert snap["session_id"] == "sess-a"
    assert snap["target_lo"] == "LO1"
    assert snap.get("suspected_misconception") is None
    assert "extensions_preview" not in snap


def test_snapshot_caps_candidate_move_types():
    moves = [{"move_type": f"t{i}"} for i in range(10)]
    snap = build_tutor_pedagogy_snapshot(
        handoff_context={
            "pedagogy_context": {
                "target_lo": "T",
                "teaching_moves": moves,
            },
        },
        bot_type="tutor",
        active_learner_session_id="s",
        learner_state_engine=_FakeLearnerEngine(),
    )
    assert len(snap["candidate_move_types"]) <= 4
