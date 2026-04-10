"""Round 2 tests — PracticeSessionManager lifecycle hooks."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.workflow_demo.practice.feature_flags import PracticeFeatureFlags
from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    PracticeSessionState,
    SequencerState,
)
from src.workflow_demo.practice.session import PracticeSessionManager


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_mgr(
    *,
    enabled: bool = True,
    events: List[Dict[str, Any]] | None = None,
) -> PracticeSessionManager:
    flags = PracticeFeatureFlags(practice_loop_enabled=enabled)

    def _capture_event(*args: Any, **kwargs: Any) -> None:
        if events is not None:
            events.append({"args": args, "kwargs": kwargs})

    return PracticeSessionManager(flags, event_emitter=_capture_event if events is not None else None)


def _empty_ext() -> Dict[str, Any]:
    return {}


# ------------------------------------------------------------------
# seed_extensions
# ------------------------------------------------------------------

class TestSeedExtensions:

    def test_seeds_practice_session_and_sequencing(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)

        assert "practice_session" in ext
        assert "sequencing" in ext
        ps = PracticeSessionState.from_dict(ext["practice_session"])
        seq = SequencerState.from_dict(ext["sequencing"])
        assert ps.active is True
        assert seq.mode == "off"

    def test_preserves_existing_extension_keys(self):
        mgr = _make_mgr()
        ext = {"progression": {"steps": [], "active_step_index": 0}}
        mgr.seed_extensions(ext)
        assert "progression" in ext
        assert "practice_session" in ext


# ------------------------------------------------------------------
# begin_practice_problem
# ------------------------------------------------------------------

class TestBeginPracticeProblem:

    def test_serves_problem_and_updates_state(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)

        problem = mgr.begin_practice_problem(ext)
        assert problem is not None
        assert isinstance(problem, PracticeProblemRef)

        ps = PracticeSessionState.from_dict(ext["practice_session"])
        assert ps.current_problem is not None
        assert ps.current_problem.problem_id == problem.problem_id
        assert ps.current_episode_trace is not None

    def test_difficulty_override(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)

        problem = mgr.begin_practice_problem(ext, difficulty_override=3)
        assert problem is not None
        assert problem.difficulty == 3

        seq = SequencerState.from_dict(ext["sequencing"])
        assert seq.current_difficulty == 3

    def test_returns_none_for_impossible_difficulty(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)

        result = mgr.begin_practice_problem(ext, difficulty_override=99)
        assert result is None

    def test_emits_practice_problem_started_event(self):
        events: List[Dict[str, Any]] = []
        mgr = _make_mgr(events=events)
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        assert any("practice_problem_started" in str(e) for e in events)


# ------------------------------------------------------------------
# record_problem_attempt
# ------------------------------------------------------------------

class TestRecordProblemAttempt:

    def test_records_attempt_on_active_episode(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)

        mgr.record_problem_attempt(ext, submission_text="x=2", is_correct=True)
        ps = PracticeSessionState.from_dict(ext["practice_session"])
        assert ps.current_episode_trace is not None
        assert len(ps.current_episode_trace.attempts) == 1
        assert ps.current_episode_trace.attempts[0].is_correct is True

    def test_noop_when_no_active_episode(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.record_problem_attempt(ext, submission_text="x=2")
        ps = PracticeSessionState.from_dict(ext["practice_session"])
        assert ps.current_episode_trace is None


# ------------------------------------------------------------------
# record_problem_chat_turn
# ------------------------------------------------------------------

class TestRecordProblemChatTurn:

    def test_increments_chat_turn_count(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)

        mgr.record_problem_chat_turn(ext)
        mgr.record_problem_chat_turn(ext)
        ps = PracticeSessionState.from_dict(ext["practice_session"])
        assert ps.current_episode_trace is not None
        assert ps.current_episode_trace.chat_turn_count == 2

    def test_noop_when_no_active_episode(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.record_problem_chat_turn(ext)


# ------------------------------------------------------------------
# finalize_problem_episode
# ------------------------------------------------------------------

class TestFinalizeProblemEpisode:

    def test_finalizes_and_moves_to_completed(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)

        ep = mgr.finalize_problem_episode(ext, solved=True)
        assert ep is not None
        assert ep.solved is True
        assert ep.completed_at is not None

        ps = PracticeSessionState.from_dict(ext["practice_session"])
        assert ps.current_problem is None
        assert ps.current_episode_trace is None
        assert len(ps.completed_episodes) == 1
        assert ps.problem_index == 1

    def test_finalize_abandoned(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)

        ep = mgr.finalize_problem_episode(ext, abandoned=True)
        assert ep is not None
        assert ep.abandoned is True
        assert ep.solved is False

    def test_returns_none_when_no_active_episode(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        result = mgr.finalize_problem_episode(ext, solved=True)
        assert result is None

    def test_advances_problem_index_and_step_index(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)

        mgr.begin_practice_problem(ext)
        mgr.finalize_problem_episode(ext, solved=True)

        mgr.begin_practice_problem(ext)
        mgr.finalize_problem_episode(ext, solved=True)

        ps = PracticeSessionState.from_dict(ext["practice_session"])
        seq = SequencerState.from_dict(ext["sequencing"])
        assert ps.problem_index == 2
        assert seq.step_index == 2
        assert len(ps.completed_episodes) == 2

    def test_emits_practice_problem_completed_event(self):
        events: List[Dict[str, Any]] = []
        mgr = _make_mgr(events=events)
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        mgr.finalize_problem_episode(ext, solved=True)
        assert any("practice_problem_completed" in str(e) for e in events)


# ------------------------------------------------------------------
# select_next_problem
# ------------------------------------------------------------------

class TestSelectNextProblem:

    def test_serves_next_problem_after_finalization(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)

        mgr.begin_practice_problem(ext)
        mgr.finalize_problem_episode(ext, solved=True)

        next_p = mgr.select_next_problem(ext)
        assert next_p is not None
        ps = PracticeSessionState.from_dict(ext["practice_session"])
        assert ps.current_problem is not None


# ------------------------------------------------------------------
# build_snapshot
# ------------------------------------------------------------------

class TestBuildSnapshot:

    def test_returns_none_when_no_practice_state(self):
        result = PracticeSessionManager.build_snapshot({})
        assert result is None

    def test_returns_snapshot_when_practice_active(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)

        snap = PracticeSessionManager.build_snapshot(ext)
        assert snap is not None
        assert snap["practice_session"]["active"] is True
        assert snap["practice_session"]["current_problem_id"] is not None
        assert snap["practice_session"]["problems_completed"] == 0
        assert snap["sequencing"]["mode"] == "off"

    def test_snapshot_after_finalization(self):
        mgr = _make_mgr()
        ext = _empty_ext()
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        mgr.finalize_problem_episode(ext, solved=True)

        snap = PracticeSessionManager.build_snapshot(ext)
        assert snap is not None
        assert snap["practice_session"]["current_problem_id"] is None
        assert snap["practice_session"]["problems_completed"] == 1
