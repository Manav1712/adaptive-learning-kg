"""Tests for Round 4 snapshot and debug output enrichment."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.workflow_demo.bot_sessions import BotSessionManager
from src.workflow_demo.pedagogy.heuristic_problem_sequencer import (
    HeuristicProblemSequencer,
)
from src.workflow_demo.pedagogy.observation_filter import HeuristicObservationFilter
from src.workflow_demo.practice.feature_flags import PracticeFeatureFlags
from src.workflow_demo.practice.session import PracticeSessionManager


def _flags(enabled: bool = True) -> PracticeFeatureFlags:
    return PracticeFeatureFlags(
        practice_loop_enabled=enabled,
        adaptive_sequencing_enabled=enabled,
        sequencer_mode="heuristic" if enabled else "off",
    )


def _manager(enabled: bool = True) -> PracticeSessionManager:
    flags = _flags(enabled)
    if enabled:
        obs = HeuristicObservationFilter()
        seq = HeuristicProblemSequencer(observation_filter=obs)
        return PracticeSessionManager(
            flags, problem_sequencer=seq, observation_filter=obs,
        )
    return PracticeSessionManager(flags)


def _run_two_problems(mgr: PracticeSessionManager) -> Dict[str, Any]:
    ext: Dict[str, Any] = {}
    mgr.seed_extensions(ext)
    mgr.begin_practice_problem(ext)
    mgr.record_problem_attempt(ext, submission_text="answer1")
    mgr.finalize_problem_episode(ext, solved=True)
    mgr.begin_practice_problem(ext)
    for i in range(4):
        mgr.record_problem_attempt(ext, submission_text=f"try{i}")
    for _ in range(6):
        mgr.record_problem_chat_turn(ext)
    mgr.finalize_problem_episode(ext, abandoned=True)
    return ext


# ------------------------------------------------------------------
# A. Snapshot includes difficulty history
# ------------------------------------------------------------------

class TestSnapshotDifficultyHistory:
    def test_history_populated_after_problems(self):
        mgr = _manager()
        ext = _run_two_problems(mgr)
        snap = PracticeSessionManager.build_snapshot(ext)
        assert snap is not None
        history = snap["sequencing"]["difficulty_history"]
        assert len(history) == 2

    def test_history_entries_have_required_fields(self):
        mgr = _manager()
        ext = _run_two_problems(mgr)
        snap = PracticeSessionManager.build_snapshot(ext)
        entry = snap["sequencing"]["difficulty_history"][0]
        assert "problem_id" in entry
        assert "prior_difficulty" in entry
        assert "new_difficulty" in entry
        assert "reason" in entry

    def test_history_empty_before_finalize(self):
        mgr = _manager()
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        snap = PracticeSessionManager.build_snapshot(ext)
        assert snap["sequencing"]["difficulty_history"] == []


# ------------------------------------------------------------------
# B. Snapshot includes recent outcomes
# ------------------------------------------------------------------

class TestSnapshotRecentOutcomes:
    def test_outcomes_populated(self):
        mgr = _manager()
        ext = _run_two_problems(mgr)
        snap = PracticeSessionManager.build_snapshot(ext)
        outcomes = snap["sequencing"]["recent_outcomes"]
        assert len(outcomes) == 2
        assert outcomes[0]["solved"] is True
        assert outcomes[1]["abandoned"] is True

    def test_outcome_entries_have_attempt_count(self):
        mgr = _manager()
        ext = _run_two_problems(mgr)
        snap = PracticeSessionManager.build_snapshot(ext)
        assert snap["sequencing"]["recent_outcomes"][0]["attempt_count"] == 1
        assert snap["sequencing"]["recent_outcomes"][1]["attempt_count"] == 4


# ------------------------------------------------------------------
# C. Snapshot includes observation with time_on_problem_sec
# ------------------------------------------------------------------

class TestSnapshotObservation:
    def test_last_observation_includes_time(self):
        mgr = _manager()
        ext = _run_two_problems(mgr)
        snap = PracticeSessionManager.build_snapshot(ext)
        last_obs = snap["sequencing"]["last_observation"]
        assert last_obs is not None
        assert "time_on_problem_sec" in last_obs


# ------------------------------------------------------------------
# D. Snapshot practice_loop_enabled field
# ------------------------------------------------------------------

class TestSnapshotPracticeLoopEnabled:
    def test_enabled_when_active(self):
        mgr = _manager(enabled=True)
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        snap = PracticeSessionManager.build_snapshot(ext)
        assert snap["practice_session"]["practice_loop_enabled"] is True

    def test_none_when_no_practice_session(self):
        snap = PracticeSessionManager.build_snapshot({})
        assert snap is None


# ------------------------------------------------------------------
# E. Disabled practice loop yields null/empty fields
# ------------------------------------------------------------------

class TestDisabledPracticeLoop:
    def test_snapshot_none_when_no_extensions(self):
        snap = PracticeSessionManager.build_snapshot({})
        assert snap is None

    def test_noop_sequencer_still_tracks_history(self):
        """Difficulty history is accumulated by the session manager
        regardless of sequencer type — NoOp just holds difficulty."""
        mgr = _manager(enabled=False)
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        mgr.finalize_problem_episode(ext, solved=True)
        snap = PracticeSessionManager.build_snapshot(ext)
        assert len(snap["sequencing"]["difficulty_history"]) == 1
        entry = snap["sequencing"]["difficulty_history"][0]
        assert entry["prior_difficulty"] == entry["new_difficulty"]


# ------------------------------------------------------------------
# F. !sequencing debug output
# ------------------------------------------------------------------

class TestSequencingDebugOutput:
    def test_includes_practice_loop_enabled(self):
        snap = {
            "practice_session": {
                "active": True,
                "practice_loop_enabled": True,
                "current_problem_id": "p1",
                "current_difficulty": 2,
                "problems_completed": 1,
            },
            "sequencing": {
                "mode": "heuristic",
                "current_difficulty": 2,
                "last_difficulty": 1,
                "step_index": 1,
                "struggle_level": "low",
                "difficulty_reason": "increase by 1",
                "last_observation": {
                    "meaningful_attempts": 1,
                    "raw_attempt_count": 1,
                    "help_turn_count": 0,
                    "time_on_problem_sec": None,
                    "solved": True,
                },
            },
        }
        output = BotSessionManager._format_tutor_sequencing_debug_from_snapshot(snap)
        assert "practice_loop_enabled" in output
        assert "time_on_problem_sec" in output

    def test_includes_reasoning_fields(self):
        snap = {
            "practice_session": {"active": True, "practice_loop_enabled": True},
            "sequencing": {
                "mode": "heuristic",
                "struggle_level": "moderate",
                "difficulty_reason": "hold difficulty",
                "last_observation": {},
            },
        }
        output = BotSessionManager._format_tutor_sequencing_debug_from_snapshot(snap)
        assert "struggle_level" in output
        assert "difficulty_reason" in output


# ------------------------------------------------------------------
# G. !practice debug output
# ------------------------------------------------------------------

class TestPracticeDebugOutput:
    def test_includes_difficulty_history(self):
        snap = {
            "practice_session": {
                "active": True,
                "practice_loop_enabled": True,
                "current_problem_id": "p3",
                "current_difficulty": 2,
                "problems_completed": 2,
            },
            "sequencing": {
                "mode": "heuristic",
                "current_difficulty": 2,
                "last_difficulty": 1,
                "difficulty_history": [
                    {"problem_id": "p1", "prior_difficulty": 1, "new_difficulty": 2, "reason": "low struggle"},
                    {"problem_id": "p2", "prior_difficulty": 2, "new_difficulty": 1, "reason": "abandoned"},
                ],
                "recent_outcomes": [
                    {"problem_id": "p1", "solved": True, "abandoned": False, "attempt_count": 1},
                    {"problem_id": "p2", "solved": False, "abandoned": True, "attempt_count": 4},
                ],
            },
        }
        output = BotSessionManager._format_tutor_practice_debug_from_snapshot(snap)
        assert "Difficulty history" in output
        assert "p1: 1 -> 2" in output
        assert "Recent outcomes" in output
        assert "p2: abandoned" in output

    def test_empty_history(self):
        snap = {
            "practice_session": {"active": False, "practice_loop_enabled": False},
            "sequencing": {},
        }
        output = BotSessionManager._format_tutor_practice_debug_from_snapshot(snap)
        assert "(none)" in output

    def test_practice_command_in_debug_commands(self):
        result = BotSessionManager._parse_tutor_debug_command("!practice")
        assert result == "!practice"
