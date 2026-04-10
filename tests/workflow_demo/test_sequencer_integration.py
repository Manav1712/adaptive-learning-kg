"""Integration tests for the problem-boundary flow and snapshot enrichment (Round 3).

Tests cover:
- Explicit completion utterances trigger finalize + resequence + serve-next
- Ordinary tutor turns do NOT trigger finalization
- Snapshot / debug output contains observation and difficulty reason
- Feature-flag gating (no resequencing when flags are off)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from src.workflow_demo.bot_sessions import (
    BotSessionManager,
    _PROBLEM_COMPLETE_PATTERNS,
)
from src.workflow_demo.pedagogy.heuristic_problem_sequencer import (
    HeuristicProblemSequencer,
)
from src.workflow_demo.pedagogy.observation_filter import HeuristicObservationFilter
from src.workflow_demo.practice.feature_flags import PracticeFeatureFlags
from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    ProblemAttempt,
    ProblemEpisodeTrace,
    SequencerState,
)
from src.workflow_demo.practice.session import PracticeSessionManager


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _flags(enabled: bool = True) -> PracticeFeatureFlags:
    return PracticeFeatureFlags(
        practice_loop_enabled=enabled,
        adaptive_sequencing_enabled=enabled,
        sequencer_mode="heuristic" if enabled else "off",
    )


def _flags_off() -> PracticeFeatureFlags:
    return PracticeFeatureFlags()


def _ref(**kw) -> PracticeProblemRef:
    defaults = dict(problem_id="p1", difficulty=1, prompt_text="What is 2+2?")
    return PracticeProblemRef(**(defaults | kw))


def _seeded_extensions(mgr: PracticeSessionManager) -> Dict[str, Any]:
    ext: Dict[str, Any] = {}
    mgr.seed_extensions(ext)
    return ext


# ------------------------------------------------------------------
# A: Completion utterance detection
# ------------------------------------------------------------------

class TestCompletionUtteranceDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "next problem",
            "Next Problem",
            "done",
            "Done!",
            "move on",
            "Move On.",
            "I solved it",
            "I solved it!",
            "next",
            "Next!",
            "I'm done",
        ],
    )
    def test_recognized_as_completion(self, text: str):
        result = BotSessionManager._is_problem_complete_utterance(text)
        assert result is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Can you help me with the next part?",
            "I'm stuck, what should I do next?",
            "What is the derivative?",
            "Hello!",
            "",
            "   ",
            "next problem please help",
        ],
    )
    def test_not_recognized_as_completion(self, text: str):
        result = BotSessionManager._is_problem_complete_utterance(text)
        assert result is None

    def test_solved_returns_true(self):
        assert BotSessionManager._is_problem_complete_utterance("I solved it") is True

    def test_move_on_returns_false(self):
        assert BotSessionManager._is_problem_complete_utterance("move on") is False

    def test_done_returns_false(self):
        assert BotSessionManager._is_problem_complete_utterance("done") is False


# ------------------------------------------------------------------
# B: Finalize-resequence-serve flow via PracticeSessionManager
# ------------------------------------------------------------------

class TestFinalizeResequenceServe:
    def test_low_struggle_solve_increases_difficulty(self):
        flags = _flags()
        obs_filter = HeuristicObservationFilter()
        sequencer = HeuristicProblemSequencer(observation_filter=obs_filter)
        events = []
        mgr = PracticeSessionManager(
            flags,
            problem_sequencer=sequencer,
            observation_filter=obs_filter,
            event_emitter=lambda *a, **kw: events.append((a, kw)),
        )
        ext = _seeded_extensions(mgr)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_attempt(ext, submission_text="x=1", is_correct=True)
        mgr.finalize_problem_episode(ext, solved=True)

        seq_raw = ext.get("sequencing", {})
        assert seq_raw["current_difficulty"] == 2
        assert seq_raw["last_difficulty"] == 1

    def test_abandoned_decreases_difficulty(self):
        flags = _flags()
        obs_filter = HeuristicObservationFilter()
        sequencer = HeuristicProblemSequencer(
            default_difficulty=2, observation_filter=obs_filter,
        )
        mgr = PracticeSessionManager(
            flags,
            problem_sequencer=sequencer,
            observation_filter=obs_filter,
        )
        ext = _seeded_extensions(mgr)
        mgr.begin_practice_problem(ext)
        for i in range(5):
            mgr.record_problem_attempt(ext, submission_text=f"try{i}")
        for _ in range(8):
            mgr.record_problem_chat_turn(ext)
        mgr.finalize_problem_episode(ext, abandoned=True)

        assert ext["sequencing"]["current_difficulty"] == 1

    def test_serve_next_after_finalize(self):
        flags = _flags()
        obs_filter = HeuristicObservationFilter()
        sequencer = HeuristicProblemSequencer(observation_filter=obs_filter)
        mgr = PracticeSessionManager(
            flags,
            problem_sequencer=sequencer,
            observation_filter=obs_filter,
        )
        ext = _seeded_extensions(mgr)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_attempt(ext, submission_text="answer", is_correct=True)
        mgr.finalize_problem_episode(ext, solved=True)
        assert ext["practice_session"]["problem_index"] == 1

        next_prob = mgr.select_next_problem(ext)
        assert next_prob is not None
        assert ext["practice_session"]["current_problem"] is not None


# ------------------------------------------------------------------
# C: Ordinary turns do NOT finalize
# ------------------------------------------------------------------

class TestOrdinaryTurnsNoFinalize:
    def test_chat_turns_dont_finalize(self):
        flags = _flags()
        obs_filter = HeuristicObservationFilter()
        sequencer = HeuristicProblemSequencer(observation_filter=obs_filter)
        mgr = PracticeSessionManager(
            flags,
            problem_sequencer=sequencer,
            observation_filter=obs_filter,
        )
        ext = _seeded_extensions(mgr)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_chat_turn(ext)
        mgr.record_problem_chat_turn(ext)
        ps = ext["practice_session"]
        assert ps["current_episode_trace"] is not None
        assert ps["problem_index"] == 0


# ------------------------------------------------------------------
# D: Snapshot / debug contains observation + difficulty reason
# ------------------------------------------------------------------

class TestSnapshotEnrichment:
    def test_snapshot_after_finalize(self):
        flags = _flags()
        obs_filter = HeuristicObservationFilter()
        sequencer = HeuristicProblemSequencer(observation_filter=obs_filter)
        mgr = PracticeSessionManager(
            flags,
            problem_sequencer=sequencer,
            observation_filter=obs_filter,
        )
        ext = _seeded_extensions(mgr)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_attempt(ext, submission_text="4", is_correct=True)
        mgr.finalize_problem_episode(ext, solved=True)

        snap = PracticeSessionManager.build_snapshot(ext)
        assert snap is not None
        seq_snap = snap["sequencing"]
        assert seq_snap["difficulty_reason"] is not None
        assert seq_snap["struggle_level"] is not None
        obs_snap = seq_snap.get("last_observation")
        assert obs_snap is not None
        assert "meaningful_attempts" in obs_snap
        assert "solved" in obs_snap

    def test_snapshot_before_finalize_has_no_observation(self):
        flags = _flags()
        obs_filter = HeuristicObservationFilter()
        sequencer = HeuristicProblemSequencer(observation_filter=obs_filter)
        mgr = PracticeSessionManager(
            flags,
            problem_sequencer=sequencer,
            observation_filter=obs_filter,
        )
        ext = _seeded_extensions(mgr)
        mgr.begin_practice_problem(ext)
        snap = PracticeSessionManager.build_snapshot(ext)
        assert snap is not None
        seq_snap = snap["sequencing"]
        assert seq_snap["last_observation"] is None


# ------------------------------------------------------------------
# E: Feature-flag gating
# ------------------------------------------------------------------

class TestFeatureFlagGating:
    def test_no_resequencing_when_flags_off(self):
        flags = _flags_off()
        mgr = PracticeSessionManager(flags)
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        seq_state = ext.get("sequencing", {})
        assert seq_state.get("mode") == "off"

    def test_no_adaptive_with_noop_sequencer(self):
        flags = PracticeFeatureFlags(practice_loop_enabled=True)
        mgr = PracticeSessionManager(flags)
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        assert ext["sequencing"]["current_difficulty"] == 1


# ------------------------------------------------------------------
# F: Debug command output format
# ------------------------------------------------------------------

class TestDebugFormat:
    def test_debug_output_contains_observation_fields(self):
        snap = {
            "practice_session": {
                "active": True,
                "current_problem_id": "p2",
                "current_difficulty": 2,
                "problems_completed": 1,
            },
            "sequencing": {
                "mode": "heuristic",
                "current_difficulty": 2,
                "last_difficulty": 1,
                "step_index": 1,
                "struggle_level": "low",
                "difficulty_reason": "struggle=low solved=True meaningful=1 help_turns=0 -> increase by 1",
                "last_observation": {
                    "meaningful_attempts": 1,
                    "raw_attempt_count": 1,
                    "help_turn_count": 0,
                    "solved": True,
                },
            },
        }
        output = BotSessionManager._format_tutor_sequencing_debug_from_snapshot(snap)
        assert "struggle_level" in output
        assert "difficulty_reason" in output
        assert "meaningful_attempts" in output
        assert "last_difficulty" in output


# ------------------------------------------------------------------
# G: Observation in event emission
# ------------------------------------------------------------------

class TestEventEmission:
    def test_finalize_emits_observation_and_difficulty(self):
        flags = _flags()
        obs_filter = HeuristicObservationFilter()
        sequencer = HeuristicProblemSequencer(observation_filter=obs_filter)
        events = []

        def capture(*args, **kwargs):
            events.append({"args": args, "kwargs": kwargs})

        mgr = PracticeSessionManager(
            flags,
            problem_sequencer=sequencer,
            observation_filter=obs_filter,
            event_emitter=capture,
        )
        ext = _seeded_extensions(mgr)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_attempt(ext, submission_text="4", is_correct=True)
        mgr.finalize_problem_episode(ext, solved=True)

        completed_events = [
            e for e in events if e["args"][0] == "practice_problem_completed"
        ]
        assert len(completed_events) == 1
        ev = completed_events[0]
        assert ev["kwargs"]["observation"] is not None
        assert ev["kwargs"]["current_difficulty"] == 2
        assert ev["kwargs"]["last_difficulty"] == 1
        assert ev["kwargs"]["difficulty_reason"] is not None
