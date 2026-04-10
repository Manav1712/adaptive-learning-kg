"""Tests for the replay artifact model and offline evaluator (Round 4)."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

import pytest

from src.workflow_demo.pedagogy.heuristic_problem_sequencer import (
    HeuristicProblemSequencer,
)
from src.workflow_demo.pedagogy.observation_filter import HeuristicObservationFilter
from src.workflow_demo.practice.export import export_practice_session
from src.workflow_demo.practice.feature_flags import PracticeFeatureFlags
from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    ProblemAttempt,
    ProblemEpisodeTrace,
)
from src.workflow_demo.practice.problem_sequencer import NoOpProblemSequencer
from src.workflow_demo.practice.replay import (
    EpisodeReplayRecord,
    PracticeReplayArtifact,
    ReplayDecision,
    replay_episodes,
)
from src.workflow_demo.practice.session import PracticeSessionManager


def _ref(**kw) -> PracticeProblemRef:
    defaults = dict(problem_id="p1", difficulty=1, prompt_text="Solve x+1=2")
    return PracticeProblemRef(**(defaults | kw))


def _attempt(text: str = "x=1") -> ProblemAttempt:
    return ProblemAttempt(attempt_index=0, submission_text=text)


def _episode(
    pid: str = "p1",
    difficulty: int = 1,
    *,
    solved: bool = False,
    abandoned: bool = False,
    attempts: int = 1,
    chat_turns: int = 0,
) -> ProblemEpisodeTrace:
    ep = ProblemEpisodeTrace(
        problem=_ref(problem_id=pid, difficulty=difficulty),
        attempts=[_attempt(f"try{i}") for i in range(attempts)],
        chat_turn_count=chat_turns,
        solved=solved,
        abandoned=abandoned,
    )
    return ep


# ------------------------------------------------------------------
# A. EpisodeReplayRecord serialization
# ------------------------------------------------------------------

class TestEpisodeReplayRecordSerialization:
    def test_round_trip(self):
        rec = EpisodeReplayRecord(
            episode_index=0,
            problem_id="p1",
            lo_id="lo1",
            difficulty_at_serve=2,
            solved=True,
            attempt_count=3,
        )
        d = rec.to_dict()
        restored = EpisodeReplayRecord.from_dict(d)
        assert restored.problem_id == "p1"
        assert restored.lo_id == "lo1"
        assert restored.difficulty_at_serve == 2
        assert restored.solved is True
        assert restored.attempt_count == 3

    def test_preserves_optional_fields(self):
        rec = EpisodeReplayRecord(
            episode_index=1,
            problem_id="p2",
            observation={"meaningful_attempts": 2},
            sequencer_state_before={"mode": "heuristic"},
            struggle_level="low",
            difficulty_reason="easy solve",
        )
        d = rec.to_dict()
        restored = EpisodeReplayRecord.from_dict(d)
        assert restored.observation == {"meaningful_attempts": 2}
        assert restored.struggle_level == "low"
        assert restored.difficulty_reason == "easy solve"


# ------------------------------------------------------------------
# B. PracticeReplayArtifact serialization
# ------------------------------------------------------------------

class TestPracticeReplayArtifactSerialization:
    def test_round_trip(self):
        art = PracticeReplayArtifact(
            session_id="s1",
            learner_session_id="ls1",
            sequencer_mode="heuristic",
            initial_difficulty=1,
            episodes=[
                EpisodeReplayRecord(episode_index=0, problem_id="p1", solved=True),
                EpisodeReplayRecord(episode_index=1, problem_id="p2", abandoned=True),
            ],
        )
        d = art.to_dict()
        restored = PracticeReplayArtifact.from_dict(d)
        assert restored.session_id == "s1"
        assert len(restored.episodes) == 2
        assert restored.episodes[0].problem_id == "p1"
        assert restored.episodes[1].abandoned is True

    def test_preserves_ordering(self):
        episodes = [
            EpisodeReplayRecord(episode_index=i, problem_id=f"p{i}")
            for i in range(5)
        ]
        art = PracticeReplayArtifact(episodes=episodes)
        d = art.to_dict()
        restored = PracticeReplayArtifact.from_dict(d)
        for i, ep in enumerate(restored.episodes):
            assert ep.episode_index == i
            assert ep.problem_id == f"p{i}"

    def test_includes_required_fields(self):
        art = PracticeReplayArtifact(
            session_id="s1",
            sequencer_mode="heuristic",
            initial_difficulty=2,
        )
        d = art.to_dict()
        assert "session_id" in d
        assert "sequencer_mode" in d
        assert "initial_difficulty" in d
        assert "episodes" in d
        assert "metadata" in d

    def test_json_serializable(self):
        art = PracticeReplayArtifact(
            session_id="s1",
            episodes=[
                EpisodeReplayRecord(
                    episode_index=0,
                    problem_id="p1",
                    observation={"meaningful_attempts": 1},
                ),
            ],
        )
        text = json.dumps(art.to_dict())
        assert isinstance(text, str)
        parsed = json.loads(text)
        assert parsed["episodes"][0]["problem_id"] == "p1"


# ------------------------------------------------------------------
# C. PracticeReplayArtifact.from_extensions
# ------------------------------------------------------------------

class TestFromExtensions:
    def _run_session(self) -> Dict[str, Any]:
        flags = PracticeFeatureFlags(
            practice_loop_enabled=True,
            adaptive_sequencing_enabled=True,
            sequencer_mode="heuristic",
        )
        obs = HeuristicObservationFilter()
        seq = HeuristicProblemSequencer(observation_filter=obs)
        mgr = PracticeSessionManager(
            flags, problem_sequencer=seq, observation_filter=obs,
        )
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_attempt(ext, submission_text="x=1")
        mgr.finalize_problem_episode(ext, solved=True)
        mgr.begin_practice_problem(ext)
        mgr.finalize_problem_episode(ext, abandoned=True)
        return ext

    def test_from_live_extensions(self):
        ext = self._run_session()
        art = PracticeReplayArtifact.from_extensions(
            ext, session_id="s1",
        )
        assert len(art.episodes) == 2
        assert art.episodes[0].solved is True
        assert art.episodes[1].abandoned is True

    def test_with_observation_filter(self):
        ext = self._run_session()
        obs = HeuristicObservationFilter()
        art = PracticeReplayArtifact.from_extensions(
            ext, observation_filter=obs,
        )
        assert art.episodes[0].observation is not None
        assert "meaningful_attempts" in art.episodes[0].observation


# ------------------------------------------------------------------
# D. Offline replay with NoOpProblemSequencer
# ------------------------------------------------------------------

class TestReplayNoOp:
    def test_all_decisions_hold(self):
        episodes = [
            _episode("p1", 1, solved=True),
            _episode("p2", 1, abandoned=True),
            _episode("p3", 1, solved=True, attempts=3),
        ]
        decisions = replay_episodes(episodes, NoOpProblemSequencer())
        assert len(decisions) == 3
        for d in decisions:
            assert d.difficulty_before == 1
            assert d.difficulty_after == 1

    def test_deterministic(self):
        episodes = [_episode("p1", 1, solved=True)]
        d1 = replay_episodes(episodes, NoOpProblemSequencer())
        d2 = replay_episodes(episodes, NoOpProblemSequencer())
        assert d1[0].difficulty_after == d2[0].difficulty_after


# ------------------------------------------------------------------
# E. Offline replay with HeuristicProblemSequencer
# ------------------------------------------------------------------

class TestReplayHeuristic:
    def test_low_struggle_increases(self):
        episodes = [_episode("p1", 1, solved=True, attempts=1)]
        decisions = replay_episodes(
            episodes, HeuristicProblemSequencer(),
        )
        assert decisions[0].difficulty_before == 1
        assert decisions[0].difficulty_after == 2

    def test_abandoned_decreases(self):
        episodes = [
            _episode("p1", 2, abandoned=True, attempts=5, chat_turns=8),
        ]
        seq = HeuristicProblemSequencer(default_difficulty=2)
        decisions = replay_episodes(episodes, seq)
        assert decisions[0].difficulty_after == 1

    def test_multi_episode_progression(self):
        episodes = [
            _episode("p1", 1, solved=True, attempts=1),
            _episode("p2", 2, solved=True, attempts=1),
            _episode("p3", 3, solved=True, attempts=1),
        ]
        decisions = replay_episodes(
            episodes, HeuristicProblemSequencer(),
        )
        assert decisions[0].difficulty_after == 2
        assert decisions[1].difficulty_after == 3
        assert decisions[2].difficulty_after == 3

    def test_deterministic(self):
        episodes = [
            _episode("p1", 1, solved=True, attempts=1),
            _episode("p2", 2, abandoned=True, attempts=4, chat_turns=6),
        ]
        d1 = replay_episodes(episodes, HeuristicProblemSequencer())
        d2 = replay_episodes(episodes, HeuristicProblemSequencer())
        for a, b in zip(d1, d2):
            assert a.difficulty_after == b.difficulty_after
            assert a.struggle_level == b.struggle_level

    def test_decisions_contain_struggle_and_reason(self):
        episodes = [_episode("p1", 1, solved=True, attempts=1)]
        decisions = replay_episodes(
            episodes, HeuristicProblemSequencer(),
        )
        assert decisions[0].struggle_level is not None
        assert decisions[0].difficulty_reason is not None

    def test_replay_decision_serializable(self):
        episodes = [_episode("p1", 1, solved=True, attempts=1)]
        decisions = replay_episodes(
            episodes, HeuristicProblemSequencer(),
        )
        d = decisions[0].to_dict()
        assert isinstance(json.dumps(d), str)


# ------------------------------------------------------------------
# F. JSON export
# ------------------------------------------------------------------

class TestExport:
    def test_export_to_file(self):
        flags = PracticeFeatureFlags(
            practice_loop_enabled=True,
            adaptive_sequencing_enabled=True,
            sequencer_mode="heuristic",
        )
        obs = HeuristicObservationFilter()
        seq = HeuristicProblemSequencer(observation_filter=obs)
        mgr = PracticeSessionManager(
            flags, problem_sequencer=seq, observation_filter=obs,
        )
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_attempt(ext, submission_text="42")
        mgr.finalize_problem_episode(ext, solved=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_export.json")
            result = export_practice_session(
                ext, output_path=path, session_id="test",
            )
            assert result == path
            with open(path) as f:
                data = json.load(f)
            assert len(data["episodes"]) == 1
            assert data["episodes"][0]["solved"] is True

    def test_export_auto_named(self):
        flags = PracticeFeatureFlags(practice_loop_enabled=True)
        mgr = PracticeSessionManager(flags)
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        mgr.finalize_problem_episode(ext, solved=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_practice_session(
                ext, output_dir=tmpdir, session_id="s1",
            )
            assert result is not None
            assert result.endswith(".json")
            assert os.path.isfile(result)

    def test_export_skipped_when_no_episodes(self):
        ext: Dict[str, Any] = {"practice_session": {"completed_episodes": []}}
        result = export_practice_session(ext, output_path="/tmp/nope.json")
        assert result is None

    def test_export_skipped_when_no_practice_session(self):
        result = export_practice_session({}, output_path="/tmp/nope.json")
        assert result is None
