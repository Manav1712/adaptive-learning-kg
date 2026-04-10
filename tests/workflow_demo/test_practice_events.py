"""Tests for Round 4 structured practice-loop events.

Verifies that finalize_problem_episode and begin_practice_problem emit
the expected events with full metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.workflow_demo.pedagogy.heuristic_problem_sequencer import (
    HeuristicProblemSequencer,
)
from src.workflow_demo.pedagogy.observation_filter import HeuristicObservationFilter
from src.workflow_demo.practice.feature_flags import PracticeFeatureFlags
from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    ProblemAttempt,
)
from src.workflow_demo.practice.session import PracticeSessionManager


def _flags() -> PracticeFeatureFlags:
    return PracticeFeatureFlags(
        practice_loop_enabled=True,
        adaptive_sequencing_enabled=True,
        sequencer_mode="heuristic",
    )


def _manager_with_events() -> tuple:
    events: List[Dict[str, Any]] = []

    def capture(*args, **kwargs):
        events.append({"args": args, "kwargs": kwargs})

    obs_filter = HeuristicObservationFilter()
    sequencer = HeuristicProblemSequencer(observation_filter=obs_filter)
    mgr = PracticeSessionManager(
        _flags(),
        problem_sequencer=sequencer,
        observation_filter=obs_filter,
        event_emitter=capture,
    )
    return mgr, events


def _seed_and_begin(mgr: PracticeSessionManager):
    ext: Dict[str, Any] = {}
    mgr.seed_extensions(ext)
    mgr.begin_practice_problem(ext)
    return ext


def _events_of_type(events, event_type: str) -> List[Dict[str, Any]]:
    return [e for e in events if e["args"][0] == event_type]


# ------------------------------------------------------------------
# A. practice_problem_completed event
# ------------------------------------------------------------------

class TestProblemCompletedEvent:
    def test_emitted_on_finalize(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="42", is_correct=True)
        mgr.finalize_problem_episode(ext, solved=True)
        completed = _events_of_type(events, "practice_problem_completed")
        assert len(completed) == 1

    def test_contains_problem_id(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="x=1")
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_problem_completed")[0]
        assert ev["kwargs"]["problem_id"] is not None

    def test_contains_lo_metadata(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_problem_completed")[0]
        assert "lo_id" in ev["kwargs"]
        assert "lo_title" in ev["kwargs"]

    def test_contains_sequencer_mode(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_problem_completed")[0]
        assert ev["kwargs"]["sequencer_mode"] == "heuristic"

    def test_contains_attempt_and_chat_counts(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="try1")
        mgr.record_problem_attempt(ext, submission_text="try2")
        mgr.record_problem_chat_turn(ext)
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_problem_completed")[0]
        assert ev["kwargs"]["attempt_count"] == 2
        assert ev["kwargs"]["chat_turn_count"] == 1


# ------------------------------------------------------------------
# B. practice_observation_summarized event
# ------------------------------------------------------------------

class TestObservationSummarizedEvent:
    def test_emitted_on_finalize(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        obs_events = _events_of_type(events, "practice_observation_summarized")
        assert len(obs_events) == 1

    def test_contains_meaningful_attempts(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="good answer")
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_observation_summarized")[0]
        assert ev["kwargs"]["meaningful_attempts"] == 1
        assert ev["kwargs"]["raw_attempt_count"] == 1

    def test_contains_solved_and_help(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="ans")
        mgr.record_problem_chat_turn(ext)
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_observation_summarized")[0]
        assert ev["kwargs"]["solved"] is True
        assert ev["kwargs"]["help_turn_count"] == 1


# ------------------------------------------------------------------
# C. practice_difficulty_decided event
# ------------------------------------------------------------------

class TestDifficultyDecidedEvent:
    def test_emitted_on_finalize(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        decided = _events_of_type(events, "practice_difficulty_decided")
        assert len(decided) == 1

    def test_contains_prior_and_new_difficulty(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_difficulty_decided")[0]
        assert "prior_difficulty" in ev["kwargs"]
        assert "new_difficulty" in ev["kwargs"]
        assert ev["kwargs"]["new_difficulty"] == 2
        assert ev["kwargs"]["prior_difficulty"] == 1

    def test_contains_struggle_and_reason(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_difficulty_decided")[0]
        assert ev["kwargs"]["struggle_level"] is not None
        assert ev["kwargs"]["difficulty_reason"] is not None

    def test_contains_observation(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        ev = _events_of_type(events, "practice_difficulty_decided")[0]
        assert ev["kwargs"]["observation"] is not None
        assert "meaningful_attempts" in ev["kwargs"]["observation"]


# ------------------------------------------------------------------
# D. practice_next_problem_served event
# ------------------------------------------------------------------

class TestNextProblemServedEvent:
    def test_emitted_after_first_problem_boundary(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        mgr.select_next_problem(ext)
        served = _events_of_type(events, "practice_next_problem_served")
        assert len(served) == 1

    def test_not_emitted_for_first_problem(self):
        mgr, events = _manager_with_events()
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        served = _events_of_type(events, "practice_next_problem_served")
        assert len(served) == 0

    def test_contains_difficulty_and_prior(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        mgr.select_next_problem(ext)
        ev = _events_of_type(events, "practice_next_problem_served")[0]
        assert "difficulty" in ev["kwargs"]
        assert "prior_difficulty" in ev["kwargs"]


# ------------------------------------------------------------------
# E. practice_episode_abandoned event
# ------------------------------------------------------------------

class TestEpisodeAbandonedEvent:
    def test_emitted_on_abandon(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="try")
        mgr.finalize_problem_episode(ext, abandoned=True)
        abandoned = _events_of_type(events, "practice_episode_abandoned")
        assert len(abandoned) == 1

    def test_not_emitted_on_solve(self):
        mgr, events = _manager_with_events()
        ext = _seed_and_begin(mgr)
        mgr.record_problem_attempt(ext, submission_text="answer")
        mgr.finalize_problem_episode(ext, solved=True)
        abandoned = _events_of_type(events, "practice_episode_abandoned")
        assert len(abandoned) == 0


# ------------------------------------------------------------------
# F. No events when emitter is None
# ------------------------------------------------------------------

class TestNoEmitter:
    def test_no_crash_without_emitter(self):
        obs_filter = HeuristicObservationFilter()
        sequencer = HeuristicProblemSequencer(observation_filter=obs_filter)
        mgr = PracticeSessionManager(
            _flags(),
            problem_sequencer=sequencer,
            observation_filter=obs_filter,
        )
        ext: Dict[str, Any] = {}
        mgr.seed_extensions(ext)
        mgr.begin_practice_problem(ext)
        mgr.record_problem_attempt(ext, submission_text="x")
        mgr.finalize_problem_episode(ext, solved=True)
