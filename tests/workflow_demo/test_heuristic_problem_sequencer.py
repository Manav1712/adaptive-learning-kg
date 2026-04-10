"""Tests for HeuristicProblemSequencer (Round 3)."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy.heuristic_problem_sequencer import (
    HeuristicProblemSequencer,
    StruggleThresholds,
)
from src.workflow_demo.pedagogy.observation_filter import HeuristicObservationFilter
from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    ProblemAttempt,
    ProblemEpisodeTrace,
    ProblemObservation,
    SequencerState,
)
from src.workflow_demo.practice.problem_sequencer import ProblemSequencer


def _ref(**kw) -> PracticeProblemRef:
    defaults = dict(problem_id="p1", difficulty=1, prompt_text="Solve x+1=2")
    return PracticeProblemRef(**(defaults | kw))


def _attempt(text: str = "x=1", correct: bool | None = True) -> ProblemAttempt:
    return ProblemAttempt(attempt_index=0, submission_text=text, is_correct=correct)


def _episode(
    *,
    attempts: list[ProblemAttempt] | None = None,
    chat_turns: int = 0,
    solved: bool = False,
    abandoned: bool = False,
) -> ProblemEpisodeTrace:
    ep = ProblemEpisodeTrace(
        problem=_ref(),
        attempts=attempts or [],
        chat_turn_count=chat_turns,
        solved=solved,
        abandoned=abandoned,
    )
    return ep


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------

class TestProtocol:
    def test_conforms(self):
        assert isinstance(HeuristicProblemSequencer(), ProblemSequencer)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInitialize:
    def test_default_difficulty(self):
        seq = HeuristicProblemSequencer()
        state = seq.initialize({})
        assert state.current_difficulty == 1
        assert state.mode == "heuristic"

    def test_custom_default(self):
        seq = HeuristicProblemSequencer(default_difficulty=2)
        state = seq.initialize({})
        assert state.current_difficulty == 2

    def test_choose_first(self):
        seq = HeuristicProblemSequencer()
        state = seq.initialize({})
        assert seq.choose_first_difficulty(state) == 1


# ------------------------------------------------------------------
# Low struggle → increase
# ------------------------------------------------------------------

class TestLowStruggle:
    def test_solved_one_attempt_no_help(self):
        """Solved with 1 meaningful attempt, 0 chat turns → increase."""
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=1)
        ep = _episode(attempts=[_attempt()], solved=True)
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 2
        assert new_state.last_difficulty == 1

    def test_increase_from_0(self):
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=0)
        ep = _episode(attempts=[_attempt()], solved=True)
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 1

    def test_clamped_at_max(self):
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=3)
        ep = _episode(attempts=[_attempt()], solved=True)
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 3


# ------------------------------------------------------------------
# Moderate struggle → hold
# ------------------------------------------------------------------

class TestModerateStruggle:
    def test_solved_multiple_attempts(self):
        """Solved with 2 meaningful attempts, 3 chat turns → moderate → hold."""
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=2)
        ep = _episode(
            attempts=[_attempt("ans1"), _attempt("ans2")],
            chat_turns=3,
            solved=True,
        )
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 2

    def test_solved_with_some_help(self):
        """Solved with 1 meaningful but 4 chat turns → moderate (help <= 5)."""
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=1)
        ep = _episode(
            attempts=[_attempt()],
            chat_turns=4,
            solved=True,
        )
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 1


# ------------------------------------------------------------------
# High struggle → decrease
# ------------------------------------------------------------------

class TestHighStruggle:
    def test_solved_many_attempts(self):
        """Solved but 5 meaningful attempts + 8 chat turns → high → decrease."""
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=2)
        ep = _episode(
            attempts=[_attempt(f"try{i}") for i in range(5)],
            chat_turns=8,
            solved=True,
        )
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 1

    def test_abandoned(self):
        """Abandoned after 4 attempts → high → decrease."""
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=2)
        ep = _episode(
            attempts=[_attempt(f"try{i}") for i in range(4)],
            chat_turns=6,
            abandoned=True,
        )
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 1

    def test_clamped_at_min(self):
        """Cannot go below 0."""
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=0)
        ep = _episode(
            attempts=[_attempt(f"try{i}") for i in range(5)],
            chat_turns=8,
            abandoned=True,
        )
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 0


# ------------------------------------------------------------------
# No attempts (chat-only) → hold
# ------------------------------------------------------------------

class TestNoAttempts:
    def test_chat_only_hold(self):
        """Chat-only interaction with no attempts → hold."""
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=2)
        ep = _episode(chat_turns=3)
        new_state = seq.update_after_problem(state, ep)
        assert new_state.current_difficulty == 2


# ------------------------------------------------------------------
# Struggle classification
# ------------------------------------------------------------------

class TestStruggleClassification:
    def test_none_level(self):
        obs = ProblemObservation(meaningful_attempts=0, raw_attempt_count=0)
        seq = HeuristicProblemSequencer()
        assert seq.classify_struggle(obs) == "none"

    def test_low(self):
        obs = ProblemObservation(
            meaningful_attempts=1, raw_attempt_count=1, solved=True, help_turn_count=0,
        )
        seq = HeuristicProblemSequencer()
        assert seq.classify_struggle(obs) == "low"

    def test_moderate(self):
        obs = ProblemObservation(
            meaningful_attempts=2, raw_attempt_count=2, solved=True, help_turn_count=3,
        )
        seq = HeuristicProblemSequencer()
        assert seq.classify_struggle(obs) == "moderate"

    def test_high_unsolved(self):
        obs = ProblemObservation(
            meaningful_attempts=4, raw_attempt_count=4, solved=False, help_turn_count=8,
        )
        seq = HeuristicProblemSequencer()
        assert seq.classify_struggle(obs) == "high"


# ------------------------------------------------------------------
# Custom thresholds
# ------------------------------------------------------------------

class TestCustomThresholds:
    def test_stricter_low(self):
        thresholds = StruggleThresholds(low_attempt_ceiling=0, low_help_ceiling=0)
        seq = HeuristicProblemSequencer(thresholds=thresholds)
        obs = ProblemObservation(
            meaningful_attempts=1, raw_attempt_count=1, solved=True, help_turn_count=0,
        )
        assert seq.classify_struggle(obs) != "low"


# ------------------------------------------------------------------
# Debug / reason tracking
# ------------------------------------------------------------------

class TestDebug:
    def test_debug_contains_struggle_and_reason(self):
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=1)
        ep = _episode(attempts=[_attempt()], solved=True)
        new_state = seq.update_after_problem(state, ep)
        assert "struggle_level" in new_state.debug
        assert "difficulty_reason" in new_state.debug
        assert "last_observation" in new_state.debug

    def test_recent_observations_grows(self):
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=1)
        ep = _episode(attempts=[_attempt()], solved=True)
        s1 = seq.update_after_problem(state, ep)
        s2 = seq.update_after_problem(s1, ep)
        assert len(s2.recent_observations) == 2


# ------------------------------------------------------------------
# Multi-problem sequence
# ------------------------------------------------------------------

class TestMultiProblemSequence:
    def test_progressive_increase(self):
        """Low-struggle solves should ramp difficulty from 1 to 3."""
        seq = HeuristicProblemSequencer()
        state = seq.initialize({})
        for _ in range(3):
            ep = _episode(attempts=[_attempt()], solved=True)
            state = seq.update_after_problem(state, ep)
        assert state.current_difficulty == 3

    def test_decrease_then_recover(self):
        """Abandon → decrease, then easy solve → increase back."""
        seq = HeuristicProblemSequencer()
        state = SequencerState(mode="heuristic", current_difficulty=2)
        bad_ep = _episode(
            attempts=[_attempt(f"x{i}") for i in range(5)],
            chat_turns=8,
            abandoned=True,
        )
        state = seq.update_after_problem(state, bad_ep)
        assert state.current_difficulty == 1
        easy_ep = _episode(attempts=[_attempt()], solved=True)
        state = seq.update_after_problem(state, easy_ep)
        assert state.current_difficulty == 2
