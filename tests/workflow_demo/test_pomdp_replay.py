"""Tests for POMDP replay integration (~8 tests)."""

from __future__ import annotations

from typing import Optional

import pytest

from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    ProblemAttempt,
    ProblemEpisodeTrace,
)
from src.workflow_demo.practice.replay import ReplayDecision, replay_episodes
from src.workflow_demo.practice.problem_sequencer import NoOpProblemSequencer
from src.workflow_demo.pedagogy.heuristic_problem_sequencer import HeuristicProblemSequencer
from src.workflow_demo.pedagogy.particle_filter import LearnerParams
from src.workflow_demo.pedagogy.pomdp_problem_sequencer import POMDPProblemSequencer
from src.workflow_demo.pedagogy.pomdp_calibration import default_particle_bank
from src.workflow_demo.pedagogy.mpc import MPCConfig


def _make_episode(difficulty: int, attempts: int = 2, solved: bool = True) -> ProblemEpisodeTrace:
    trace = ProblemEpisodeTrace(
        problem=PracticeProblemRef(
            problem_id=f"prob-d{difficulty}-a{attempts}",
            difficulty=difficulty,
            prompt_text="test",
        ),
    )
    for i in range(attempts):
        trace.append_attempt(ProblemAttempt(
            attempt_index=i,
            submission_text=f"answer {i} is long enough",
            is_correct=(i == attempts - 1 and solved),
        ))
    trace.finalize(solved=solved)
    return trace


def _make_episodes():
    return [
        _make_episode(difficulty=1, attempts=2, solved=True),
        _make_episode(difficulty=2, attempts=4, solved=True),
        _make_episode(difficulty=1, attempts=1, solved=True),
    ]


def _make_pomdp_sequencer(seed=42) -> POMDPProblemSequencer:
    return POMDPProblemSequencer(
        mpc_config=MPCConfig(rollouts=10, horizon=3),
        random_seed=seed,
    )


# ------------------------------------------------------------------
# Basic replay
# ------------------------------------------------------------------

class TestPOMDPReplay:
    def test_replay_works_with_pomdp_sequencer(self):
        episodes = _make_episodes()
        seq = _make_pomdp_sequencer()
        decisions = replay_episodes(episodes, seq)
        assert len(decisions) == 3

    def test_determinism_same_seed(self):
        episodes = _make_episodes()
        d1 = replay_episodes(episodes, _make_pomdp_sequencer(seed=42))
        d2 = replay_episodes(episodes, _make_pomdp_sequencer(seed=42))
        for a, b in zip(d1, d2):
            assert a.difficulty_after == b.difficulty_after
            assert a.decision_margin == pytest.approx(b.decision_margin)

    def test_comparison_with_heuristic_and_noop(self):
        episodes = _make_episodes()
        pomdp = replay_episodes(episodes, _make_pomdp_sequencer())
        heuristic = replay_episodes(episodes, HeuristicProblemSequencer())
        noop = replay_episodes(episodes, NoOpProblemSequencer())
        assert len(pomdp) == len(heuristic) == len(noop) == 3


# ------------------------------------------------------------------
# POMDP-specific fields
# ------------------------------------------------------------------

class TestPOMDPReplayFields:
    def test_posterior_summary_populated(self):
        episodes = _make_episodes()
        decisions = replay_episodes(episodes, _make_pomdp_sequencer())
        for d in decisions:
            assert d.posterior_summary is not None
            assert "posterior_expected_effort" in d.posterior_summary
            assert "posterior_expected_tau" in d.posterior_summary
            assert "active_particle_count" in d.posterior_summary
            assert "belief_ess" in d.posterior_summary

    def test_posterior_summary_none_for_heuristic(self):
        episodes = _make_episodes()
        decisions = replay_episodes(episodes, HeuristicProblemSequencer())
        for d in decisions:
            assert d.posterior_summary is None

    def test_decision_margin_populated(self):
        episodes = _make_episodes()
        decisions = replay_episodes(episodes, _make_pomdp_sequencer())
        for d in decisions:
            assert d.decision_margin is not None
            assert d.decision_margin >= 0.0

    def test_served_difficulty_matches_episode(self):
        episodes = _make_episodes()
        decisions = replay_episodes(episodes, _make_pomdp_sequencer())
        for ep, dec in zip(episodes, decisions):
            assert dec.served_difficulty == ep.problem.difficulty

    def test_served_and_next_difficulty_distinguished(self):
        episodes = _make_episodes()
        decisions = replay_episodes(episodes, _make_pomdp_sequencer())
        for dec in decisions:
            # served_difficulty is what was just completed
            # difficulty_after is the next difficulty chosen by MPC
            assert dec.served_difficulty is not None
            assert dec.difficulty_after is not None


# ------------------------------------------------------------------
# Default particle bank count
# ------------------------------------------------------------------

class TestDefaultParticleBank:
    def test_exactly_27_particles(self):
        bank = default_particle_bank()
        assert len(bank) == 27
