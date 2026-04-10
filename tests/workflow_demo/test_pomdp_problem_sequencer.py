"""Tests for src/workflow_demo/pedagogy/pomdp_problem_sequencer.py (~14 tests)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    ProblemAttempt,
    ProblemEpisodeTrace,
    ProblemObservation,
    SequencerState,
)
from src.workflow_demo.practice.problem_sequencer import ProblemSequencer
from src.workflow_demo.pedagogy.particle_filter import LearnerParams
from src.workflow_demo.pedagogy.pomdp_problem_sequencer import POMDPProblemSequencer
from src.workflow_demo.pedagogy.pomdp_model import POMDPConstants
from src.workflow_demo.pedagogy.mpc import MPCConfig


def _make_trace(difficulty: int = 1, attempts: int = 2, solved: bool = True,
                llm_meaningful: Optional[int] = None) -> ProblemEpisodeTrace:
    trace = ProblemEpisodeTrace(
        problem=PracticeProblemRef(
            problem_id=f"p-{difficulty}",
            difficulty=difficulty,
            prompt_text="test problem",
        ),
    )
    for i in range(attempts):
        trace.append_attempt(ProblemAttempt(
            attempt_index=i,
            submission_text=f"answer attempt {i}",
            is_correct=(i == attempts - 1 and solved),
        ))
    trace.finalize(solved=solved)
    if llm_meaningful is not None:
        trace.llm_meaningful_attempts = llm_meaningful
    return trace


def _make_sequencer(**kwargs) -> POMDPProblemSequencer:
    defaults = dict(
        mpc_config=MPCConfig(rollouts=10, horizon=3),
        random_seed=42,
    )
    defaults.update(kwargs)
    return POMDPProblemSequencer(**defaults)


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------

class TestProtocolConformance:
    def test_isinstance_check(self):
        seq = _make_sequencer()
        assert isinstance(seq, ProblemSequencer)


# ------------------------------------------------------------------
# initialize
# ------------------------------------------------------------------

class TestInitialize:
    def test_fallback_prior_27_particles(self):
        seq = _make_sequencer()
        state = seq.initialize({})
        particles = state.debug.get("belief_particles", [])
        assert len(particles) == 27

    def test_mode_is_pomdp(self):
        seq = _make_sequencer()
        state = seq.initialize({})
        assert state.mode == "pomdp"

    def test_init_source_fallback(self):
        seq = _make_sequencer()
        state = seq.initialize({})
        assert state.debug["init_source"] == "fallback"

    def test_init_source_calibrated(self):
        bank = [LearnerParams(2.0, 1.5, 0.3)]
        seq = _make_sequencer(particle_bank=bank)
        state = seq.initialize({})
        assert state.debug["init_source"] == "calibrated"
        assert len(state.debug["belief_particles"]) == 1


# ------------------------------------------------------------------
# update_after_problem — served_difficulty
# ------------------------------------------------------------------

class TestServedDifficulty:
    def test_uses_trace_difficulty_not_state(self):
        seq = _make_sequencer()
        state = seq.initialize({})
        state.current_difficulty = 0  # state says 0
        trace = _make_trace(difficulty=3)  # trace says 3
        new_state = seq.update_after_problem(state, trace)
        assert new_state.debug["served_difficulty"] == 3
        assert new_state.last_difficulty == 3

    def test_different_served_difficulty_different_posteriors(self):
        bank = [
            LearnerParams(c1=2.0, c2=1.5, tau=0.2),
            LearnerParams(c1=3.0, c2=2.0, tau=0.4),
        ]
        seq1 = _make_sequencer(particle_bank=bank, random_seed=42)
        seq2 = _make_sequencer(particle_bank=bank, random_seed=42)

        s1 = seq1.initialize({})
        s2 = seq2.initialize({})

        t1 = _make_trace(difficulty=0, attempts=2, llm_meaningful=2)
        t2 = _make_trace(difficulty=3, attempts=2, llm_meaningful=2)

        r1 = seq1.update_after_problem(s1, t1)
        r2 = seq2.update_after_problem(s2, t2)

        assert r1.posterior_expected_effort != pytest.approx(r2.posterior_expected_effort, abs=1e-6)


# ------------------------------------------------------------------
# Observation precedence
# ------------------------------------------------------------------

class TestObservationPrecedence:
    def test_finalized_llm_used_first(self):
        seq = _make_sequencer()
        state = seq.initialize({})
        trace = _make_trace(attempts=5, llm_meaningful=1)
        new_state = seq.update_after_problem(state, trace)
        assert new_state.debug["meaningful_attempts"] == 1

    def test_filter_used_when_no_llm(self):
        mock_filter = MagicMock()
        mock_filter.summarize.return_value = ProblemObservation(
            meaningful_attempts=7, raw_attempt_count=10,
        )
        seq = _make_sequencer(observation_filter=mock_filter)
        state = seq.initialize({})
        trace = _make_trace(attempts=10, llm_meaningful=None)
        new_state = seq.update_after_problem(state, trace)
        assert new_state.debug["meaningful_attempts"] == 7
        mock_filter.summarize.assert_called_once()

    def test_raw_fallback_when_no_filter_no_llm(self):
        seq = _make_sequencer(observation_filter=None)
        state = seq.initialize({})
        trace = _make_trace(attempts=4, llm_meaningful=None)
        new_state = seq.update_after_problem(state, trace)
        assert new_state.debug["meaningful_attempts"] == 4


# ------------------------------------------------------------------
# Output fields
# ------------------------------------------------------------------

class TestOutputFields:
    def test_posterior_summaries_populated(self):
        seq = _make_sequencer()
        state = seq.initialize({})
        trace = _make_trace()
        new_state = seq.update_after_problem(state, trace)
        assert new_state.posterior_expected_effort is not None
        assert new_state.posterior_expected_tau is not None
        assert new_state.active_particle_count is not None

    def test_decision_margin_in_debug(self):
        seq = _make_sequencer()
        state = seq.initialize({})
        trace = _make_trace()
        new_state = seq.update_after_problem(state, trace)
        assert "decision_margin" in new_state.debug
        assert new_state.debug["decision_margin"] >= 0.0

    def test_valid_difficulty_output(self):
        seq = _make_sequencer()
        state = seq.initialize({})
        trace = _make_trace()
        new_state = seq.update_after_problem(state, trace)
        assert new_state.current_difficulty in {0, 1, 2, 3}


# ------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------

class TestDeterminism:
    def test_seeded_produces_identical_results(self):
        bank = [
            LearnerParams(c1=2.0, c2=1.5, tau=0.2),
            LearnerParams(c1=3.0, c2=2.0, tau=0.4),
        ]
        trace = _make_trace(difficulty=1, attempts=3, llm_meaningful=2)

        seq1 = _make_sequencer(particle_bank=bank, random_seed=99)
        s1 = seq1.initialize({})
        r1 = seq1.update_after_problem(s1, trace)

        seq2 = _make_sequencer(particle_bank=bank, random_seed=99)
        s2 = seq2.initialize({})
        r2 = seq2.update_after_problem(s2, trace)

        assert r1.current_difficulty == r2.current_difficulty
        assert r1.posterior_expected_effort == pytest.approx(r2.posterior_expected_effort)
        assert r1.debug["decision_margin"] == pytest.approx(r2.debug["decision_margin"])
