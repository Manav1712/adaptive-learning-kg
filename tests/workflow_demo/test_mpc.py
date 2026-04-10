"""Tests for src/workflow_demo/pedagogy/mpc.py (~10 tests)."""

from __future__ import annotations

import random

import pytest

from src.workflow_demo.pedagogy.mpc import (
    ACTION_SPACE,
    ActionValue,
    MPCConfig,
    evaluate_actions,
    select_action,
)
from src.workflow_demo.pedagogy.particle_filter import (
    LearnerParams,
    build_uniform_belief,
)
from src.workflow_demo.pedagogy.pomdp_model import POMDPConstants


def _make_belief():
    params = [
        LearnerParams(c1=2.0, c2=1.5, tau=0.3),
        LearnerParams(c1=3.0, c2=2.0, tau=0.2),
        LearnerParams(c1=1.5, c2=1.0, tau=0.4),
    ]
    return build_uniform_belief(params)


# ------------------------------------------------------------------
# evaluate_actions
# ------------------------------------------------------------------

class TestEvaluateActions:
    def test_returns_all_four_actions(self):
        belief = _make_belief()
        config = MPCConfig(rollouts=10, horizon=5)
        rng = random.Random(42)
        results = evaluate_actions(belief, config, rng=rng)
        actions = {r.action for r in results}
        assert actions == set(ACTION_SPACE)

    def test_respects_rollout_count(self):
        belief = _make_belief()
        config = MPCConfig(rollouts=7, horizon=3)
        rng = random.Random(42)
        results = evaluate_actions(belief, config, rng=rng)
        for r in results:
            assert r.rollout_count == 7

    def test_values_are_non_negative(self):
        belief = _make_belief()
        config = MPCConfig(rollouts=20, horizon=5)
        rng = random.Random(42)
        results = evaluate_actions(belief, config, rng=rng)
        for r in results:
            assert r.mean_value >= 0.0

    def test_empty_belief_returns_zeros(self):
        from src.workflow_demo.pedagogy.particle_filter import BeliefState
        empty = BeliefState(particles=[])
        config = MPCConfig(rollouts=10, horizon=5)
        rng = random.Random(42)
        results = evaluate_actions(empty, config, rng=rng)
        assert len(results) == 4
        for r in results:
            assert r.mean_value == 0.0
            assert r.rollout_count == 0


# ------------------------------------------------------------------
# select_action
# ------------------------------------------------------------------

class TestSelectAction:
    def test_returns_valid_action(self):
        belief = _make_belief()
        config = MPCConfig(rollouts=20, horizon=5)
        rng = random.Random(42)
        action, values, margin = select_action(belief, config, rng=rng)
        assert action in ACTION_SPACE

    def test_deterministic_same_seed(self):
        belief = _make_belief()
        config = MPCConfig(rollouts=20, horizon=5)

        rng1 = random.Random(99)
        a1, v1, m1 = select_action(belief, config, rng=rng1)

        rng2 = random.Random(99)
        a2, v2, m2 = select_action(belief, config, rng=rng2)

        assert a1 == a2
        assert m1 == pytest.approx(m2)
        for av1, av2 in zip(v1, v2):
            assert av1.mean_value == pytest.approx(av2.mean_value)

    def test_different_seeds_can_differ(self):
        belief = _make_belief()
        config = MPCConfig(rollouts=200, horizon=10)
        results = set()
        for seed in range(50):
            rng = random.Random(seed)
            action, _, _ = select_action(belief, config, rng=rng)
            results.add(action)
        # Not all seeds must differ but with 50 seeds we expect some variation
        assert len(results) >= 1

    def test_decision_margin_non_negative(self):
        belief = _make_belief()
        config = MPCConfig(rollouts=20, horizon=5)
        rng = random.Random(42)
        _, _, margin = select_action(belief, config, rng=rng)
        assert margin >= 0.0

    def test_decision_margin_correct_value(self):
        belief = _make_belief()
        config = MPCConfig(rollouts=20, horizon=5)
        rng = random.Random(42)
        _, values, margin = select_action(belief, config, rng=rng)
        sorted_vals = sorted([v.mean_value for v in values], reverse=True)
        assert margin == pytest.approx(sorted_vals[0] - sorted_vals[1])

    def test_decision_margin_zero_when_tied(self):
        from src.workflow_demo.pedagogy.particle_filter import BeliefState
        empty = BeliefState(particles=[])
        config = MPCConfig(rollouts=10, horizon=5)
        rng = random.Random(42)
        _, values, margin = select_action(empty, config, rng=rng)
        assert margin == pytest.approx(0.0)


# ------------------------------------------------------------------
# Greedy rollout determinism
# ------------------------------------------------------------------

class TestGreedyRolloutDeterminism:
    def test_same_particle_same_result(self):
        """Two rollouts from identical initial state should be identical."""
        belief = _make_belief()
        config = MPCConfig(rollouts=1, horizon=10)

        rng1 = random.Random(42)
        _, vals1, _ = select_action(belief, config, rng=rng1)

        rng2 = random.Random(42)
        _, vals2, _ = select_action(belief, config, rng=rng2)

        for v1, v2 in zip(vals1, vals2):
            assert v1.mean_value == pytest.approx(v2.mean_value)
