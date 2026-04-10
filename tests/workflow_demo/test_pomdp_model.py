"""Tests for src/workflow_demo/pedagogy/pomdp_model.py (~15 tests)."""

from __future__ import annotations

import math

import pytest

from src.workflow_demo.pedagogy.pomdp_model import (
    POMDPConstants,
    desired_difficulty,
    initial_effort,
    normalized_skill,
    poisson_pmf,
    poisson_rate,
    reward,
    transition,
)


# ------------------------------------------------------------------
# initial_effort
# ------------------------------------------------------------------

class TestInitialEffort:
    def test_basic(self):
        assert initial_effort(2.0, 3.0) == 5.0

    def test_c1_plus_c2(self):
        assert initial_effort(1.0, 1.0) == 2.0
        assert initial_effort(4.0, 5.0) == 9.0


# ------------------------------------------------------------------
# transition
# ------------------------------------------------------------------

class TestTransition:
    def test_hand_computed(self):
        c1, c2, tau = 3.0, 2.0, 0.3
        se_t = 5.0
        action = 1
        constants = POMDPConstants()
        delta_t = tau * (constants.eta + action * constants.delta_knowledge)
        expected = (se_t - c2) * math.exp(-delta_t) + c2
        result = transition(se_t, c1, c2, tau, action, constants)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_action_0_decays_slower_than_action_3(self):
        c1, c2, tau = 3.0, 2.0, 0.3
        se_t = 5.0
        constants = POMDPConstants()
        se_a0 = transition(se_t, c1, c2, tau, 0, constants)
        se_a3 = transition(se_t, c1, c2, tau, 3, constants)
        assert se_a0 > se_a3, "Higher action should drive SE toward c2 faster"

    def test_converges_toward_c2(self):
        c1, c2, tau = 3.0, 1.0, 0.5
        se = initial_effort(c1, c2)
        constants = POMDPConstants()
        for _ in range(50):
            se = transition(se, c1, c2, tau, 2, constants)
        assert se == pytest.approx(c2, abs=0.01)

    def test_custom_constants(self):
        custom = POMDPConstants(eta=2.0, delta_knowledge=0.5)
        se = transition(5.0, 3.0, 2.0, 0.3, 1, custom)
        delta_t = 0.3 * (2.0 + 1 * 0.5)
        expected = (5.0 - 2.0) * math.exp(-delta_t) + 2.0
        assert se == pytest.approx(expected, rel=1e-10)


# ------------------------------------------------------------------
# poisson_rate
# ------------------------------------------------------------------

class TestPoissonRate:
    def test_scales_with_action(self):
        constants = POMDPConstants()
        r0 = poisson_rate(3.0, 0, constants)
        r3 = poisson_rate(3.0, 3, constants)
        assert r3 > r0

    def test_scales_with_se(self):
        constants = POMDPConstants()
        r_low = poisson_rate(1.0, 1, constants)
        r_high = poisson_rate(5.0, 1, constants)
        assert r_high > r_low

    def test_floor_at_epsilon(self):
        constants = POMDPConstants()
        r = poisson_rate(0.0, 0, constants)
        assert r == constants.eps


# ------------------------------------------------------------------
# desired_difficulty
# ------------------------------------------------------------------

class TestDesiredDifficulty:
    def test_high_mastery_gives_3(self):
        c1, c2 = 4.0, 1.0
        se_t = c2 + 0.1 * c1  # NSE = 0.1 <= all thresholds
        assert desired_difficulty(se_t, c1, c2) == 3

    def test_low_mastery_gives_0(self):
        c1, c2 = 4.0, 1.0
        se_t = c2 + 0.9 * c1  # NSE = 0.9 > all thresholds
        assert desired_difficulty(se_t, c1, c2) == 0

    def test_boundary_at_threshold1(self):
        c1, c2 = 5.0, 1.0
        constants = POMDPConstants(thresholds=(0.2, 0.4, 0.6))
        nse_at_t1 = 0.2
        se_t = c2 + nse_at_t1 * c1
        assert desired_difficulty(se_t, c1, c2, constants) == 3  # <= all three

    def test_nse_between_t2_and_t3(self):
        c1, c2 = 5.0, 1.0
        constants = POMDPConstants(thresholds=(0.2, 0.4, 0.6))
        nse = 0.5  # > t1(0.2), > t2(0.4), <= t3(0.6)
        se_t = c2 + nse * c1
        assert desired_difficulty(se_t, c1, c2, constants) == 1


# ------------------------------------------------------------------
# reward
# ------------------------------------------------------------------

class TestReward:
    def test_perfect_match(self):
        c1, c2 = 4.0, 1.0
        se_t = c2 + 0.1 * c1  # a_exp = 3
        assert reward(3, se_t, c1, c2) == 1.0

    def test_mismatch_penalty(self):
        c1, c2 = 4.0, 1.0
        se_t = c2 + 0.1 * c1  # a_exp = 3
        r = reward(1, se_t, c1, c2)
        assert 0.0 <= r < 1.0

    def test_clamped_at_zero(self):
        c1, c2 = 4.0, 1.0
        se_t = c2 + 0.1 * c1  # a_exp = 3
        r = reward(0, se_t, c1, c2)
        assert r >= 0.0


# ------------------------------------------------------------------
# poisson_pmf
# ------------------------------------------------------------------

class TestPoissonPMF:
    def test_spot_check_against_formula(self):
        lam = 3.0
        k = 2
        expected = math.exp(-lam) * (lam ** k) / math.factorial(k)
        assert poisson_pmf(k, lam) == pytest.approx(expected, rel=1e-10)

    def test_pmf_sums_close_to_one(self):
        lam = 2.5
        total = sum(poisson_pmf(k, lam) for k in range(30))
        assert total == pytest.approx(1.0, abs=1e-8)

    def test_zero_lambda(self):
        assert poisson_pmf(0, 0.0) == 1.0
        assert poisson_pmf(1, 0.0) == 0.0

    def test_negative_k(self):
        assert poisson_pmf(-1, 2.0) == 0.0
