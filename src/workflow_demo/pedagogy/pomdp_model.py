"""
Paper-faithful POMDP model — deterministic math functions.

All functions are pure, stateless, and match the equations in the paper:
transition, observation (Poisson), reward shaping, desired difficulty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class POMDPConstants:
    """Configurable constants for the POMDP model (paper defaults)."""

    eta: float = 1.2
    delta_knowledge: float = 1.0
    slip: float = 0.1
    eps: float = 1e-6
    thresholds: Tuple[float, float, float] = (0.2, 0.4, 0.6)
    reward_penalty: float = 0.25


_DEFAULT_CONSTANTS = POMDPConstants()


def initial_effort(c1: float, c2: float) -> float:
    """SE_0 = c1 + c2."""
    return c1 + c2


def transition(
    se_t: float,
    c1: float,
    c2: float,
    tau: float,
    action: int,
    constants: POMDPConstants = _DEFAULT_CONSTANTS,
) -> float:
    """SE_{t+1} = (SE_t - c2) * exp(-delta_t) + c2.

    delta_t = tau * (eta + action * delta_knowledge)
    """
    delta_t = tau * (constants.eta + action * constants.delta_knowledge)
    return (se_t - c2) * math.exp(-delta_t) + c2


def poisson_rate(
    se_t: float,
    action: int,
    constants: POMDPConstants = _DEFAULT_CONSTANTS,
) -> float:
    """lambda_{t+1} = max(SE_{t+1} * (1 + a_t) * (1 - slip), epsilon)."""
    return max(se_t * (1 + action) * (1.0 - constants.slip), constants.eps)


def normalized_skill(se_t: float, c1: float, c2: float) -> float:
    """NSE_t = (SE_t - c2) / c1."""
    if c1 == 0:
        return 0.0
    return (se_t - c2) / c1


def desired_difficulty(
    se_t: float,
    c1: float,
    c2: float,
    constants: POMDPConstants = _DEFAULT_CONSTANTS,
) -> int:
    """Map normalized skill to desired difficulty bucket {0,1,2,3}.

    a_exp = 1(NSE <= t1) + 1(NSE <= t2) + 1(NSE <= t3)
    Higher mastery (lower NSE) -> higher desired difficulty.
    """
    nse = normalized_skill(se_t, c1, c2)
    t1, t2, t3 = constants.thresholds
    return int(nse <= t1) + int(nse <= t2) + int(nse <= t3)


def reward(
    action: int,
    se_t: float,
    c1: float,
    c2: float,
    constants: POMDPConstants = _DEFAULT_CONSTANTS,
) -> float:
    """Shaped reward: 1.0 if action matches desired difficulty, else penalized."""
    a_exp = desired_difficulty(se_t, c1, c2, constants)
    if action == a_exp:
        return 1.0
    return max(0.0, 1.0 - constants.reward_penalty * abs(action - a_exp))


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function P(X=k; lambda).

    Direct computation avoiding scipy dependency.
    """
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
