"""
MPC (Model Predictive Control) — Monte Carlo rollout planner.

Evaluates candidate difficulty actions via sampled rollouts through the
POMDP model.  Randomness is limited to initial particle sampling;
the greedy rollout policy for subsequent steps is deterministic.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from .particle_filter import BeliefState
from .pomdp_model import (
    POMDPConstants,
    desired_difficulty,
    reward,
    transition,
)

_DEFAULT_CONSTANTS = POMDPConstants()

ACTION_SPACE = [0, 1, 2, 3]


@dataclass(frozen=True)
class MPCConfig:
    """Configuration for Monte Carlo rollout planner."""

    rollouts: int = 50
    horizon: int = 10
    gamma: float = 1.0


@dataclass
class ActionValue:
    """Estimated value for one candidate action."""

    action: int
    mean_value: float
    rollout_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_actions(
    belief: BeliefState,
    config: MPCConfig,
    constants: POMDPConstants = _DEFAULT_CONSTANTS,
    *,
    rng: random.Random,
) -> List[ActionValue]:
    """Evaluate all candidate actions via Monte Carlo rollouts.

    For each action in {0,1,2,3}, runs ``config.rollouts`` rollouts.
    Each rollout:
      1. Samples a particle from belief weights (only source of randomness).
      2. Forces the first action.
      3. Uses deterministic greedy rollout policy for steps 2..H.
    """
    if not belief.particles:
        return [ActionValue(action=a, mean_value=0.0, rollout_count=0) for a in ACTION_SPACE]

    weights = [p.weight for p in belief.particles]
    results: List[ActionValue] = []

    for candidate_action in ACTION_SPACE:
        total_value = 0.0
        for _ in range(config.rollouts):
            (sampled_particle,) = rng.choices(belief.particles, weights=weights, k=1)

            se = sampled_particle.se_t
            c1 = sampled_particle.params.c1
            c2 = sampled_particle.params.c2
            tau = sampled_particle.params.tau

            cumulative = 0.0
            discount = 1.0

            for step in range(config.horizon):
                action = candidate_action if step == 0 else desired_difficulty(se, c1, c2, constants)
                cumulative += discount * reward(action, se, c1, c2, constants)
                se = transition(se, c1, c2, tau, action, constants)
                discount *= config.gamma

            total_value += cumulative

        results.append(ActionValue(
            action=candidate_action,
            mean_value=total_value / config.rollouts,
            rollout_count=config.rollouts,
        ))

    return results


def select_action(
    belief: BeliefState,
    config: MPCConfig,
    constants: POMDPConstants = _DEFAULT_CONSTANTS,
    *,
    rng: random.Random,
) -> Tuple[int, List[ActionValue], float]:
    """Choose the best action and return (action, all_values, decision_margin).

    decision_margin = best_value - second_best_value.
    """
    action_values = evaluate_actions(belief, config, constants, rng=rng)

    sorted_by_value = sorted(action_values, key=lambda av: av.mean_value, reverse=True)
    best_action = sorted_by_value[0].action
    best_value = sorted_by_value[0].mean_value

    if len(sorted_by_value) >= 2:
        decision_margin = best_value - sorted_by_value[1].mean_value
    else:
        decision_margin = 0.0

    return best_action, action_values, decision_margin
