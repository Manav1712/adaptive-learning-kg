"""
POMDP-based problem sequencer — paper-faithful implementation.

Conforms to the ``ProblemSequencer`` protocol.  Uses ``served_difficulty``
(the difficulty of the just-completed problem) for belief updates, and
``next_difficulty`` (MPC output) for the problem to serve next.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..practice.models import ProblemEpisodeTrace, ProblemObservation, SequencerState
from .mpc import MPCConfig, select_action
from .particle_filter import (
    BeliefState,
    LearnerParams,
    build_uniform_belief,
    deserialize_belief,
    serialize_belief,
)
from .pomdp_model import POMDPConstants


@runtime_checkable
class _ObservationFilterLike(Protocol):
    def summarize(self, trace: ProblemEpisodeTrace) -> ProblemObservation: ...


class POMDPProblemSequencer:
    """Paper-faithful POMDP sequencer behind ``ProblemSequencer`` protocol.

    Uses particle-filter belief over (c1, c2, tau) with MPC action selection.
    """

    def __init__(
        self,
        *,
        constants: Optional[POMDPConstants] = None,
        mpc_config: Optional[MPCConfig] = None,
        particle_bank: Optional[List[LearnerParams]] = None,
        observation_filter: Optional[_ObservationFilterLike] = None,
        random_seed: Optional[int] = None,
        rollouts: Optional[int] = None,
        horizon: Optional[int] = None,
    ) -> None:
        self._constants = constants or POMDPConstants()

        effective_rollouts = rollouts if rollouts is not None else (
            mpc_config.rollouts if mpc_config else 50
        )
        effective_horizon = horizon if horizon is not None else (
            mpc_config.horizon if mpc_config else 10
        )
        self._mpc_config = MPCConfig(
            rollouts=effective_rollouts,
            horizon=effective_horizon,
            gamma=mpc_config.gamma if mpc_config else 1.0,
        )

        self._particle_bank = particle_bank
        self._obs_filter = observation_filter
        self._rng = random.Random(random_seed)

    # ------------------------------------------------------------------
    # ProblemSequencer protocol
    # ------------------------------------------------------------------

    def initialize(self, context: Dict[str, Any]) -> SequencerState:
        bank = self._particle_bank
        init_source = "calibrated"

        if not bank:
            from .pomdp_calibration import default_particle_bank
            bank = default_particle_bank()
            init_source = "fallback"

        belief = build_uniform_belief(bank)

        return SequencerState(
            mode="pomdp",
            current_difficulty=1,
            debug={
                "belief_particles": serialize_belief(belief),
                "init_source": init_source,
            },
        )

    def choose_first_difficulty(self, state: SequencerState) -> int:
        belief = deserialize_belief(state.debug.get("belief_particles", []))
        action, action_values, decision_margin = select_action(
            belief, self._mpc_config, self._constants, rng=self._rng,
        )
        state.current_difficulty = action
        state.debug["action_values"] = [av.to_dict() for av in action_values]
        state.debug["decision_margin"] = decision_margin
        return action

    def update_after_problem(
        self,
        state: SequencerState,
        trace: ProblemEpisodeTrace,
    ) -> SequencerState:
        served_difficulty = trace.problem.difficulty

        belief = deserialize_belief(state.debug.get("belief_particles", []))

        meaningful_attempts = self._extract_meaningful_attempts(trace)

        belief = belief.predict(served_difficulty, self._constants)
        belief = belief.update(meaningful_attempts, served_difficulty, self._constants)

        next_difficulty, action_values, decision_margin = select_action(
            belief, self._mpc_config, self._constants, rng=self._rng,
        )

        init_source = state.debug.get("init_source", "unknown")

        return SequencerState(
            mode="pomdp",
            step_index=state.step_index,
            last_difficulty=served_difficulty,
            current_difficulty=next_difficulty,
            recent_observations=state.recent_observations + [meaningful_attempts],
            posterior_expected_effort=belief.posterior_expected_effort(),
            posterior_expected_tau=belief.posterior_expected_tau(),
            active_particle_count=belief.active_particle_count(),
            debug={
                "belief_particles": serialize_belief(belief),
                "served_difficulty": served_difficulty,
                "next_difficulty": next_difficulty,
                "meaningful_attempts": meaningful_attempts,
                "action_values": [av.to_dict() for av in action_values],
                "decision_margin": decision_margin,
                "belief_ess": belief.effective_sample_size(),
                "init_source": init_source,
            },
        )

    def choose_next_difficulty(self, state: SequencerState) -> int:
        return state.current_difficulty

    # ------------------------------------------------------------------
    # Observation extraction with strict precedence
    # ------------------------------------------------------------------

    def _extract_meaningful_attempts(self, trace: ProblemEpisodeTrace) -> int:
        """Extract meaningful_attempts with strict source-of-truth ordering.

        1. Finalized llm_meaningful_attempts on the episode trace.
        2. ObservationFilter.summarize() if filter is available.
        3. Raw len(trace.attempts) as last resort.
        """
        if trace.llm_meaningful_attempts is not None:
            return trace.llm_meaningful_attempts

        if self._obs_filter is not None:
            obs = self._obs_filter.summarize(trace)
            return obs.meaningful_attempts

        return len(trace.attempts)
