"""
Heuristic problem sequencer — first functional adaptive version (Round 3).

Chooses next difficulty in ``{0, 1, 2, 3}`` based on local struggle
signals from the most recent ``ProblemObservation``.  No POMDP, no
particle filter, no LLM calls.

Struggle classification
-----------------------
The filter produces a *struggle level* from the observation:

=============  ==========================================================
Level          Condition
=============  ==========================================================
``low``        solved AND meaningful_attempts <= ``low_ceiling``
               AND help_turn_count <= ``low_help_ceiling``
``moderate``   solved AND (meaningful_attempts <= ``moderate_ceiling``
               OR help_turn_count <= ``moderate_help_ceiling``)
``high``       anything else (including abandoned)
=============  ==========================================================

Difficulty update rule
----------------------
=============  ======  ============================
Struggle       Solved  Action
=============  ======  ============================
low            yes     increase difficulty by 1
moderate       yes     hold difficulty
high           yes     decrease difficulty by 1
high           no      decrease difficulty by 1
(no attempts)  —       hold (chat-only interaction)
=============  ======  ============================

Result is always clamped to ``[0, 3]``.

All thresholds are constructor parameters so tests and later rounds can
override them freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from ..practice.models import ProblemEpisodeTrace, ProblemObservation, SequencerState
from .observation_filter import HeuristicObservationFilter, ObservationFilter

StruggleLevel = Literal["low", "moderate", "high", "none"]

MIN_DIFFICULTY = 0
MAX_DIFFICULTY = 3


@dataclass(frozen=True)
class StruggleThresholds:
    """Configurable thresholds for the heuristic struggle classifier."""

    low_attempt_ceiling: int = 1
    low_help_ceiling: int = 2
    moderate_attempt_ceiling: int = 3
    moderate_help_ceiling: int = 5


class HeuristicProblemSequencer:
    """First adaptive sequencer — bounded difficulty updates from struggle level.

    Conforms to the ``ProblemSequencer`` protocol.
    """

    def __init__(
        self,
        *,
        default_difficulty: int = 1,
        thresholds: StruggleThresholds | None = None,
        observation_filter: ObservationFilter | None = None,
    ) -> None:
        self._default = default_difficulty
        self.thresholds = thresholds or StruggleThresholds()
        self.obs_filter: ObservationFilter = observation_filter or HeuristicObservationFilter()

    # ------------------------------------------------------------------
    # ProblemSequencer protocol
    # ------------------------------------------------------------------

    def initialize(self, context: Dict[str, Any]) -> SequencerState:
        return SequencerState(
            mode="heuristic",
            current_difficulty=self._default,
        )

    def choose_first_difficulty(self, state: SequencerState) -> int:
        return state.current_difficulty

    def update_after_problem(
        self,
        state: SequencerState,
        trace: ProblemEpisodeTrace,
    ) -> SequencerState:
        obs = self.obs_filter.summarize(trace)
        struggle = self.classify_struggle(obs)
        delta = self._difficulty_delta(struggle, obs)
        new_diff = _clamp(state.current_difficulty + delta, MIN_DIFFICULTY, MAX_DIFFICULTY)

        return SequencerState(
            mode=state.mode,
            step_index=state.step_index,
            last_difficulty=state.current_difficulty,
            current_difficulty=new_diff,
            recent_observations=state.recent_observations + [obs.meaningful_attempts],
            posterior_expected_effort=state.posterior_expected_effort,
            posterior_expected_tau=state.posterior_expected_tau,
            active_particle_count=state.active_particle_count,
            debug={
                "last_observation": obs.to_dict(),
                "struggle_level": struggle,
                "difficulty_delta": delta,
                "difficulty_reason": self._reason(struggle, delta, obs),
            },
        )

    def choose_next_difficulty(self, state: SequencerState) -> int:
        return state.current_difficulty

    # ------------------------------------------------------------------
    # Struggle classification
    # ------------------------------------------------------------------

    def classify_struggle(self, obs: ProblemObservation) -> StruggleLevel:
        if obs.raw_attempt_count == 0 and obs.meaningful_attempts == 0:
            return "none"
        t = self.thresholds
        if (
            obs.solved
            and obs.meaningful_attempts <= t.low_attempt_ceiling
            and obs.help_turn_count <= t.low_help_ceiling
        ):
            return "low"
        if obs.solved and (
            obs.meaningful_attempts <= t.moderate_attempt_ceiling
            or obs.help_turn_count <= t.moderate_help_ceiling
        ):
            return "moderate"
        return "high"

    # ------------------------------------------------------------------
    # Difficulty delta
    # ------------------------------------------------------------------

    @staticmethod
    def _difficulty_delta(struggle: StruggleLevel, obs: ProblemObservation) -> int:
        if struggle == "none":
            return 0
        if struggle == "low":
            return 1
        if struggle == "moderate":
            return 0
        # high
        if not obs.solved:
            return -1
        return -1

    @staticmethod
    def _reason(struggle: StruggleLevel, delta: int, obs: ProblemObservation) -> str:
        direction = "increase" if delta > 0 else ("decrease" if delta < 0 else "hold")
        return (
            f"struggle={struggle} solved={obs.solved} "
            f"meaningful={obs.meaningful_attempts} help_turns={obs.help_turn_count} "
            f"-> {direction} by {abs(delta)}"
        )


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))
