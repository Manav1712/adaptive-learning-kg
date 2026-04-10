"""
Problem sequencer interface and no-op implementation.

A ``ProblemSequencer`` decides the next difficulty bucket after each completed
problem. The ``NoOpProblemSequencer`` always returns a fixed difficulty so that
the practice loop skeleton can run without adaptive logic.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from .models import ProblemEpisodeTrace, SequencerState


@runtime_checkable
class ProblemSequencer(Protocol):
    """Between-problem sequencing engine."""

    def initialize(self, context: Dict[str, Any]) -> SequencerState: ...

    def choose_first_difficulty(self, state: SequencerState) -> int: ...

    def update_after_problem(
        self,
        state: SequencerState,
        trace: ProblemEpisodeTrace,
    ) -> SequencerState: ...

    def choose_next_difficulty(self, state: SequencerState) -> int: ...


class NoOpProblemSequencer:
    """Fixed-difficulty sequencer (Round 2 safe default).

    Always returns ``default_difficulty`` without modifying state.
    """

    def __init__(self, default_difficulty: int = 1) -> None:
        self._default = default_difficulty

    def initialize(self, context: Dict[str, Any]) -> SequencerState:
        return SequencerState(
            mode="off",
            current_difficulty=self._default,
        )

    def choose_first_difficulty(self, state: SequencerState) -> int:
        return self._default

    def update_after_problem(
        self,
        state: SequencerState,
        trace: ProblemEpisodeTrace,
    ) -> SequencerState:
        return state

    def choose_next_difficulty(self, state: SequencerState) -> int:
        return self._default
