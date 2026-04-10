"""
Observation filter — summarises a completed problem episode into a
``ProblemObservation`` for consumption by the sequencer.

Round 3 provides a pure-heuristic implementation that uses only local
runtime signals (attempt counts, chat turns, timestamps, solve status).
No LLM calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from ..practice.models import ProblemEpisodeTrace, ProblemObservation


@runtime_checkable
class ObservationFilter(Protocol):
    """Derive a ``ProblemObservation`` from a completed episode trace."""

    def summarize(self, trace: ProblemEpisodeTrace) -> ProblemObservation: ...


# ------------------------------------------------------------------
# Heuristic implementation
# ------------------------------------------------------------------

class HeuristicObservationFilter:
    """Pure-local observation filter using attempt / chat-turn / time signals.

    Meaningful-attempt heuristic
    ----------------------------
    An attempt is counted as *meaningful* when it carries a non-empty
    ``submission_text`` whose stripped length exceeds
    ``min_substantive_length`` characters.  This is a conservative proxy:
    it filters out blank, whitespace-only, and trivially short ("ok", "?")
    submissions while counting anything that looks like a real answer.

    When no recorded attempts exist (chat-only interaction), the filter
    falls back to ``chat_turn_count // chat_turns_per_virtual_attempt`` to
    produce a synthetic attempt count, capped at
    ``max_virtual_attempts_from_chat``.

    Time-on-problem derivation
    --------------------------
    If the episode has both ``started_at`` and ``completed_at`` ISO
    timestamps, elapsed seconds are computed.  Otherwise ``None``.
    """

    def __init__(
        self,
        *,
        min_substantive_length: int = 3,
        chat_turns_per_virtual_attempt: int = 4,
        max_virtual_attempts_from_chat: int = 5,
    ) -> None:
        self.min_substantive_length = min_substantive_length
        self.chat_turns_per_virtual_attempt = chat_turns_per_virtual_attempt
        self.max_virtual_attempts_from_chat = max_virtual_attempts_from_chat

    def summarize(self, trace: ProblemEpisodeTrace) -> ProblemObservation:
        raw_count = len(trace.attempts)
        meaningful = self._count_meaningful(trace)
        time_sec = self._derive_time(trace)

        return ProblemObservation(
            meaningful_attempts=meaningful,
            raw_attempt_count=raw_count,
            time_on_problem_sec=time_sec,
            help_turn_count=trace.chat_turn_count,
            solved=trace.solved,
            debug={
                "filter": "heuristic",
                "min_substantive_length": self.min_substantive_length,
                "chat_turns_per_virtual_attempt": self.chat_turns_per_virtual_attempt,
            },
        )

    def _count_meaningful(self, trace: ProblemEpisodeTrace) -> int:
        if trace.attempts:
            return sum(
                1
                for a in trace.attempts
                if a.submission_text
                and len(a.submission_text.strip()) >= self.min_substantive_length
            )
        if trace.chat_turn_count > 0 and self.chat_turns_per_virtual_attempt > 0:
            virtual = trace.chat_turn_count // self.chat_turns_per_virtual_attempt
            return min(virtual, self.max_virtual_attempts_from_chat)
        return 0

    @staticmethod
    def _derive_time(trace: ProblemEpisodeTrace) -> float | None:
        if trace.time_on_problem_sec is not None:
            return trace.time_on_problem_sec
        if trace.started_at and trace.completed_at:
            try:
                t0 = datetime.fromisoformat(trace.started_at)
                t1 = datetime.fromisoformat(trace.completed_at)
                delta = (t1 - t0).total_seconds()
                return max(delta, 0.0)
            except (ValueError, TypeError):
                return None
        return None
