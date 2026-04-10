"""
Problem bank interface and stub implementation for the practice loop.

A ``ProblemBank`` provides candidate problems for a given LO and difficulty
bucket. The ``StubProblemBank`` returns hard-coded placeholders so that the
skeleton practice loop can be exercised without a real content backend.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from .models import PracticeProblemRef


@runtime_checkable
class ProblemBank(Protocol):
    """Retrieve candidate practice problems by LO and difficulty."""

    def get_candidates(
        self,
        lo_id: Optional[str],
        difficulty: int,
        *,
        limit: int = 5,
    ) -> List[PracticeProblemRef]: ...


class StubProblemBank:
    """Hard-coded problem bank for Round 2 scaffolding.

    Returns synthetic problems at every difficulty level so that lifecycle
    hooks can be tested without a real content source.
    """

    _BANK: List[PracticeProblemRef] = [
        PracticeProblemRef(
            problem_id="stub-easy-1",
            difficulty=0,
            prompt_text="Compute the derivative of f(x) = 3x + 2.",
            problem_type="short_answer",
        ),
        PracticeProblemRef(
            problem_id="stub-medium-1",
            difficulty=1,
            prompt_text="Find the derivative of f(x) = x^2 + 5x - 3.",
            problem_type="short_answer",
        ),
        PracticeProblemRef(
            problem_id="stub-hard-1",
            difficulty=2,
            prompt_text="Differentiate f(x) = sin(x) * e^x.",
            problem_type="short_answer",
        ),
        PracticeProblemRef(
            problem_id="stub-vhard-1",
            difficulty=3,
            prompt_text="Evaluate the integral of ln(x)/x^2 dx from 1 to infinity.",
            problem_type="short_answer",
        ),
    ]

    def get_candidates(
        self,
        lo_id: Optional[str],
        difficulty: int,
        *,
        limit: int = 5,
    ) -> List[PracticeProblemRef]:
        matches = [p for p in self._BANK if p.difficulty == difficulty]
        if lo_id is not None:
            lo_matches = [p for p in matches if p.lo_id == lo_id]
            if lo_matches:
                matches = lo_matches
        return matches[:limit]
