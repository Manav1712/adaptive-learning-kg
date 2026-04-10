"""
Problem selector interface and default implementation.

A ``ProblemSelector`` picks one concrete problem from a list of candidates
returned by a ``ProblemBank``.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from .models import PracticeProblemRef


@runtime_checkable
class ProblemSelector(Protocol):
    """Choose one problem from a candidate list."""

    def select(self, candidates: List[PracticeProblemRef]) -> PracticeProblemRef: ...


class FirstMatchSelector:
    """Trivial selector that returns the first candidate."""

    def select(self, candidates: List[PracticeProblemRef]) -> PracticeProblemRef:
        if not candidates:
            raise ValueError("Cannot select from an empty candidate list.")
        return candidates[0]
