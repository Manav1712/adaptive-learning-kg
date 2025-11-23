"""
Dataclasses shared across the workflow_demo package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlanStep:
    """
    Describes a single tutoring step such as explain, example, or practice.
    """

    step_id: str
    step_type: str
    goal: str
    lo_id: Optional[int] = None
    content_id: Optional[str] = None
    budget_tokens: int = 250


@dataclass
class TeachingPack:
    """
    Bundled retrieval output containing key points and supporting content.
    """

    key_points: List[str] = field(default_factory=list)
    examples: List[dict] = field(default_factory=list)
    practice: List[dict] = field(default_factory=list)
    prerequisites: List[dict] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)


@dataclass
class SessionPlan:
    """
    High-level tutoring plan comprised of the current and future steps.
    """

    subject: str
    learning_objective: str
    mode: str
    current_plan: List[PlanStep]
    future_plan: List[PlanStep]
    first_question: str
    teaching_pack: TeachingPack

