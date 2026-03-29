"""
Canonical teaching-move and retrieval-intent labels for the pedagogy layer.

These values are stable API surface for planners, policy, and telemetry.
"""

from __future__ import annotations

from enum import Enum


class TeachingMoveType(str, Enum):
    """High-level categories of tutor/coach pedagogical actions."""

    EXPLAIN_CONCEPT = "explain_concept"
    WORKED_EXAMPLE = "worked_example"
    SCAFFOLDED_QUESTION = "scaffolded_question"
    PROBING_QUESTION = "probing_question"
    CORRECTIVE_FEEDBACK = "corrective_feedback"
    SUMMARIZE_AND_CHECK = "summarize_and_check"
    METACOGNITIVE_PROMPT = "metacognitive_prompt"
    CONNECT_PRIOR_KNOWLEDGE = "connect_prior_knowledge"


class RetrievalIntent(str, Enum):
    """Reasons to invoke or bias retrieval from the existing retriever."""

    PREREQUISITE_REFRESH = "prerequisite_refresh"
    DEFINITION_SNIPPET = "definition_snippet"
    WORKED_PARALLEL = "worked_parallel"
    COUNTER_EXAMPLE = "counter_example"
    PRACTICE_ITEM = "practice_item"
    VISUAL_OR_DIAGRAM = "visual_or_diagram"
    MISCONCEPTION_REPAIR = "misconception_repair"
