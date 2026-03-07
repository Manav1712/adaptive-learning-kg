"""
Workflow demo package that wires up a lightweight Coach → Retriever → Tutor loop.

Exports convenience factories for quickly instantiating the demo pipeline.
"""

from .coach_agent import CoachAgent
from .planner import FAQPlanner, TutoringPlanner
from .retriever import TeachingPackRetriever
from .runtime_factory import build_coach_runtime

__all__ = [
    "CoachAgent",
    "TeachingPackRetriever",
    "TutoringPlanner",
    "FAQPlanner",
    "build_coach_runtime",
]

