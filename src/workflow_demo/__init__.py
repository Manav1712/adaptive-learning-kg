"""
Workflow demo package that wires up a lightweight Coach → Retriever → Tutor loop.

Exports convenience factories for quickly instantiating the demo pipeline.
"""

from .coach import CoachAgent
from .planner import FAQPlanner, TutoringPlanner
from .retriever import TeachingPackRetriever

__all__ = [
    "CoachAgent",
    "TeachingPackRetriever",
    "TutoringPlanner",
    "FAQPlanner",
]

