"""
Workflow demo package that wires up a lightweight Coach → Retriever → Tutor loop.

Exports convenience factories for quickly instantiating the demo pipeline.
"""

from .coach import CoachAgent
from .retriever import TeachingPackRetriever
from .planner import TutoringPlanner, FAQPlanner
from .tutor import TutorAgent, FAQAgent

__all__ = [
    "CoachAgent",
    "TeachingPackRetriever",
    "TutoringPlanner",
    "FAQPlanner",
    "TutorAgent",
    "FAQAgent",
]

