"""
Minimal data models for Coach and Retriever.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievalRequest:
    """Minimal retrieval request built from user input."""
    query: str
    subject: Optional[str] = None
    intent: Optional[str] = None


@dataclass
class RetrievalResponse:
    """Minimal retrieval response with a flag and context bullets."""
    can_answer: bool
    minimal_context: List[str] = field(default_factory=list)


@dataclass
class CoachDecision:
    """Coach decision: action plus optional request to pass along."""
    action: str  # e.g., "retrieve", "ask_clarification", "answer"
    request: Optional[RetrievalRequest] = None
