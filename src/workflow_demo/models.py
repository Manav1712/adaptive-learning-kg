"""
Dataclasses shared across the workflow_demo package.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class RetrievalCandidate:
    """
    A candidate LO or content item returned from retrieval.
    
    Used to pass retrieval results to the planner LLM.
    """
    lo_id: int
    title: str
    score: float  # Similarity score (0.0 - 1.0)
    source: str   # "text_embedding" | "image_embedding" | "bm25"
    
    # KG fields passed through
    book: Optional[str] = None
    unit: Optional[str] = None
    chapter: Optional[str] = None
    how_to_teach: Optional[str] = None
    why_to_teach: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lo_id": self.lo_id,
            "title": self.title,
            "score": self.score,
            "source": self.source,
            "book": self.book,
            "unit": self.unit,
            "chapter": self.chapter,
            "how_to_teach": self.how_to_teach,
            "why_to_teach": self.why_to_teach,
        }


@dataclass
class RetrievalResult:
    """
    Complete retrieval output including candidates from all sources.
    
    Passed to the planner LLM for plan generation.
    """
    query: str
    text_candidates: List[RetrievalCandidate]   # From text/OCR embeddings
    image_candidates: List[RetrievalCandidate]  # From image embeddings (CLIP)
    merged_candidates: List[RetrievalCandidate] # Combined + deduplicated
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "text_candidates": [c.to_dict() for c in self.text_candidates],
            "image_candidates": [c.to_dict() for c in self.image_candidates],
            "merged_candidates": [c.to_dict() for c in self.merged_candidates],
        }


# Dataclasses used by the current planning and retrieval flow.

@dataclass
class PlanStep:
    """
    LEGACY: Describes a single tutoring step.
    Kept for backwards compatibility.
    """
    step_id: str
    step_type: str
    goal: str
    lo_id: Optional[int] = None
    content_id: Optional[str] = None
    budget_tokens: int = 250
    how_to_teach: Optional[str] = None
    why_to_teach: Optional[str] = None


@dataclass
class TeachingPack:
    """
    LEGACY: Bundled retrieval output.
    Kept for backwards compatibility.
    """
    key_points: List[str] = field(default_factory=list)
    examples: List[dict] = field(default_factory=list)
    practice: List[dict] = field(default_factory=list)
    prerequisites: List[dict] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)
    images: List[dict] = field(default_factory=list)


@dataclass
class SessionPlan:
    """
    LEGACY: High-level tutoring plan.
    Kept for backwards compatibility.
    """
    subject: str
    learning_objective: str
    mode: str
    current_plan: List[PlanStep]
    future_plan: List[PlanStep]
    first_question: str
    teaching_pack: TeachingPack
    session_guidance: Optional[str] = None
    book: Optional[str] = None
    unit: Optional[str] = None
    chapter: Optional[str] = None

