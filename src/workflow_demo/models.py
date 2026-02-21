"""
Dataclasses shared across the workflow_demo package.

Simplified plan structure per meeting requirements:
- current_plan: 1 primary LO + up to 2 dependent LOs (all same mode)
- future_plan: 1 LO
- Each LO includes proficiency score and teaching notes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class LearningObjectiveEntry:
    """
    A single learning objective with proficiency and teaching guidance.
    
    Used in both current_plan and future_plan.
    """
    lo_id: int
    title: str
    proficiency: float  # 0.0 - 1.0, from student profile
    how_to_teach: str   # Instructional approach from KG
    why_to_teach: str   # Pedagogical rationale from KG
    notes: str = ""     # Short qualitative notes (e.g., "mastered because...")
    is_primary: bool = False  # True for the main LO, False for dependents


@dataclass
class SimplifiedPlan:
    """
    Simplified tutoring plan structure.
    
    current_plan: 1 primary LO + up to 2 dependent LOs (prereqs)
    future_plan: 1 LO for next session
    All LOs share the same mode.
    """
    subject: str
    mode: str  # "conceptual_review" | "examples" | "practice"
    current_plan: List[LearningObjectiveEntry]  # 1 primary + up to 2 dependents
    future_plan: List[LearningObjectiveEntry]   # 1 LO
    
    # Metadata from KG
    book: Optional[str] = None
    unit: Optional[str] = None
    chapter: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "subject": self.subject,
            "mode": self.mode,
            "current_plan": [
                {
                    "lo_id": lo.lo_id,
                    "title": lo.title,
                    "proficiency": lo.proficiency,
                    "how_to_teach": lo.how_to_teach,
                    "why_to_teach": lo.why_to_teach,
                    "notes": lo.notes,
                    "is_primary": lo.is_primary,
                }
                for lo in self.current_plan
            ],
            "future_plan": [
                {
                    "lo_id": lo.lo_id,
                    "title": lo.title,
                    "proficiency": lo.proficiency,
                    "how_to_teach": lo.how_to_teach,
                    "why_to_teach": lo.why_to_teach,
                    "notes": lo.notes,
                    "is_primary": lo.is_primary,
                }
                for lo in self.future_plan
            ],
            "book": self.book,
            "unit": self.unit,
            "chapter": self.chapter,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimplifiedPlan":
        """Create from dict."""
        current_plan = [
            LearningObjectiveEntry(
                lo_id=lo["lo_id"],
                title=lo["title"],
                proficiency=lo["proficiency"],
                how_to_teach=lo["how_to_teach"],
                why_to_teach=lo["why_to_teach"],
                notes=lo.get("notes", ""),
                is_primary=lo.get("is_primary", False),
            )
            for lo in data.get("current_plan", [])
        ]
        future_plan = [
            LearningObjectiveEntry(
                lo_id=lo["lo_id"],
                title=lo["title"],
                proficiency=lo["proficiency"],
                how_to_teach=lo["how_to_teach"],
                why_to_teach=lo["why_to_teach"],
                notes=lo.get("notes", ""),
                is_primary=lo.get("is_primary", False),
            )
            for lo in data.get("future_plan", [])
        ]
        return cls(
            subject=data.get("subject", ""),
            mode=data.get("mode", "conceptual_review"),
            current_plan=current_plan,
            future_plan=future_plan,
            book=data.get("book"),
            unit=data.get("unit"),
            chapter=data.get("chapter"),
        )


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


# Legacy dataclasses kept for backwards compatibility during transition
# TODO: Remove after full migration to SimplifiedPlan

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


@dataclass
class MultimodalInput:
    """
    Student input that may include an image reference.
    """
    text: str
    image: Optional[str] = None  # File path or URL
    image_query: Optional[str] = None  # Text query derived from the image (OCR)
