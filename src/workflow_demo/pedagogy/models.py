"""
Pydantic models for the optional pedagogical decision layer (Phase 0).

These types are not yet consumed by coach, planner, or tutor; they define
the contract for later phases.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import RetrievalIntent, TeachingMoveType


class AttemptRecord(BaseModel):
    """One observable student attempt or response slice within a session."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    attempt_id: str = Field(min_length=1, max_length=128)
    turn_index: int = Field(ge=0)
    lo_id: Optional[int] = Field(default=None, ge=0)
    is_correct: Optional[bool] = None
    student_response_excerpt: str = Field(default="", max_length=4000)
    latency_ms: Optional[int] = Field(default=None, ge=0, le=3_600_000)
    created_at_iso: Optional[str] = Field(default=None, max_length=64)


class LearnerState(BaseModel):
    """Compact, layer-local view of the learner (orthogonal to SessionMemory shape)."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    # Canonical Phase 1 fields
    active_session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        validation_alias=AliasChoices("active_session_id", "session_id"),
    )
    current_focus_lo: Optional[str] = Field(
        default=None,
        max_length=256,
        validation_alias=AliasChoices("current_focus_lo", "active_lo_id"),
    )
    mastery: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("mastery", "lo_mastery_proxy"),
    )
    misconceptions: dict[str, list[str]] = Field(default_factory=dict)
    recent_attempts: list[AttemptRecord] = Field(default_factory=list, max_length=32)
    hint_history: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_misconceptions(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        misconceptions = data.get("misconceptions")
        if isinstance(misconceptions, list):
            data = dict(data)
            data["misconceptions"] = {"__legacy__": [str(item) for item in misconceptions]}
        return data

    @field_validator("current_focus_lo", mode="before")
    @classmethod
    def _coerce_focus_to_string(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("mastery")
    @classmethod
    def _mastery_in_unit_interval(cls, v: dict[str, float]) -> dict[str, float]:
        for key, val in v.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"mastery[{key!r}] must be in [0.0, 1.0], got {val!r}"
                )
        return v

    @field_validator("misconceptions")
    @classmethod
    def _normalize_misconceptions(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for target_lo, entries in value.items():
            key = str(target_lo).strip() or "unknown"
            deduped: list[str] = []
            for item in entries:
                text = str(item).strip()
                if text and text not in deduped:
                    deduped.append(text)
            normalized[key] = deduped
        return normalized


class MisconceptionDiagnosis(BaseModel):
    """Structured output of a misconception hypothesis (diagnoser phase)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    target_lo: str = Field(
        min_length=1,
        max_length=256,
        validation_alias=AliasChoices("target_lo", "target"),
    )
    suspected_misconception: str = Field(
        min_length=1,
        max_length=512,
        validation_alias=AliasChoices("suspected_misconception", "label", "code"),
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=4000)
    prerequisite_gap_los: list[str] = Field(default_factory=list, max_length=32)
    evidence_quotes: list[str] = Field(default_factory=list, max_length=16)
    # Legacy compatibility inputs (excluded from serialization).
    code: Optional[str] = Field(default=None, exclude=True, max_length=128)
    label: Optional[str] = Field(default=None, exclude=True, max_length=512)
    related_lo_ids: list[int] = Field(default_factory=list, exclude=True, max_length=32)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        upgraded = dict(data)
        if "target_lo" not in upgraded:
            upgraded["target_lo"] = (
                upgraded.get("target")
                or upgraded.get("current_focus_lo")
                or "unknown"
            )
        if "suspected_misconception" not in upgraded:
            upgraded["suspected_misconception"] = (
                upgraded.get("label")
                or upgraded.get("code")
                or "uncertain_or_unknown"
            )
        if "rationale" not in upgraded:
            if upgraded.get("evidence_quotes"):
                upgraded["rationale"] = "Evidence observed in learner response."
            else:
                upgraded["rationale"] = ""
        if "prerequisite_gap_los" not in upgraded:
            related = upgraded.get("related_lo_ids") or []
            upgraded["prerequisite_gap_los"] = [str(item) for item in related]
        return upgraded


class TeachingMoveCandidate(BaseModel):
    """One scored option the policy may choose."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    move_type: TeachingMoveType
    priority_score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=4000)
    retrieval_intents: list[RetrievalIntent] = Field(default_factory=list, max_length=16)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=32)


class PolicyDecision(BaseModel):
    """Result of policy scoring over candidate moves."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chosen: TeachingMoveCandidate
    alternatives: list[TeachingMoveCandidate] = Field(default_factory=list, max_length=16)
    policy_version: str = Field(default="0", max_length=64)
    trace_notes: str = Field(default="", max_length=4000)


class CriticVerdict(BaseModel):
    """Outcome of a post-generation pedagogical check."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    approved: bool
    severity: str = Field(default="none", max_length=32)
    violations: list[str] = Field(default_factory=list, max_length=32)
    revision_hint: Optional[str] = Field(default=None, max_length=2000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PedagogicalContext(BaseModel):
    """Bundle carried alongside tutoring handoffs when the layer is active."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    layer_version: str = Field(default="0", max_length=32)
    learner_state: LearnerState
    diagnosis: Optional[MisconceptionDiagnosis] = None
    policy: Optional[PolicyDecision] = None
    last_critic: Optional[CriticVerdict] = None
    active_move: Optional[TeachingMoveCandidate] = None
    extensions: dict[str, Any] = Field(default_factory=dict, max_length=16)
