"""
Data models for the adaptive practice loop and problem-sequencing subsystem.

All models are plain dataclasses with dict serialization so they can be stored
inside ``pedagogy_context["extensions"]`` without Pydantic dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------
# Problem reference
# ------------------------------------------------------------------

@dataclass
class PracticeProblemRef:
    """Immutable reference to one practice problem."""

    problem_id: str
    difficulty: int  # 0..3
    prompt_text: str
    lo_id: Optional[str] = None
    lo_title: Optional[str] = None
    problem_type: str = "short_answer"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PracticeProblemRef":
        return cls(
            problem_id=data["problem_id"],
            difficulty=data["difficulty"],
            prompt_text=data["prompt_text"],
            lo_id=data.get("lo_id"),
            lo_title=data.get("lo_title"),
            problem_type=data.get("problem_type", "short_answer"),
            metadata=data.get("metadata") or {},
        )


# ------------------------------------------------------------------
# Problem attempt
# ------------------------------------------------------------------

@dataclass
class ProblemAttempt:
    """One student submission or answer attempt within a problem episode."""

    attempt_index: int
    created_at: str = field(default_factory=_utc_now_iso)
    submission_text: Optional[str] = None
    is_correct: Optional[bool] = None
    feedback_text: Optional[str] = None
    trace_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemAttempt":
        return cls(
            attempt_index=data["attempt_index"],
            created_at=data.get("created_at", ""),
            submission_text=data.get("submission_text"),
            is_correct=data.get("is_correct"),
            feedback_text=data.get("feedback_text"),
            trace_metadata=data.get("trace_metadata") or {},
        )


# ------------------------------------------------------------------
# Problem episode trace
# ------------------------------------------------------------------

@dataclass
class ProblemEpisodeTrace:
    """Full trace of one practice-problem episode (from serve to completion)."""

    problem: PracticeProblemRef
    attempts: List[ProblemAttempt] = field(default_factory=list)
    chat_turn_count: int = 0
    started_at: str = field(default_factory=_utc_now_iso)
    completed_at: Optional[str] = None
    solved: bool = False
    abandoned: bool = False
    llm_meaningful_attempts: Optional[int] = None
    time_on_problem_sec: Optional[float] = None

    def append_attempt(self, attempt: ProblemAttempt) -> None:
        self.attempts.append(attempt)

    def record_chat_turn(self) -> None:
        self.chat_turn_count += 1

    def finalize(self, *, solved: bool = False, abandoned: bool = False) -> None:
        self.completed_at = _utc_now_iso()
        self.solved = solved
        self.abandoned = abandoned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "attempts": [a.to_dict() for a in self.attempts],
            "chat_turn_count": self.chat_turn_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "solved": self.solved,
            "abandoned": self.abandoned,
            "llm_meaningful_attempts": self.llm_meaningful_attempts,
            "time_on_problem_sec": self.time_on_problem_sec,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemEpisodeTrace":
        return cls(
            problem=PracticeProblemRef.from_dict(data["problem"]),
            attempts=[ProblemAttempt.from_dict(a) for a in data.get("attempts") or []],
            chat_turn_count=data.get("chat_turn_count", 0),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
            solved=data.get("solved", False),
            abandoned=data.get("abandoned", False),
            llm_meaningful_attempts=data.get("llm_meaningful_attempts"),
            time_on_problem_sec=data.get("time_on_problem_sec"),
        )


# ------------------------------------------------------------------
# Problem observation (output of observation filter)
# ------------------------------------------------------------------

@dataclass
class ProblemObservation:
    """Summarized observation from a completed problem episode."""

    meaningful_attempts: int
    raw_attempt_count: int
    time_on_problem_sec: Optional[float] = None
    help_turn_count: int = 0
    solved: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemObservation":
        return cls(
            meaningful_attempts=data["meaningful_attempts"],
            raw_attempt_count=data["raw_attempt_count"],
            time_on_problem_sec=data.get("time_on_problem_sec"),
            help_turn_count=data.get("help_turn_count", 0),
            solved=data.get("solved", False),
            debug=data.get("debug") or {},
        )


# ------------------------------------------------------------------
# Sequencer state
# ------------------------------------------------------------------

SequencerMode = Literal["off", "heuristic", "pomdp"]


@dataclass
class SequencerState:
    """Mutable state for the between-problem sequencing engine."""

    mode: SequencerMode = "off"
    step_index: int = 0
    last_difficulty: Optional[int] = None
    current_difficulty: int = 1
    recent_observations: List[int] = field(default_factory=list)
    posterior_expected_effort: Optional[float] = None
    posterior_expected_tau: Optional[float] = None
    active_particle_count: Optional[int] = None
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SequencerState":
        return cls(
            mode=data.get("mode", "off"),
            step_index=data.get("step_index", 0),
            last_difficulty=data.get("last_difficulty"),
            current_difficulty=data.get("current_difficulty", 1),
            recent_observations=data.get("recent_observations") or [],
            posterior_expected_effort=data.get("posterior_expected_effort"),
            posterior_expected_tau=data.get("posterior_expected_tau"),
            active_particle_count=data.get("active_particle_count"),
            debug=data.get("debug") or {},
        )


# ------------------------------------------------------------------
# Practice session state (aggregate stored in extensions)
# ------------------------------------------------------------------

@dataclass
class PracticeSessionState:
    """Aggregate state for one active practice-loop session."""

    active: bool = False
    current_problem: Optional[PracticeProblemRef] = None
    current_episode_trace: Optional[ProblemEpisodeTrace] = None
    completed_episodes: List[ProblemEpisodeTrace] = field(default_factory=list)
    problem_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "current_problem": self.current_problem.to_dict() if self.current_problem else None,
            "current_episode_trace": (
                self.current_episode_trace.to_dict() if self.current_episode_trace else None
            ),
            "completed_episodes": [e.to_dict() for e in self.completed_episodes],
            "problem_index": self.problem_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PracticeSessionState":
        cp = data.get("current_problem")
        cet = data.get("current_episode_trace")
        return cls(
            active=data.get("active", False),
            current_problem=PracticeProblemRef.from_dict(cp) if cp else None,
            current_episode_trace=ProblemEpisodeTrace.from_dict(cet) if cet else None,
            completed_episodes=[
                ProblemEpisodeTrace.from_dict(e) for e in data.get("completed_episodes") or []
            ],
            problem_index=data.get("problem_index", 0),
        )
