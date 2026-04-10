"""
Practice replay artifact and offline evaluator (Round 4).

Provides:
- ``PracticeReplayArtifact`` — serializable record of a completed practice
  session, sufficient for offline comparison of different sequencer
  implementations.
- ``replay_episodes`` — lightweight evaluator that feeds a list of completed
  episode traces through any ``ProblemSequencer`` and returns the sequence
  of difficulty decisions for comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .models import (
    ProblemEpisodeTrace,
    ProblemObservation,
    SequencerState,
)
from .problem_sequencer import ProblemSequencer


# ------------------------------------------------------------------
# Per-episode replay record
# ------------------------------------------------------------------

@dataclass
class EpisodeReplayRecord:
    """One completed problem episode with its sequencer context."""

    episode_index: int
    problem_id: str
    lo_id: Optional[str] = None
    lo_title: Optional[str] = None
    difficulty_at_serve: int = 1
    solved: bool = False
    abandoned: bool = False
    attempt_count: int = 0
    chat_turn_count: int = 0
    time_on_problem_sec: Optional[float] = None
    observation: Optional[Dict[str, Any]] = None
    sequencer_state_before: Optional[Dict[str, Any]] = None
    sequencer_state_after: Optional[Dict[str, Any]] = None
    chosen_next_difficulty: Optional[int] = None
    struggle_level: Optional[str] = None
    difficulty_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeReplayRecord":
        return cls(
            episode_index=data.get("episode_index", 0),
            problem_id=data.get("problem_id", ""),
            lo_id=data.get("lo_id"),
            lo_title=data.get("lo_title"),
            difficulty_at_serve=data.get("difficulty_at_serve", 1),
            solved=data.get("solved", False),
            abandoned=data.get("abandoned", False),
            attempt_count=data.get("attempt_count", 0),
            chat_turn_count=data.get("chat_turn_count", 0),
            time_on_problem_sec=data.get("time_on_problem_sec"),
            observation=data.get("observation"),
            sequencer_state_before=data.get("sequencer_state_before"),
            sequencer_state_after=data.get("sequencer_state_after"),
            chosen_next_difficulty=data.get("chosen_next_difficulty"),
            struggle_level=data.get("struggle_level"),
            difficulty_reason=data.get("difficulty_reason"),
        )


# ------------------------------------------------------------------
# Session-level replay artifact
# ------------------------------------------------------------------

@dataclass
class PracticeReplayArtifact:
    """Serializable replay artifact for one practice session.

    Contains enough information to replay sequencer decisions offline.
    """

    session_id: Optional[str] = None
    learner_session_id: Optional[str] = None
    sequencer_mode: str = "off"
    initial_difficulty: int = 1
    episodes: List[EpisodeReplayRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "learner_session_id": self.learner_session_id,
            "sequencer_mode": self.sequencer_mode,
            "initial_difficulty": self.initial_difficulty,
            "episodes": [e.to_dict() for e in self.episodes],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PracticeReplayArtifact":
        return cls(
            session_id=data.get("session_id"),
            learner_session_id=data.get("learner_session_id"),
            sequencer_mode=data.get("sequencer_mode", "off"),
            initial_difficulty=data.get("initial_difficulty", 1),
            episodes=[
                EpisodeReplayRecord.from_dict(e) for e in data.get("episodes") or []
            ],
            metadata=data.get("metadata") or {},
        )

    @classmethod
    def from_extensions(
        cls,
        extensions: Dict[str, Any],
        *,
        session_id: Optional[str] = None,
        learner_session_id: Optional[str] = None,
        observation_filter: Optional[Any] = None,
    ) -> "PracticeReplayArtifact":
        """Build a replay artifact from live ``extensions`` state.

        Iterates completed episodes and reconstructs per-episode records.
        """
        raw_ps = extensions.get("practice_session", {})
        raw_seq = extensions.get("sequencing", {})
        completed = raw_ps.get("completed_episodes") or []

        episodes: List[EpisodeReplayRecord] = []
        for idx, ep_raw in enumerate(completed):
            if isinstance(ep_raw, dict):
                ep = ProblemEpisodeTrace.from_dict(ep_raw)
            else:
                ep = ep_raw

            obs_dict: Optional[Dict[str, Any]] = None
            if observation_filter is not None:
                try:
                    obs = observation_filter.summarize(ep)
                    obs_dict = obs.to_dict()
                except Exception:
                    pass

            episodes.append(EpisodeReplayRecord(
                episode_index=idx,
                problem_id=ep.problem.problem_id,
                lo_id=ep.problem.lo_id,
                lo_title=ep.problem.lo_title,
                difficulty_at_serve=ep.problem.difficulty,
                solved=ep.solved,
                abandoned=ep.abandoned,
                attempt_count=len(ep.attempts),
                chat_turn_count=ep.chat_turn_count,
                time_on_problem_sec=ep.time_on_problem_sec,
                observation=obs_dict,
            ))

        return cls(
            session_id=session_id,
            learner_session_id=learner_session_id,
            sequencer_mode=raw_seq.get("mode", "off"),
            initial_difficulty=raw_seq.get("current_difficulty", 1),
            episodes=episodes,
            metadata={},
        )


# ------------------------------------------------------------------
# Offline replay / evaluator
# ------------------------------------------------------------------

@dataclass
class ReplayDecision:
    """One step of a replayed sequencer decision."""

    episode_index: int
    problem_id: str
    difficulty_before: int
    difficulty_after: int
    struggle_level: Optional[str] = None
    difficulty_reason: Optional[str] = None
    posterior_summary: Optional[Dict[str, Any]] = None
    decision_margin: Optional[float] = None
    served_difficulty: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def replay_episodes(
    episodes: List[ProblemEpisodeTrace],
    sequencer: ProblemSequencer,
    *,
    initial_context: Optional[Dict[str, Any]] = None,
) -> List[ReplayDecision]:
    """Replay a list of completed episodes through a sequencer.

    Initialises the sequencer, then feeds each episode through
    ``update_after_problem`` in order, recording the difficulty decision
    at each step.

    This is intentionally deterministic: the same episodes + sequencer
    will always produce the same sequence of decisions.
    """
    state = sequencer.initialize(initial_context or {})
    decisions: List[ReplayDecision] = []

    for idx, episode in enumerate(episodes):
        difficulty_before = state.current_difficulty
        state = sequencer.update_after_problem(state, episode)

        debug = state.debug or {}
        posterior_summary: Optional[Dict[str, Any]] = None
        decision_margin: Optional[float] = None
        served_diff: Optional[int] = None

        if state.mode == "pomdp":
            posterior_summary = {
                "posterior_expected_effort": state.posterior_expected_effort,
                "posterior_expected_tau": state.posterior_expected_tau,
                "active_particle_count": state.active_particle_count,
                "belief_ess": debug.get("belief_ess"),
            }
            decision_margin = debug.get("decision_margin")
            served_diff = debug.get("served_difficulty")
            if served_diff is None:
                served_diff = state.last_difficulty

        decisions.append(ReplayDecision(
            episode_index=idx,
            problem_id=episode.problem.problem_id,
            difficulty_before=difficulty_before,
            difficulty_after=state.current_difficulty,
            struggle_level=debug.get("struggle_level"),
            difficulty_reason=debug.get("difficulty_reason"),
            posterior_summary=posterior_summary,
            decision_margin=decision_margin,
            served_difficulty=served_diff,
        ))

    return decisions
