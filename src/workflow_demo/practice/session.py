"""
Practice session manager — lifecycle hooks for the practice loop.

``PracticeSessionManager`` owns all reads and writes to
``pedagogy_context["extensions"]["practice_session"]`` and
``pedagogy_context["extensions"]["sequencing"]``.

It is instantiated by ``BotSessionManager`` when the practice-loop feature
flag is enabled.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from .feature_flags import PracticeFeatureFlags
from .models import (
    PracticeProblemRef,
    ProblemAttempt,
    ProblemEpisodeTrace,
    ProblemObservation,
    PracticeSessionState,
    SequencerState,
)
from .problem_bank import ProblemBank, StubProblemBank
from .problem_selector import FirstMatchSelector, ProblemSelector
from .problem_sequencer import NoOpProblemSequencer, ProblemSequencer

EventEmitter = Callable[..., Any]


@runtime_checkable
class _ObservationFilterLike(Protocol):
    def summarize(self, trace: ProblemEpisodeTrace) -> ProblemObservation: ...


class PracticeSessionManager:
    """Manages the practice-loop lifecycle within a tutor session.

    This class is the single owner of all practice and sequencing state in
    ``pedagogy_context["extensions"]``.  ``BotSessionManager`` delegates to it
    but never writes those keys directly.
    """

    def __init__(
        self,
        flags: PracticeFeatureFlags,
        *,
        problem_bank: Optional[ProblemBank] = None,
        problem_selector: Optional[ProblemSelector] = None,
        problem_sequencer: Optional[ProblemSequencer] = None,
        observation_filter: Optional[_ObservationFilterLike] = None,
        event_emitter: Optional[EventEmitter] = None,
    ) -> None:
        self.flags = flags
        self.bank: ProblemBank = problem_bank or StubProblemBank()
        self.selector: ProblemSelector = problem_selector or FirstMatchSelector()
        self.sequencer: ProblemSequencer = problem_sequencer or NoOpProblemSequencer()
        self.obs_filter: Optional[_ObservationFilterLike] = observation_filter
        self._emit = event_emitter

    # ------------------------------------------------------------------
    # Initialization (called once when a tutor session begins)
    # ------------------------------------------------------------------

    def seed_extensions(
        self,
        extensions: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write initial practice-session and sequencing state into *extensions*.

        Returns the updated extensions dict (mutated in place for convenience).
        """
        seq_state = self.sequencer.initialize(context or {})
        ps_state = PracticeSessionState(active=True)

        extensions["practice_session"] = ps_state.to_dict()
        extensions["sequencing"] = seq_state.to_dict()
        return extensions

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_practice_state(extensions: Dict[str, Any]) -> PracticeSessionState:
        raw = extensions.get("practice_session")
        if isinstance(raw, dict):
            return PracticeSessionState.from_dict(raw)
        return PracticeSessionState()

    @staticmethod
    def _load_sequencer_state(extensions: Dict[str, Any]) -> SequencerState:
        raw = extensions.get("sequencing")
        if isinstance(raw, dict):
            return SequencerState.from_dict(raw)
        return SequencerState()

    @staticmethod
    def _save(
        extensions: Dict[str, Any],
        ps: PracticeSessionState,
        seq: SequencerState,
    ) -> None:
        extensions["practice_session"] = ps.to_dict()
        extensions["sequencing"] = seq.to_dict()

    # ------------------------------------------------------------------
    # Lifecycle hooks (called explicitly in Round 2)
    # ------------------------------------------------------------------

    def begin_practice_problem(
        self,
        extensions: Dict[str, Any],
        *,
        lo_id: Optional[str] = None,
        difficulty_override: Optional[int] = None,
    ) -> Optional[PracticeProblemRef]:
        """Serve the next practice problem and update extensions.

        Returns the selected ``PracticeProblemRef``, or ``None`` if no
        candidates are available.
        """
        ps = self._load_practice_state(extensions)
        seq = self._load_sequencer_state(extensions)

        difficulty = (
            difficulty_override
            if difficulty_override is not None
            else (
                self.sequencer.choose_first_difficulty(seq)
                if ps.problem_index == 0
                else self.sequencer.choose_next_difficulty(seq)
            )
        )

        candidates = self.bank.get_candidates(lo_id, difficulty)
        if not candidates:
            return None

        problem = self.selector.select(candidates)
        episode = ProblemEpisodeTrace(problem=problem)

        ps.current_problem = problem
        ps.current_episode_trace = episode
        ps.active = True
        seq.current_difficulty = difficulty

        self._save(extensions, ps, seq)

        if self._emit:
            self._emit(
                "practice_problem_started",
                "Served practice problem.",
                phase="practice",
                problem_id=problem.problem_id,
                difficulty=difficulty,
                problem_index=ps.problem_index,
            )

        return problem

    def record_problem_attempt(
        self,
        extensions: Dict[str, Any],
        *,
        submission_text: Optional[str] = None,
        is_correct: Optional[bool] = None,
        feedback_text: Optional[str] = None,
    ) -> None:
        """Record one student attempt within the current problem episode."""
        ps = self._load_practice_state(extensions)
        if ps.current_episode_trace is None:
            return

        attempt = ProblemAttempt(
            attempt_index=len(ps.current_episode_trace.attempts),
            submission_text=submission_text,
            is_correct=is_correct,
            feedback_text=feedback_text,
        )
        ps.current_episode_trace.append_attempt(attempt)
        seq = self._load_sequencer_state(extensions)
        self._save(extensions, ps, seq)

    def record_problem_chat_turn(self, extensions: Dict[str, Any]) -> None:
        """Increment the chat-turn counter for the current problem episode."""
        ps = self._load_practice_state(extensions)
        if ps.current_episode_trace is None:
            return
        ps.current_episode_trace.record_chat_turn()
        seq = self._load_sequencer_state(extensions)
        self._save(extensions, ps, seq)

    def finalize_problem_episode(
        self,
        extensions: Dict[str, Any],
        *,
        solved: bool = False,
        abandoned: bool = False,
    ) -> Optional[ProblemEpisodeTrace]:
        """Finalize the current problem episode and advance the practice loop.

        When an ``observation_filter`` is configured, it is run on the
        episode trace and the result is stored in the sequencing debug
        state.  The sequencer's ``update_after_problem`` is called
        regardless (it receives the raw trace).

        Returns the finalized ``ProblemEpisodeTrace``, or ``None`` if no
        active episode exists.
        """
        ps = self._load_practice_state(extensions)
        seq = self._load_sequencer_state(extensions)

        if ps.current_episode_trace is None:
            return None

        episode = ps.current_episode_trace
        episode.finalize(solved=solved, abandoned=abandoned)

        observation: Optional[ProblemObservation] = None
        if self.obs_filter is not None:
            observation = self.obs_filter.summarize(episode)

        seq = self.sequencer.update_after_problem(seq, episode)

        if observation is not None:
            seq.debug["last_observation"] = observation.to_dict()

        ps.completed_episodes.append(episode)
        ps.current_problem = None
        ps.current_episode_trace = None
        ps.problem_index += 1
        seq.step_index = ps.problem_index

        self._save(extensions, ps, seq)

        if self._emit:
            obs_dict = observation.to_dict() if observation else None
            self._emit(
                "practice_problem_completed",
                "Practice problem finalized.",
                phase="practice",
                problem_id=episode.problem.problem_id,
                solved=solved,
                abandoned=abandoned,
                attempt_count=len(episode.attempts),
                chat_turn_count=episode.chat_turn_count,
                problem_index=ps.problem_index,
                observation=obs_dict,
                current_difficulty=seq.current_difficulty,
                last_difficulty=seq.last_difficulty,
                difficulty_reason=seq.debug.get("difficulty_reason"),
            )

        return episode

    def select_next_problem(
        self,
        extensions: Dict[str, Any],
        *,
        lo_id: Optional[str] = None,
    ) -> Optional[PracticeProblemRef]:
        """Convenience: choose next difficulty via sequencer, then serve."""
        return self.begin_practice_problem(extensions, lo_id=lo_id)

    # ------------------------------------------------------------------
    # Snapshot helper
    # ------------------------------------------------------------------

    @staticmethod
    def build_snapshot(extensions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a compact practice-session snapshot for the API/debug layer.

        Returns ``None`` when the practice session extension is absent.
        """
        raw_ps = extensions.get("practice_session")
        if not isinstance(raw_ps, dict):
            return None
        raw_seq = extensions.get("sequencing")
        seq = raw_seq if isinstance(raw_seq, dict) else {}
        seq_debug = seq.get("debug") if isinstance(seq.get("debug"), dict) else {}

        last_obs = seq_debug.get("last_observation")
        last_obs_summary: Optional[Dict[str, Any]] = None
        if isinstance(last_obs, dict):
            last_obs_summary = {
                "meaningful_attempts": last_obs.get("meaningful_attempts"),
                "raw_attempt_count": last_obs.get("raw_attempt_count"),
                "help_turn_count": last_obs.get("help_turn_count"),
                "solved": last_obs.get("solved"),
            }

        return {
            "practice_session": {
                "active": raw_ps.get("active", False),
                "current_problem_id": (
                    (raw_ps.get("current_problem") or {}).get("problem_id")
                ),
                "current_difficulty": seq.get("current_difficulty"),
                "problems_completed": len(raw_ps.get("completed_episodes") or []),
            },
            "sequencing": {
                "mode": seq.get("mode", "off"),
                "current_difficulty": seq.get("current_difficulty"),
                "last_difficulty": seq.get("last_difficulty"),
                "step_index": seq.get("step_index", 0),
                "struggle_level": seq_debug.get("struggle_level"),
                "difficulty_reason": seq_debug.get("difficulty_reason"),
                "last_observation": last_obs_summary,
            },
        }
