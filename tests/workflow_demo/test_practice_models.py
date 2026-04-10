"""Round 2 tests — practice loop data models."""

from __future__ import annotations

import pytest

from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    PracticeSessionState,
    ProblemAttempt,
    ProblemEpisodeTrace,
    ProblemObservation,
    SequencerState,
)


# ------------------------------------------------------------------
# PracticeProblemRef
# ------------------------------------------------------------------

class TestPracticeProblemRef:

    def test_create_with_required_fields(self):
        p = PracticeProblemRef(problem_id="p1", difficulty=2, prompt_text="Solve x^2=4.")
        assert p.problem_id == "p1"
        assert p.difficulty == 2
        assert p.prompt_text == "Solve x^2=4."
        assert p.problem_type == "short_answer"
        assert p.lo_id is None
        assert p.metadata == {}

    def test_create_with_optional_fields(self):
        p = PracticeProblemRef(
            problem_id="p2", difficulty=0, prompt_text="Easy.",
            lo_id="lo-42", lo_title="Derivatives", problem_type="math",
            metadata={"source": "bank"},
        )
        assert p.lo_id == "lo-42"
        assert p.lo_title == "Derivatives"
        assert p.metadata["source"] == "bank"

    def test_to_dict_round_trip(self):
        original = PracticeProblemRef(
            problem_id="p3", difficulty=1, prompt_text="Diff.",
            lo_id="lo-1", metadata={"k": "v"},
        )
        d = original.to_dict()
        restored = PracticeProblemRef.from_dict(d)
        assert restored.problem_id == original.problem_id
        assert restored.difficulty == original.difficulty
        assert restored.lo_id == original.lo_id
        assert restored.metadata == original.metadata


# ------------------------------------------------------------------
# ProblemAttempt
# ------------------------------------------------------------------

class TestProblemAttempt:

    def test_create_and_defaults(self):
        a = ProblemAttempt(attempt_index=0)
        assert a.attempt_index == 0
        assert a.submission_text is None
        assert a.is_correct is None
        assert a.created_at  # non-empty timestamp

    def test_create_with_values(self):
        a = ProblemAttempt(
            attempt_index=1, submission_text="x=2", is_correct=True,
            feedback_text="Correct!", trace_metadata={"latency_ms": 1200},
        )
        assert a.is_correct is True
        assert a.trace_metadata["latency_ms"] == 1200

    def test_to_dict_round_trip(self):
        original = ProblemAttempt(attempt_index=2, submission_text="y=3")
        d = original.to_dict()
        restored = ProblemAttempt.from_dict(d)
        assert restored.attempt_index == original.attempt_index
        assert restored.submission_text == original.submission_text


# ------------------------------------------------------------------
# ProblemEpisodeTrace
# ------------------------------------------------------------------

class TestProblemEpisodeTrace:

    @pytest.fixture()
    def problem(self):
        return PracticeProblemRef(problem_id="ep1", difficulty=1, prompt_text="Solve.")

    def test_create_empty_episode(self, problem):
        ep = ProblemEpisodeTrace(problem=problem)
        assert ep.problem.problem_id == "ep1"
        assert ep.attempts == []
        assert ep.chat_turn_count == 0
        assert ep.solved is False
        assert ep.abandoned is False
        assert ep.completed_at is None

    def test_append_attempt(self, problem):
        ep = ProblemEpisodeTrace(problem=problem)
        a1 = ProblemAttempt(attempt_index=0, submission_text="x=1")
        a2 = ProblemAttempt(attempt_index=1, submission_text="x=2", is_correct=True)
        ep.append_attempt(a1)
        ep.append_attempt(a2)
        assert len(ep.attempts) == 2
        assert ep.attempts[1].is_correct is True

    def test_record_chat_turn(self, problem):
        ep = ProblemEpisodeTrace(problem=problem)
        ep.record_chat_turn()
        ep.record_chat_turn()
        assert ep.chat_turn_count == 2

    def test_finalize_solved(self, problem):
        ep = ProblemEpisodeTrace(problem=problem)
        ep.finalize(solved=True)
        assert ep.solved is True
        assert ep.abandoned is False
        assert ep.completed_at is not None

    def test_finalize_abandoned(self, problem):
        ep = ProblemEpisodeTrace(problem=problem)
        ep.finalize(abandoned=True)
        assert ep.abandoned is True
        assert ep.solved is False

    def test_to_dict_round_trip(self, problem):
        ep = ProblemEpisodeTrace(problem=problem)
        ep.append_attempt(ProblemAttempt(attempt_index=0, submission_text="a"))
        ep.record_chat_turn()
        ep.finalize(solved=True)

        d = ep.to_dict()
        restored = ProblemEpisodeTrace.from_dict(d)
        assert restored.problem.problem_id == "ep1"
        assert len(restored.attempts) == 1
        assert restored.chat_turn_count == 1
        assert restored.solved is True
        assert restored.completed_at is not None


# ------------------------------------------------------------------
# ProblemObservation
# ------------------------------------------------------------------

class TestProblemObservation:

    def test_create_with_required_fields(self):
        obs = ProblemObservation(meaningful_attempts=3, raw_attempt_count=5)
        assert obs.meaningful_attempts == 3
        assert obs.raw_attempt_count == 5
        assert obs.time_on_problem_sec is None
        assert obs.help_turn_count == 0
        assert obs.solved is False

    def test_all_fields(self):
        obs = ProblemObservation(
            meaningful_attempts=2, raw_attempt_count=4,
            time_on_problem_sec=120.5, help_turn_count=3,
            solved=True, debug={"method": "heuristic"},
        )
        assert obs.time_on_problem_sec == 120.5
        assert obs.debug["method"] == "heuristic"

    def test_to_dict_round_trip(self):
        original = ProblemObservation(meaningful_attempts=1, raw_attempt_count=2, solved=True)
        d = original.to_dict()
        restored = ProblemObservation.from_dict(d)
        assert restored.meaningful_attempts == original.meaningful_attempts
        assert restored.solved is True


# ------------------------------------------------------------------
# SequencerState
# ------------------------------------------------------------------

class TestSequencerState:

    def test_defaults(self):
        s = SequencerState()
        assert s.mode == "off"
        assert s.step_index == 0
        assert s.last_difficulty is None
        assert s.current_difficulty == 1
        assert s.recent_observations == []
        assert s.posterior_expected_effort is None

    def test_to_dict_round_trip(self):
        original = SequencerState(
            mode="heuristic", step_index=3, last_difficulty=1,
            current_difficulty=2, recent_observations=[1, 2, 3],
        )
        d = original.to_dict()
        restored = SequencerState.from_dict(d)
        assert restored.mode == "heuristic"
        assert restored.step_index == 3
        assert restored.last_difficulty == 1
        assert restored.current_difficulty == 2
        assert restored.recent_observations == [1, 2, 3]

    def test_from_dict_with_missing_keys(self):
        restored = SequencerState.from_dict({})
        assert restored.mode == "off"
        assert restored.current_difficulty == 1


# ------------------------------------------------------------------
# PracticeSessionState
# ------------------------------------------------------------------

class TestPracticeSessionState:

    def test_defaults(self):
        ps = PracticeSessionState()
        assert ps.active is False
        assert ps.current_problem is None
        assert ps.current_episode_trace is None
        assert ps.completed_episodes == []
        assert ps.problem_index == 0

    def test_to_dict_round_trip_empty(self):
        original = PracticeSessionState(active=True)
        d = original.to_dict()
        restored = PracticeSessionState.from_dict(d)
        assert restored.active is True
        assert restored.current_problem is None
        assert restored.completed_episodes == []

    def test_to_dict_round_trip_with_data(self):
        prob = PracticeProblemRef(problem_id="rt1", difficulty=2, prompt_text="Q.")
        ep = ProblemEpisodeTrace(problem=prob)
        ep.append_attempt(ProblemAttempt(attempt_index=0))
        ep.finalize(solved=True)

        original = PracticeSessionState(
            active=True,
            current_problem=prob,
            completed_episodes=[ep],
            problem_index=1,
        )
        d = original.to_dict()
        restored = PracticeSessionState.from_dict(d)
        assert restored.active is True
        assert restored.current_problem is not None
        assert restored.current_problem.problem_id == "rt1"
        assert len(restored.completed_episodes) == 1
        assert restored.completed_episodes[0].solved is True
        assert restored.problem_index == 1
