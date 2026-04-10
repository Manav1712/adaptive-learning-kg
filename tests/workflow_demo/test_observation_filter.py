"""Tests for the heuristic ObservationFilter (Round 3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.workflow_demo.pedagogy.observation_filter import (
    HeuristicObservationFilter,
    ObservationFilter,
)
from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    ProblemAttempt,
    ProblemEpisodeTrace,
)


def _make_ref(**kw) -> PracticeProblemRef:
    defaults = dict(problem_id="p1", difficulty=1, prompt_text="What is 2+2?")
    return PracticeProblemRef(**(defaults | kw))


def _make_episode(
    *,
    attempts: list[ProblemAttempt] | None = None,
    chat_turns: int = 0,
    solved: bool = False,
    abandoned: bool = False,
    started_at: str | None = None,
    completed_at: str | None = None,
    time_on_problem_sec: float | None = None,
) -> ProblemEpisodeTrace:
    ep = ProblemEpisodeTrace(
        problem=_make_ref(),
        attempts=attempts or [],
        chat_turn_count=chat_turns,
        solved=solved,
        abandoned=abandoned,
        time_on_problem_sec=time_on_problem_sec,
    )
    if started_at is not None:
        ep.started_at = started_at
    if completed_at is not None:
        ep.completed_at = completed_at
    return ep


def _attempt(text: str = "some answer", *, correct: bool | None = None) -> ProblemAttempt:
    return ProblemAttempt(attempt_index=0, submission_text=text, is_correct=correct)


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------

class TestProtocol:
    def test_conforms(self):
        assert isinstance(HeuristicObservationFilter(), ObservationFilter)


# ------------------------------------------------------------------
# Raw attempt counting
# ------------------------------------------------------------------

class TestRawAttemptCount:
    def test_empty_episode(self):
        obs = HeuristicObservationFilter().summarize(_make_episode())
        assert obs.raw_attempt_count == 0

    def test_single_attempt(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[_attempt()])
        )
        assert obs.raw_attempt_count == 1

    def test_three_attempts(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[_attempt(), _attempt(), _attempt()])
        )
        assert obs.raw_attempt_count == 3


# ------------------------------------------------------------------
# Meaningful attempt counting
# ------------------------------------------------------------------

class TestMeaningfulAttempts:
    def test_substantive_text(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[_attempt("The answer is 4")])
        )
        assert obs.meaningful_attempts == 1

    def test_empty_text_not_meaningful(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[_attempt("")])
        )
        assert obs.meaningful_attempts == 0

    def test_short_text_not_meaningful(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[_attempt("ok")])
        )
        assert obs.meaningful_attempts == 0

    def test_none_text_not_meaningful(self):
        a = ProblemAttempt(attempt_index=0, submission_text=None)
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[a])
        )
        assert obs.meaningful_attempts == 0

    def test_whitespace_only_not_meaningful(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[_attempt("   ")])
        )
        assert obs.meaningful_attempts == 0

    def test_mixed(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[
                _attempt("real answer"),
                _attempt(""),
                _attempt("another"),
            ])
        )
        assert obs.meaningful_attempts == 2


# ------------------------------------------------------------------
# Chat-only virtual attempts
# ------------------------------------------------------------------

class TestChatOnlyVirtualAttempts:
    def test_no_attempts_no_chat(self):
        obs = HeuristicObservationFilter().summarize(_make_episode())
        assert obs.meaningful_attempts == 0

    def test_chat_turns_generate_virtual(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(chat_turns=8)
        )
        # 8 // 4 = 2
        assert obs.meaningful_attempts == 2

    def test_virtual_capped_at_max(self):
        obs = HeuristicObservationFilter(max_virtual_attempts_from_chat=3).summarize(
            _make_episode(chat_turns=100)
        )
        assert obs.meaningful_attempts == 3

    def test_virtual_not_applied_when_attempts_exist(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[_attempt("")], chat_turns=20)
        )
        assert obs.meaningful_attempts == 0

    def test_custom_chat_per_virtual(self):
        obs = HeuristicObservationFilter(chat_turns_per_virtual_attempt=2).summarize(
            _make_episode(chat_turns=7)
        )
        assert obs.meaningful_attempts == 3


# ------------------------------------------------------------------
# Time on problem
# ------------------------------------------------------------------

class TestTimeOnProblem:
    def test_explicit_time_used_directly(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(time_on_problem_sec=42.5)
        )
        assert obs.time_on_problem_sec == 42.5

    def test_derived_from_timestamps(self):
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=90)
        obs = HeuristicObservationFilter().summarize(
            _make_episode(
                started_at=t0.isoformat(),
                completed_at=t1.isoformat(),
            )
        )
        assert obs.time_on_problem_sec == pytest.approx(90.0)

    def test_no_timestamps_returns_none(self):
        obs = HeuristicObservationFilter().summarize(_make_episode())
        assert obs.time_on_problem_sec is None

    def test_malformed_timestamps_returns_none(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(started_at="not-a-date", completed_at="also-bad")
        )
        assert obs.time_on_problem_sec is None

    def test_started_only_returns_none(self):
        t0 = datetime.now(timezone.utc).isoformat()
        obs = HeuristicObservationFilter().summarize(
            _make_episode(started_at=t0)
        )
        assert obs.time_on_problem_sec is None


# ------------------------------------------------------------------
# Solved / help turn propagation
# ------------------------------------------------------------------

class TestOutputFields:
    def test_solved_propagated(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(solved=True, attempts=[_attempt()])
        )
        assert obs.solved is True

    def test_help_turn_count(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(chat_turns=5, attempts=[_attempt()])
        )
        assert obs.help_turn_count == 5

    def test_debug_contains_filter_name(self):
        obs = HeuristicObservationFilter().summarize(_make_episode())
        assert obs.debug.get("filter") == "heuristic"


# ------------------------------------------------------------------
# Round-trip serialization
# ------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_from_dict(self):
        obs = HeuristicObservationFilter().summarize(
            _make_episode(attempts=[_attempt()], chat_turns=3, solved=True)
        )
        d = obs.to_dict()
        from src.workflow_demo.practice.models import ProblemObservation
        restored = ProblemObservation.from_dict(d)
        assert restored.meaningful_attempts == obs.meaningful_attempts
        assert restored.raw_attempt_count == obs.raw_attempt_count
        assert restored.solved == obs.solved
