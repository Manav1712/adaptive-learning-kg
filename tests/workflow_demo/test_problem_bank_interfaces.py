"""Round 2 tests — problem bank, selector, and sequencer interfaces."""

from __future__ import annotations

import pytest

from src.workflow_demo.practice.models import (
    PracticeProblemRef,
    ProblemEpisodeTrace,
    SequencerState,
)
from src.workflow_demo.practice.problem_bank import ProblemBank, StubProblemBank
from src.workflow_demo.practice.problem_selector import (
    FirstMatchSelector,
    ProblemSelector,
)
from src.workflow_demo.practice.problem_sequencer import (
    NoOpProblemSequencer,
    ProblemSequencer,
)


# ------------------------------------------------------------------
# StubProblemBank
# ------------------------------------------------------------------

class TestStubProblemBank:

    def test_isinstance_protocol(self):
        bank = StubProblemBank()
        assert isinstance(bank, ProblemBank)

    @pytest.mark.parametrize("difficulty", [0, 1, 2, 3])
    def test_returns_candidates_for_each_difficulty(self, difficulty):
        bank = StubProblemBank()
        candidates = bank.get_candidates(lo_id=None, difficulty=difficulty)
        assert len(candidates) >= 1
        assert all(c.difficulty == difficulty for c in candidates)

    def test_returns_empty_for_unknown_difficulty(self):
        bank = StubProblemBank()
        candidates = bank.get_candidates(lo_id=None, difficulty=99)
        assert candidates == []

    def test_lo_filter_falls_back_to_difficulty_only(self):
        bank = StubProblemBank()
        candidates = bank.get_candidates(lo_id="nonexistent-lo", difficulty=1)
        assert len(candidates) >= 1

    def test_limit_parameter(self):
        bank = StubProblemBank()
        candidates = bank.get_candidates(lo_id=None, difficulty=1, limit=1)
        assert len(candidates) <= 1


# ------------------------------------------------------------------
# FirstMatchSelector
# ------------------------------------------------------------------

class TestFirstMatchSelector:

    def test_isinstance_protocol(self):
        sel = FirstMatchSelector()
        assert isinstance(sel, ProblemSelector)

    def test_returns_first_candidate(self):
        sel = FirstMatchSelector()
        c1 = PracticeProblemRef(problem_id="a", difficulty=0, prompt_text="A.")
        c2 = PracticeProblemRef(problem_id="b", difficulty=0, prompt_text="B.")
        result = sel.select([c1, c2])
        assert result.problem_id == "a"

    def test_raises_on_empty_list(self):
        sel = FirstMatchSelector()
        with pytest.raises(ValueError, match="empty"):
            sel.select([])


# ------------------------------------------------------------------
# NoOpProblemSequencer
# ------------------------------------------------------------------

class TestNoOpProblemSequencer:

    def test_isinstance_protocol(self):
        seq = NoOpProblemSequencer()
        assert isinstance(seq, ProblemSequencer)

    def test_initialize_returns_valid_state(self):
        seq = NoOpProblemSequencer()
        state = seq.initialize({})
        assert isinstance(state, SequencerState)
        assert state.mode == "off"
        assert state.current_difficulty == 1

    def test_choose_first_difficulty_returns_default(self):
        seq = NoOpProblemSequencer(default_difficulty=2)
        state = seq.initialize({})
        assert seq.choose_first_difficulty(state) == 2

    def test_choose_next_difficulty_returns_default(self):
        seq = NoOpProblemSequencer(default_difficulty=0)
        state = seq.initialize({})
        assert seq.choose_next_difficulty(state) == 0

    def test_update_after_problem_is_noop(self):
        seq = NoOpProblemSequencer()
        state = seq.initialize({})
        prob = PracticeProblemRef(problem_id="x", difficulty=1, prompt_text="Q.")
        trace = ProblemEpisodeTrace(problem=prob)
        trace.finalize(solved=True)

        new_state = seq.update_after_problem(state, trace)
        assert new_state is state  # identity — truly no-op

    def test_custom_default_difficulty(self):
        seq = NoOpProblemSequencer(default_difficulty=3)
        state = seq.initialize({})
        assert seq.choose_first_difficulty(state) == 3
        assert seq.choose_next_difficulty(state) == 3
