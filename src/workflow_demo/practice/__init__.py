"""
Adaptive practice loop and problem-sequencing subsystem.

Round 3 — heuristic observation filter and adaptive sequencer behind feature
flags.  Defaults to no-op when flags are off.
"""

from .feature_flags import PracticeFeatureFlags
from .models import (
    PracticeProblemRef,
    PracticeSessionState,
    ProblemAttempt,
    ProblemEpisodeTrace,
    ProblemObservation,
    SequencerState,
)
from .problem_bank import ProblemBank, StubProblemBank
from .problem_selector import FirstMatchSelector, ProblemSelector
from .problem_sequencer import NoOpProblemSequencer, ProblemSequencer
from .session import PracticeSessionManager

__all__ = [
    "FirstMatchSelector",
    "NoOpProblemSequencer",
    "PracticeFeatureFlags",
    "PracticeProblemRef",
    "PracticeSessionManager",
    "PracticeSessionState",
    "ProblemAttempt",
    "ProblemBank",
    "ProblemEpisodeTrace",
    "ProblemObservation",
    "ProblemSelector",
    "ProblemSequencer",
    "SequencerState",
    "StubProblemBank",
]
