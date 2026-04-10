"""
Adaptive practice loop and problem-sequencing subsystem.

Round 4 — instrumentation, replay artifacts, offline evaluator, and export
hooks.  Defaults to no-op when flags are off.
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
from .replay import (
    EpisodeReplayRecord,
    PracticeReplayArtifact,
    ReplayDecision,
    replay_episodes,
)
from .session import PracticeSessionManager

__all__ = [
    "EpisodeReplayRecord",
    "FirstMatchSelector",
    "NoOpProblemSequencer",
    "PracticeFeatureFlags",
    "PracticeProblemRef",
    "PracticeReplayArtifact",
    "PracticeSessionManager",
    "PracticeSessionState",
    "ProblemAttempt",
    "ProblemBank",
    "ProblemEpisodeTrace",
    "ProblemObservation",
    "ProblemSelector",
    "ProblemSequencer",
    "ReplayDecision",
    "SequencerState",
    "StubProblemBank",
    "replay_episodes",
]
