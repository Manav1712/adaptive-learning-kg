"""
Feature flags for the adaptive practice loop and sequencing subsystem.

All flags default to the safe/off state so the system is behaviour-identical
when none are set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


SequencerModeFlag = Literal["off", "heuristic", "pomdp"]


@dataclass(frozen=True)
class PracticeFeatureFlags:
    """Typed, immutable snapshot of practice-related feature flags."""

    practice_loop_enabled: bool = False
    adaptive_sequencing_enabled: bool = False
    sequencer_mode: SequencerModeFlag = "off"
    llm_attempt_observer_enabled: bool = False
    sequencer_rollouts: int = 50
    sequencer_horizon: int = 10

    @classmethod
    def from_env(cls) -> "PracticeFeatureFlags":
        """Read all ``WORKFLOW_DEMO_*`` practice flags from the environment."""
        practice_loop = os.getenv("WORKFLOW_DEMO_ENABLE_PRACTICE_LOOP", "0") == "1"
        adaptive = os.getenv("WORKFLOW_DEMO_ENABLE_ADAPTIVE_SEQUENCING", "0") == "1"

        raw_mode = os.getenv("WORKFLOW_DEMO_SEQUENCER_MODE", "off").lower()
        mode: SequencerModeFlag = raw_mode if raw_mode in ("off", "heuristic", "pomdp") else "off"

        llm_observer = os.getenv("WORKFLOW_DEMO_ENABLE_LLM_ATTEMPT_OBSERVER", "0") == "1"

        try:
            rollouts = int(os.getenv("WORKFLOW_DEMO_SEQUENCER_ROLLOUTS", "50"))
        except ValueError:
            rollouts = 50

        try:
            horizon = int(os.getenv("WORKFLOW_DEMO_SEQUENCER_HORIZON", "10"))
        except ValueError:
            horizon = 10

        return cls(
            practice_loop_enabled=practice_loop,
            adaptive_sequencing_enabled=adaptive,
            sequencer_mode=mode,
            llm_attempt_observer_enabled=llm_observer,
            sequencer_rollouts=rollouts,
            sequencer_horizon=horizon,
        )
