"""Round 2 tests — feature flags, sequencing state, and snapshot integration."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.workflow_demo.practice.feature_flags import PracticeFeatureFlags
from src.workflow_demo.practice.models import SequencerState
from src.workflow_demo.practice.session import PracticeSessionManager
from src.workflow_demo.pedagogy.tutor_pedagogy_snapshot import build_tutor_pedagogy_snapshot


# ------------------------------------------------------------------
# PracticeFeatureFlags
# ------------------------------------------------------------------

class TestPracticeFeatureFlags:

    def test_defaults_all_off(self):
        flags = PracticeFeatureFlags()
        assert flags.practice_loop_enabled is False
        assert flags.adaptive_sequencing_enabled is False
        assert flags.sequencer_mode == "off"
        assert flags.llm_attempt_observer_enabled is False
        assert flags.sequencer_rollouts == 50
        assert flags.sequencer_horizon == 10

    def test_from_env_all_off(self, monkeypatch):
        monkeypatch.delenv("WORKFLOW_DEMO_ENABLE_PRACTICE_LOOP", raising=False)
        monkeypatch.delenv("WORKFLOW_DEMO_ENABLE_ADAPTIVE_SEQUENCING", raising=False)
        monkeypatch.delenv("WORKFLOW_DEMO_SEQUENCER_MODE", raising=False)
        monkeypatch.delenv("WORKFLOW_DEMO_ENABLE_LLM_ATTEMPT_OBSERVER", raising=False)
        monkeypatch.delenv("WORKFLOW_DEMO_SEQUENCER_ROLLOUTS", raising=False)
        monkeypatch.delenv("WORKFLOW_DEMO_SEQUENCER_HORIZON", raising=False)
        flags = PracticeFeatureFlags.from_env()
        assert flags.practice_loop_enabled is False
        assert flags.sequencer_mode == "off"

    def test_from_env_practice_enabled(self, monkeypatch):
        monkeypatch.setenv("WORKFLOW_DEMO_ENABLE_PRACTICE_LOOP", "1")
        monkeypatch.setenv("WORKFLOW_DEMO_ENABLE_ADAPTIVE_SEQUENCING", "1")
        monkeypatch.setenv("WORKFLOW_DEMO_SEQUENCER_MODE", "heuristic")
        monkeypatch.setenv("WORKFLOW_DEMO_ENABLE_LLM_ATTEMPT_OBSERVER", "1")
        monkeypatch.setenv("WORKFLOW_DEMO_SEQUENCER_ROLLOUTS", "100")
        monkeypatch.setenv("WORKFLOW_DEMO_SEQUENCER_HORIZON", "20")
        flags = PracticeFeatureFlags.from_env()
        assert flags.practice_loop_enabled is True
        assert flags.adaptive_sequencing_enabled is True
        assert flags.sequencer_mode == "heuristic"
        assert flags.llm_attempt_observer_enabled is True
        assert flags.sequencer_rollouts == 100
        assert flags.sequencer_horizon == 20

    def test_from_env_invalid_mode_falls_back(self, monkeypatch):
        monkeypatch.setenv("WORKFLOW_DEMO_SEQUENCER_MODE", "invalid_mode")
        flags = PracticeFeatureFlags.from_env()
        assert flags.sequencer_mode == "off"

    def test_from_env_invalid_int_falls_back(self, monkeypatch):
        monkeypatch.setenv("WORKFLOW_DEMO_SEQUENCER_ROLLOUTS", "not_a_number")
        monkeypatch.setenv("WORKFLOW_DEMO_SEQUENCER_HORIZON", "abc")
        flags = PracticeFeatureFlags.from_env()
        assert flags.sequencer_rollouts == 50
        assert flags.sequencer_horizon == 10

    def test_frozen(self):
        flags = PracticeFeatureFlags()
        with pytest.raises(AttributeError):
            flags.practice_loop_enabled = True  # type: ignore[misc]


# ------------------------------------------------------------------
# SequencerState in extensions round-trip
# ------------------------------------------------------------------

class TestSequencerStateInExtensions:

    def test_store_and_read_from_extensions(self):
        state = SequencerState(mode="heuristic", current_difficulty=2, step_index=5)
        ext: Dict[str, Any] = {"sequencing": state.to_dict()}
        restored = SequencerState.from_dict(ext["sequencing"])
        assert restored.mode == "heuristic"
        assert restored.current_difficulty == 2
        assert restored.step_index == 5


# ------------------------------------------------------------------
# Snapshot includes practice/sequencing fields
# ------------------------------------------------------------------

class TestSnapshotPracticeFields:

    def _build_handoff(self, *, with_practice: bool = False) -> Dict[str, Any]:
        """Build a minimal tutor handoff context."""
        ext: Dict[str, Any] = {}
        if with_practice:
            flags = PracticeFeatureFlags(practice_loop_enabled=True)
            mgr = PracticeSessionManager(flags)
            mgr.seed_extensions(ext)
            mgr.begin_practice_problem(ext)

        return {
            "session_params": {"learning_objective": "Test LO"},
            "pedagogy_context": {
                "target_lo": "Test LO",
                "extensions": ext,
            },
        }

    def test_snapshot_includes_practice_when_enabled(self):
        ctx = self._build_handoff(with_practice=True)
        snap = build_tutor_pedagogy_snapshot(
            handoff_context=ctx,
            bot_type="tutor",
            active_learner_session_id="test:1",
            learner_state_engine=None,
        )
        assert snap is not None
        assert snap.get("practice_session") is not None
        assert snap["practice_session"]["active"] is True
        assert snap.get("sequencing") is not None
        assert snap["sequencing"]["mode"] == "off"

    def test_snapshot_omits_practice_when_disabled(self):
        ctx = self._build_handoff(with_practice=False)
        snap = build_tutor_pedagogy_snapshot(
            handoff_context=ctx,
            bot_type="tutor",
            active_learner_session_id="test:2",
            learner_state_engine=None,
        )
        assert snap is not None
        assert snap.get("practice_session") is None
        assert snap.get("sequencing") is None

    def test_snapshot_none_for_faq(self):
        snap = build_tutor_pedagogy_snapshot(
            handoff_context={"session_params": {}},
            bot_type="faq",
            active_learner_session_id=None,
            learner_state_engine=None,
        )
        assert snap is None


# ------------------------------------------------------------------
# BotSessionManager practice seeding (integration-light)
# ------------------------------------------------------------------

class TestBotSessionManagerPracticeSeeding:
    """Verify that BotSessionManager creates PracticeSessionManager only when
    the feature flag is enabled, and does NOT create one when flags are off."""

    def test_no_practice_mgr_when_disabled(self):
        from src.workflow_demo.bot_sessions import BotSessionManager

        class _FakeAgent:
            emit_event = staticmethod(lambda *a, **kw: {})

        flags = PracticeFeatureFlags(practice_loop_enabled=False)
        mgr = BotSessionManager(_FakeAgent(), practice_flags=flags)  # type: ignore[arg-type]
        assert mgr.practice_session_manager is None

    def test_practice_mgr_created_when_enabled(self):
        from src.workflow_demo.bot_sessions import BotSessionManager

        class _FakeAgent:
            emit_event = staticmethod(lambda *a, **kw: {})

        flags = PracticeFeatureFlags(practice_loop_enabled=True)
        mgr = BotSessionManager(_FakeAgent(), practice_flags=flags)  # type: ignore[arg-type]
        assert mgr.practice_session_manager is not None
