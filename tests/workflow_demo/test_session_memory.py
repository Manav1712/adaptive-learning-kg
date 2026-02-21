"""Unit tests for workflow_demo.session_memory utilities."""

import json
import os
import tempfile

import pytest

from src.workflow_demo.session_memory import SessionMemory, create_handoff_context
from src.workflow_demo.coach_agent import CoachAgent
from src.workflow_demo.bot_sessions import BotSessionManager


@pytest.mark.unit
def test_session_memory_records_recent_sessions(sample_session_memory: SessionMemory):
    """Fixture should expose a single recent tutoring session by default."""
    recent = sample_session_memory.get_recent_sessions()
    assert len(recent) == 1
    assert recent[0]["type"] == "tutor"
    assert recent[0]["summary"]["student_understanding"] == "good"


@pytest.mark.unit
def test_session_memory_max_entries_enforced():
    """SessionMemory should evict oldest entries beyond max_entries."""
    memory = SessionMemory(max_entries=2)
    for idx in range(3):
        memory.add_session(
            session_type="faq" if idx % 2 else "tutor",
            params={"index": idx},
            summary={"topics_covered": [f"Topic {idx}"]},
        )
    recent = memory.get_recent_sessions()
    assert len(recent) == 2
    assert recent[0]["params"]["index"] == 1
    assert recent[1]["params"]["index"] == 2


@pytest.mark.unit
def test_last_tutoring_session_returns_latest():
    """last_tutoring_session should retrieve the most recent tutor entry."""
    memory = SessionMemory(max_entries=5)
    memory.add_session("faq", {"topic": "exam schedule"}, {"topics_addressed": ["exam schedule"]})
    memory.add_session("tutor", {"learning_objective": "Derivatives"}, {"topics_covered": ["Derivatives"]})
    memory.add_session("tutor", {"learning_objective": "Integrals"}, {"topics_covered": ["Integrals"]})
    last = memory.last_tutoring_session()
    assert last is not None
    assert last["params"]["learning_objective"] == "Integrals"


@pytest.mark.unit
def test_create_handoff_context(sample_session_memory: SessionMemory):
    """create_handoff_context should embed metadata and recent sessions."""
    params = {"subject": "calculus", "learning_objective": "Derivatives"}
    context = create_handoff_context(
        from_agent="coach",
        to_agent="tutor",
        session_params=params,
        conversation_summary="Student confirmed tutoring plan.",
        session_memory=sample_session_memory,
        student_state={"lo_mastery": {1893: 0.7}},
    )
    assert context["session_params"] == params
    assert context["handoff_metadata"]["from_agent"] == "coach"
    assert context["recent_sessions"] == sample_session_memory.get_recent_sessions()
    assert context["student_state"]["lo_mastery"][1893] == 0.7


# ---------------------------------------------------------------------------
# Mastery wiring tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_lo_mastery_from_excellent_understanding():
    """_update_lo_mastery should map 'excellent' to 0.9."""
    agent = CoachAgent.__new__(CoachAgent)
    agent.student_profile = {"lo_mastery": {}}
    mgr = BotSessionManager(agent)
    params = {"learning_objective": "Derivatives"}
    summary = {"student_understanding": "excellent"}
    mgr._update_lo_mastery(params, summary)
    assert agent.student_profile["lo_mastery"]["Derivatives"] == 0.9


@pytest.mark.unit
def test_update_lo_mastery_from_needs_practice():
    """_update_lo_mastery should map 'needs_practice' to 0.4."""
    agent = CoachAgent.__new__(CoachAgent)
    agent.student_profile = {"lo_mastery": {}}
    mgr = BotSessionManager(agent)
    params = {"learning_objective": "Integrals"}
    summary = {"student_understanding": "needs_practice"}
    mgr._update_lo_mastery(params, summary)
    assert agent.student_profile["lo_mastery"]["Integrals"] == 0.4


@pytest.mark.unit
def test_update_lo_mastery_defaults_on_unknown_label():
    """_update_lo_mastery should default to 0.4 for unknown labels."""
    agent = CoachAgent.__new__(CoachAgent)
    agent.student_profile = {"lo_mastery": {}}
    mgr = BotSessionManager(agent)
    params = {"learning_objective": "Limits"}
    summary = {"student_understanding": "unknown_label"}
    mgr._update_lo_mastery(params, summary)
    assert agent.student_profile["lo_mastery"]["Limits"] == 0.4


@pytest.mark.unit
def test_update_lo_mastery_skips_when_no_lo_key():
    """_update_lo_mastery should skip update if no LO key in params."""
    agent = CoachAgent.__new__(CoachAgent)
    agent.student_profile = {"lo_mastery": {}}
    mgr = BotSessionManager(agent)
    params = {}
    summary = {"student_understanding": "excellent"}
    mgr._update_lo_mastery(params, summary)
    assert agent.student_profile["lo_mastery"] == {}


# ---------------------------------------------------------------------------
# Continuity-aware greeting tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_return_greeting_tutor_with_mode():
    """_build_return_greeting should produce a custom greeting for tutor sessions."""
    agent = CoachAgent.__new__(CoachAgent)
    mgr = BotSessionManager(agent)
    params = {"learning_objective": "Derivatives", "mode": "practice"}
    greeting = mgr._build_return_greeting(params, session_type="tutor")
    assert "Derivatives" in greeting
    assert "practice" in greeting
    assert "Nice work" in greeting


@pytest.mark.unit
def test_build_return_greeting_tutor_without_mode():
    """_build_return_greeting should handle missing mode gracefully."""
    agent = CoachAgent.__new__(CoachAgent)
    mgr = BotSessionManager(agent)
    params = {"learning_objective": "Integrals"}
    greeting = mgr._build_return_greeting(params, session_type="tutor")
    assert "Integrals" in greeting
    assert "Nice work" in greeting


@pytest.mark.unit
def test_build_return_greeting_faq_with_topic():
    """_build_return_greeting should produce a custom greeting for FAQ sessions."""
    agent = CoachAgent.__new__(CoachAgent)
    mgr = BotSessionManager(agent)
    params = {"topic": "exam schedule"}
    greeting = mgr._build_return_greeting(params, session_type="faq")
    assert "exam schedule" in greeting
    assert "Glad I could help" in greeting


@pytest.mark.unit
def test_build_return_greeting_fallback():
    """_build_return_greeting should fall back to generic greeting when info is missing."""
    agent = CoachAgent.__new__(CoachAgent)
    mgr = BotSessionManager(agent)
    params = {}
    greeting = mgr._build_return_greeting(params, session_type="tutor")
    assert "learning coach" in greeting


# ---------------------------------------------------------------------------
# JSON Persistence tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_memory_persistence_saves_and_loads():
    """SessionMemory should persist entries to JSON and reload on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "session_memory.json")
        
        # Create memory, add sessions, and let it auto-save
        memory1 = SessionMemory(max_entries=5, persistence_path=path)
        memory1.add_session("tutor", {"learning_objective": "Derivatives"}, {"student_understanding": "good"})
        memory1.add_session("faq", {"topic": "exam schedule"}, {"topics_addressed": ["exam schedule"]})
        
        # Verify file was created
        assert os.path.exists(path)
        
        # Create new memory instance pointing to same file
        memory2 = SessionMemory(max_entries=5, persistence_path=path)
        recent = memory2.get_recent_sessions()
        
        # Verify sessions were loaded
        assert len(recent) == 2
        assert recent[0]["type"] == "tutor"
        assert recent[0]["params"]["learning_objective"] == "Derivatives"
        assert recent[1]["type"] == "faq"
        assert recent[1]["params"]["topic"] == "exam schedule"


@pytest.mark.unit
def test_session_memory_persistence_handles_missing_file():
    """SessionMemory should start empty when persistence file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nonexistent.json")
        
        memory = SessionMemory(max_entries=5, persistence_path=path)
        assert memory.get_recent_sessions() == []


@pytest.mark.unit
def test_session_memory_persistence_handles_empty_file():
    """SessionMemory should handle an empty JSON file gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty.json")
        
        # Create empty file
        with open(path, "w") as f:
            f.write("")
        
        memory = SessionMemory(max_entries=5, persistence_path=path)
        assert memory.get_recent_sessions() == []


@pytest.mark.unit
def test_session_memory_persistence_handles_invalid_json():
    """SessionMemory should handle corrupted JSON file gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "corrupt.json")
        
        # Create file with invalid JSON
        with open(path, "w") as f:
            f.write("{ not valid json }")
        
        memory = SessionMemory(max_entries=5, persistence_path=path)
        assert memory.get_recent_sessions() == []


@pytest.mark.unit
def test_session_memory_no_persistence_when_path_not_set():
    """SessionMemory should not create files when persistence_path is None."""
    memory = SessionMemory(max_entries=5, persistence_path=None)
    memory.add_session("tutor", {"learning_objective": "Limits"}, {"student_understanding": "excellent"})
    
    # No file should be created
    assert memory.persistence_path is None
    assert len(memory.get_recent_sessions()) == 1


@pytest.mark.unit
def test_session_memory_saves_new_schema_with_profile():
    """SessionMemory should write sessions + student_profile to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "session_memory.json")

        memory = SessionMemory(max_entries=5, persistence_path=path)
        memory.add_session("tutor", {"learning_objective": "Derivatives"}, {"student_understanding": "good"})
        memory.student_profile["lo_mastery"]["Derivatives"] = 0.8
        memory.save()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert set(data.keys()) == {"sessions", "student_profile"}
        assert data["sessions"][0]["params"]["learning_objective"] == "Derivatives"
        assert data["student_profile"]["lo_mastery"]["Derivatives"] == 0.8


@pytest.mark.unit
def test_session_memory_loads_legacy_list_file():
    """SessionMemory should remain backward compatible with legacy list files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "legacy.json")
        legacy_payload = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "type": "tutor",
                "params": {"learning_objective": "Chain Rule"},
                "summary": {"student_understanding": "good"},
                "conversation_exchanges": [],
            }
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(legacy_payload, f)

        memory = SessionMemory(max_entries=5, persistence_path=path)
        recent = memory.get_recent_sessions()

        assert len(recent) == 1
        assert recent[0]["params"]["learning_objective"] == "Chain Rule"
        assert memory.student_profile == {"lo_mastery": {}}


@pytest.mark.unit
def test_session_memory_persists_lo_mastery_scores():
    """lo_mastery entries should persist across SessionMemory instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "session_memory.json")

        memory1 = SessionMemory(max_entries=5, persistence_path=path)
        memory1.student_profile["lo_mastery"]["Derivatives"] = 0.9
        memory1.save()

        memory2 = SessionMemory(max_entries=5, persistence_path=path)
        assert memory2.student_profile["lo_mastery"]["Derivatives"] == 0.9
