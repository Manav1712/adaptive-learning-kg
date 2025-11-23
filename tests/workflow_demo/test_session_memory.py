"""Unit tests for workflow_demo.session_memory utilities."""

import pytest

from src.workflow_demo.session_memory import SessionMemory, create_handoff_context


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
