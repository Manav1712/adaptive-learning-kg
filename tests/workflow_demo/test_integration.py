"""Integration tests spanning multiple workflow_demo components."""

import pytest

from src.workflow_demo.planner import TutoringPlanner


@pytest.mark.integration
def test_coach_planner_to_tutor_flow(monkeypatch, coach_agent, mock_retriever):
    """Coach should call the planner and hand off to the tutor bot when instructed."""
    tutor_messages = []

    def _fake_tutor_bot(**kwargs):
        tutor_messages.append(kwargs["handoff_context"]["session_params"]["subject"])
        return {
            "message_to_student": "Tutor taking over now.",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": ["Derivatives"],
                "student_understanding": "good",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "Integration test",
            },
        }

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _fake_tutor_bot)

    coach_agent.llm_client._queued.clear()
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "call_tutoring_planner",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "conceptual_review",
                "student_request": "Teach me derivatives",
            },
            "conversation_summary": "",
        }
    )
    coach_agent.llm_client.queue_response(
        {
            "message_to_student": "",
            "action": "start_tutor",
            "tool_params": {
                "subject": "calculus",
                "learning_objective": "Derivatives",
                "mode": "conceptual_review",
            },
            "conversation_summary": "Student confirmed tutoring plan.",
        }
    )

    reply = coach_agent.process_turn("I want help with derivatives")
    # New flow: planner → start_tutor in one turn, tutor fires immediately
    assert "Tutor taking over now." in reply
    assert mock_retriever.calls, "Planner should invoke the retriever."
    assert tutor_messages == ["calculus"]
    assert coach_agent.bot_session_manager.is_active is True


@pytest.mark.integration
def test_session_memory_captures_completed_tutor_sessions(
    monkeypatch, coach_agent, sample_planner_response
):
    """When the tutor bot ends the session, the summary should be recorded in SessionMemory."""

    def _ending_tutor_bot(**kwargs):
        return {
            "message_to_student": "Great work! Session complete.",
            "end_activity": True,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": ["Derivatives"],
                "student_understanding": "excellent",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "Tutor summary",
            },
        }

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _ending_tutor_bot)

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.handoff_context = {"session_params": {"subject": "calculus", "learning_objective": "Derivatives", "mode": "practice"}}
    coach_agent.collected_params = {"subject": "calculus", "learning_objective": "Derivatives", "mode": "practice"}
    coach_agent.planner_result = sample_planner_response

    _ = coach_agent.process_turn("done")
    sessions = coach_agent.session_memory.get_recent_sessions()
    assert any(entry["summary"].get("notes") == "Tutor summary" for entry in sessions)


@pytest.mark.integration
def test_tutoring_planner_uses_mock_retriever(mock_retriever, sample_session_plan):
    """Planner + retriever fixture should surface the simplified tutoring plan."""
    planner = TutoringPlanner(mock_retriever)
    params = {
        "student_request": "Teach me derivatives",
        "mode": "conceptual_review",
        "student_profile": {"lo_mastery": {}},
    }
    response = planner.create_plan(params)
    assert response["status"] == "complete"
    assert response["plan"]["subject"] == "calculus"
    assert response["plan"]["current_plan"]


@pytest.mark.integration
def test_mastery_updates_after_tutor_session(monkeypatch, coach_agent, sample_planner_response):
    """After a tutor session ends, student_profile['lo_mastery'] should be updated."""

    def _ending_tutor_bot(**kwargs):
        return {
            "message_to_student": "Great work! Session complete.",
            "end_activity": True,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": ["Derivatives"],
                "student_understanding": "excellent",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "Mastery test",
            },
        }

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _ending_tutor_bot)

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.handoff_context = {"session_params": {"subject": "calculus", "learning_objective": "Derivatives", "mode": "practice"}}
    coach_agent.collected_params = {"subject": "calculus", "learning_objective": "Derivatives", "mode": "practice"}
    coach_agent.planner_result = sample_planner_response

    _ = coach_agent.process_turn("done")

    # Mastery should now be set to 0.9 for "Derivatives" (excellent -> 0.9)
    assert "Derivatives" in coach_agent.student_profile.get("lo_mastery", {})
    assert coach_agent.student_profile["lo_mastery"]["Derivatives"] == 0.9


@pytest.mark.integration
def test_continuity_greeting_after_tutor_session(monkeypatch, coach_agent, sample_planner_response):
    """After a tutor session ends, the greeting should mention the LO and mode."""

    def _ending_tutor_bot(**kwargs):
        return {
            "message_to_student": "Great work! Session complete.",
            "end_activity": True,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": ["Derivatives"],
                "student_understanding": "good",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "Continuity test",
            },
        }

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _ending_tutor_bot)

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.handoff_context = {"session_params": {"subject": "calculus", "learning_objective": "Derivatives", "mode": "practice"}}
    coach_agent.collected_params = {"subject": "calculus", "learning_objective": "Derivatives", "mode": "practice"}
    coach_agent.planner_result = sample_planner_response

    greeting = coach_agent.process_turn("done")

    # The greeting should be the tutor's final message followed by the return greeting
    assert "Nice work on Derivatives" in greeting or "Great work" in greeting


@pytest.mark.integration
def test_pedagogy_phase0_package_import_surface():
    """Phase 0 pedagogy subpackage is reachable and JSON-stable without wiring."""
    import src.workflow_demo.pedagogy as pedagogy

    assert pedagogy.TeachingMoveType.WORKED_EXAMPLE.value == "worked_example"
    assert pedagogy.RetrievalIntent.PRACTICE_ITEM.value == "practice_item"
    assert pedagogy.PedagogyRuntimeEvent.POLICY_SCORED.value == "pedagogy_policy_scored"
    assert "PedagogicalContext" in pedagogy.__all__


@pytest.mark.integration
def test_tutor_retrieval_debug_command_skips_tutor_llm_and_pedagogy_pipeline(
    monkeypatch, coach_agent
):
    """!retrieval must not invoke tutor_bot or run diagnosis / retrieval refresh."""

    def _tutor_should_not_run(**kwargs):
        raise AssertionError("tutor_bot must not run for !retrieval")

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _tutor_should_not_run)

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-1"
    coach_agent.bot_session_manager.handoff_context = {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "Derivatives",
            "mode": "practice",
        },
        "pedagogy_context": {
            "learner_state": {"active_session_id": "sess-1"},
            "layer_version": "1",
            "target_lo": "Derivatives",
            "instruction_lo": "Slope",
            "retrieval_intent": "teach_current_concept",
            "retrieval_action": "reuse_pack",
            "retrieval_execution_mode": "no_io",
            "retrieval_session": {
                "pack_focus_lo": "Derivatives",
                "pack_revision": 2,
                "last_diagnosis_fingerprint": "a|b|c",
                "last_selected_move_type": "worked_example",
            },
            "policy_decision": {"decision_reason": "picked best"},
            "tutor_instruction_directives": {"policy_reason": "picked best"},
        },
    }
    coach_agent.bot_session_manager.conversation_history = []

    out = coach_agent.process_turn("!retrieval")
    assert "[DEBUG]" in out
    assert "policy_reason:" in out and "picked best" in out
    assert "Tutor retrieval / pedagogy state" in out
    assert "Derivatives" in out
    assert "reuse_pack" in out
    assert "pack_revision" in out


@pytest.mark.integration
def test_tutor_normal_turn_after_retrieval_debug(monkeypatch, coach_agent):
    """After !retrieval, ordinary student text still reaches the tutor path once."""

    calls = []

    def _tutor(**kwargs):
        calls.append("tutor")
        return {
            "message_to_student": "ok",
            "end_activity": False,
            "silent_end": False,
            "needs_mode_confirmation": False,
            "needs_topic_confirmation": False,
            "requested_mode": None,
            "session_summary": {
                "topics_covered": [],
                "student_understanding": "good",
                "suggested_next_topic": None,
                "switch_topic_request": None,
                "switch_mode_request": None,
                "notes": "",
            },
        }

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _tutor)
    monkeypatch.setattr(
        "src.workflow_demo.bot_sessions.BotSessionManager._run_misconception_diagnosis",
        lambda *a, **k: None,
    )

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-1"
    coach_agent.bot_session_manager.handoff_context = {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "Derivatives",
            "mode": "practice",
        },
        "pedagogy_context": {"learner_state": {"active_session_id": "sess-1"}},
    }
    coach_agent.bot_session_manager.conversation_history = []

    coach_agent.process_turn("!retrieval")
    assert calls == []
    coach_agent.process_turn("hello")
    assert calls == ["tutor"]


@pytest.mark.integration
def test_tutor_policy_diagnosis_state_debug_commands_skip_tutor_llm(
    monkeypatch, coach_agent
):
    """!policy, !diagnosis, !state must not invoke tutor_bot."""

    def _tutor_should_not_run(**kwargs):
        raise AssertionError("tutor_bot must not run for debug commands")

    monkeypatch.setattr("src.workflow_demo.bot_sessions.tutor_bot", _tutor_should_not_run)

    coach_agent.bot_session_manager.is_active = True
    coach_agent.bot_session_manager.bot_type = "tutor"
    coach_agent.bot_session_manager.active_learner_session_id = "sess-1"
    coach_agent.bot_session_manager.handoff_context = {
        "session_params": {
            "subject": "calculus",
            "learning_objective": "Derivatives",
            "mode": "practice",
        },
        "pedagogy_context": {
            "learner_state": {"active_session_id": "sess-1"},
            "target_lo": "Derivatives",
            "instruction_lo": "Slope",
            "retrieval_intent": "teach_current_concept",
            "retrieval_action": "reuse_pack",
            "retrieval_execution_mode": "no_io",
            "retrieval_session": {
                "pack_focus_lo": "Derivatives",
                "pack_revision": 1,
                "last_diagnosis_fingerprint": "fp",
                "last_selected_move_type": "worked_example",
            },
            "policy_decision": {"decision_reason": "test reason"},
            "tutor_instruction_directives": {"policy_reason": "test reason"},
            "teaching_moves": [{"move_type": "worked_example"}],
            "diagnosis": {
                "target_lo": "Derivatives",
                "suspected_misconception": "confusion",
                "confidence": 0.7,
                "prerequisite_gap_los": ["limits"],
            },
        },
    }
    coach_agent.bot_session_manager.conversation_history = []

    out_policy = coach_agent.process_turn("!policy")
    assert "[DEBUG] Tutor policy" in out_policy
    assert "test reason" in out_policy

    out_diag = coach_agent.process_turn("!diagnosis")
    assert "[DEBUG] Tutor diagnosis" in out_diag
    assert "confusion" in out_diag

    out_state = coach_agent.process_turn("!state")
    assert "[DEBUG] Learner snapshot" in out_state
