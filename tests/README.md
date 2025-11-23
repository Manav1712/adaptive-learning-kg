# workflow_demo Test Suite

This directory hosts the pytest-based test harness for the multi-agent workflow demo.

## Layout

- `conftest.py` – shared fixtures (mock LLM client, sample KG data, retriever/tutor helpers)
- `workflow_demo/test_models.py` – dataclass/unit tests
- `workflow_demo/test_session_memory.py` – memory + handoff helpers
- `workflow_demo/test_retriever.py` – embedding backend + teaching pack assembly
- `workflow_demo/test_planner.py` – tutoring/FAQ planner behavior
- `workflow_demo/test_tutor.py` – tutor_bot + faq_bot contract tests
- `workflow_demo/test_coach.py` – CoachAgent state-machine tests
- `workflow_demo/test_integration.py` – planner/bot/session-memory integrations
- `workflow_demo/test_e2e.py` – conversational end-to-end flows

## Running tests

```bash
# Activate the project virtualenv first
source venv/bin/activate

# Fast unit tests only
pytest -m unit

# Integration tests (planner/coach/bot wiring)
pytest -m integration

# Full suite with coverage report
pytest --cov=src/workflow_demo --cov-report=term-missing

# End-to-end conversational flows
pytest -m e2e
```

## Notes

- The fixtures mock LLM calls and retriever behavior to keep tests deterministic.
- Integration/end-to-end tests patch tutor/FAQ bots to avoid real OpenAI usage.
- When adding new tests, reuse the shared fixtures instead of duplicating mock setup.
