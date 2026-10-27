# workflow_demo Test Suite

Pytest suite for the adaptive tutoring runtime. Markers are declared in `pytest.ini` (`--strict-markers`).

## Markers

| Marker | What it selects |
|--------|-----------------|
| `unit` | Isolated tests, including pedagogy policy, retrieval decisions, practice models, and the POMDP stack |
| `integration` | Multi-component wiring (planner, coach, bots, session memory) |
| `e2e` | Conversational flows with bots patched so they do not call OpenAI |
| `acceptance` | Phase 9 acceptance scenarios |
| `pedagogy_eval` | Pedagogy eval harness drift check |
| `slow` | Reserved for tests that take longer than about a second |

Files with no marker are not selected by `pytest -m unit` or `pytest -m integration`. The practice and POMDP modules are marked `unit`.

## Running tests

```bash
source venv/bin/activate

pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m acceptance
pytest -m pedagogy_eval

pytest tests/workflow_demo -q
pytest --cov=src/workflow_demo --cov-report=term-missing
```

## Notes

- `conftest.py` holds shared fixtures: mock LLM client, sample graph data, retriever and tutor helpers.
- Integration and end-to-end tests patch tutor and FAQ bots so they do not call OpenAI.
- Reuse those fixtures instead of duplicating mock setup.
- `test_web_api.py` covers the FastAPI bridge with `build_coach_runtime` mocked.
- Retrieval-policy behavior is in `test_pedagogy_retrieval_phase5.py`. `test_pedagogical_retrieval.py` is a marked placeholder with no cases.
