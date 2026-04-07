# Adaptive Learning Knowledge Graph

Local-first project that combines **(1) building and evaluating curriculum knowledge graphs** from OpenStax-style materials with **(2) an adaptive tutoring runtime**: a coach routes students into tutoring or FAQ sessions, retrieval grounds answers in graph-backed learning objectives and content, and a pedagogy layer chooses teaching moves and retrieval behavior turn by turn.

**Architecture (single canonical doc):** [`docs/ADAPTIVE_TUTORING_ARCHITECTURE_SPEC.md`](docs/ADAPTIVE_TUTORING_ARCHITECTURE_SPEC.md)

**KG pipeline notes (experiments / ingestion):** [`architecture/`](architecture/)

---

## Quick start

**Requirements:** Python 3.10+ recommended (3.11+ works). An OpenAI API key for embeddings and LLM calls.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

### Tutoring CLI demo

```bash
python -m src.workflow_demo.run_demo
```

Type `quit` or `exit` to leave the REPL. You can pass image paths in the conversation per the demo’s help text.

### Pedagogy evaluation harness
```bash
python -m src.workflow_demo.pedagogy_eval
# With math-example guard scenarios enabled:
WORKFLOW_DEMO_TUTOR_MATH_GUARD=1 python -m src.workflow_demo.pedagogy_eval
```

### Web UI (FastAPI + React)

**Backend** (default `http://127.0.0.1:8001`):

```bash
source venv/bin/activate
python -m src.workflow_demo.web_api
```

**Frontend** (`http://127.0.0.1:5173`):

```bash
cd frontend
npm install
npm run dev
```

If the API is not on port 8001:

```bash
export VITE_API_BASE_URL=http://127.0.0.1:8001
```

### Tests

```bash
source venv/bin/activate
pytest tests/workflow_demo -q
```

---

## Repository layout 

| Path | Role |
|------|------|
| `src/workflow_demo/` | Adaptive tutoring runtime (coach, retrieval, pedagogy, tutor bots, optional web API) |
| `demo/` | CSV-backed knowledge graph + embeddings cache used by the retriever |
| `src/experiments_manual/` | Offline KG edge discovery and evaluation scripts |
| `architecture/` | Phase notes and manual ingestion / retrieval design sketches |
| `frontend/` | Vite + React demo UI |

---

## Knowledge graph construction (offline)

If you are building or refreshing graphs from sources, see commands and context in **`architecture/manual_ingestion.md`** and **`README`-era scripts under `src/experiments_manual/`** (paths in those docs may vary by branch). The **running tutor** in `workflow_demo` consumes data under **`demo/`** as loaded by the runtime.
