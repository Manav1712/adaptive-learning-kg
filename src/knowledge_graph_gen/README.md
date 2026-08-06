# Knowledge Graph Generation

Turn draft content CSVs ("chunks") into a curriculum knowledge graph.

Corpus size does not change the process — larger inputs just take longer and cost more API calls.

## What you put in

Put draft CSVs under `data/raw/` matching these names:

- `concept_draft_contents*.csv`
- `example_draft_contents*.csv`
- `try_it_draft_contents*.csv`

Each row is one content chunk with columns:

| Column | Meaning |
|--------|---------|
| `lo_id` | Learning objective id |
| `raw_content` | JSON body (concept text, or example/try_it problem + steps) |
| `type` | `concept`, `example`, or `try_it` |
| `book` | Book title |
| `learning_objective` | LO text |
| `unit` | Unit name |
| `chapter` | Chapter name |

## How chunks become a graph

```
data/raw/*_draft_contents*.csv
        │
        ▼  prepare_nodes
   lo_index.csv          (LO nodes + curriculum order)
   content_items.csv     (concept / example / try_it nodes; answers stripped)
        │
        ├──────────────────────────┐
        ▼                          ▼
  discover_prereqs           discover_content_links
  (earlier LO → later LO)    (every content × every LO, scored)
        │                          │
        ▼                          ▼
  edges_prereqs.csv          edges_content.csv
        │                          │
        └──────────┬───────────────┘
                   ▼
     knowledge_graph/runs/<run_id>/
```

1. **Prepare nodes** — Parse each chunk. Build LO nodes and content nodes. Add order columns so prerequisites can only go forward in the curriculum.
2. **Discover prerequisites** — For each later LO, propose earlier LOs as candidates. An LLM scores pairs. Keep edges at score ≥ 0.7 (default). Relation: `prerequisite`.
3. **Discover content links** — For each content item, propose all LOs as candidates. An LLM scores pairs. Keep edges at score ≥ 0.8 (default). Relation by type:
   - concept → `explained_by`
   - example → `exemplified_by`
   - try_it → `practiced_by`
4. **Evaluate (optional)** — Heuristic checks (structure / integrity). Optional LLM second-opinion on each edge. Neither step rewrites the graph; they only report.

## How to run

Requires `OPENAI_API_KEY` and dependencies from the repo `requirements.txt`.

**Full pipeline (standard process):**

```bash
python -m src.knowledge_graph_gen.run --raw-dir data/raw
```

**Smoke test** (limit how many targets/items are scored):

```bash
python -m src.knowledge_graph_gen.run --raw-dir data/raw --limit 5
```

**With evaluation:**

```bash
python -m src.knowledge_graph_gen.run --raw-dir data/raw --eval
python -m src.knowledge_graph_gen.run --raw-dir data/raw --eval --eval-llm
```

**Step by step** (same run folder):

```bash
RUN=knowledge_graph/runs/<run_id>

python src/knowledge_graph_gen/prepare_nodes.py --raw-dir data/raw --run-dir "$RUN"
python src/knowledge_graph_gen/discover_prereqs.py --run-dir "$RUN"
python src/knowledge_graph_gen/discover_content_links.py --run-dir "$RUN"

python src/knowledge_graph_gen/evaluate_heuristic.py --run-dir "$RUN" --edges-kind prereqs
python src/knowledge_graph_gen/evaluate_heuristic.py --run-dir "$RUN" --edges-kind content
python src/knowledge_graph_gen/evaluate_llm.py --run-dir "$RUN" --edges-kind prereqs
python src/knowledge_graph_gen/evaluate_llm.py --run-dir "$RUN" --edges-kind content
```

## What you get

Each run lands in `knowledge_graph/runs/<YYYYMMDD_HHMMSS>/`:

| File | Role |
|------|------|
| `lo_index.csv` | LO nodes |
| `content_items.csv` | Content nodes |
| `edges_prereqs.csv` | LO → LO prerequisite edges |
| `edges_content.csv` | LO ↔ content edges |
| `manifest.json` | Run metadata (counts, thresholds, model) |
| `intermediates/` | Candidate pairs + optional eval outputs |

Those four CSVs are the knowledge graph artifact. Promote a run into `demo/` manually if the tutoring runtime should use it.

## Scripts in this folder

| Script | Job |
|--------|-----|
| `run.py` | One-command orchestrator |
| `prepare_nodes.py` | Chunks → nodes |
| `discover_prereqs.py` | Score LO→LO edges |
| `discover_content_links.py` | Score content↔LO edges |
| `evaluate_heuristic.py` | Structural / integrity QA (no LLM) |
| `evaluate_llm.py` | Semantic second-opinion QA (LLM) |
