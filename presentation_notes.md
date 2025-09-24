### Slide 1 — Overarching Problem

- **Goal**: Help learners get the right content at the right time.
- **Issue with off‑the‑shelf KGs**: Too generic, noisy edges, hard to control quality.
- **Our approach**: Manually generate a high‑quality knowledge graph from OpenStax, use Zep’s temporal layer for memory, and build a simple, reliable RAG system (Retriever → Coach Generator → Tutor).

### Slide 2 — Scope and Data

- **Domain**: OpenStax mathematics content (local‑first workflow).
- **Nodes**: Learning Objectives (LOs), Content Items (concepts, examples, try‑its).
- **Edges**: 
  - LO→LO: `prerequisite`
  - LO→Content: `explained_by`, `exemplified_by`, `practiced_by`
- **Demo dataset**: 138 LOs, 270 content items, 577 edges (for previews/evaluation).

### Slide 3 — Ideation and Approach Shift

- Phase 1: Zep ingestion baselines (V1–V4) validated ontology ideas but were not production‑ready.
- Decision: Move to a manual, offline, multimodal pipeline for precise control.
- Keep Zep in the loop for what it does best now: temporal memory in conversations.

### Slide 4 — Manual KG Construction (Overview)

- **Inputs**: `data/processed/lo_index.csv`, `data/processed/content_items.csv`.
- **Discovery**: LLM‑driven edges with strict prompts and pruning.
- **Validation**: Structural checks + semantic LLM checks.
- **Visualization**: Interactive graph preview HTML.

### Slide 5 — Code‑First Pipeline Overview (with references)

- Plain-language: This is the exact path your data follows from raw tables to a clean graph.
- data extraction → build source tables for LOs and Content
  - Source: database export (SQL) or OpenStax CSVs [CODE]
  - Files: `data/processed/lo_index.csv`, `data/processed/content_items.csv` [SS]
- wrangle + order → normalize fields, arrange LOs chronologically by unit/chapter
  - Files: `src/experiments_manual/prepare_lo_view.py` [CODE]
- candidate generation → propose edges with strict filters
  - Files: `src/experiments_manual/discover_content_links.py`, `src/experiments_manual/discover_prereqs.py` [CODE]
- semantic judging → LLM labels per relation type
  - Files: `src/experiments_manual/evaluate_llm.py` [CODE]
- structural checks + viz → integrity, cycles, preview
  - Files: `src/experiments_manual/evaluate_outputs.py`, `data/processed/graph_preview.html` [CODE][SS]
- deliverables → CSVs + final machine‑readable `kg.json`
  - Files: `edges_content.csv`, `edges_prereqs.csv`, `kg.json` [SS]

### Slide 6 — Data Extraction (SQL → Tables)

- Plain-language: We export two clean tables: learning objectives and content items.
- Source of truth: database tables for LOs and Content. [SS]
- Example SQL (template; adjust to your schema):
```sql
-- Learning Objectives
SELECT lo_id, learning_objective, unit, chapter, book
FROM learning_objectives
WHERE book = 'Calculus Volume 1';

-- Content Items
SELECT content_id, content_type, lo_id_parent, text, learning_objective, unit, chapter, book
FROM content_items
WHERE book = 'Calculus Volume 1';
```
- Export to CSVs → `lo_index.csv`, `content_items.csv` for downstream steps. [CODE][SS]

### Slide 7 — Data Wrangling & LO Ordering

- Plain-language: Clean column values and build short summaries so prompts stay focused.
- Normalize fields (trim, consistent enums, dedupe IDs). [CODE]
- Build compact summaries: aggregate content text per LO.
- Chronological arrangement:
  - Sort key: (unit, chapter, within_chapter_index).
  - Double for‑loop passes ensure stable order and predictable edge direction.
- Code: `src/experiments_manual/prepare_lo_view.py` (ordering/transforms). [CODE][SS]

- For‑loop directionality (forward‑only) — expanded:
  - Outer loop iterates target LOs in the sorted order; inner loop considers only source LOs where `chronological_key(source) < chronological_key(target)`.
  - Guardrails: skip equivalence/part‑of candidates; after scoring/filtering, drop reciprocals (keep only forward edge if `A→B` and `B→A` appear).
  - Reuse: the same `create_chronological_key` is used for generation and validation; chronology is re‑checked post‑scoring.
  - [CODE]
    ```python
    # Pseudocode for forward‑only prerequisite proposal
    ordered = sort(los, key=chronological_key)
    for target in ordered:
        for source in candidates_in_scope(target):  # same unit/chapter filters
            if chronological_key(source) < chronological_key(target):
                emit_candidate(source, target)
    # Post‑process: remove reciprocals; assert DAG via NetworkX
    ```
  - [SS] Show `discover_prereqs.py` → `generate_prereq_candidates(...)` loops and `create_chronological_key(...)` definition.

### Slide 8 — Candidate Generation (Edge Discovery Focus)

- Plain-language: We propose only likely links; later steps judge them.
- What edges we propose:
  - LO → LO `prerequisite`: A must be understood to achieve B at its intended depth.
  - LO → Content `explained_by`: Content teaches the core idea(s) of the LO.
  - LO → Content `exemplified_by`: Worked examples mirroring the LO’s problem types.
  - LO → Content `practiced_by`: Practice aligned to the LO’s skills.
- Signals we look for before judging:
  - Same unit/chapter scope to reduce false positives.
  - Title/summary overlap on key terms and verbs.
  - Role cues: examples, exercises, conceptual walkthroughs.
  - Prereqs: enabling concepts, not equivalence or part‑of.
- Directionality and granularity:
  - Avoid equivalence/part‑of in prerequisites.
  - Prefer explicit role signals in content.
- Candidate set sizing:
  - Keep shortlists small and relevant for quality and speed.
- Code: `src/experiments_manual/discover_content_links.py` (filters and thresholds). [CODE][SS]

- Optional LLM scoring for candidates (numeric):
  - Schema: `{"results":[{"lo_id","score","confidence","rationale"}]}` where `score ∈ [-1,1]`, `confidence ∈ [0,1]`. [CODE]
  - Thresholds: keep only `score >= score_threshold`; for prereqs, also keep only if `confidence >= min_confidence` (when set).
  - Heuristic fallback: if LLM is unavailable, use Jaccard overlap on aggregated texts to assign `score` and a simple `confidence` heuristic. [CODE]
  - Post-scoring guards: re-check forward chronology for prereqs; remove reciprocals; write `score` and `confidence` to CSV.
  - Files: `discover_prereqs.py` → `score_prereq_candidates(...)`; `discover_content_links.py` → `score_candidates(...)`. [CODE]

### Slide 9 — Prompt Strategy (Relation‑Specific)

- Plain-language: We ask the LLM different yes/no questions depending on link type.
- Role‑specific prompts:
  - Prerequisite: “Is source required for target at intended depth?” Disallow related/equivalent/part‑of.
  - Content roles: “Does this content teach / demonstrate with worked examples / provide practice for the LO?”
- Inputs kept compact:
  - LO/content titles and short summaries only (no long passages), to minimize drift.
- Labels used to reflect nuance:
  - Prereq → `prerequisite`, `supports`, `incorrect`.
  - Content → `correct`, `supports`, `incorrect`.
  - “supports” captures helpful but not perfect matches to preserve useful connections.
- Strict output and tight wording:
  - One JSON object (label + short reason) to keep decisions focused and comparable.
  - Consistent definitions across judgments to stabilize results and reduce variance.
 - Code refs: `evaluate_llm.py` → `build_prereq_prompt`, `build_content_prompt` [CODE][SS]
  - Note: This slide covers label-only evaluation prompts. Discovery scoring prompts use the numeric schema in Slide 8 (`score` + `confidence`).

### Slide 10A — Edge Evaluation: Structure & Cycle Avoidance

- Plain-language: First we make sure the graph is structurally sound (no missing IDs, no loops, reasonable degrees).
- Structural checks:
  - Referential integrity; degree stats; cycle detection (NetworkX) in `src/experiments_manual/evaluate_outputs.py`. [CODE][SS]
- Cycle avoidance in generation:
  - Forward‑only proposal: iterate targets in chronological order; consider sources only if `chronological_key(source) < chronological_key(target)`.
  - Remove reciprocals post‑scoring; keep only the forward chronological edge.
  - Pointer: See Slide 7 for the ordering and for‑loop details. [SS]

### Slide 10B — Edge Evaluation: LLM Judge & Manual Review

- Plain-language: Then the LLM judges whether each proposed edge makes sense semantically.
- Inputs to the judge:
  - Prereq: source/target LO titles and short summaries.
  - Content: LO title/summary and content title/summary.
- Judgment questions:
  - Prereq: is source required for target at the intended depth (beyond recall)?
  - Content: does this teach, demonstrate, or provide practice for the LO?
- Labels and meaning:
  - Prereq → prerequisite (required), supports (helpful), incorrect (unrelated/equivalent/part‑of).
  - Content → correct (direct fit), supports (related/helpful), incorrect (irrelevant/misleading/harmful).
- Decision rules:
  - Accept: prerequisite (prereqs) and correct (content).
  - Keep as context: supports.
  - Remove: incorrect.
- LLM judge (strict):
  - `src/experiments_manual/evaluate_llm.py` with JSON‑only outputs (schema: {"label","reason<=200"}); temperature 0.0; retries+JSON mode. [CODE][SS]
  - Confidence (optional): compute from retry stability or map labels → scores (e.g., prerequisite/correct=1.0, supports=0.6, incorrect=0.0).
- Manual review:
  - Queue borderline edges (e.g., supports, near conflicts); sample and adjudicate.
 - Note: Evaluation is label-only (no scores/confidence). Scores/confidence apply in discovery (Slide 8).

### Slide 15 — RAG System (Design at a Glance)

- Plain-language: The KG powers search and tutoring; Zep remembers the conversation.
- **Retriever**: Uses the KG to fetch relevant LOs and content. [SS]
- **Coach Generator**: Produces learning strategies and paths using validated edges. [SS]
- **Tutor**: Guides learners interactively; context comes from the Retriever and Zep’s temporal layer. [SS]
- **Zep temporal layer**: Provides conversation memory and context continuity. [SS]

### Appendix — Environment & Repro Notes

- Use Python venv: `python3 -m venv venv && source venv/bin/activate`.
- Install: `pip install -r requirements.txt`.
- Keys: set `.env` with `OPENAI_API_KEY`; code calls `load_dotenv()`.
- Outputs to watch: `edges_content.csv`, `edges_prereqs.csv`, `llm_edge_checks*.jsonl`, `graph_preview.html`.

### Slide 16 — Cost Estimate (Discovery + Evaluation)

- GPT‑4o‑mini Pricing
  - Input Tokens: $2.50 / 1M tokens ($0.0025 / 1K)
  - Output Tokens: $10.00 / 1M tokens ($0.0100 / 1K)

- Per LO→LO Pair (based on our batching and observed prompt sizes)
  - Discovery: ~0.225K input ($0.00056) + ~0.023K output ($0.00023) ≈ **$0.00079**
  - Evaluation (label‑only): ~1.10K input ($0.00275) + ~0.02K output ($0.00020) ≈ **$0.00295**
  - Total per LO→LO pair ≈ **$0.00374**

- Current Demo Run (scale of what we actually ran)
  - LO→LO pairs judged: 326 candidates discovered; 290 final edges evaluated
  - LO→LO subtotal: ~326 × $0.00079 + 290 × $0.00295 ≈ **$1.1**
  - End‑to‑end (including LO↔Content discovery + eval): **≈ $3–$5** total for the demo

- Full Per‑Textbook Projections
  - Naive all‑pairs (if 500 LOs and you scored every pair: 500×499/2 = 124,750 pairs): 124,750 × $0.00374 ≈ **$467**
  - Our pipeline with pruning (use observed avg ~2.96 candidates/target): ~500 × 2.96 ≈ 1,480 LO→LO pairs → 1,480 × $0.00374 ≈ **$5.5** (LO→LO only)
  - End‑to‑end (including LO↔Content discovery + evaluation at current sizes/config): **≈ $305–$360 per textbook**; **≈ $1.2K–$1.45K for four textbooks**

- Notes
  - Images are passed as URLs; costs accounted via tokenized text, not separate vision fees.
  - Assumptions from current `config.yaml` (max 8 targets/call) and observed averages (candidates/call, token sizes).
### Slide 17 — Practical Notes

- Cost control: Parallelize by chapter; cache identical judgments; review patterns to tighten candidates.
- Include screenshots next to code bullets for each phase (extraction, wrangling, discovery, eval, export).
- Provide `kg.json` early for team integration; keep CSVs for audits.

