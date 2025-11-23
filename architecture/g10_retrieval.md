## G10 — Did it work? Retrieval evaluation (LLM-only vs embeddings)

### G1 problem recap
- We want the system to pull **just enough** correct, cited information from the knowledge graph so the Coach can answer a student’s question.
- It should be **fast enough** for live tutoring, not too expensive, and should avoid making things up.

### Scope of this G10
- We compared two ways of doing retrieval on the current graph (about 138 LOs and 270 content items, and growing):
  - **LLM-only retrieval** – the LLM reads everything (or a big chunk), picks what seems best, and ranks it.
  - **Embeddings-based retrieval** – use vector search + keyword search to get candidates, then re-rank, follow graph links, and compress the final context.
- We judged them on:
  - How often they surface the “right” LOs and content.
  - How small and focused the final context is.
  - How fast and cheap they are.
  - How easy they are to keep stable and safe.

### What we saw (high level)
- **LLM-only**
  - Good at understanding fuzzy or short questions.
  - Simple to wire up at this size of graph.
  - But slower and more expensive, and results can shift more when prompts or models change.
  - Harder to see *why* something was chosen; easier for the model to quietly slip in unsupported facts if we are not strict.

- **Embeddings-based (multi-stage)**
  - Uses dense + lexical search to get a solid candidate set, then a smaller model to re-rank and the graph to fill in prereqs and linked content.
  - Much more predictable on latency and cost, and easier to cache.
  - Natural fit for “just enough” answers: a few LOs, a few content items, clear IDs, and short bullets.
  - Needs a bit more infra (indexes, refresh jobs), but behavior is easier to reason about and tune.

### Did it solve the G1 problem?
- **Yes, with the embeddings-based approach as the default.**
  - It reliably returns 1–2 best-fit LOs, 0–1 prerequisite LOs, and 1–3 good content items, all cited.
  - The amount of text stays within our budget so the Coach prompt doesn’t blow up.
- LLM-only *can* work at this scale, but for ongoing use it’s too sensitive to prompt/model changes and too expensive to be the main path.

### Decision
- **Default:** use embeddings-based, multi-stage retrieval:
  - Stage A: dense + lexical recall, fused.
  - Stage B: small LLM / cross-encoder re-ranking with diversity.
  - Stage C: 1-hop graph expansion for prereqs and linked content.
- **Backup:** keep an LLM-only mode around for experiments and special cases, behind timeouts and clear safety checks.

### Risks and mitigations
- **LLM-only mode can get unstable over time**
  - Lock down prompts, keep temperature at 0, and always check that the model’s output matches the JSON format we expect.
- **Recall can drop as we add more content**
  - Keep both dense and lexical search turned on, watch simple “does the right LO show up in top‑k?” checks, and turn on HyDE-style query expansion if short questions start to miss things.
- **Noisy or inconsistent edge types in the graph**
  - Prefer a small, clean set of relations (like `PREREQUISITE_OF` and `ASSESSED_BY`) and downweight or ignore everything else when scoring.

### Recommended G2s (next)
1) **Create a small “truth set” of queries**
   - Pick 50–100 real student questions, mark which LOs and content *should* show up, and use that to track basic metrics over time.
2) **Add simple retrieval logging**
   - For each query, log how long each stage took, whether we hit the cache, and which LOs/content we returned; review this regularly (for example, weekly).
3) **Wire in clarifying questions**
   - Define a handful of standard clarifying questions and a simple rule like “if confidence is low, ask instead of answering,” and track how often this happens and whether it helps.
4) **Tighten the graph schema**
   - Where we can, enforce or map edges into a small set of relation types so most edges are `PREREQUISITE_OF`, `PART_OF`, or `ASSESSED_BY` instead of hundreds of one‑off labels.
5) **Add caching and batching**
   - Cache common embedding and re-ranking results and batch similar calls together; measure how much this lowers p95 latency.
6) **Put hard limits on cost**
   - Set token and time budgets for re-ranking and compression, and alert if we start going over them.

### Bottom line (G10)
- The embeddings-based, multi-stage retriever does what we need: it finds the right pieces of the graph, keeps responses small and grounded, and stays fast and affordable. LLM-only retrieval is useful as a backup or research path, but not as the main way we answer student questions. 

### Examples from CSVs (4 best comparisons)
- Case 1: Sum & Difference identities → cosine formulas
  - Target LO: 961 “Using the Sum and Difference Formulas for Cosine”
  - Prereq path (from edges_prereqs.csv):
    - 960 “Sum and Difference Identities” → 961 (prerequisite, score 0.9, conf 0.95)
  - Supporting content (from edges_content.csv):
    - 961_concept_1 explained_by 961 (score 1.0) — “specifically focuses on using the sum and difference formulas for cosine”
  - Why this is a good comparison:
    - Embeddings+lexical recall reliably surfaces 960 and 961; re-ranking prefers the direct “cosine” item; graph step adds just the single prerequisite.

- Case 2: Law of Sines → solve oblique triangles
  - Target LO: 1035 “Using the Law of Sines to Solve Oblique Triangles”
  - Prereq path:
    - 1034 “Non-right Triangles: Law of Sines” → 1035 (prerequisite, score 0.9, conf 0.9)
  - Supporting content:
    - 1035_concept_1 explained_by 1035 (score 1.0) — “aligning perfectly with the learning objective”
  - Why it’s strong:
    - Clean, single-hop path; top content exactly matches the LO. Minimal, well-cited bundle (1 LO + 1 prereq + 1 content).

- Case 3: Polar coordinates → transform equations
  - Target LO: 1047 “Transforming Equations between Polar and Rectangular Forms”
  - Prereq path:
    - 1043 “Polar Coordinates” → 1047 (prerequisite, score 0.9, conf 0.9)
    - 1045 “Converting from Polar to Rectangular” → 1047 (prerequisite, score 0.9, conf 0.9)
  - Supporting content:
    - 1047_concept_1 explained_by 1047 (score 1.0)
  - Why it’s strong:
    - Graph step adds exactly the two skills learners typically need before transforming equations; content is a perfect match.

- Case 4: Vectors → operations in i and j
  - Target LO: 1077 “Performing Operations with Vectors in Terms of i and j”
  - Prereq path:
    - 1073 “Vector addition and scalar multiplication” → 1077 (prerequisite, score 0.9, conf 0.9)
  - Supporting content:
    - 1077_concept_1 explained_by 1077 (score 1.0)
  - Why it’s strong:
    - Retrieval surfaces the general “vectors” family, re-ranking picks the precise LO, and the graph supplies one clear prerequisite — yielding a tight, teachable set.