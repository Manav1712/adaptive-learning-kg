## Coach–Retriever Architecture and Behaviors

### Goals
- **Just-enough context**: return only minimal, high-signal facts, LOs, and content pointers.
- **Graph-aware retrieval**: semantic match first, then traverse `PREREQUISITE_OF` and `ASSESSED_BY`.
- **Safety and precision**: constrain to KG; avoid hallucinations; attribute all context.

### High-level flow
1. **Coach: Intent + Constraints**
   - Classify intent (tutoring, practice, definition, comparison, troubleshooting).
   - Derive constraints: content types, difficulty, token budget, depth, subject, and whether prerequisites are needed.

2. **Coach: Query Understanding**
   - Normalize/consolidate query; optionally generate paraphrases within a small budget.
   - Produce structured retrieval request with filters and budgets.

3. **Retriever: Multi-stage Retrieval**
   - Stage A (Hybrid recall): dense semantic + lexical (BM25/SPLADE) over LO and content.
   - Stage B (Re-rank): cross-encoder or LLM re-rank within a tight window; apply MMR for diversity.
   - Stage C (Graph expansion): limited-radius traversal to collect minimal 
   prerequisites and aligned content.
   
4. **Retriever: Context Compression**
   - Extract 3–7 bullet facts; include 1–3 content items; return citations.
5. **Coach: Answer or Clarify**
   - If confidence low or ambiguous: ask a clarifying question.
   - Else: build explanation or practice plan using returned minimal context.

### Coach behavior
- **Inputs**: student utterance, short conversation snippet, optional learner profile.
- **Steps**:
  - Classify intent; detect ambiguity/out-of-scope.
  - Build `retrieval_request` with:
    - query, subject, conversation_snippet
    - intent, desired content types, difficulty
    - budgets: top_k, max_tokens, graph_depth, timeout_ms
  - If ambiguous: include `seek_clarification=true`.
  - Call Retriever and evaluate `can_answer` and `confidence`.
  - If `needs_clarification`: ask 1 concise question; else compose tutoring response or route to Planner/Tutor.

### Retriever behavior
- **Stage A – Hybrid recall**
  - Dense encoder on LO titles/descriptions/summaries and content metadata; combine with lexical scores.
  - Filters: subject, content types, difficulty; respect timeouts.
- **Stage B – Re-ranking + diversity**
  - Cross-encoder/LLM re-ranking on top-N; apply MMR to avoid same-node near-duplicates.
- **Stage C – Graph-aware expansion (budgeted)**
  - For top LO candidates, traverse `PREREQUISITE_OF` up to `depth_prereq` to assemble the minimal path to readiness.
  - Pull `ASSESSED_BY` content aligned to those LOs; prefer examples vs exercises based on intent.
- **Context compression**
  - Selective extract of key sentences; generate 3–7 atomic facts; include 1–3 content items with snippets.
  - Hard token cap; drop anything not needed to answer the query.
- **Guardrails**
  - No new facts beyond KG; LLM only for rewriting/reranking/compression.
  - Attribute every fact to LO IDs/content IDs.

### Retrieval Mode Options

#### Hybrid Mode (Recommended)
- **Stage A**: Dense embeddings (FAISS) + lexical (BM25/SPLADE), fused via RRF
- **Stage B**: Cross-encoder or LLM re-ranking on top 30-50 candidates
- **Pros**: High recall, stable performance, cost-effective at scale
- **Cons**: Requires embedding index setup
- **Best for**: Production systems, large corpora, cost-sensitive deployments

#### LLM-Only Mode (Small Scale)
- **Stage A**: LLM ranks all items or broad candidate set directly
- **Stage B**: Optional second-pass LLM refinement
- **Pros**: Minimal infrastructure, handles ambiguous queries well
- **Cons**: Higher latency/cost, less deterministic, recall fragility
- **Best for**: Prototyping, small corpora (<500 nodes), research experiments

#### Mode Selection Guidelines
- **Current scale (138 LOs, 270 content)**: Either mode viable; hybrid preferred for production
- **Latency budget**: <1.2s p95 for hybrid; <3s for LLM-only
- **Cost budget**: Hybrid ~$0.01/1k queries; LLM-only ~$0.10/1k queries
- **Fallback**: Always include lexical fallback and timeout handling

### Coach ↔ Retriever I/O contract

- **Request**
```json
{
  "query": "How do I solve a quadratic equation?",
  "subject": "algebra",
  "intent": "tutoring",
  "conversation_snippet": "We discussed factoring earlier.",
  "constraints": {
    "content_types": ["Example", "Exercise"],
    "difficulty": "intro",
    "top_k": { "lo": 5, "content": 5 },
    "graph_depth": { "prereq": 1, "content": 1 },
    "token_budget": 900,
    "timeout_ms": 1200
  },
  "seek_clarification": false
}
```

- **Response**
```json
{
  "can_answer": true,
  "needs_clarification": false,
  "confidence": 0.76,
  "matched_los": [
    { "id": "LO-ALG-021", "title": "Solve quadratic equations", "reason": "semantic match + rerank", "score": 0.83 }
  ],
  "supporting_los": [
    { "id": "LO-ALG-009", "title": "Factor trinomials", "edge": "PREREQUISITE_OF", "path_len": 1, "score": 0.71 }
  ],
  "content_items": [
    { "id": "EX-342", "type": "Example", "title": "Completing the square, step-by-step", "for_lo": "LO-ALG-021", "score": 0.78 },
    { "id": "EXR-118", "type": "Exercise", "title": "Solve x^2-5x+6=0", "for_lo": "LO-ALG-021", "score": 0.74 }
  ],
  "minimal_context": [
    "A quadratic equation has the form ax^2 + bx + c = 0.",
    "Factorable quadratics can be solved by finding two numbers that multiply to ac and sum to b.",
    "If not factorable, use quadratic formula: x = (-b ± √(b^2-4ac)) / (2a)."
  ],
  "citations": [
    { "type": "LO", "id": "LO-ALG-021" },
    { "type": "LO", "id": "LO-ALG-009" },
    { "type": "Content", "id": "EX-342" }
  ],
  "telemetry": {
    "stages": { "hybrid": 30, "rerank": 15, "graph": 10, "compress": 18 },
    "applied_filters": ["subject:algebra", "types:Example,Exercise"]
  }
}
```

### SOTA retrieval practices embedded
- **Hybrid retrieval**: dense + lexical; reciprocal rank fusion for robust recall.
- **Generative query rewriting**: light-weight paraphrases/HyDE constrained to budget; route by intent.
- **Cross-encoder re-ranking**: high-precision on small candidate pool.
- **MMR diversity**: reduce redundancy across near-duplicate nodes/content.
- **Graph-aware scoring**: boost LOs with coherent prerequisite paths and aligned `ASSESSED_BY` content.
- **Parent–child chunking**: index short chunks but attribute to parent LO/content to keep context coherent.
- **Selective context compression**: extract-only summaries; attribute every bullet to sources.
- **Confidence and refusal**: degrade gracefully with clarifications or safe fallback when confidence is low.
- **Observability**: capture per-stage latency, hit ratios, coverage of cited nodes, and recall@k on a held-out set.

### Scenario playbooks
- **Ambiguous query (“do derivatives use limits?”)**
  - Coach: sets `seek_clarification=true`; asks, “Do you want a definition, an example, or practice?”
  - Retriever: returns top-2 LO clusters + 1 example each; low `confidence`; minimal context only.
- **Direct tutoring ask (“explain chain rule”)**
  - Coach: tutoring intent, examples preferred; `graph_depth.prereq=1`.
  - Retriever: 1–2 LOs, 1 prerequisite, 1 example + 1 quick exercise; 3–5 bullets.
- **Practice-only request (“give me problems on integrals”)**
  - Coach: practice intent; filter `content_types=["Exercise"]`; skip prerequisite expansion.
  - Retriever: diverse exercises across difficulty; no long bullets; ensure coverage.
- **Troubleshooting (“my factoring keeps failing”)**
  - Coach: flags misconception; request examples that contrast correct/incorrect.
  - Retriever: examples with common pitfalls; short bullets on checks.
- **Multi-step concept chain (“Taylor → series → error bounds”)**
  - Coach: set `graph_depth.prereq=2`, token_budget tight.
  - Retriever: minimal path of 2–3 LOs with 1 example; bullets reflect progression.
- **Out-of-scope**
  - Retriever: `can_answer=false`, suggest closest in-scope LOs; Coach informs limits.
- **Follow-up reference (“that last step”)**
  - Coach: pass conversation snippet; retriever favors previously cited IDs.
  - Retriever: fetch adjacent content to prior LO IDs; short delta-only bullets.
- **Low-confidence data region**
  - Retriever: mark `needs_clarification=true`; return 1 clarifying question plus best-effort minimal context.

### Configuration defaults (tunable)
- **top_k**: `{ lo: 6, content: 6 }`; **graph_depth**: `{ prereq: 1, content: 1 }`
- **timeouts**: 1.2s p95 for retrieval; 200ms budget for LLM rewrite; 250ms for re-rank.
- **budgets**: 800–1000 tokens response cap; 3–7 bullets; ≤3 content items.

### Guardrails
- **Attribution required**: every bullet must cite LO/content IDs.
- **No free-form generation of facts**: LLM only for rewrite/rerank/compress.
- **Privacy/safety**: filtered subject domains; block unsafe content types.

### Integration notes
- **Indexes**: reuse your FAISS dense index and add a lightweight lexical index; fuse scores.
- **Graph ops**: use acyclic `PREREQUISITE_OF` paths; score coherence and path length.
- **Zep graphs (optional)**: if using Zep, apply ontology and use graph queries with `user_ids/graph_ids` filters; ontology controls entity/edge types and improves precision.

## Appendix: NLP Terms Explained

### Core Retrieval Concepts

**Dense Embeddings**
- Converting text into numerical vectors (lists of numbers) that capture meaning
- Similar concepts get similar vectors, so "car" and "automobile" would be close in this number space
- Think of it like giving each word/concept a unique "fingerprint" based on its meaning

**Lexical Retrieval (BM25/SPLADE)**
- Traditional text search that looks for exact word matches and word frequency
- BM25: counts how often query words appear in documents, weighted by rarity
- SPLADE: sparse neural model that learns which words are important for each document
- Like a librarian finding books by looking up specific keywords in the catalog

**Reciprocal Rank Fusion (RRF)**
- Combines results from multiple search methods (dense + lexical) into one ranked list
- Takes rankings from each method and merges them mathematically
- Ensures no single method dominates; balances different types of relevance

**Cross-encoder Re-ranking**
- Takes a small set of candidates (like top 30-50) and compares each one directly to the query
- More expensive but more accurate than initial retrieval
- Like having a human expert review the top candidates and pick the best ones

**MMR (Maximal Marginal Relevance)**
- Ensures diversity in results by avoiding near-duplicate items
- Prevents returning 5 very similar examples when you want variety
- Balances relevance with diversity

### Why Multi-Stage Retrieval?

**Single-stage approaches have problems:**
- Dense embeddings alone: fast but miss exact word matches
- Lexical search alone: finds exact words but misses meaning
- LLM-only ranking: expensive, slow, inconsistent

**Multi-stage pipeline solves this:**
- Stage A: Cast wide net (high recall) with hybrid search
- Stage B: Pick best candidates (high precision) with re-ranking
- Stage C: Use graph structure to find related content
- Result: Find relevant stuff + ensure it's actually good + cost-effective

### Query Processing

**HyDE (Hypothetical Document Embeddings)**
- Generates fake "ideal" documents that would answer the query
- Uses these fake documents to find real similar content
- Like asking "what would a perfect answer look like?" then finding real content that matches

**Query Rewriting/Paraphrasing**
- Takes the user's question and creates alternative ways to ask the same thing
- Helps catch cases where the user's wording doesn't match how concepts are described in the knowledge base
- Like translating between different ways of asking the same question

### Graph Concepts

**Graph Traversal**
- Following connections between nodes in the knowledge graph
- Like following a trail of breadcrumbs: if A leads to B, and B leads to C, then A→B→C is a path
- Used to find prerequisites: if you need to learn derivatives, what should you learn first?

**Acyclic Graph**
- A graph where you can't go in circles (no A→B→C→A loops)
- Ensures prerequisite chains make logical sense
- Like a family tree where you can't be your own ancestor

**Edge Scoring**
- Each connection between concepts gets a strength score (0-1)
- Higher scores mean stronger/more important relationships
- Like rating how important each prerequisite is on a scale

### Evaluation Metrics

**Recall@K**
- Of all the correct answers, how many did we find in our top K results?
- If there are 10 correct answers and we found 7 in top-10, recall@10 = 70%

**MRR (Mean Reciprocal Rank)**
- Average of 1/rank for the first correct answer found
- If correct answer is at position 3, MRR = 1/3 = 0.33
- Higher is better; perfect would be 1.0

**Precision**
- Of all the results we returned, how many were actually correct?
- If we return 10 items and 8 are correct, precision = 80%

**F1 Score**
- Combines precision and recall into one number
- Balances finding the right things (recall) with not returning wrong things (precision)

### Technical Terms

**FAISS**
- Facebook's library for fast similarity search on large collections of vectors
- Like a super-fast search engine for finding similar numerical fingerprints

**Token Budget**
- Limit on how many "pieces" of text (tokens) can be processed or returned
- Tokens are roughly words or word-parts; helps control response length and cost

**Semantic Matching**
- Finding content based on meaning rather than exact word matches
- "car" matches "automobile" even though they're different words
- Uses dense embeddings to understand meaning

**Hallucination**
- When AI systems make up information that isn't in their training data
- Like confidently stating false facts that sound plausible
- We prevent this by only using information from our knowledge graph

**Attribution**
- Clearly marking which source (LO or content item) each piece of information came from
- Like footnotes in a research paper - every fact can be traced back to its source
