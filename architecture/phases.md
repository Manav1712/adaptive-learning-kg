# Phases

## Current Status: Phase 1 - Advanced Experiments 🚧

**Last Updated**: Baseline V3 ontology enforcement experiment launched
**Next Milestone**: Evaluate V3 results → decide production readiness vs further iteration
**Key Learning**: FREE plan processing limits (2-6 hours for 270 episodes), ontology enforcement critical for production

---

## Phase 1 — Foundations & Ground Truth (Weeks 1–3) 🚧 **IN PROGRESS**

**Agents**
- ✅ Corpus Ingestion → pull OpenStax, chunk
- ✅ Entity/Fact Extractor → LO/Concept/Problem + edges  
- ✅ Resolver/Deduper → merge synonyms, keep provenance
- ✅ KG Writer → write to Zep/Neo4j; light Community Detector

**Flow**
1. ✅ Fetch → chunk → extract entities/relations
2. ✅ Dedup → write nodes/edges (+ section/exercise IDs)

**You get:** A clean static subject KG (no personalization yet).
Example: `PREREQUISITE_OF(Factoring → Completing the Square)`; `ASSESSED_BY(CTS → Ex 9.4 #17)`.

**Progress Status:**
- **Baseline V1**: Built initial KG with 474 nodes, 493 edges (baseline performance)
- **Baseline V2**: Schema hints experiment - 270 episodes, 24.6% constraint effectiveness, 169 edge types
- **Baseline V3**: Ontology enforcement experiment - 270 episodes with custom entity/edge types, fact rating, processing overnight
- **Key Achievements**: Fixed async processing pipeline, solved rate limiting, implemented comprehensive evaluation framework
- **Critical Discovery**: Schema hints insufficient for production - need ontology enforcement for edge type control

**Current Investigation (Baseline V3):**
- **Custom Ontology**: Concept/Example/Exercise/TryIt entities with property definitions
- **Edge Constraints**: PREREQUISITE_OF/PART_OF/ASSESSED_BY with source-target restrictions  
- **Fact Rating**: Applied instruction to filter low-relevance relationships
- **Type Balancing**: 250 max per type to ensure concept dominance
- **Status**: Processing on Zep (2-6 hour completion time for 270 episodes)

**Decision Point**: 
- ✅ If V3 achieves 80%+ constraint effectiveness + <10 edge types → Ready for Phase 2
- ⚠️ If still high edge-type noise → Investigate alternative approaches before Phase 2

**Next**: Evaluate V3 overnight results to determine production readiness

---

## Phase 2 — Retrieval Pipeline with Coach (Weeks 4–5)
**Agents**
- Coach (front-door): greets, clarifies, maps the turn to LO/community; decides continue/branch/switch session
- Retrieval Agent: BM25 + embeddings + BFS (centered on LO)
- Pack Constructor: tight, cited “teaching pack”
- Tutor: teaches from the pack

**Flow**
1. Coach chats briefly → identifies target LO/community → decides continue/branch/switch
2. (If switch) close current Episode; open new one
3. Retrieval pulls LO, 1–2 prereqs, example, 3 practices
4. Constructor builds cited pack → Tutor replies

**You get:** Reliable, focused Q&A with citations and small prompts.
Example: “What is vertex form?” → LO summary + 1 worked example + 2 exercises, cited to “Ch9 §4”.

---

## Phase 3 — Personalization Layer (Weeks 6–8)
**Agents**
- Event Logger → turns/attempts → Episode
- Grader → scores attempts (separate model/rubric)
- Mastery Estimator → updates MASTERED_BY, CONFUSES_WITH with decay/time windows
- Overlay Writer → writes per-student temporal edges (keeps history)

**Flow**
1. Log attempts → Grader scores
2. Mastery Estimator updates overlay edges with timestamps & evidence

**You get:** A per-student overlay KG atop the shared map.
Example: `STRUGGLES_WITH(Student A ↔ Factoring, evidence=2)`; `MASTERED_BY(CTS → Student A, p=0.78)`.

---

## Phase 4 — Adaptive Learning (Weeks 9–10)
**Agents**
- Lesson Planner → blends global KG + student overlay into a plan
- Retrieval Agent → biased by overlay (recency/mastery/confusions)
- Tutor / Grader → teach, then score; updates flow back to overlay

**Flow**
1. Planner picks target LO + missing prereqs
2. Retrieval builds a personalized pack
3. Tutor teaches → Grader scores → Mastery Estimator updates overlay

**You get:** An adaptive loop that changes content/pacing per student.
Example: For Student A on “quadratics,” the pack auto-includes a quick factoring refresher.

---

## Phase 5 — Scale & Optimization (Week 11)
**Agents**
- Indexer/Embedder → precompute embeddings
- Cache/Shard Manager → hot communities, partitioning
- Observability → tokens, p95 latency, hit rates

**Flow**
1. Precompute & cache frequent lookups
2. Tune search/rerank; cap BFS hops; watch latency

**You get:** Sub-2s responses at book scale with stable costs.

---

## Phase 6 — GraphRAG Depth Layer (Week 12)
**Agents**
- GraphRAG Summarizer (offline) → C0/C1 community summaries + claims
- Importer → store summaries/claims in Zep (as community nodes/attachments)
- Answer Composer → map-reduce from summaries, then personalize with overlay

**Flow**
1. Build C0/C1 overviews per unit/topic (with references)
2. Import to Zep → for broad prompts, compose a global answer → blend student needs

**You get:** Global sensemaking (great unit overviews) + live personalization.
Example: “1-hour review of Quadratics” → structured outline with key concepts/pitfalls + targeted factoring fixes for Student A.
