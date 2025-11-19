# Adaptive Learning System - Presentation Script

## Slide 1: Coach, Tutor, and Overarching System

**Title: Adaptive Learning System Architecture**

**Coach Agent**
- Front-door interface for student interactions
- Classifies user intent (tutoring, practice, definition, comparison)
- Manages session state and conversation flow
- Builds structured retrieval requests
- Asks clarifying questions when needed

**Tutor Agent**
- Generates personalized learning responses
- Uses retrieved content to construct explanations
- Adapts to student's knowledge level
- Provides examples, practice problems, and step-by-step guidance

**System Flow**
1. Student asks question → Coach classifies intent
2. Coach builds retrieval request → Retriever finds relevant content
3. Retriever returns LOs + content → Tutor generates response
4. Tutor delivers personalized explanation → Student learns

**Key Principle**: Just-enough context (3-7 bullet facts) with full source attribution

---

## Slide 2: Knowledge Graph and GraphRAG

**Title: Knowledge Graph Foundation**

**Knowledge Graph Structure**
- 138 Learning Objectives (LOs) covering calculus topics
- 270 Content items (concepts, examples, try_it exercises)
- Prerequisite relationships: LO → LO (290 edges)
- Content relationships: LO → Content (428 edges)

**GraphRAG Approach**
- Uses graph structure for retrieval and reasoning
- Enriches documents with graph context (prerequisites, related content)
- Enables multi-hop traversal for comprehensive coverage
- Current: Graph metadata enrichment (not full GraphRAG yet)

**Why Graph Structure Matters**
- Prerequisites show learning dependencies
- Related content provides examples and practice
- Graph expansion improves retrieval relevance
- Foundation for adaptive learning paths

**Data Sources**: MatchGPT KG data (CSV format: lo_index.csv, content_items.csv, edges_prereqs.csv, edges_content.csv)

---

## Slide 3: Retriever Agent and Current Focus

**Title: Retrieval Agent - Current Focus**

**Current Implementation**
- **Embedding-based retrieval**: OpenAI text-embedding-3-small with FAISS
- Semantic search over LO titles and content text
- Optional graph expansion (1-hop traversal)
- Fast, scalable, cost-effective

**Retrieval Pipeline**
1. Build FAISS indexes from document embeddings
2. Encode query → search vector similarity
3. Return top-k LOs and top-k content items
4. Optionally expand with prerequisites and related content

**Evaluation Status**
- Comparing embedding retrieval vs. other methods
- 25 test queries evaluated
- Manual relevance rating in progress
- Results: ~394 evaluation rows exported

**Current Focus**
- Refining embedding-based retrieval (removed BM25)
- Testing LLM-as-retriever baselines (GPT-5 full-context)
- Evaluating retrieval quality and relevance
- Preparing for integration with Coach/Tutor agents

**Next Steps**: Manual evaluation → Aggregate metrics → Production integration

---

## Slide 4: Personalized LO Selection with Knowledge Graph

**Title: From Retrieval to Personalized Teaching**

**Step 1: Embedding Retrieval**
- Query: "What is a derivative?"
- Embedding search returns top 5-10 LOs (e.g., "The Tangent Problem", "Average Rate of Change", "Limits")
- Initial candidate set: ~10 LOs, ~10 content items

**Step 2: LLM + Knowledge Graph Personalization**
- LLM analyzes student context (knowledge level, progress, learning goals)
- Uses KG structure to personalize:
  - Checks prerequisites: Are foundational LOs already learned?
  - Traverses relationships: Which content supports each LO?
  - Identifies learning path: What order makes sense?
  - Filters by difficulty: Matches student's level

**Step 3: Curated Subset for Tutor**
- LLM selects 3-5 most relevant LOs from initial set
- Prioritizes based on:
  - Prerequisite satisfaction (can student learn this now?)
  - Learning sequence (what comes next logically?)
  - Content availability (are examples/practice available?)
  - Student readiness (not too advanced, not too basic)

**Example Flow**
1. Embeddings return: [LO_1893, LO_1231, LO_1230, LO_1897, LO_1872]
2. KG shows: LO_1893 requires LO_1231 as prerequisite
3. LLM checks: Student hasn't learned LO_1231 yet
4. LLM selects: [LO_1231 (foundation), LO_1893 (target), LO_1230 (related)]
5. Tutor receives: Personalized sequence with supporting content

**Key Insight**: Knowledge graph enables intelligent filtering beyond semantic similarity

## G10: Real Queries, Pipeline Outputs (text + visuals), Verdict

| User Message (verbatim) | Modality Hint | Embeddings Retrieval (top hits) | GPT‑5 Full‑Context Retrieval (top hits) | Which Output Lands Better? |
| --- | --- | --- | --- | --- |
| What is a derivative? | Text-first + simple tangent sketch | LOs: The Tangent Problem and Differential Calculus; Finding the Average Rate of Change. Content: Tangent Problem example + try‑it (secant line slopes). | LOs: The Tangent Problem and Differential Calculus; Finding the Average Rate of Change. Content: Same Tangent Problem examples/try‑its. | Tie (embeddings edge for speed/cost). Both return the right concept + visuals; GPT‑5 doesn’t add extra value here. |
| How do I solve a quadratic equation? | Text answer + worksheets/graphs | LOs: Several trig “quadratic form” items (off‑topic drift). Content: Trig examples/try‑its. | LOs: Polynomials; Algebraic Functions; Transformations. Content: Quadratic zero‑finding example + practice. | GPT‑5 wins (quality). It corrects the “quadratic form” ambiguity and picks true polynomial items. |
| Explain the chain rule | Text explanation + step visual | LOs: Composition of Functions; Evaluating Composite Functions (confuses “composition” with derivative chain rule). | LOs: Composition of Functions; Evaluating Composite Functions (same confusion). | Neither fully nails it. Needs better labeled chain‑rule assets; both pipelines drift to function composition. |
| How do I find limits? | Text walkthrough + limit graphs | LOs: Evaluating Limits with the Limit Laws; Infinite Limits; One‑Sided Limits. Content: Limit‑law examples + practice with graphs. | LOs: Evaluating Limit Laws; Additional Techniques; One‑Sided/Polynomial Limits. Content: Squeeze/limit practice. | Embeddings win (efficiency). Similar quality; embeddings are faster/cheaper and pull the right visuals. |
| What is continuity? | Definition + piecewise graph | LOs: Continuity at a Point; Continuity over an Interval; Types of Discontinuities. Content: Continuity concept + piecewise practice. | LOs: Continuity at a Point; Continuity over an Interval; Types of Discontinuities; IVT. Content: Continuity concept + IVT practice. | Embeddings edge. Both good; embeddings surface the expected text + visuals with lower latency. |
| Give me practice problems on integrals | Problem set + simple area visuals | LOs: Area Problem; then some limits/continuity drift. Content: Area try‑it/example appear but not consistently at the top. | LOs: Area Problem; Polynomials; Trig/Exp/Log (some drift). Content: Area try‑it + example right at the top; some unrelated limit drills. | Slight GPT‑5 edge (practice focus). It puts the area try‑it/example on top; both have some noise. |
| How do I use the product rule? | Step‑by‑step derivative + annotated steps | LOs: Sum‑to‑Product/Product‑to‑Sum identities (algebraic drift); exp/log examples. | LOs: Tangent Problem; Combining Functions; Function Notation (not derivative product rule). | Neither. Both misinterpret “product rule” (derivative) as algebraic identities. Needs better labeling/training data. |
| What is the difference between a limit and a derivative? | Contrast text + simple visuals | LOs: Intuitive Definition of a Limit; Existence of a Limit; Limit Laws. Content: Limit tables/graphs. | LOs: Tangent Problem (derivative lens); Intuitive Definition of a Limit; Rates of Change. Content: Tangent Problem examples. | GPT‑5 edge (framing). It brings in derivative‑centric tangent items, making the contrast clearer. |
| Explain the tangent problem | Text intuition + secant→tangent visuals | LOs: Tangent Problem and Differential Calculus; Average Rate of Change. Content: Classic secant‑to‑tangent examples/try‑its. | LOs: Tangent Problem; Intuitive Definition of a Limit; Average Rate of Change. Content: Same examples/try‑its. | Tie. Both surface the canonical text + visuals. Embeddings are cheaper. |
| How do I find the area under a curve? | Area narrative + Riemann rectangles | LOs: Average Rate of Change; Area Problem; Parameterizing a Curve (some drift). Content: Area example + try‑it present. | LOs: Area Problem; Intuitive Limit; Limit Laws. Content: Area example + try‑it at the very top. | GPT‑5 edge (focus). It places area items first (text + visuals). Embeddings include extras (parametric/geometry). |

### Why this matters for multimodality
- Embeddings reliably pull an LO with its linked visuals (examples, try‑its, figures) via metadata, giving the Tutor a clean text+image bundle fast.
- GPT‑5 can “judge” which single visual is most instructive (even when tags are weak) and can curate a mini‑lesson across sources—but at higher cost/latency.

### Practical guidance
- Default to embeddings → fast, cheap candidate pool (text + visuals) with strong coverage.
- Use GPT‑5 selectively when queries are ambiguous (e.g., “quadratic form”), you need bespoke curation, or you must pick the best visual among many similar options.

### Bottom line
Embeddings are the right default. They’re fast, inexpensive, and consistently return the right text + visuals; let the LLM personalize from that pool when you truly need deep synthesis or nuanced selection.