# Adaptive Learning Platform — Software Specification

## 1. Overview
This project builds a local-first adaptive learning system over a static math knowledge graph (OpenStax corpus), with support for:
- Retrieval-augmented tutoring
- Per-student personalization
- D3-based visualization of the knowledge graph

The system operates entirely locally for now, with file-based or in-memory storage for the knowledge graph and embeddings. Later, it can be extended to cloud and Neo4j/Zep if needed.

---

## 2. Core Scenarios

### Scenario 1 — First-time User Session
**Actors:** Student, Coach Agent, Retrieval Agent, Pack Constructor, Tutor  
**Trigger:** Student asks a topic-level question (“What is vertex form?”).  
**Flow:**
1. **Coach** identifies the relevant Learning Objective (LO) from the static KG.
2. **Retrieval Agent** finds:
   - Target LO node
   - 1–2 prerequisite LOs
   - 1 worked example
   - 2–3 practice exercises
3. **Pack Constructor** builds a small teaching pack (with citations).
4. **Tutor** presents explanation and exercises.
5. **End state:** Session log file is created for personalization later.

---

### Scenario 2 — Continuing an Active Session
**Actors:** Student, Coach Agent, Retrieval Agent, Tutor  
**Trigger:** Student continues a topic already in progress.  
**Flow:**
1. **Coach** retrieves session context from local store (JSON file or SQLite).
2. System resumes from previous LO without reloading entire context.
3. **Retrieval Agent** refreshes examples/exercises if needed.
4. **Tutor** continues teaching.

---

### Scenario 3 — Session Switch
**Actors:** Student, Coach Agent  
**Trigger:** Student decides to change topic (“Let’s switch to factoring”).  
**Flow:**
1. **Coach** closes the current session (marks in local log).
2. Starts a new session with target LO = “Factoring”.
3. New retrieval → teaching pack → tutoring loop.

---

### Scenario 4 — Exercise Grading & Feedback
**Actors:** Student, Grader, Mastery Estimator, Overlay Writer  
**Trigger:** Student submits solution to an exercise.  
**Flow:**
1. **Grader** checks answer (exact match for now, rubric later).
2. **Mastery Estimator** updates local personalization overlay:
   - MASTERED_BY edges (p-value up)
   - STRUGGLES_WITH edges (if incorrect)
3. **Overlay Writer** stores updated overlay in local JSON.

---

### Scenario 5 — Personalized Review Session
**Actors:** Student, Lesson Planner, Retrieval Agent, Tutor  
**Trigger:** Student requests a review (“Review quadratics”).  
**Flow:**
1. **Lesson Planner** reads:
   - Global KG
   - Student overlay (mastery/confusion edges)
2. Selects target LOs with weak mastery/confusion.
3. **Retrieval Agent** builds personalized pack biased towards weak areas.
4. **Tutor** delivers review.

---

### Scenario 6 — Broad Unit Overview (GraphRAG Offline Summaries)
**Actors:** Student, Answer Composer, Retrieval Agent, Tutor  
**Trigger:** Student requests high-level summary (“Give me a 1-hour review of Quadratics”).  
**Flow:**
1. **Answer Composer** retrieves offline GraphRAG C0/C1 summaries from local store.
2. Merges global summary with personalized weak points.
3. **Retrieval Agent** fetches examples/exercises aligned to outline.
4. **Tutor** presents structured review.

---

## 3. Non-Functional Requirements
- **Local-first**: All data stored in files or lightweight DB (e.g., SQLite).
- **Portable**: No cloud dependencies in initial version.
- **Visualizable**: KG exportable to JSON for D3 rendering.
- **Extensible**: Architecture should allow later swap to Zep/Neo4j/cloud hosting.

---

## 4. Data Storage
- **Static KG**: `kg_nodes.json`, `kg_edges.json`
- **Embedding Index**: `embeddings.json`
- **Session Logs**: One JSON per session
- **Personalization Overlay**: `overlay_{student_id}.json`
- **GraphRAG Summaries**: `summaries.json`

