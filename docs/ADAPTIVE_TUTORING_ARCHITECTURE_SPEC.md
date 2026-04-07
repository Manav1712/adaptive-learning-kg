# Adaptive Tutoring System — Implementation Architecture Specification

---

## SECTION 1 — Executive system summary

**What the system does:** A **local-first** Python runtime (`CoachAgent`) orchestrates an LLM-driven **coach** that routes students into **tutoring** or **FAQ** bot sessions. Tutoring combines (a) **retrieval over a CSV-backed knowledge graph** in `demo/`, (b) a **session plan** of learning objectives (LOs) from `TutoringPlanner` in `src/workflow_demo/planner.py` via `retrieve_candidates`, and (c) a **pedagogy layer** on each tutor student turn: **misconception diagnosis** → **teaching move candidates** → **deterministic policy scoring** → **pedagogical retrieval policy** that decides reuse vs refresh of a **teaching pack**, then **tutor LLM** JSON responses with optional **math example guard**.

**Major layers:**

1. **Data / KG artifacts** — CSVs (`lo_index`, `content_items`, `edges_prereqs`, `edges_content`) loaded by [`load_demo_frames`](src/workflow_demo/data_loader.py).
2. **Retrieval** — [`TeachingPackRetriever`](src/workflow_demo/retriever.py): hybrid dense + BM25 (RRF), optional LLM rerank, graph expansion into teaching packs; parallel `retrieve_candidates` for planner LO picking.
3. **Orchestration** — [`CoachAgent`](src/workflow_demo/coach_agent.py) + [`CoachRouter`](src/workflow_demo/coach_router.py) + [`CoachLLMClient`](src/workflow_demo/coach_llm_client.py); [`BotSessionManager`](src/workflow_demo/bot_sessions.py) for tutor/FAQ lifecycle.
4. **Pedagogy** — [`MisconceptionDiagnoser`](src/workflow_demo/pedagogy/diagnosis.py), [`TeachingMoveGenerator`](src/workflow_demo/pedagogy/teaching_moves.py), [`PolicyScorer`](src/workflow_demo/pedagogy/policy.py), [`PedagogicalRetrievalPolicy`](src/workflow_demo/pedagogy/retrieval_policy.py), [`compute_turn_progression_signals`](src/workflow_demo/pedagogy/turn_progression.py).
5. **Integration** — [`web_api.py`](src/workflow_demo/web_api.py) FastAPI: REST + optional SSE; [`runtime_factory.build_coach_runtime`](src/workflow_demo/runtime_factory.py).

**Main runtime path (happy path):** Student → coach LLM directive → `call_tutoring_planner` → `TutoringPlanner.create_plan` (`retrieve_candidates` + heuristic or optional planner LLM) → `start_tutor` → `BotSessionManager.begin` → `create_handoff_context` + `ensure_tutor_learner_context` → `tutor_bot` opening message. Subsequent turns: diagnosis + policy + retrieval → `tutor_bot` → optional math guard.

**Design principles visible in code:** (1) **Separation of planner LO selection** (`retrieve_candidates`, simple plan JSON) from **rich pack assembly** (`retrieve_plan`, used in pedagogy refresh paths). (2) **Deterministic policy** with explicit tie-breaks and progression gates. (3) **Feature flags** for LLM-heavy components (planner LLM, diagnosis LLM, math guard). (4) **JSON-only** tutor/FAQ contracts with retry.

**Difference from generic RAG tutor:** Explicit **target_lo vs instruction_lo**, **move-typed conditioning** (`tutor_instruction_directives`), **retrieval policy** with logical actions (`reuse_pack` / `augment_pack` / `refresh_pack`) mapped to **execution modes** (`no_io` / `constrained_refresh` / `full_refresh`), **turn progression** to suppress repeated diagnostics, and **in-process pedagogy events** for observability.

---

## SECTION 2 — Repository and subsystem map

| Subsystem | Purpose | Key modules | Upstream | Downstream |
|-----------|---------|-------------|----------|------------|
| **Demo KG + loader** | Static curriculum graph | [`data_loader.py`](src/workflow_demo/data_loader.py), [`demo/*.csv`](demo) | CSV files | `TeachingPackRetriever`, planners |
| **Embeddings / retrieval** | Hybrid search, packs, CLIP | [`retriever.py`](src/workflow_demo/retriever.py), [`clip_embeddings.py`](src/workflow_demo/clip_embeddings.py) | OpenAI embeddings API, optional CLIP | Planners, `PedagogicalRetrievalPolicy` |
| **Coach orchestration** | Routing, state | [`coach_agent.py`](src/workflow_demo/coach_agent.py), [`coach_router.py`](src/workflow_demo/coach_router.py), [`coach_llm_client.py`](src/workflow_demo/coach_llm_client.py) | OpenAI | Planners, `BotSessionManager` |
| **Planners** | Tutoring + FAQ plan JSON | [`planner.py`](src/workflow_demo/planner.py) | Retriever, optional planner LLM | Session params in handoff |
| **Bot sessions** | Tutor/FAQ turns, pedagogy pipeline | [`bot_sessions.py`](src/workflow_demo/bot_sessions.py) | Diagnoser, policy, retrieval | `tutor_bot` / `faq_bot` |
| **Pedagogy core** | Diagnosis, moves, policy, retrieval | [`pedagogy/`](src/workflow_demo/pedagogy/) | Learner state, retriever | `pedagogy_context`, events |
| **Tutor / FAQ LLM** | Student-facing JSON bots | [`tutor.py`](src/workflow_demo/tutor.py) | Handoff payload | Student message |
| **Session + profile** | History, `lo_mastery` | [`session_memory.py`](src/workflow_demo/session_memory.py), [`demo_profiles.py`](src/workflow_demo/demo_profiles.py) | Disk (optional) | Coach, mastery updates |
| **Learner state (pedagogy)** | In-session attempts, misconceptions | [`learner_state.py`](src/workflow_demo/pedagogy/learner_state.py), [`state_store.py`](src/workflow_demo/pedagogy/state_store.py) | Profile seed | Policy, snapshot |
| **Runtime events** | Structured telemetry | [`runtime_events.py`](src/workflow_demo/runtime_events.py), [`pedagogy/events.py`](src/workflow_demo/pedagogy/events.py) | Coach emit | Web sink, logs |
| **Web API** | HTTP bridge | [`web_api.py`](src/workflow_demo/web_api.py) | FastAPI | Coach |
| **Eval harness** | Scenario pedagogy checks | [`pedagogy_eval/harness.py`](src/workflow_demo/pedagogy_eval/harness.py) | Mocks/patches | Reports |
| **Offline experiments** | LLM edge discovery (partial repo) | [`src/experiments_manual/`](src/experiments_manual) | Config YAML | Processed CSV edges (external to `demo/` unless copied) |

**Navigation tip:** Start at [`runtime_factory.py`](src/workflow_demo/runtime_factory.py) → [`coach_agent.py`](src/workflow_demo/coach_agent.py) → [`bot_sessions.py`](src/workflow_demo/bot_sessions.py) → [`retriever.py`](src/workflow_demo/retriever.py) + [`pedagogy/retrieval_policy.py`](src/workflow_demo/pedagogy/retrieval_policy.py).

---

## SECTION 3 — Data ingestion and knowledge graph pipeline

**Source data types (runtime demo):** Four CSVs under [`demo/`](demo): `lo_index.csv`, `content_items.csv`, `edges_prereqs.csv`, `edges_content.csv` (see [`load_demo_frames`](src/workflow_demo/data_loader.py)).

**Schema (as used by loader):**

- **LOs:** DataFrame `los` + `lo_lookup: dict[lo_id → row dict]` including fields consumed by retrieval such as `learning_objective`, book/unit/chapter, pedagogical hints via `_get_how_to_teach` / `_get_why_to_teach` in retriever (from LO rows).
- **Content:** `content` DataFrame; items joined to LOs for snippets and `content_type`.
- **Prerequisites:** `edges_prereqs` → `prereq_in_map` (adjacency for prerequisite expansion).
- **Content–LO links:** `edges_content` → `content_ids_map`.

**How content becomes structured:** The **online** tutor does not run ingestion; it reads **pre-built** CSVs. **Offline** construction is described in [`README.md`](../README.md) and [`architecture/manual_ingestion.md`](../architecture/manual_ingestion.md) and implemented in part under [`src/experiments_manual/`](../src/experiments_manual) (`discover_content_links.py`, `discover_prereqs.py`, `evaluate_heuristic.py`, `evaluate_llm.py`, `config.yaml`). **Gap:** Root README references `prepare_lo_view.py` and `evaluate_outputs.py` — **these files are not present** in this repository snapshot; treat README pipeline steps as **partially aspirational** unless those scripts exist in your branch.

**Validation:** [`load_demo_frames`](src/workflow_demo/data_loader.py) builds maps; structural validation for manual experiments is **not centralized** in `workflow_demo` (experiments scripts bear their own checks).

**Storage format:** Pandas in memory; optional **embedding cache** under `demo/.embedding_cache/` (`.npy` files, hash-keyed) in [`TeachingPackRetriever`](src/workflow_demo/retriever.py).

**Ambiguities:** Notes under `architecture/` may describe **ideal** or pipeline-specific schemas; the **running tutor** uses the **demo CSV layout** consumed by `load_demo_frames`.

---

## SECTION 4 — Retrieval / RAG architecture

### 4.1 Components

- **Dense embeddings:** OpenAI `text-embedding-3-large` or `-small` via [`EmbeddingBackend`](src/workflow_demo/retriever.py) (requires `OPENAI_API_KEY`).
- **Lexical:** BM25 via `rank_bm25` when installed; if import fails, hybrid degrades to semantic-only (logged).
- **Fusion:** [`_hybrid_fusion`](src/workflow_demo/retriever.py) — **RRF** over semantic + BM25 hit lists (separate pipelines for LO vs content indices).
- **Reranking:** Optional OpenAI chat completion (`rerank_model`, default `gpt-5.4-mini` in retriever `__init__`) in `_rerank_hits`; **disabled** in pedagogical `retrieve_plan` calls from [`PedagogicalRetrievalPolicy.run`](src/workflow_demo/pedagogy/retrieval_policy.py) (`enable_rerank=False`).
- **Graph-aware behavior:** [`_build_teaching_pack`](src/workflow_demo/retriever.py) pulls prerequisite rows from `kg.prereq_in_map`; content rows typed as `concept` / `example` / `try_it` / etc. feed **examples** vs **practice** slots; **images** from `_search_images` when metadata exists.

### 4.2 Two retrieval entry points

| Method | Used by | Output role |
|--------|---------|-------------|
| `retrieve_candidates` | `TutoringPlanner` | `RetrievalCandidate` list → **coach plan** (LO titles, how/why, book metadata) |
| `retrieve_plan` | `PedagogicalRetrievalPolicy` (augment/refresh), image preprocessor | [`SessionPlan`](src/workflow_demo/models.py) with `TeachingPack` + internal `PlanStep` lists |

**Important inconsistency:** The **tutor handoff `current_plan`** comes from the **planner** (list of LO dicts: `lo_id`, `title`, `proficiency`, `how_to_teach`, `why_to_teach`, `notes`, `is_primary`). The `retrieve_plan` path builds a **different** `current_plan` as `List[PlanStep]` inside `SessionPlan` — that structure is **not** what the coach passes to the tutor in the main flow; it is used when assembling **teaching_pack** inside retrieval policy refresh.

### 4.3 Teaching pack construction

[`_build_teaching_pack`](src/workflow_demo/retriever.py) builds:

- `key_points` (synthetic strings + related LOs),
- `examples` / `practice` from content hits by `content_type`,
- `prerequisites` from graph prereq IDs,
- `citations`, `images` (image search).

### 4.4 `retrieval_intent` vs `retrieval_action` vs `retrieval_execution_mode`

- **`PedagogicalRetrievalIntent`** ([`constants.py`](src/workflow_demo/pedagogy/constants.py)): Step-1 **pedagogical** intent — `teach_current_concept`, `repair_prerequisite`, `retrieve_worked_example`, `retrieve_misconception_support`. Set by [`decide_pedagogical_retrieval_intent`](src/workflow_demo/pedagogy/retrieval_policy.py) from **move type + diagnosis**.
- **`RetrievalPolicyAction`** (stored on context as string **`retrieval_action`**): Logical decision — `reuse_pack`, `augment_pack`, `refresh_pack`. From [`decide_retrieval_action`](src/workflow_demo/pedagogy/retrieval_policy.py) using **material triggers** `t1`–`t5` (session target change, instruction unsupported by pack, missing artifact for intent, diagnosis fingerprint change, empty/invalid pack).
- **`RetrievalExecutionMode`:** **Physical** mapping ([`map_action_to_execution_mode`](src/workflow_demo/pedagogy/retrieval_policy.py)):
  - `reuse_pack` → `no_io`
  - `augment_pack` → `constrained_refresh` (**implemented as full `retrieve_plan`**, not incremental merge — comment in code: v1 prefers `retrieve_plan` over weak merge)
  - `refresh_pack` → `full_refresh`
- **`legacy_retrieval_intent`:** From move’s [`RetrievalIntent`](src/workflow_demo/pedagogy/constants.py) enum via `_map_move_to_intent`; carried in policy output but tutor payload emphasizes `PedagogicalRetrievalIntent` strings on `pedagogy_context`.

**Approximation:** “Augment” does not merge retrieved rows into the old pack in the success path; it **replaces** with a fresh `retrieve_plan` teaching pack (constrained `top_k` in augment path).

### 4.5 Candidate retrieval vs plan retrieval

- **Planner:** Only needs ranked LOs → `retrieve_candidates` (text + optional CLIP image path).
- **Pedagogy refresh:** Needs full pack → `retrieve_plan` with query composed from student text, LO strings, and session params.

### 4.6 Errors / gaps

- If `retrieve_plan` raises, policy returns **`fallback_used=True`**, keeps prior pack when possible, appends error strings ([`PedagogicalRetrievalOutput.errors`](src/workflow_demo/pedagogy/retrieval_policy.py)).
- **Opening tutor message:** `retrieve_plan` is **not** called in `TutoringPlanner` or `begin()`; **`teaching_pack` may be absent or empty** until the first **student** turn runs pedagogy (triggers `t5` → refresh). *Implemented gap worth flagging for product accuracy.*

---

## SECTION 5 — Planning and agent orchestration

### 5.1 Roles

- **Coach (`CoachLLMClient`):** JSON directive: `action` ∈ `none | call_tutoring_planner | call_faq_planner | start_tutor | start_faq | show_proficiency` ([`COACH_SYSTEM_PROMPT`](src/workflow_demo/coach_llm_client.py)).
- **Tutoring planner (`TutoringPlanner`):** `create_plan` → `{ status, plan, message }`; uses `retrieve_candidates` + proficiency map + optional LLM (`WORKFLOW_DEMO_ENABLE_PLANNER_LLM`) or **heuristic** plan ([`_build_heuristic_plan`](src/workflow_demo/planner.py)).
- **FAQ planner (`FAQPlanner`):** Maps to canned `FAQ_TOPICS` strings or LLM-assisted topic pick when enabled.
- **Tutor bot (`tutor_bot`):** JSON tutor responses from [`TUTOR_SYSTEM_PROMPT`](src/workflow_demo/tutor.py).
- **FAQ bot (`faq_bot`):** JSON from [`FAQ_SYSTEM_PROMPT`](src/workflow_demo/tutor.py).
- **Session manager (`BotSessionManager`):** Owns handoff, conversation history, pedagogy pipeline on tutor turns.

### 5.2 Routing

[`CoachRouter.handle_turn`](src/workflow_demo/coach_router.py): pre-classification (FAQ keywords, session history regex, syllabus escalation, fast-track topic after clarification) → loop (max 5) calling `_get_directive` → planner or `begin()`.

**Plan conflict:** [`_detect_plan_conflicts`](src/workflow_demo/coach_router.py) only on `REPLANNABLE_KEYS` = `{mode, topic}` — **subject** excluded by comment.

### 5.3 Session start and handoff

[`create_handoff_context`](src/workflow_demo/session_memory.py): `handoff_metadata`, `session_params`, `conversation_summary`, `recent_sessions`, `student_state`, `image`.

Tutor: [`ensure_tutor_learner_context`](src/workflow_demo/coach_agent.py) seeds `pedagogy_context` as JSON from [`PedagogicalContext`](src/workflow_demo/pedagogy/models.py) (learner state, `target_lo`, `instruction_lo`, `retrieval_session` snapshot).

---

## SECTION 6 — Pedagogy architecture

### 6.1 Learner state engine ([`LearnerStateEngine`](src/workflow_demo/pedagogy/learner_state.py))

- **Purpose:** Session-local state: attempts, misconception history, hints; snapshot for API.
- **Inputs:** `student_profile` seed (`lo_mastery`, `confidence_seed`), per-turn updates.
- **Outputs:** `LearnerState` model; events `pedagogy_learner_state_initialized` / `_updated`.
- **Storage:** [`LearnerStateStore`](src/workflow_demo/pedagogy/state_store.py) — **in-memory only** (not persisted across process restart).
- **Limitation:** Explicitly **no BKT** (see docstring).

### 6.2 Misconception diagnosis ([`MisconceptionDiagnoser`](src/workflow_demo/pedagogy/diagnosis.py))

- **Heuristic first** ([`HeuristicDiagnoser`](src/workflow_demo/pedagogy/diagnosis_rules.py)); if confidence ≥ `HEURISTIC_ACCEPT_CONFIDENCE` (0.55), return.
- Else **optional LLM** ([`LLMDiagnosisAdapter`](src/workflow_demo/pedagogy/diagnosis_llm.py)) if `WORKFLOW_DEMO_ENABLE_DIAGNOSIS_LLM` set.
- Else return heuristic (possibly low confidence).

**`MisconceptionDiagnosis` fields:** `target_lo`, `suspected_misconception`, `confidence`, `rationale`, `prerequisite_gap_los`, `evidence_quotes` ([`models.py`](src/workflow_demo/pedagogy/models.py)).

### 6.3 Teaching move generation ([`TeachingMoveGenerator`](src/workflow_demo/pedagogy/teaching_moves.py))

- Produces **2–4** candidates among: `diagnostic_question`, `prereq_remediation`, `graduated_hint`, `worked_example` (plus filler from fallback order).
- **Note:** Enum includes `explain_concept` but **generator does not emit it** in `generate_candidates` — tutor prompt still lists “other” move types; **policy may never select `explain_concept` from this generator**.

### 6.4 Policy scorer ([`PolicyScorer`](src/workflow_demo/pedagogy/policy.py))

- Scores each candidate with weighted features (expected gain, priority, leakage risk) + situation flags (low confidence, prereq gap, stuck, etc.).
- **Progression:** [`TurnProgressionSignals`](src/workflow_demo/pedagogy/turn_progression.py) — `suppress_repeat_diagnostic` applies **penalty** `_REPEAT_DIAGNOSTIC_SCORE_PENALTY` to `diagnostic_question` when gate fires; example-request boosts `worked_example`.
- **Output:** [`PolicyDecision`](src/workflow_demo/pedagogy/models.py) with `selected_move`, `scores`, `decision_reason`.

### 6.5 `target_lo` vs `instruction_lo`

- **`session_target_lo` / `target_lo`:** Stable session goal — from prior `pedagogy_context.target_lo` or plan focus ([`bot_sessions._run_misconception_diagnosis`](src/workflow_demo/bot_sessions.py)).
- **`instruction_lo`:** Per-turn focus from [`derive_instruction_lo`](src/workflow_demo/pedagogy/instruction_lo.py): prereq gap first, else non-unknown `diagnosis.target_lo`, else `session_target_lo`.

### 6.6 Retrieval policy (see Section 4)

Triggers `t1`–`t5`; fingerprint via [`diagnosis_fingerprint`](src/workflow_demo/pedagogy/retrieval_policy.py).

### 6.7 Tutor conditioning

[`tutor_instruction_directives`](src/workflow_demo/tutor.py) — six keys: `session_target_lo`, `instruction_lo`, `selected_move_type`, `retrieval_intent`, `retrieval_action`, `policy_reason`. Dual-written as `tutor_directives` on `pedagogy_context`. **`retrieval_execution_mode` is NOT in directives** (tutor system prompt: execution mode on `pedagogy_context` only).

### 6.8 Turn progression / repeated-check suppression

[`compute_turn_progression_signals`](src/workflow_demo/pedagogy/turn_progression.py): `suppress_repeat_diagnostic` when prior move was `diagnostic_question` and student shows advance intent, adequate check response, substantive math attempt, or example request — and not confusion / short ack.

### 6.9 Math guard

[`maybe_apply_math_example_guard`](src/workflow_demo/pedagogy/math_example_guard.py): only if `WORKFLOW_DEMO_TUTOR_MATH_GUARD` and `selected_move_type == worked_example`; sympy verifies **single** integral or derivative **polynomial** pattern; repair or append note.

### 6.10 Observability

[`PedagogyRuntimeEvent`](src/workflow_demo/pedagogy/events.py) emitted from [`bot_sessions._run_misconception_diagnosis`](src/workflow_demo/bot_sessions.py) (diagnosis, moves, policy, retrieval decided/executed) and math guard callbacks.

---

## SECTION 7 — Student profile and learner-state lifecycle

**Student profile (`SessionMemory.student_profile`):** Default `{"lo_mastery": {}}`. Seeded in [`build_coach_runtime`](src/workflow_demo/runtime_factory.py) from [`get_active_profile()`](src/workflow_demo/demo_profiles.py) (`ACTIVE_PROFILE` 1=strong, 2=weak — **hardcoded** demo switch).

**Learner state vs profile:** Profile is **durable** (when `session_memory_path` set) for `lo_mastery`; [`LearnerState`](src/workflow_demo/pedagogy/models.py) is **session-scoped in-memory** via `LearnerStateStore`.

**Initialization:** [`initialize_from_profile`](src/workflow_demo/pedagogy/learner_state.py) merges `lo_mastery` into `mastery` map; optional `confidence_seed`.

**Updates:** [`record_turn`](src/workflow_demo/pedagogy/learner_state.py) on student messages (non-debug); [`record_misconception`](src/workflow_demo/pedagogy/learner_state.py) after diagnosis.

**Mastery persistence:** On tutor session end, [`_update_lo_mastery`](src/workflow_demo/bot_sessions.py) maps `session_summary.student_understanding` via [`UNDERSTANDING_TO_MASTERY`](src/workflow_demo/bot_sessions.py) onto `student_profile["lo_mastery"][lo_key]`.

**Aspirational:** Long-term personalization beyond `lo_mastery` dict is **not** implemented.

---

## SECTION 8 — Prompting and runtime payload contracts

### 8.1 Coach

[`COACH_SYSTEM_PROMPT`](src/workflow_demo/coach_llm_client.py): strict JSON with `message_to_student`, `action`, `tool_params` (subject, learning_objective, mode, topic, student_request), `conversation_summary`. **Not** `json_object` response_format in code (plain chat).

### 8.2 Tutoring planner LLM

[`TUTORING_PLANNER_PROMPT`](src/workflow_demo/planner.py): JSON `status` + `plan` with `current_plan` / `future_plan` LO objects — **only when** `WORKFLOW_DEMO_ENABLE_PLANNER_LLM` enabled.

### 8.3 Tutor

- **System:** [`TUTOR_SYSTEM_PROMPT`](src/workflow_demo/tutor.py): rules for plan adherence, off-topic detection, move types, JSON schema.
- **User payload:** JSON with:
  - `handoff_context` (includes `session_params` with `current_plan`, `future_plan`, `mode`, `teaching_pack`, …),
  - `tutor_instruction_directives` (six fields),
  - `tutor_directives` (duplicate),
  - `conversation_history` (last 12),
  - `retrieved_images`.
- **API:** `chat.completions.create` with `response_format={"type":"json_object"}`, temperature 0, up to 2 attempts with [`_JSON_ONLY_RETRY_PROMPT`](src/workflow_demo/tutor.py).

**Authoritative fields:** Move-specific behavior: **`tutor_instruction_directives`** override “Teaching Flow” when non-empty (per prompt). **`teaching_pack`** is grounding source when present.

**Output schema (normalized):** [`_normalize_tutor_response`](src/workflow_demo/tutor.py): `message_to_student`, `end_activity`, `silent_end`, `needs_mode_confirmation`, `needs_topic_confirmation`, `requested_mode`, `session_summary` with topics_covered, student_understanding, etc.

### 8.4 FAQ

[`FAQ_SYSTEM_PROMPT`](src/workflow_demo/tutor.py) + payload `handoff_context` + `conversation_history`; same JSON retry pattern.

### 8.5 Malformed output

Returns [`_fallback_tutor_response`](src/workflow_demo/tutor.py) / `_fallback_faq_response` — **does not** end session (`end_activity=False`).

---

## SECTION 9 — Runtime events, snapshots, debug, backend integration

**Runtime events:** [`emit_runtime_event`](src/workflow_demo/runtime_events.py) → dict with `id`, `type`, `phase`, `message`, `created_at`, `metadata`.

**Web API:** [`ThreadSafeEventSink`](src/workflow_demo/web_api.py) batches events; **`POST /api/chat`** returns drained events; **`POST /api/chat/stream`** SSE streams events during turn, final `done` with `pedagogy_snapshot`.

**Pedagogy snapshot:** [`build_tutor_pedagogy_snapshot`](src/workflow_demo/pedagogy/tutor_pedagogy_snapshot.py); exposed via `CoachAgent.get_pedagogy_snapshot_for_api()` → `ChatResponse.pedagogy_snapshot`.

**Debug commands (tutor-only, no LLM):** `!retrieval`, `!policy`, `!diagnosis`, `!state` in [`bot_sessions.py`](src/workflow_demo/bot_sessions.py) — formatted from same snapshot builder.

**Env:** `WORKFLOW_DEMO_API_HOST`, `WORKFLOW_DEMO_API_PORT`, `WORKFLOW_DEMO_CORS_ORIGINS`, `.env` loaded from repo root in `web_api`.

---

## SECTION 10 — Response generation, post-processing, guardrails

1. **Tutor LLM** produces JSON → `coerce_json` → normalize.
2. **Math guard** (optional env): may mutate `message_to_student`; sets `pedagogy_context.last_guard_result`.
3. **No separate “critic”** in production path (`CriticVerdict` exists in models but not wired in `bot_sessions`).

**FAQ isolation:** FAQ path does not run pedagogy pipeline — only `faq_bot`.

---

## SECTION 11 — Evaluation and testing architecture

**Unit / integration tests** under [`tests/workflow_demo/`](../tests/workflow_demo): representative files — `test_tutor.py`, `test_retriever.py`, `test_pedagogy_retrieval_phase5.py`, `test_policy_scorer.py`, `test_turn_progression.py`, `test_math_example_guard.py`, `test_math_guard_integration.py`, `test_tutor_pedagogy_snapshot.py`, `test_acceptance_phase9.py`, `test_integration.py`, `test_pedagogy_eval_harness.py`.

**Pedagogy eval harness:** [`pedagogy_eval/harness.py`](../src/workflow_demo/pedagogy_eval/harness.py) runs scenarios with patched `tutor_bot` / diagnosis, asserts on `pedagogy_context` fields and snapshot.

**Evidence (local parity run):** Using a Python 3.11+ virtualenv with `requirements.txt` installed: `pytest tests/workflow_demo` reported **224 passed**; `python -m src.workflow_demo.pedagogy_eval --verbose` reported **6 passed, 1 skipped** (math-guard scenario skipped when guard disabled). System Python 3.8 in this environment fails import (`dict[str, ...]` typing); use **Python 3.9+** (as required by modern Pydantic usage in `pedagogy/models.py`) for parity.

---

## SECTION 12 — Deployment, configuration, environment assumptions

**Environment variables (non-exhaustive):**

- `OPENAI_API_KEY` — required for embeddings, coach, tutor, optional rerank.
- `WORKFLOW_DEMO_LLM_MODEL` — defaults `gpt-5.4-mini` in coach/tutor init paths.
- `WORKFLOW_DEMO_ENABLE_PLANNER_LLM`, `WORKFLOW_DEMO_ENABLE_DIAGNOSIS_LLM`
- `WORKFLOW_DEMO_TUTOR_MATH_GUARD`
- `WORKFLOW_DEMO_API_HOST`, `WORKFLOW_DEMO_API_PORT`, `WORKFLOW_DEMO_CORS_ORIGINS`

**Python deps:** [`requirements.txt`](../requirements.txt) — openai, pandas, numpy, pydantic, rank-bm25, fastapi, uvicorn, sympy, pytest, etc.

**Artifacts:** [`demo/`](../demo) CSVs + optional `.embedding_cache/`; optional CLIP `image_corpus` under workflow_demo per retriever.

**Startup:** `python -m src.workflow_demo.web_api` (see [`web_api.py`](../src/workflow_demo/web_api.py) `main()`).

---

## SECTION 13 — Known limitations, ambiguities, open questions

**Implemented:** Coach/planner/tutor loop; pedagogy on tutor **student** turns; retrieval policy; progression gates; math guard (narrow); FastAPI bridge.

**Partial / gaps:**

- Initial tutor opening may lack **`teaching_pack`** until first student turn.
- **`augment_pack`** ≈ full `retrieve_plan` replace, not true additive merge (except failed fallback path).
- **README offline pipeline** references missing scripts — verify before relying on ingestion commands.
- **TeachingMoveGenerator** does not cover all enum values in tutor prompt.
- **Learner state** not persisted to disk (unlike optional session memory).

**Model-quality dependence:** Coach directives, tutor wording, optional planner/diagnosis LLMs.

**Open questions (needs product/architect confirmation):**

1. Intended long-term relationship between **planner `current_plan` LO dicts** vs **`retrieve_plan` `PlanStep`** formats.
2. Whether the **opening tutor turn** should call `retrieve_plan` proactively so `teaching_pack` is populated before the first student message.

**Suggested interpretations (inferred; not authoritative):**

- **Dual plan formats:** Treat the planner’s LO dict list as the **authoritative session agenda** for the tutor prompt (`current_plan` / `future_plan`). Treat `SessionPlan.current_plan` as **`TeachingPackRetriever`’s internal retrieval view** used when building packs during `retrieve_plan`, not as something the coach must merge into the handoff unless a future refactor unifies them.
- **Opening `retrieve_plan`:** If product requires grounding on the first tutor message, add a call after plan creation (e.g. in `BotSessionManager.begin` or post-planner) using the same query/subject/mode as pedagogy refresh; until then, the documented behavior (pack fills on first student turn via `t5`) is the implemented contract.
- **`EXPLAIN_CONCEPT`:** Either extend `TeachingMoveGenerator` to emit it when appropriate, or narrow the tutor prompt to the four generator move types to avoid dead prompt branches.

---

## SECTION 14 — Reproduction guidance for an engineering team

1. Read [`runtime_factory.py`](../src/workflow_demo/runtime_factory.py) and [`coach_agent.py`](../src/workflow_demo/coach_agent.py).
2. Validate **`demo/`** CSVs and run retriever tests with stubbed embeddings.
3. Wire **OpenAI** credentials; confirm embedding cache builds.
4. Implement or restore **offline ingestion** if replacing demo data (`experiments_manual` + any notebooks not in repo).
5. Run **`pytest tests/workflow_demo`** and the **pedagogy_eval** module entrypoint.

**Core vs optional:** Core: `workflow_demo` runtime + demo CSVs + OpenAI. Optional: CLIP image index, planner LLM, diagnosis LLM, math guard, persistent session file.

**Risky areas:** Dual plan representations; empty initial teaching pack; in-memory learner store; env-flag matrix.

---

## Appendices

### A. Critical implementation artifacts

- [`coach_agent.py`](../src/workflow_demo/coach_agent.py), [`bot_sessions.py`](../src/workflow_demo/bot_sessions.py), [`retriever.py`](../src/workflow_demo/retriever.py), [`pedagogy/retrieval_policy.py`](../src/workflow_demo/pedagogy/retrieval_policy.py), [`tutor.py`](../src/workflow_demo/tutor.py), [`planner.py`](../src/workflow_demo/planner.py), [`web_api.py`](../src/workflow_demo/web_api.py), [`demo/`](../demo) CSVs.

### B. Highest-risk ambiguities for human clarification

1. Should **`retrieve_plan`** run at **session start** to populate **`teaching_pack`**?
2. Official **offline pipeline** outputs vs current **`demo/`** provenance.
3. Whether **`EXPLAIN_CONCEPT`** should be generated by **`TeachingMoveGenerator`**.

### C. Quick-start reading order

1. [`runtime_factory.py`](../src/workflow_demo/runtime_factory.py) → [`coach_agent.py`](../src/workflow_demo/coach_agent.py)
2. [`coach_router.py`](../src/workflow_demo/coach_router.py) + [`planner.py`](../src/workflow_demo/planner.py)
3. [`bot_sessions.py`](../src/workflow_demo/bot_sessions.py)
4. [`retriever.py`](../src/workflow_demo/retriever.py) + [`pedagogy/retrieval_policy.py`](../src/workflow_demo/pedagogy/retrieval_policy.py)
5. [`tutor.py`](../src/workflow_demo/tutor.py)
6. [`tests/workflow_demo/test_integration.py`](../tests/workflow_demo/test_integration.py) (contract examples)
