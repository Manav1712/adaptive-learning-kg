# User Stories

## 1. Product Context

This system is designed to provide a **single conversational tutoring experience** that can:

- coach students into the right mode (tutoring vs FAQ),
- launch focused tutoring sessions around retrieved learning objectives,
- answer course logistics/syllabus questions with constrained responses,
- preserve continuity across sessions (recent work + proficiency).

The code indicates a practical educational assistant for math topics (especially calculus/trigonometry), optimized for local demo use via CLI while retaining internal multi-role orchestration.

## 2. User Personas

### Persona A: Concept-Focused Student
- **Goals**: Understand difficult math concepts with guided explanations.
- **Frustrations**: Generic explanations, unclear prerequisites, no continuity.
- **Needs**: Fast route to topic-focused tutoring with context-aware teaching.

### Persona B: Practice-Oriented Student
- **Goals**: Move from concept understanding to practice/examples mode.
- **Frustrations**: Static explanations that ignore desired learning mode.
- **Needs**: Mode-aware tutoring and smooth mode switching.

### Persona C: Administrative/FAQ Student
- **Goals**: Quickly get policy/logistics answers (exam schedule, grading, syllabus topics).
- **Frustrations**: Getting tutoring content when they asked operations questions.
- **Needs**: Dedicated FAQ handling with concise, reliable answers.

### Persona D: Returning Student
- **Goals**: Resume learning from prior sessions and see progress.
- **Frustrations**: Repeating context every time; no progress visibility.
- **Needs**: Session memory and proficiency report.

### Persona E (Inferred): Demo Operator / Educator
- **Goals**: Run and demonstrate tutoring flows with minimal setup.
- **Frustrations**: Cold-start delays and opaque runtime behavior.
- **Needs**: Reusable retriever runtime, predictable prompts, runtime events.

## 3. Core User Journeys

### Journey 1: Asking a conceptual question
Student asks for help on a topic (for example, derivatives). The coach identifies tutoring intent, invokes planning/retrieval, and starts tutor mode without extra confirmation. The tutor explains according to the generated `current_plan`.

### Journey 2: Requesting examples or practice
Student asks to switch to examples/practice or starts directly with that intent. The system routes through planner with selected mode and starts tutor session with mode-specific guidance.

### Journey 3: Asking syllabus/administrative questions
Student asks about exams/homework/grading/syllabus topics. The coach routes to FAQ planning; FAQ bot answers from predefined script and asks follow-up.

### Journey 4: Continuing from prior sessions
Student asks “what did we cover last time?” The router intercepts this pattern and replies from stored session memory without requiring LLM routing.

### Journey 5: Image-assisted support
Student includes an image path/URL. System extracts image context, merges it with text query, and can pass image directly into tutor vision call for grounded explanation.

## 4. Detailed User Stories

### Theme A: Onboarding and Session Start

**Story ID**: US-001  
**Title**: Start with coach greeting  
**Persona**: Concept-Focused Student  
**User Story**: As a student, I want an immediate clear greeting so that I know what help is available.  
**Context / Notes**: `COACH_GREETING` returned by `CoachAgent.initial_greeting()`.  
**Acceptance Criteria**:
- On runtime start, assistant presents one greeting describing tutoring + FAQ help.
- Greeting is returned even when no prior memory exists.

**Story ID**: US-002  
**Title**: Ask for missing topic when request is vague  
**Persona**: Concept-Focused Student  
**User Story**: As a student, I want clarifying prompts when I do not name a topic so that tutoring can start correctly.  
**Context / Notes**: Coach prompt instructs clarifying question for generic “start tutoring” intent.  
**Acceptance Criteria**:
- If no usable topic is detected, response asks for specific topic/LO.
- No tutor session is started in this state.

**Story ID**: US-003  
**Title**: Fast-track after topic clarification  
**Persona**: Concept-Focused Student  
**User Story**: As a student, I want short topic follow-ups to start tutoring immediately so that conversation feels efficient.  
**Context / Notes**: `CoachRouter._maybe_fast_track_tutoring_topic`.  
**Acceptance Criteria**:
- If previous assistant message asked for topic and student replies with short topic text, planner is called directly.
- Tutor session starts without another clarification loop when planner returns complete.

### Theme B: Question Answering and Tutoring

**Story ID**: US-004  
**Title**: Route conceptual questions to tutoring planner  
**Persona**: Concept-Focused Student  
**User Story**: As a student, I want conceptual math questions routed to tutoring so that I receive guided teaching instead of policy answers.  
**Context / Notes**: Coach directives trigger `call_tutoring_planner`.  
**Acceptance Criteria**:
- Tutor-planner action is selected for learning intents.
- Planning result is stored and used to start tutor mode.

**Story ID**: US-005  
**Title**: Start tutor session immediately after complete plan  
**Persona**: Concept-Focused Student  
**User Story**: As a student, I want teaching to begin right away so that momentum is not lost.  
**Context / Notes**: Prompt policy in `COACH_SYSTEM_PROMPT` and `TUTOR_SYSTEM_PROMPT`.  
**Acceptance Criteria**:
- After planner status `complete`, coach starts tutor session without asking confirmation.
- Tutor provides opening instructional message.

**Story ID**: US-006  
**Title**: Enforce plan-focused tutoring  
**Persona**: Concept-Focused Student  
**User Story**: As a student, I want tutoring to stay focused on session objectives so that explanations are coherent.  
**Context / Notes**: Tutor prompt “STAY ON PLAN” and out-of-plan handling.  
**Acceptance Criteria**:
- Tutor references `current_plan` and mode in response behavior.
- Off-topic requests trigger confirmation to switch instead of immediate off-topic teaching.

**Story ID**: US-007  
**Title**: Offer future-topic continuity  
**Persona**: Returning Student  
**User Story**: As a student, I want a suggestion for what to learn next so that I can continue progress across sessions.  
**Context / Notes**: Tutor prompt includes `future_plan` mention on wrap-up.  
**Acceptance Criteria**:
- Session-end tutor responses can suggest next LO from `future_plan`.
- Suggested next topic is captured in session summary when provided.

### Theme C: Planning and Guidance

**Story ID**: US-008  
**Title**: Build one focused current plan  
**Persona**: Concept-Focused Student  
**User Story**: As a student, I want a focused plan anchored on one primary objective so that the session has clear direction.  
**Context / Notes**: `TutoringPlanner` simplified plan contract.  
**Acceptance Criteria**:
- `current_plan` includes exactly one primary LO (`is_primary=true`).
- Additional current-plan items are optional supporting LOs.

**Story ID**: US-009  
**Title**: Adapt plan using proficiency  
**Persona**: Returning Student  
**User Story**: As a returning student, I want planning to account for my prior mastery so that time is spent where I need help.  
**Context / Notes**: `proficiency_map` in planner; note generation by score tiers.  
**Acceptance Criteria**:
- Candidate proficiency scores are derived from `student_profile.lo_mastery`.
- Plan notes differ for high vs low proficiency contexts.

**Story ID**: US-010  
**Title**: Keep tutoring mode explicit  
**Persona**: Practice-Oriented Student  
**User Story**: As a student, I want conceptual/examples/practice mode to remain explicit so that explanations match my preference.  
**Context / Notes**: Mode passed through planner and tutor prompts.  
**Acceptance Criteria**:
- Planner output `mode` matches requested mode.
- Tutor behavior reflects mode-switch cues and confirmation workflow.

### Theme D: FAQ and Syllabus Assistance

**Story ID**: US-011  
**Title**: Route FAQ keywords to FAQ planning  
**Persona**: Administrative/FAQ Student  
**User Story**: As a student, I want admin questions recognized immediately so that I get logistics answers quickly.  
**Context / Notes**: `_FAQ_KEYWORD_MAP` + `_detect_faq_topic`.  
**Acceptance Criteria**:
- Queries containing exam/quiz/homework/grading/office-hours keywords map to FAQ topics.
- FAQ planner is invoked with topic and original student wording.

**Story ID**: US-012  
**Title**: Handle syllabus-topic ambiguity safely  
**Persona**: Administrative/FAQ Student  
**User Story**: As a student, I want clarification attempts to eventually resolve so that I do not get stuck in loops.  
**Context / Notes**: syllabus escalation in `CoachRouter._maybe_force_syllabus_plan`.  
**Acceptance Criteria**:
- After configured clarification limit for syllabus requests, system force-calls FAQ planner.
- If planner completes, FAQ session starts automatically.

**Story ID**: US-013  
**Title**: Constrain FAQ answers to known scripts  
**Persona**: Administrative/FAQ Student  
**User Story**: As a student, I want policy answers grounded in known course script so that answers remain consistent.  
**Context / Notes**: FAQ prompt rule: “use only provided script.”  
**Acceptance Criteria**:
- FAQ bot receives `session_params.script`.
- FAQ output includes follow-up question from plan.

### Theme E: Memory and Continuity

**Story ID**: US-014  
**Title**: Persist completed sessions  
**Persona**: Returning Student  
**User Story**: As a student, I want my completed sessions saved so that future interactions can build on prior work.  
**Context / Notes**: `SessionMemory.add_session` in bot finalization.  
**Acceptance Criteria**:
- On `end_activity=true`, session params/summary/exchanges are stored.
- Stored sessions are bounded by configured max entries.

**Story ID**: US-015  
**Title**: Retrieve last tutoring context  
**Persona**: Returning Student  
**User Story**: As a student, I want to ask what I learned last session so that I can continue effectively.  
**Context / Notes**: regex interception + `last_tutoring_session()`.  
**Acceptance Criteria**:
- Session-history questions return last session LO/mode/subject summary.
- If no sessions exist, assistant offers to start new tutoring.

**Story ID**: US-016  
**Title**: Show proficiency report on demand  
**Persona**: Returning Student  
**User Story**: As a student, I want to see my progress scores so that I can prioritize weak areas.  
**Context / Notes**: `show_proficiency` action + `format_proficiency_report`.  
**Acceptance Criteria**:
- Progress request phrases trigger proficiency report action.
- Report segments strong areas and areas to focus on.

**Story ID**: US-017  
**Title**: Update mastery from tutor outcomes  
**Persona**: Returning Student  
**User Story**: As a student, I want my mastery to update after sessions so that future plans are personalized.  
**Context / Notes**: qualitative-to-numeric mapping in `UNDERSTANDING_TO_MASTERY`.  
**Acceptance Criteria**:
- Tutor session summary `student_understanding` updates LO mastery score.
- Updated profile is persisted to session memory storage.

### Theme F: Retrieval and Grounding

**Story ID**: US-018  
**Title**: Retrieve LO candidates via hybrid search  
**Persona**: Concept-Focused Student  
**User Story**: As a student, I want relevant objectives selected from my question so that tutoring starts on the right concept.  
**Context / Notes**: dense + BM25 + fusion in retriever.  
**Acceptance Criteria**:
- Text query retrieval produces ranked candidate list.
- Planner receives merged candidates with LO metadata.

**Story ID**: US-019  
**Title**: Support image-informed retrieval  
**Persona**: Concept-Focused Student  
**User Story**: As a student sharing an image, I want the system to use visual context so that explanations match what I uploaded.  
**Context / Notes**: image preprocessing + optional CLIP + tutor multimodal payload.  
**Acceptance Criteria**:
- Image input updates query context and can influence candidate retrieval.
- Tutor receives image in multimodal message when available.

### Theme G: Reliability and Safety

**Story ID**: US-020  
**Title**: Survive malformed LLM JSON  
**Persona**: All personas  
**User Story**: As a user, I want the assistant to recover from model-format errors so that conversation does not crash.  
**Context / Notes**: retry + fallback in `tutor.py`, `coach_llm_client.py`, `json_utils.py`.  
**Acceptance Criteria**:
- Invalid JSON triggers one retry (tutor/faq) with JSON-only reminder.
- On repeated failure, assistant returns safe fallback message.

**Story ID**: US-021  
**Title**: Prevent orchestration loops  
**Persona**: All personas  
**User Story**: As a user, I want bounded decision loops so that requests terminate with a response.  
**Context / Notes**: router loop cap and loop-exhausted fallback message.  
**Acceptance Criteria**:
- Router loop does not exceed configured max iterations.
- Loop exhaustion returns clear retry/rephrase guidance.

**Story ID**: US-022  
**Title**: Preserve continuity after session end  
**Persona**: Returning Student  
**User Story**: As a student, I want a contextual return message after finishing a session so that transition back to coach feels natural.  
**Context / Notes**: `_build_return_greeting` in `BotSessionManager`.  
**Acceptance Criteria**:
- Tutor completion greeting references LO (and mode when present).
- FAQ completion greeting references FAQ topic when available.

## 5. Functional Requirements Derived from the Stories

### FR Group A: Orchestration
- System must classify intents into tutoring, FAQ, proficiency, or clarification.
- System must support planner-first then immediate session handoff.
- System must maintain one active bot session at a time.

### FR Group B: Tutoring Experience
- Planner must produce a compact plan with primary + supporting LOs and future LO.
- Tutor must operate in one explicit mode (`conceptual_review`, `examples`, `practice`).
- Tutor must handle off-plan/topic or mode-switch requests with confirmation workflow.

### FR Group C: FAQ Experience
- FAQ planner must map to known topic list only.
- FAQ bot must answer from provided script and avoid unsupported invention.
- Syllabus requests must have escalation path after repeated ambiguity.

### FR Group D: Memory and Personalization
- Completed sessions must persist with summaries and exchanges.
- System must expose recent-session continuity and explicit proficiency reporting.
- Tutor outcomes must update mastery values for later planning.

### FR Group E: Reliability
- API/model failures must degrade to safe user-visible fallbacks.
- JSON outputs must be normalized/validated before use.
- Decision loops must be bounded and fail gracefully.

## 6. Edge Cases / Failure Scenarios

- **Ambiguous tutoring request**: planner returns `need_info`; coach relays clarification.
- **No retrieval candidates**: tutoring planner asks for rephrasing.
- **Unknown FAQ topic**: FAQ planner returns known-topic guidance prompt.
- **Malformed LLM JSON**: retry then fallback response.
- **Session state corruption**: bot manager guard resets and returns greeting.
- **Repeated syllabus clarifications**: forced FAQ planning to avoid dead-end loops.
- **Missing API key**:
  - retriever embeddings fail hard without `OPENAI_API_KEY`,
  - planner LLM features can degrade to deterministic mode when disabled.

## 7. Future User Stories (Inferred Opportunities)

These are plausible next steps inferred from architecture and README references; they are **not fully implemented in the inspected `src/workflow_demo` module set**.

**Story ID**: FUS-001  
**Title**: Browser-based live tutoring session  
**Persona**: Student  
**User Story**: As a student, I want a web chat interface so that I can use the tutor without CLI commands.  
**Reasonable Basis**: README references web bridge/frontend architecture, but corresponding backend module is absent here.

**Story ID**: FUS-002  
**Title**: Session resume by explicit selection  
**Persona**: Returning Student  
**User Story**: As a student, I want to choose from recent sessions to resume exactly where I left off.  
**Reasonable Basis**: memory already stores recent sessions, but selection UX is not present.

**Story ID**: FUS-003  
**Title**: Instructor-configurable FAQ scripts  
**Persona**: Educator/Operator  
**User Story**: As an instructor, I want to edit FAQ policies without code changes so that course logistics remain current.  
**Reasonable Basis**: FAQ scripts are currently hardcoded constants.

**Story ID**: FUS-004  
**Title**: Retrieval confidence transparency  
**Persona**: Student  
**User Story**: As a student, I want confidence-aware responses when context is weak so that I know when to rephrase or narrow scope.  
**Reasonable Basis**: retrieval scores exist but are not surfaced in user messaging.

