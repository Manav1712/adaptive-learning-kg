# Workflow Demo -- Architecture Explanation

---

# bot_sessions.py

## Purpose

Manages the full lifecycle of tutor and FAQ bot sessions. Owns all session-specific state (active flag, bot type, handoff context, conversation history) and coordinates session start, turn handling, finalization, and cleanup. The CoachAgent delegates here rather than tracking session fields itself.

## Dependencies

- `session_memory.create_handoff_context` -- packages session params, memory, student state, and image into a context dict the bot reads.
- `tutor.tutor_bot` / `tutor.faq_bot` -- the LLM-backed bots that generate responses.
- `CoachAgent` (via `self.agent`) -- shared state: LLM client, session memory, student profile, planner result, image fields, conversation history helpers.

## Constants

### UNDERSTANDING_TO_MASTERY

Maps the tutor bot's qualitative student_understanding label ("excellent", "good", etc.) to a numeric 0-1 mastery score. Used after a tutor session ends to update the student profile.

## Class: BotSessionManager

### State (owned by the manager)

| Field | Type | Description |
|---|---|---|
| `is_active` | bool | Whether a bot session is currently running. |
| `bot_type` | str or None | "tutor" or "faq" during a session, None otherwise. |
| `handoff_context` | dict or None | The context dict passed to the bot (session params, memory, student state, image). |
| `conversation_history` | list[dict] | Chronological list of `{"speaker": ..., "text": ...}` exchanges within the current session. |

### Public Methods

#### `__init__(agent)`

Stores a reference to the CoachAgent and initializes all session state to inactive/empty.

#### `handle_turn(user_input) -> str`

Called by CoachAgent.process_turn when `is_active` is True. Appends the student message to conversation_history and calls `_invoke_bot()` to get the bot's response.

**Called from:** `CoachAgent._handle_bot_turn`

#### `begin(bot_type, tool_params, conversation_summary) -> str`

Entry point for starting a new session. Performs a four-step pipeline:

1. `_build_session_params(tool_params)` -- merges planner output with tool_params.
2. `create_handoff_context(...)` -- packages everything the bot needs into a context dict.
3. `_reset()` -- clears all transient state (image is captured beforehand and restored after).
4. `_invoke_bot(initial=True)` -- calls the bot with empty history to get its opening message.

**Called from:** `CoachRouter.handle_turn` (actions `start_tutor`, `start_faq`) and `CoachRouter._maybe_force_syllabus_plan`.

#### `format_proficiency_report() -> str`

Reads `lo_mastery` from the student profile and formats it into a human-readable progress report. Groups LOs into "strong areas" (>=65%) and "areas to focus on" (<65%), and suggests the weakest LO for practice.

**Called from:** `CoachRouter.handle_turn` (action `show_proficiency`).

### Private Methods

#### `_build_session_params(tool_params) -> dict`

Merges the planner's plan dict with tool_params from the coach directive. Plan values take precedence -- tool_params only fills keys the plan didn't set. Ensures `student_request` is always present (falls back to the last student message). Attaches `image_query` if the student submitted an image.

#### `_reset()`

Single-point cleanup for all session and transient state. Clears:
- Manager-owned: is_active, bot_type, handoff_context, conversation_history.
- Agent-owned: collected_params, planner_result, syllabus escalation flags, current_image, current_image_query.

Called by `begin` (before setting up new session) and `_finalize_bot_session` (after session ends).

#### `_invoke_bot(initial=False) -> str`

The single method that calls `tutor_bot` or `faq_bot`. Handles:
- Guard against missing session state (resets and returns coach greeting).
- History selection: empty list on first call, full conversation_history on subsequent calls.
- Dispatches to tutor (with image) or faq bot.
- Records the bot's reply in conversation_history.
- If the bot signals `end_activity`, delegates to `_finalize_bot_session`.

Called by `begin` (initial=True) and `handle_turn` (initial=False).

#### `_finalize_bot_session(bot_response) -> str`

Runs when the bot sets `end_activity: True`. Steps:
1. Captures session data (summary, params, exchanges) before reset.
2. Persists the session record to session memory.
3. For tutor sessions: updates lo_mastery and saves to disk.
4. Extracts switch_topic / switch_mode requests from the summary.
5. Calls `_reset()`.
6. If a switch was requested, routes back to coach with a synthetic turn.
7. Otherwise returns a continuity-aware greeting.

#### `_update_lo_mastery(params, summary)`

Resolves the LO key from params, maps the tutor's qualitative understanding label to a numeric score via UNDERSTANDING_TO_MASTERY, and writes it to `student_profile["lo_mastery"]`.

#### `_build_return_greeting(params, session_type) -> str`

Builds a greeting after a normal session end. References the LO and mode for tutor sessions, the topic for FAQ sessions. Falls back to the generic coach greeting.

## Flow

```
CoachRouter decides to start a session
  |
  v
begin(bot_type, tool_params, conversation_summary)
  |-- _build_session_params(tool_params)
  |-- create_handoff_context(...)
  |-- _reset()
  |-- set is_active=True, bot_type, handoff_context, restore image
  |-- _invoke_bot(initial=True) --> bot's opening message
  |
  v
Student sends messages (while is_active)
  |
  v
handle_turn(user_input)
  |-- append to conversation_history
  |-- _invoke_bot()
        |-- dispatch to tutor_bot or faq_bot
        |-- record reply in conversation_history
        |-- if end_activity: _finalize_bot_session()
              |-- save session to memory
              |-- update mastery (tutor only)
              |-- _reset()
              |-- switch topic/mode? --> route back to coach
              |-- else --> return greeting
```

---

# clip_embeddings.py

## Purpose

Provides CLIP-based embedding utilities for encoding text and images into a shared 512-dimensional vector space. This allows direct similarity comparison between text queries and images using dot products. Used by the retriever for text-to-image and image-to-image search.

## Dependencies

- `sentence-transformers` (`SentenceTransformer`) -- loads and runs the CLIP model locally.
- `Pillow` (`PIL.Image`) -- opens and converts image files to RGB before encoding.
- `numpy` -- all embeddings are returned as float32 ndarrays.

## Standalone Function

### `normalize_dense(arr) -> np.ndarray`

L2-normalizes each row of a 2D array to unit length. Adds a small epsilon (1e-12) to norms to avoid division by zero. Returns the input unchanged if the array is empty.

**Called from:** `retriever.py` -- applied to image embeddings after `encode_images`, and to single image embeddings during image search.

## Class: CLIPEmbeddingBackend

### `__init__(model_name="clip-ViT-B-32")`

Loads the CLIP model via SentenceTransformer. The default model (`clip-ViT-B-32`) produces 512-dimensional embeddings for both text and images.

### `encode_text(texts) -> np.ndarray`

Takes a list of strings, encodes them through CLIP with L2 normalization, and returns a float32 ndarray of shape `(len(texts), 512)`.

**Called from:** `retriever.py` -- encodes a text query for text-to-image similarity search.

### `encode_images(image_paths) -> np.ndarray`

Takes a list of `Path` objects, opens each as RGB, encodes them through CLIP with L2 normalization, and returns a float32 ndarray of shape `(len(image_paths), 512)`. Raises `ValueError` if any image file cannot be read.

**Called from:** `retriever.py` -- batch-encodes all images during index build, and encodes a single image during image-based search.

## Flow

```
retriever builds image index
  |
  v
encode_images(image_paths)
  |-- open each image as RGB
  |-- run through CLIP model
  |-- return float32 ndarray (n, 512)
  |
  v
normalize_dense(embeddings)  [called by retriever]
  |-- L2-normalize each row to unit vector
  |-- store as self.image_embeddings

student submits a text query for image search
  |
  v
encode_text([query])
  |-- run text through CLIP model
  |-- return float32 vector (1, 512)
  |
  v
dot product: image_embeddings @ query_vec
  |-- rank images by similarity score
```

## Notes

- Both `encode_text` and `encode_images` pass `normalize_embeddings=True` to the model, so vectors are already unit-length. The retriever then calls `normalize_dense` again, which is a harmless no-op on already-normalized vectors.
- CLIP's key property: text and images live in the same vector space, so a text embedding can be directly compared against an image embedding for meaningful similarity.

---

# coach_llm_client.py

## Purpose

Single point where the coach agent talks to the LLM. Sends the current conversation state to OpenAI, receives a structured JSON directive telling the coach what to do next, and handles all error/retry logic so the coach never crashes.

## Dependencies

- `openai` -- OpenAI client and error types (`APIError`, `RateLimitError`, `APIConnectionError`, `APITimeoutError`).
- `.json_utils.coerce_json` -- parses the raw LLM text response into a Python dict, handling markdown fences and malformed JSON.

## Module-Level Constants

### `COACH_SYSTEM_PROMPT`

The system prompt that defines the coach LLM's behaviour. Tells the LLM:
- What input it receives (conversation history, recent sessions, planner results, collected params).
- What JSON shape to return (action, message_to_student, tool_params, conversation_summary).
- Decision rules: intent detection (tutoring vs FAQ vs proficiency), when to ask clarifying questions, when to call planners, when to hand off to bots, how to handle returning from sessions.

### `_FALLBACK_DIRECTIVE`

Safe default response used when anything goes wrong. Action is `"none"` with a generic error message. Every error path spreads this dict and optionally overrides `message_to_student`.

### `_MAX_RETRIES`

Number of attempts before giving up on transient errors (default: 3).

### `_RETRY_DELAY_BASE`

Starting delay in seconds for exponential backoff (default: 1.0). Actual wait is `_RETRY_DELAY_BASE * 2^attempt` (1s, 2s, 4s).

## Class: CoachLLMClient

### `__init__(openai_client, model)`

Stores the OpenAI client instance and model name (e.g. `gpt-4o-mini`).

### `_should_retry(attempt, error) -> bool` (static)

Shared retry helper used by both transient-error and 5xx-error handlers. If retries remain, logs the error, sleeps with exponential backoff, and returns `True`. Otherwise returns `False` so the caller falls through to its final error response.

**Called from:** `get_directive` -- in the `RateLimitError`/`APIConnectionError`/`APITimeoutError` block and in the `APIError` 5xx block.

### `get_directive(payload) -> dict`

The main (and only real) method. Takes a dict with conversation state and returns a parsed directive. Steps:

1. Builds a two-message prompt: system prompt + JSON-serialized payload.
2. Calls OpenAI chat completions with `temperature=0` for deterministic output.
3. Parses the response via `coerce_json`.
4. On failure, categorizes the error and either retries or returns a fallback:
   - **Transient errors** (rate limit, connection, timeout): retry via `_should_retry`.
   - **Server 5xx errors**: retry via `_should_retry`.
   - **Parse errors** (bad JSON, missing fields): no retry, return fallback immediately.
   - **Catch-all**: no retry, return generic fallback.

**Called from:** `CoachAgent._get_coach_directive` -- every coach turn calls this to decide what action to take.

## Flow

```
CoachAgent builds payload (history, sessions, planner result, etc.)
  |
  v
get_directive(payload)
  |-- build messages: [system prompt, JSON payload]
  |
  v
for attempt in range(3):
  |
  |-- call OpenAI chat.completions.create
  |     |
  |     |-- success? --> coerce_json(content) --> return directive
  |     |
  |     |-- empty content? --> return fallback
  |
  |-- RateLimitError / ConnectionError / Timeout?
  |     |-- _should_retry? --> sleep, continue
  |     |-- else --> return "connection issues" fallback
  |
  |-- APIError with 5xx status?
  |     |-- _should_retry? --> sleep, continue
  |     |-- else --> return "encountered an error" fallback
  |
  |-- JSONDecodeError / parse failure?
  |     |-- return "trouble processing" fallback (no retry)
  |
  |-- unexpected exception?
        |-- return generic fallback (no retry)

all retries exhausted --> return "trouble connecting" fallback
```
