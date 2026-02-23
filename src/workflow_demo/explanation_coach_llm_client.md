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
