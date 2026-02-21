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
