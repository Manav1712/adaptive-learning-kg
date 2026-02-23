"""OpenAI API wrapper with retry logic and directive parsing for the coach agent.

The coach agent delegates every routing decision to an LLM. This module
is the single point where that LLM call happens. It sends the current
conversation state (history, session summaries, planner results, etc.)
to OpenAI with a system prompt that defines the coach's behaviour, and
parses the JSON response into a structured directive containing:
  - action: what the coach should do next (e.g. call a planner, start
    a tutor/FAQ session, show proficiency, or do nothing).
  - message_to_student: text to display to the student (if any).
  - tool_params: subject/topic/mode metadata for downstream tools.
  - conversation_summary: short summary used during handoffs.

Transient errors (rate limits, timeouts, 5xx) are retried with
exponential backoff. Non-recoverable errors return a safe fallback
directive so the coach never crashes.
"""

import json
import time
from typing import Any, Dict, List
from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError, OpenAI
from .json_utils import coerce_json

COACH_SYSTEM_PROMPT = """
You are the orchestrator of a learning assistant. The student only talks to you,
but behind the scenes you can call tools (tutoring planner, FAQ planner, tutor bot, FAQ bot).

INPUT: A JSON payload with:
- conversation_history: last 10 messages between coach and student.
- recent_sessions: up to 5 session summaries (tutoring or FAQ) with params and summaries.
- last_tutoring_session: shortcut to the most recent tutoring session (or null).
- planner_result: most recent planner call result, if any.
- collected_params: currently known subject/topic/mode/faq topic metadata.
- returning_from_session: true when the student just came back from a tutor/FAQ session via a switch request.

YOU MUST RETURN STRICT JSON:
{
  "message_to_student": "plain text or empty string when handing off",
  "action": "none|call_tutoring_planner|call_faq_planner|start_tutor|start_faq|show_proficiency",
  "tool_params": {
    "subject": "...",
    "learning_objective": "...",
    "mode": "conceptual_review|examples|practice",
    "topic": "...",
    "student_request": "original wording"
  },
  "conversation_summary": "short summary used when handing off (optional)"
}

Guidelines:
1. Intent detection:
   - Tutoring intent: wants to learn/practice/review a topic. Gather subject + learning_objective + mode.
   - FAQ intent: wants logistics/policy info. Gather topic only.
   - Proficiency intent: wants to see their learning progress. Phrases like "show my proficiency", "how am I doing",
     "what's my progress", "show my scores", "my learning progress" → set action="show_proficiency".
   - Syllabus / course-outline / "major concepts" questions should go to FAQ mode using topic "syllabus_topics" and student_request containing the learner's wording.
   - Topic switch phrases ("teach me X instead", "let's do Y") after a session should jump straight into planning with the NEW topic mentioned.
   - "Back to" or "continue" phrases should reuse last_tutoring_session.params (subject + learning_objective) as defaults.
   - IMPORTANT: If the student does NOT name a topic/subject (e.g., "start a tutoring session", "I want to learn"), you MUST ASK:
     "What specific topic or learning objective would you like to focus on?" and set action="none".
   - Do NOT infer subject or learning_objective from recent_sessions or last_tutoring_session unless the student explicitly says to continue/return (e.g., "continue where I left off", "back to previous topic").
   - Mode switch ("switch to practice/examples/conceptual review") should reuse last_tutoring_session.params for subject + learning_objective,
     override the mode, and immediately call the tutoring planner.
2. Planner usage:
   - When required info is missing, ask one clarifying question (message_to_student) and set action="none".
   - Once all tutoring params are ready, call tutoring planner via action="call_tutoring_planner" with the params.
   - Same for FAQ planner with action="call_faq_planner".
   - When planner returns status need_info, relay the planner's message verbatim and continue collecting info.
3. Handoff to Tutor/FAQ:
   - Once a plan is generated, IMMEDIATELY set action="start_tutor" (for tutoring) or "start_faq" (for FAQ).
   - Do NOT ask for the student's confirmation.
   - Do NOT summarize the plan to the student.
   - Set message_to_student to an empty string so the tutor/faq bot can speak first.
   - Include conversation_summary describing why we are starting the session.
4. Returning from sessions:
   - If returning_from_session is true, immediately honor the switch request contained in the latest message.
   - Do NOT greet; instead analyze the request and call the appropriate planner or start a new plan, using last_tutoring_session when helpful.
5. Always keep tool_params specific to the current intent. Do not mix FAQ params with tutoring params.
"""

_FALLBACK_DIRECTIVE: Dict[str, Any] = {
    "action": "none",
    "message_to_student": "Something went wrong. Please try again.",
    "tool_params": {},
    "conversation_summary": None,
}

_MAX_RETRIES = 3
# Starting delay for exponential backoff: 1s, 2s, 4s, ...
_RETRY_DELAY_BASE = 1.0


class CoachLLMClient:
    """Wrapper around OpenAI API with retry logic, exponential backoff, and directive parsing."""

    def __init__(self, openai_client: OpenAI, model: str) -> None:
        """Initialize the LLM client.

        Args:
            openai_client: OpenAI client instance.
            model: Model name (e.g., 'gpt-4o-mini').
        """
        self.openai_client = openai_client
        self.model = model

    @staticmethod
    def _should_retry(
        attempt: int, error: Exception
    ) -> bool:
        """Log the error and sleep with exponential backoff if
        retries remain. Returns True if the caller should retry.
        """
        if attempt >= _MAX_RETRIES - 1:
            return False
        wait_time = _RETRY_DELAY_BASE * (2 ** attempt)
        print(
            f"[Coach] {type(error).__name__} "
            f"(attempt {attempt + 1}/{_MAX_RETRIES}). "
            f"Retrying in {wait_time:.1f}s..."
        )
        time.sleep(wait_time)
        return True

    def get_directive(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the OpenAI API with the coach brain payload and
        return a parsed directive.

        Implements retry logic with exponential backoff for
        transient errors.

        Args:
            payload: JSON-serializable dict with conversation
                history, session state, etc.

        Returns:
            Parsed directive dict with action, message_to_student,
            tool_params, conversation_summary.
        """
        # Build the two-message prompt: system instructions + state.
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, indent=2),
            },
        ]

        for attempt in range(_MAX_RETRIES):
            try:
                # Send conversation state to OpenAI; temp=0 for
                # deterministic output.
                response = (
                    self.openai_client.chat.completions.create(
                        model=self.model,
                        temperature=0,
                        messages=messages,
                        timeout=30.0,
                    )
                )

                # Extract the text content from the response.
                content = response.choices[0].message.content
                if not content:
                    return {
                        **_FALLBACK_DIRECTIVE,
                        "message_to_student": (
                            "I didn't receive a response. "
                            "Could you try again?"
                        ),
                    }

                # Parse the JSON string into a directive dict.
                return coerce_json(content)

            # -- Transient network / rate-limit errors: retry. --
            except (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
            ) as e:
                if self._should_retry(attempt, e):
                    continue
                print(
                    f"[Coach] {type(e).__name__} "
                    f"after {_MAX_RETRIES} attempts: {e}"
                )
                return {
                    **_FALLBACK_DIRECTIVE,
                    "message_to_student": (
                        "I'm having connection issues. "
                        "Please try again."
                    ),
                }

            # -- Server-side 5xx errors: retry if attempts remain. --
            except APIError as e:
                status_code = getattr(e, "status_code", 0)
                if status_code and status_code >= 500:
                    if self._should_retry(attempt, e):
                        continue
                print(f"[Coach] API error: {e}")
                return {
                    **_FALLBACK_DIRECTIVE,
                    "message_to_student": (
                        "I encountered an error. "
                        "Could you rephrase your request?"
                    ),
                }

            # -- Response parsing failures: no retry. --
            except (
                json.JSONDecodeError,
                IndexError,
                AttributeError,
            ) as e:
                print(
                    f"[Coach] Failed to parse LLM response: {e}"
                )
                return {
                    **_FALLBACK_DIRECTIVE,
                    "message_to_student": (
                        "I'm having trouble processing that. "
                        "Could you rephrase your request?"
                    ),
                }

            # -- Catch-all for anything unexpected. --
            except Exception as e:
                print(f"[Coach] Unexpected error: {e}")
                return {**_FALLBACK_DIRECTIVE}

        # All retries exhausted without a successful response.
        return {
            **_FALLBACK_DIRECTIVE,
            "message_to_student": (
                "I'm having trouble connecting right now. "
                "Could you try again?"
            ),
        }
