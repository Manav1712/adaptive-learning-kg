"""OpenAI API wrapper with retry logic and directive parsing for the coach agent."""

import json
import time
from typing import Any, Dict, List, Optional

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

    def get_directive(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the OpenAI API with the coach brain payload and return a parsed directive.

        Implements retry logic with exponential backoff for transient errors.

        Args:
            payload: JSON-serializable dict with conversation history, session state, etc.

        Returns:
            Parsed directive dict with action, message_to_student, tool_params, conversation_summary.
        """
        MAX_RETRIES = 3
        RETRY_DELAY_BASE = 1.0

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]

        for attempt in range(MAX_RETRIES):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=messages,
                    timeout=30.0,
                )
                content = response.choices[0].message.content
                if not content:
                    return {
                        **_FALLBACK_DIRECTIVE,
                        "message_to_student": "I didn't receive a response. Could you try again?",
                    }
                return coerce_json(content)

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                    print(
                        f"[Coach] {type(e).__name__} (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                print(f"[Coach] {type(e).__name__} after {MAX_RETRIES} attempts: {e}")
                return {
                    **_FALLBACK_DIRECTIVE,
                    "message_to_student": "I'm having connection issues. Please try again.",
                }

            except APIError as e:
                status_code = getattr(e, "status_code", 0)
                if status_code and status_code >= 500 and attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                    print(
                        f"[Coach] API error {status_code} (attempt {attempt + 1}/{MAX_RETRIES}). "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                print(f"[Coach] API error: {e}")
                return {
                    **_FALLBACK_DIRECTIVE,
                    "message_to_student": "I encountered an error. Could you rephrase your request?",
                }

            except (json.JSONDecodeError, IndexError, AttributeError) as e:
                print(f"[Coach] Failed to parse LLM response: {e}")
                return {
                    **_FALLBACK_DIRECTIVE,
                    "message_to_student": "I'm having trouble processing that. Could you rephrase your request?",
                }

            except Exception as e:
                print(f"[Coach] Unexpected error: {e}")
                return {**_FALLBACK_DIRECTIVE}

        return {
            **_FALLBACK_DIRECTIVE,
            "message_to_student": "I'm having trouble connecting right now. Could you try again?",
        }
