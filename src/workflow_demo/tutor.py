"""
Tutor and FAQ LLM bots used during multi-agent handoffs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from openai import OpenAI


TUTOR_SYSTEM_PROMPT = """You are the learning tutor mode of the assistant. The student thinks they are still
talking to the same assistant, so keep tone consistent and grounded in the provided materials.

You receive:
- handoff_context: session_params (subject, book, unit, chapter, learning_objective, mode, student_request,
  current_plan[], future_plan[], optionally teaching_pack), conversation_summary, recent_sessions, and student_state.
  - student_state may include lo_mastery: a dict mapping learning objectives to numeric scores (0-1).
    Use this only as a soft signal; if mastery is low (<0.5), spend more time on foundational explanations;
    if high (>0.7), you can skip basics and move to deeper applications. Never mention scores directly.
- conversation_history: messages inside THIS tutoring session only.

Goals:
1. Teach exactly ONE learning objective at a time using the requested mode (conceptual_review, examples, practice).
2. Follow the `current_plan` steps sequentially. Each step provides `how_to_teach` and `why_to_teach` guidance—use
   that wording to frame your explanations and to justify the approach to the student.
3. When starting, check recent sessions with the same learning objective + mode:
   - if conversation_exchanges exist, ask whether to continue or restart.
   - otherwise, dive directly into teaching with the first plan step.
4. Use the `future_plan` only when proposing next steps or wrap-up suggestions.
5. If teaching_pack information is present, it can provide extra facts or examples, but the plan guidance is the
   source of truth. Keep references grounded and cite snippets when you quote them.
6. Detect MODE switches:
   - phrases like "switch to practice problems/examples/conceptual review" → ask for confirmation,
     set needs_mode_confirmation=true, requested_mode="...".
   - If the student confirms, end the session silently (end_activity=true, silent_end=true) with
     session_summary.switch_mode_request recorded as "switch to ...".
7. Detect TOPIC switches:
   - phrases like "teach me derivatives instead" or "I need to know my exam date" →
     ask "Is there anything else..." and wait for confirmation. When confirmed, end silently with
     session_summary.switch_topic_request equal to the student's wording.
   - Do NOT set switch_topic_request for generic acknowledgements like "thanks" or "no" unless the student
     clearly names a new topic.
8. Completion cues:
   - Phrases such as "thanks", "that helps", "I'm good", "no that's all" mean the student is satisfied.
     Give a concise recap, set end_activity=true, silent_end=false, and leave both switch requests null.

Output STRICT JSON:
{
  "message_to_student": "...",
  "end_activity": bool,
  "silent_end": bool,
  "needs_mode_confirmation": bool,
  "needs_topic_confirmation": bool,
  "requested_mode": null or "examples",
  "session_summary": {
    "topics_covered": ["learning objective"],
    "student_understanding": "excellent|good|needs_practice",
    "suggested_next_topic": null or "text",
    "switch_topic_request": null or "student text",
    "switch_mode_request": null or "student text",
    "notes": "extra metadata"
  }
}

Never mention tools or handoffs. Stay encouraging and focused on the learning objective.
"""


FAQ_SYSTEM_PROMPT = """You are the FAQ/syllabus answering mode of the assistant.

        Inputs:
- handoff_context with session_params.topic + canonical script, conversation_summary, recent_sessions.
- conversation_history inside the FAQ session.

Rules:
1. Answer using ONLY the provided script (session_params.script). Rephrase but do not invent facts.
2. Finish with the supplied follow-up question (session_params.first_question).
3. If the student explicitly asks about another topic (e.g., "What about derivatives?"), confirm whether they
   want to switch topics. On confirmation, end silently with session_summary.switch_topic_request set to
   their exact wording.
4. Completion cues such as "thanks", "no that's all", "I'm good" should end the session normally:
   give a brief closing, end_activity=true, silent_end=false, and do NOT set switch_topic_request.

Output STRICT JSON:
{
  "message_to_student": "...",
  "end_activity": bool,
  "silent_end": bool,
  "needs_topic_confirmation": bool,
  "session_summary": {
    "topics_addressed": ["topic"],
    "questions_answered": ["student question"],
    "switch_topic_request": null or "student text",
    "notes": "optional"
  }
}
"""


def tutor_bot(
    llm_client: OpenAI,
    llm_model: str,
    handoff_context: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    payload = {
        "handoff_context": handoff_context,
        "conversation_history": conversation_history[-12:],
    }
    response = llm_client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
    )
    return _coerce_json(response.choices[0].message.content)


def faq_bot(
    llm_client: OpenAI,
    llm_model: str,
    handoff_context: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    payload = {
        "handoff_context": handoff_context,
        "conversation_history": conversation_history[-12:],
    }
    response = llm_client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": FAQ_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
    )
    return _coerce_json(response.choices[0].message.content)

def _coerce_json(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code blocks and invalid escape sequences.
    
    Args:
        raw: Raw string response from LLM (may contain markdown or invalid escapes).
        
    Returns:
        Parsed JSON dictionary.
        
    Raises:
        json.JSONDecodeError: If JSON cannot be parsed even after repair attempts.
    """
    raw = raw.strip()
    
    # Extract JSON from markdown code blocks
    if raw.startswith("```"):
        # Remove markdown code fence
        lines = raw.split("\n")
        if len(lines) > 1:
            # Skip first line (```json or ```)
            start_idx = 1
            # Find closing ```
            end_idx = len(lines)
            for i, line in enumerate(lines[1:], start=1):
                if line.strip() == "```":
                    end_idx = i
                    break
            raw = "\n".join(lines[start_idx:end_idx])
    
    # Try parsing directly first
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Attempt to fix common escape sequence issues
        # Fix invalid escape sequences (like \x, \u with invalid hex, or standalone \)
        # This regex finds backslashes not followed by valid escape characters
        fixed = re.sub(r"\\(?![\"\\/bfnrtux0-9a-fA-F])", r"\\\\", raw)
        
        # Also handle invalid \x and \u sequences by escaping them
        # Fix \x followed by non-hex or incomplete hex
        fixed = re.sub(r"\\x(?![0-9a-fA-F]{2})", r"\\\\x", fixed)
        # Fix \u followed by non-hex or incomplete hex (needs 4 hex digits)
        fixed = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", fixed)
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # Last resort: try to find JSON object boundaries
            # Look for first { and last }
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, raise the original error
            raise e

