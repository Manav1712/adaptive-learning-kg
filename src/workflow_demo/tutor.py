"""
Tutor and FAQ LLM bots used during multi-agent handoffs.
"""

from __future__ import annotations

import json
import base64
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from json import JSONDecodeError

from openai import OpenAI

from .json_utils import coerce_json
from .pedagogy.math_example_guard import maybe_apply_math_example_guard

# Canonical Phase 6 conditioning object (also dual-written as tutor_directives on pedagogy_context).
TUTOR_INSTRUCTION_DIRECTIVE_KEYS: tuple[str, ...] = (
    "session_target_lo",
    "instruction_lo",
    "selected_move_type",
    "retrieval_intent",
    "retrieval_action",
    "policy_reason",
)


def extract_tutor_instruction_directives(
    pedagogy_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the six-field conditioning dict for the tutor payload; prefer canonical key, else legacy."""
    if not isinstance(pedagogy_context, dict):
        return {}
    raw = pedagogy_context.get("tutor_instruction_directives")
    if isinstance(raw, dict) and raw:
        return {k: raw.get(k) for k in TUTOR_INSTRUCTION_DIRECTIVE_KEYS}
    legacy = pedagogy_context.get("tutor_directives")
    if isinstance(legacy, dict) and legacy:
        return {k: legacy.get(k) for k in TUTOR_INSTRUCTION_DIRECTIVE_KEYS}
    return {}


TUTOR_SYSTEM_PROMPT = """You are the learning tutor for K-12/college math and science sessions. The student thinks they are talking to the same assistant,
so keep tone consistent, concise, encouraging, and grounded only in the provided materials.

CRITICAL RULES:
1. NO PLAN CONFIRMATION - Start teaching immediately. Do NOT ask "ready to begin?" or similar.
2. STAY ON PLAN - Follow the current_plan exactly. Each LO has how_to_teach and why_to_teach guidance.
3. OUT-OF-PLAN DETECTION - If the student asks about something NOT in current_plan:
   - Say: "That's a different topic. Would you like to end this session and work on that instead?"
   - Set needs_topic_confirmation=true
   - Do NOT answer the off-topic question yet
   - If they confirm, end with silent_end=true and switch_topic_request set to their request
4. PROVIDED MATERIALS ONLY - Keep instruction grounded in the provided materials and teaching_pack when present. Do not invent unsupported facts, examples, numeric values, or symbolic details.

You receive:
- handoff_context: session_params containing:
  - current_plan: List of LOs to teach. Each has:
    - title, proficiency, how_to_teach, why_to_teach, notes, is_primary
  - future_plan: 1 LO for next session
  - mode: conceptual_review | examples | practice
  - subject, book, unit, chapter
  - teaching_pack: key_points, examples, practice, etc. (when present)
- tutor_instruction_directives (top-level JSON; may be {}): exactly when non-empty:
  - session_target_lo: stable session learning goal for this tutoring session
  - instruction_lo: immediate instructional focus for THIS turn (may differ from session_target_lo)
  - selected_move_type: pedagogical move the system chose
  - retrieval_intent, retrieval_action: how materials were chosen this turn (hint only)
  - policy_reason: short rationale for the move (hint only; do not quote as if the student said it)
- conversation_history: messages inside THIS tutoring session only
- image: If provided, you can see it directly

Move conditioning (mandatory when tutor_instruction_directives is non-empty):
- Precedence: If this section conflicts with "Teaching Flow" below on how much to say, or whether to ask vs explain, or what to lead with, the rules HERE win for this turn. Still respect out-of-plan detection and JSON output schema.
- Always anchor the visible teaching move to instruction_lo. Use session_target_lo when stating what the learner is ultimately working toward (especially for prerequisite work).
- Use policy_reason only as a private hint for why this move was chosen; never present it as student wording.
- Optional: pedagogy_context may include turn_progression_signals (e.g. explicit_advance_intent). When explicit_advance_intent is true and selected_move_type is NOT diagnostic_question: acknowledge the student's readiness in one short sentence, then proceed directly with teaching aligned to instruction_lo—do not ask another comprehension-check or prerequisite-check question on this turn.

When selected_move_type is diagnostic_question:
- Lead with one focused question (or at most one short setup sentence, then the question). Do not give a full explanation or worked solution before the student answers.
- Avoid solution leakage: no final answers, no "the answer is" before the student responds.
- The question must clearly target instruction_lo (or a concrete instance of it).

When selected_move_type is graduated_hint:
- Give one step or nudge toward the next student action; do not present a complete solution path.
- Prefer progressive language ("try this next", "notice that") over closing the problem.

When selected_move_type is worked_example:
- Use an example-shaped response (setup, steps, brief takeaway). When retrieval_action is reuse_pack and teaching_pack has examples or key_points, ground the example primarily in that pack text; avoid inventing unsupported numeric or symbolic details. If the pack is thin, say so briefly and stay conservative.

When selected_move_type is prereq_remediation:
- Explicitly frame instruction_lo as a prerequisite needed for session_target_lo (in student-facing language).
- Teach the prerequisite in the body of the message.
- End with one or two sentences reconnecting how this prerequisite supports or enables session_target_lo.
- Do not behave as if the session goal permanently changed to instruction_lo; session_target_lo remains the long-term lesson goal.

For any other selected_move_type:
- Follow instruction_lo and policy_reason with a short response appropriate to the label (e.g. explain_concept: concise conceptual explanation without a full exam solution unless mode clearly requires it).

Teaching Flow:
1. Start with the PRIMARY LO (is_primary=true) - dive straight into teaching
2. Use the `how_to_teach` to frame your explanation style
3. Use the `why_to_teach` to explain pedagogical connections
4. Use the `notes` field to adapt (e.g., "skip basics" if proficiency is high)
5. If there are dependent LOs (is_primary=false), cover them as needed based on student's responses
6. When wrapping up, mention what's in future_plan as a suggestion

Image Handling:
- If an image is provided, reference it directly
- Describe what you see (axes, equations, errors in handwritten work)
- Tie explanations to visible elements

Mode Switching:
- If student says "switch to practice/examples/conceptual review":
  - Set needs_mode_confirmation=true, requested_mode="..."
  - Ask: "Would you like to switch to [mode] mode?"
  - If confirmed, end silently with switch_mode_request

Completion:
- On "thanks", "I'm done", "that's all", "go back", etc.:
  - Give a 1-2 sentence recap
  - Set end_activity=true, silent_end=false
  - Do NOT continue teaching

Output STRICT JSON:
{
  "message_to_student": "...",
  "end_activity": bool,
  "silent_end": bool,
  "needs_mode_confirmation": bool,
  "needs_topic_confirmation": bool,
  "requested_mode": null or "...",
  "session_summary": {
    "topics_covered": ["LO title"],
    "student_understanding": "excellent|good|needs_practice",
    "suggested_next_topic": null or "from future_plan",
    "switch_topic_request": null or "student's off-topic request",
    "switch_mode_request": null or "switch to ..."
  }
}

Primary success metrics:
- High JSON validity and exact compliance with the required output schema
- Strong compliance with move-specific behavior for selected_move_type
- No solution leakage during diagnostic_question turns
- Consistent in-plan tutoring behavior
- Concise, encouraging, instructionally grounded responses

Never mention tools, handoffs, or internal state. Be encouraging and focused.

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
4. Completion cues (CRITICAL - these MUST trigger end_activity=true):
   - Explicit end signals: "thanks", "that helps", "I'm good", "no that's all", "I'm done", "that's enough",
     "I'm finished", "we're done", "that's all", "all done", "finished", "done", "stop", "end session",
     "no more", "nothing else", "no thanks", "no thank you"
   - Navigation/restart phrases: "go back", "back to start", "back to the start", "return to start",
     "start over", "restart", "go back to coach", "back to the coach", "return to coach"
   - When ANY of these phrases appear, IMMEDIATELY end the session:
     Give a brief closing (1-2 sentences max), end_activity=true, silent_end=false, 
     and do NOT set switch_topic_request. Do NOT continue answering after these cues.

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

_JSON_ONLY_RETRY_PROMPT = (
    "Return only valid JSON that matches the required schema. "
    "Do not include markdown, prose, or code fences."
)
_BOT_JSON_RETRY_LIMIT = 2


def tutor_bot(
    llm_client: OpenAI,
    llm_model: str,
    handoff_context: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    image: Optional[str] = None,
    on_math_guard_outcome: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Tutor bot with native multimodal support (Tier 1).

    Images are passed through to the configured OpenAI chat/vision model (see
    ``llm_model``) for multimodal understanding.

    Flow:
    1. Image → same model as text (vision-capable when an image is attached)
    2. Retrieved images from corpus → included in context with file paths
    3. Tutor sees both the student's image and relevant teaching materials
    """
    teaching_pack = (
        handoff_context.get("session_params", {})
        .get("teaching_pack", {})
    )
    retrieved_images = teaching_pack.get("images") if isinstance(teaching_pack, dict) else None

    pedagogy_ctx = (
        handoff_context.get("pedagogy_context")
        if isinstance(handoff_context, dict)
        else None
    )
    tutor_instruction_directives = extract_tutor_instruction_directives(
        pedagogy_ctx if isinstance(pedagogy_ctx, dict) else None
    )

    payload = {
        "handoff_context": handoff_context,
        "tutor_instruction_directives": tutor_instruction_directives,
        # Legacy payload key: same six fields as tutor_instruction_directives for log/API compatibility.
        "tutor_directives": tutor_instruction_directives,
        "conversation_history": conversation_history[-12:],
        "retrieved_images": retrieved_images or [],  # Includes 'path' field for student viewing
    }
    payload_text = json.dumps(payload, indent=2)
    
    # Convert image to OpenAI API format (base64 or URL)
    image_content = _image_to_content(image) if image else None
    if image_content:
        # Multimodal: use list format
        print(f"[Tutor] Image provided; including in user message: {image}")
        content_parts: List[Dict[str, Any]] = [
            image_content,
            {"type": "text", "text": payload_text}
        ]
        print("[Tutor] Multimodal message constructed with image and payload.")
        user_content = content_parts
    else:
        # Text-only: use string format (for backward compatibility)
        user_content = payload_text

    content = _request_json_content(
        llm_client=llm_client,
        llm_model=llm_model,
        system_prompt=TUTOR_SYSTEM_PROMPT,
        user_content=user_content,
        warning_label="Tutor",
    )
    if content is not None:
        normalized = _normalize_tutor_response(coerce_json(content))
        if os.getenv("WORKFLOW_DEMO_TUTOR_MATH_GUARD", "").lower() in ("1", "true", "yes"):

            def _guard_outcome(oc: Dict[str, Any]) -> None:
                pc = handoff_context.get("pedagogy_context")
                if isinstance(pc, dict):
                    pc["last_guard_result"] = {
                        "candidate_type": oc.get("candidate_type"),
                        "verified": oc.get("verified"),
                        "repaired": oc.get("repaired"),
                        "reason": oc.get("reason"),
                    }
                if on_math_guard_outcome is not None:
                    on_math_guard_outcome(oc)

            normalized = maybe_apply_math_example_guard(
                normalized,
                handoff_context,
                on_outcome=_guard_outcome,
            )
        elif isinstance(handoff_context.get("pedagogy_context"), dict):
            handoff_context["pedagogy_context"].pop("last_guard_result", None)
        return normalized

    print("[Tutor] Warning: invalid tutor JSON response after retry. Using fallback.")
    return _fallback_tutor_response(
        "I'm having trouble continuing this tutoring session. Could you try that again?"
    )


def _request_json_content(
    llm_client: OpenAI,
    llm_model: str,
    system_prompt: str,
    user_content: Any,
    warning_label: str,
) -> Optional[str]:
    """
    Request JSON output from the chat API with one retry on malformed output.

    Inputs:
        llm_client: OpenAI client instance.
        llm_model: Model name for the request.
        system_prompt: System prompt describing the JSON contract.
        user_content: Main user payload for the model.
        warning_label: Short label used in warning logs.

    Outputs:
        Raw JSON string when the model returns parseable JSON, else None.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(_BOT_JSON_RETRY_LIMIT):
        response = llm_client.chat.completions.create(
            model=llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            if attempt < _BOT_JSON_RETRY_LIMIT - 1:
                print(f"[{warning_label}] Empty JSON response; retrying once.")
                messages.append({"role": "user", "content": _JSON_ONLY_RETRY_PROMPT})
                continue
            return None

        try:
            coerce_json(content)
            return content
        except (JSONDecodeError, TypeError, ValueError) as exc:
            if attempt < _BOT_JSON_RETRY_LIMIT - 1:
                print(f"[{warning_label}] Invalid JSON response ({exc}); retrying once.")
                messages.append({"role": "user", "content": _JSON_ONLY_RETRY_PROMPT})
                continue
            return None

    return None


def _normalize_tutor_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize tutor output so required keys are always present.

    Inputs:
        response: Parsed tutor response dictionary.

    Outputs:
        Tutor-shaped response dictionary with safe defaults.
    """
    summary = response.get("session_summary")
    if not isinstance(summary, dict):
        summary = {}
    return {
        "message_to_student": response.get("message_to_student") or "",
        "end_activity": bool(response.get("end_activity", False)),
        "silent_end": bool(response.get("silent_end", False)),
        "needs_mode_confirmation": bool(
            response.get("needs_mode_confirmation", False)
        ),
        "needs_topic_confirmation": bool(
            response.get("needs_topic_confirmation", False)
        ),
        "requested_mode": response.get("requested_mode"),
        "session_summary": {
            "topics_covered": list(summary.get("topics_covered") or []),
            "student_understanding": summary.get(
                "student_understanding", "needs_practice"
            ),
            "suggested_next_topic": summary.get("suggested_next_topic"),
            "switch_topic_request": summary.get("switch_topic_request"),
            "switch_mode_request": summary.get("switch_mode_request"),
            "notes": summary.get("notes", ""),
        },
    }


def _image_to_content(image: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Convert image path/URL to OpenAI API format for native multimodal input.
    
    Returns format compatible with GPT-4o vision API:
    - URLs: {"type": "image_url", "image_url": {"url": "..."}}
    - Local files: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    """
    if not image:
        return None

    if image.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": image}}

    try:
        path = Path(image)
        if not path.exists() or not path.is_file():
            return None
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/jpeg")
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    except Exception:
        return None


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
    content = _request_json_content(
        llm_client=llm_client,
        llm_model=llm_model,
        system_prompt=FAQ_SYSTEM_PROMPT,
        user_content=json.dumps(payload, indent=2),
        warning_label="FAQ",
    )
    if content is not None:
        return _normalize_faq_response(coerce_json(content))

    print("[FAQ] Warning: invalid FAQ JSON response after retry. Using fallback.")
    return _fallback_faq_response(
        "I'm having trouble answering that right now. Could you try asking again?"
    )


def _normalize_faq_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize FAQ output so required keys are always present.

    Inputs:
        response: Parsed FAQ response dictionary.

    Outputs:
        FAQ-shaped response dictionary with safe defaults.
    """
    summary = response.get("session_summary")
    if not isinstance(summary, dict):
        summary = {}
    return {
        "message_to_student": response.get("message_to_student") or "",
        "end_activity": bool(response.get("end_activity", False)),
        "silent_end": bool(response.get("silent_end", False)),
        "needs_topic_confirmation": bool(
            response.get("needs_topic_confirmation", False)
        ),
        "session_summary": {
            "topics_addressed": list(summary.get("topics_addressed") or []),
            "questions_answered": list(summary.get("questions_answered") or []),
            "switch_topic_request": summary.get("switch_topic_request"),
            "notes": summary.get("notes", ""),
        },
    }


def _fallback_tutor_response(message: str) -> Dict[str, Any]:
    """
    Build a safe tutor response when the LLM fails to return valid JSON.

    Inputs:
        message: Fallback text shown to the student.

    Outputs:
        Tutor-shaped response dictionary that keeps the bot session alive.
    """
    return {
        "message_to_student": message,
        "end_activity": False,
        "silent_end": False,
        "needs_mode_confirmation": False,
        "needs_topic_confirmation": False,
        "requested_mode": None,
        "session_summary": {
            "topics_covered": [],
            "student_understanding": "needs_practice",
            "suggested_next_topic": None,
            "switch_topic_request": None,
            "switch_mode_request": None,
            "notes": "Fallback response due to invalid tutor JSON.",
        },
    }


def _fallback_faq_response(message: str) -> Dict[str, Any]:
    """
    Build a safe FAQ response when the LLM fails to return valid JSON.

    Inputs:
        message: Fallback text shown to the student.

    Outputs:
        FAQ-shaped response dictionary that keeps the bot session alive.
    """
    return {
        "message_to_student": message,
        "end_activity": False,
        "silent_end": False,
        "needs_topic_confirmation": False,
        "session_summary": {
            "topics_addressed": [],
            "questions_answered": [],
            "switch_topic_request": None,
            "notes": "Fallback response due to invalid FAQ JSON.",
        },
    }

