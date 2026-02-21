"""
Tutor and FAQ LLM bots used during multi-agent handoffs.
"""

from __future__ import annotations

import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .json_utils import coerce_json


TUTOR_SYSTEM_PROMPT = """You are the learning tutor. The student thinks they are talking to the same assistant,
so keep tone consistent and grounded in the provided materials.

CRITICAL RULES:
1. NO PLAN CONFIRMATION - Start teaching immediately. Do NOT ask "ready to begin?" or similar.
2. STAY ON PLAN - Follow the current_plan exactly. Each LO has how_to_teach and why_to_teach guidance.
3. OUT-OF-PLAN DETECTION - If the student asks about something NOT in current_plan:
   - Say: "That's a different topic. Would you like to end this session and work on that instead?"
   - Set needs_topic_confirmation=true
   - Do NOT answer the off-topic question yet
   - If they confirm, end with silent_end=true and switch_topic_request set to their request

You receive:
- handoff_context: session_params containing:
  - current_plan: List of LOs to teach. Each has:
    - title, proficiency, how_to_teach, why_to_teach, notes, is_primary
  - future_plan: 1 LO for next session
  - mode: conceptual_review | examples | practice
  - subject, book, unit, chapter
- conversation_history: messages inside THIS tutoring session only
- image: If provided, you can see it directly

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


def tutor_bot(
    llm_client: OpenAI,
    llm_model: str,
    handoff_context: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    image: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tutor bot with native multimodal support (Tier 1).
    
    The image is passed directly to GPT-4o for native vision understanding.
    This allows the tutor to see exactly what the student sees and provide
    precise visual instruction without information loss from text conversion.
    
    Flow:
    1. Image → GPT-4o Vision (native, no preprocessing needed)
    2. Retrieved images from corpus → Included in context with file paths
    3. Tutor sees both student's image and relevant teaching materials
    """
    teaching_pack = (
        handoff_context.get("session_params", {})
        .get("teaching_pack", {})
    )
    retrieved_images = teaching_pack.get("images") if isinstance(teaching_pack, dict) else None

    payload = {
        "handoff_context": handoff_context,
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

    response = llm_client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        # LLM returned empty response - return a fallback
        return {"reply": "I'm sorry, I didn't catch that. Could you repeat your question?", "done": False}
    return coerce_json(content)


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
    response = llm_client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": FAQ_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
    )
    return coerce_json(response.choices[0].message.content)

