"""
LLM adapter for optional misconception diagnosis refinement.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Protocol

from .diagnosis_config import (
    LLM_MAX_RECENT_MESSAGES,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    diagnosis_llm_enabled,
)
from .models import LearnerState, MisconceptionDiagnosis

LOGGER = logging.getLogger(__name__)


class _CompletionMessage(Protocol):
    content: Optional[str]


class _CompletionChoice(Protocol):
    message: _CompletionMessage


class _CompletionResponse(Protocol):
    choices: List[_CompletionChoice]


class _CompletionsAPI(Protocol):
    def create(
        self,
        *,
        model: str,
        temperature: float,
        response_format: Dict[str, str],
        messages: List[Dict[str, str]],
    ) -> _CompletionResponse:
        ...


class _ChatAPI(Protocol):
    completions: _CompletionsAPI


class SupportsChatCompletions(Protocol):
    chat: _ChatAPI


class LLMDiagnosisAdapter:
    """Adapter around chat-completions to produce MisconceptionDiagnosis."""

    def __init__(
        self,
        llm_client: Optional[SupportsChatCompletions],
        llm_model: Optional[str],
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model

    def diagnose(
        self,
        *,
        session_id: str,
        target_lo: str,
        user_input: str,
        learner_state: LearnerState,
        recent_messages: List[Dict[str, str]],
    ) -> Optional[MisconceptionDiagnosis]:
        if not diagnosis_llm_enabled():
            return None
        if not self.llm_client or not self.llm_model:
            return None
        if not user_input.strip():
            return None

        payload = {
            "session_id": session_id,
            "target_lo": target_lo,
            "user_input": user_input,
            "recent_messages": recent_messages[-LLM_MAX_RECENT_MESSAGES:],
            "learner_state": learner_state.model_dump(mode="json"),
        }

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                LOGGER.warning("Diagnosis LLM returned empty content.")
                return None
            raw = json.loads(content)
            return MisconceptionDiagnosis.model_validate(raw)
        except Exception as exc:
            LOGGER.warning("Diagnosis LLM adapter failed safely: %s", exc)
            return None
