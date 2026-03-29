"""
Misconception diagnoser used on tutor turns.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .models import LearnerState, MisconceptionDiagnosis

_LOW_SIGNAL_RE = re.compile(r"^(ok|okay|thanks|thank you|got it|idk|i don't know|done|yes|no)[.! ]*$", re.IGNORECASE)
_CONFUSION_RE = re.compile(
    r"\b(confused|don't understand|do not understand|not sure|i am lost|i'm lost|can't follow|cannot follow)\b",
    re.IGNORECASE,
)
_PREREQ_GAP_RE = re.compile(
    r"\b(prerequisite|review basics|forgot|what is|before this|foundation|foundations)\b",
    re.IGNORECASE,
)
_POWER_RULE_RE = re.compile(
    r"\bderivative\b.*\bx\^?2\b.*\bis\b.*\bx\b",
    re.IGNORECASE,
)


class MisconceptionDiagnoser:
    """
    Hybrid misconception diagnoser:
    - heuristic-first (default path)
    - optional LLM (feature-flagged)
    - guaranteed fallback object
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model

    def diagnose_turn(
        self,
        session_id: str,
        user_input: str,
        current_focus_lo: Optional[str],
        learner_state: LearnerState,
        recent_messages: Optional[List[Dict[str, str]]] = None,
    ) -> MisconceptionDiagnosis:
        """Return one valid diagnosis object for the tutor turn."""
        target_lo = (current_focus_lo or learner_state.current_focus_lo or "unknown").strip() or "unknown"
        text = (user_input or "").strip()
        recent = recent_messages or []

        heuristic = self._heuristic_diagnosis(target_lo=target_lo, user_input=text, recent_messages=recent)
        if heuristic.confidence >= 0.55:
            return heuristic

        llm_diag = self._maybe_llm_diagnosis(
            session_id=session_id,
            target_lo=target_lo,
            user_input=text,
            learner_state=learner_state,
            recent_messages=recent,
        )
        if llm_diag is not None:
            return llm_diag
        return heuristic

    def _heuristic_diagnosis(
        self,
        target_lo: str,
        user_input: str,
        recent_messages: List[Dict[str, str]],
    ) -> MisconceptionDiagnosis:
        text = (user_input or "").strip()
        lowered = text.lower()
        if not text or len(text) < 6 or _LOW_SIGNAL_RE.match(text):
            return self._fallback_diagnosis(target_lo, "low-signal learner turn.")

        if _POWER_RULE_RE.search(text):
            return MisconceptionDiagnosis(
                target_lo=target_lo,
                suspected_misconception="power_rule_exponent_misapplied",
                confidence=0.88,
                rationale="Learner likely reduced exponent without multiplying by original exponent.",
                prerequisite_gap_los=["Exponent Rules", "Power Rule"],
                evidence_quotes=[text[:240]],
            )

        if _PREREQ_GAP_RE.search(lowered):
            prereqs = self._infer_prerequisite_gap_los(target_lo)
            return MisconceptionDiagnosis(
                target_lo=target_lo,
                suspected_misconception="prerequisite_gap",
                confidence=0.74,
                rationale="Learner asks for foundational refreshers before current topic progression.",
                prerequisite_gap_los=prereqs,
                evidence_quotes=[text[:240]],
            )

        if _CONFUSION_RE.search(lowered):
            return MisconceptionDiagnosis(
                target_lo=target_lo,
                suspected_misconception="conceptual_confusion",
                confidence=0.66,
                rationale="Learner explicitly reports confusion/uncertainty.",
                prerequisite_gap_los=[],
                evidence_quotes=[text[:240]],
            )

        # Mild signal from punctuation/hedging.
        if "?" in text or "maybe" in lowered or "i think" in lowered:
            return MisconceptionDiagnosis(
                target_lo=target_lo,
                suspected_misconception="uncertain_reasoning",
                confidence=0.41,
                rationale="Learner appears uncertain but without a specific misconception pattern.",
                prerequisite_gap_los=[],
                evidence_quotes=[text[:240]],
            )

        return self._fallback_diagnosis(target_lo, "no strong heuristic evidence.")

    def _maybe_llm_diagnosis(
        self,
        session_id: str,
        target_lo: str,
        user_input: str,
        learner_state: LearnerState,
        recent_messages: List[Dict[str, str]],
    ) -> Optional[MisconceptionDiagnosis]:
        if not self._llm_enabled():
            return None
        if not self.llm_client or not self.llm_model:
            return None
        if not user_input.strip():
            return None

        payload = {
            "session_id": session_id,
            "target_lo": target_lo,
            "user_input": user_input,
            "recent_messages": recent_messages[-4:],
            "learner_state": learner_state.model_dump(mode="json"),
        }
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Diagnose possible learner misconception. "
                            "Return strict JSON with fields: target_lo, suspected_misconception, "
                            "confidence (0..1), rationale, prerequisite_gap_los (string list)."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                return None
            raw = json.loads(content)
            return MisconceptionDiagnosis.model_validate(raw)
        except Exception:
            return None

    @staticmethod
    def _llm_enabled() -> bool:
        flag = os.getenv("WORKFLOW_DEMO_ENABLE_DIAGNOSIS_LLM", "")
        return flag.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _fallback_diagnosis(target_lo: str, reason: str) -> MisconceptionDiagnosis:
        return MisconceptionDiagnosis(
            target_lo=target_lo or "unknown",
            suspected_misconception="uncertain_or_low_signal",
            confidence=0.15,
            rationale=f"Fallback diagnosis: {reason}",
            prerequisite_gap_los=[],
            evidence_quotes=[],
        )

    @staticmethod
    def _infer_prerequisite_gap_los(target_lo: str) -> List[str]:
        topic = (target_lo or "").lower()
        if "deriv" in topic:
            return ["Functions", "Limits"]
        if "integral" in topic:
            return ["Derivatives", "Area interpretation"]
        if "trig" in topic:
            return ["Unit Circle", "Basic Trigonometric Identities"]
        return ["Foundational algebra"]
