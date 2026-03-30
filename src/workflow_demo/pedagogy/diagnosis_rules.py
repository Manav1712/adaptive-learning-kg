"""
Heuristic misconception diagnosis rules and fallback behavior.

Rules run in a fixed order inside HeuristicDiagnoser.diagnose; first match wins.
Regex patterns are built from phrase tuples in diagnosis_config (no inline literals).
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Pattern, Tuple

from .diagnosis_config import (
    CONFUSION_LEXICAL_PHRASES,
    FALLBACK_CONFIDENCE,
    FALLBACK_REASON_LOW_SIGNAL,
    FALLBACK_REASON_NO_HEURISTIC_EVIDENCE,
    FALLBACK_RATIONALE_PREFIX,
    HEURISTIC_CONF_CONFUSION,
    HEURISTIC_CONF_POWER_RULE,
    HEURISTIC_CONF_PREREQ_GAP,
    HEURISTIC_CONF_UNCERTAIN_REASONING,
    HEURISTIC_LABEL_CONCEPTUAL_CONFUSION,
    HEURISTIC_LABEL_POWER_RULE,
    HEURISTIC_LABEL_PREREQUISITE_GAP,
    HEURISTIC_LABEL_UNCERTAIN_OR_LOW_SIGNAL,
    HEURISTIC_LABEL_UNCERTAIN_REASONING,
    HEURISTIC_POWER_RULE_TEXT_PATTERN,
    HEURISTIC_RATIONALE_CONFUSION,
    HEURISTIC_RATIONALE_POWER_RULE,
    HEURISTIC_RATIONALE_PREREQ_GAP,
    HEURISTIC_RATIONALE_UNCERTAIN_REASONING,
    LOW_SIGNAL_ACK_PHRASES,
    LOW_SIGNAL_MIN_CHARS,
    MAX_EVIDENCE_QUOTE_CHARS,
    POWER_RULE_PREREQ_LABELS,
    PREREQ_GAP_LEXICAL_PHRASES,
    UNCERTAIN_HEDGE_TOKENS,
    UNCERTAIN_QUESTION_MARK,
    UNKNOWN_TARGET_LO,
)
from .models import MisconceptionDiagnosis


def _alternation_from_phrases(phrases: Tuple[str, ...]) -> str:
    """Join alternatives with |, escaping each phrase for safe regex use."""
    return "|".join(re.escape(p) for p in phrases)


# Whole-line acknowledgments only (short replies that carry little diagnostic signal).
_LOW_SIGNAL_RE: Pattern[str] = re.compile(
    rf"^(?:{_alternation_from_phrases(LOW_SIGNAL_ACK_PHRASES)})[.! ]*$",
    re.IGNORECASE,
)

# Explicit confusion language; checked before generic uncertainty hedges.
_CONFUSION_RE: Pattern[str] = re.compile(
    rf"\b(?:{_alternation_from_phrases(CONFUSION_LEXICAL_PHRASES)})\b",
    re.IGNORECASE,
)

# Student signals they need foundations or a refresher before the current topic.
_PREREQ_GAP_RE: Pattern[str] = re.compile(
    rf"\b(?:{_alternation_from_phrases(PREREQ_GAP_LEXICAL_PHRASES)})\b",
    re.IGNORECASE,
)

# Demo-specific: text that looks like a classic power-rule slip on x^2.
_POWER_RULE_RE: Pattern[str] = re.compile(
    HEURISTIC_POWER_RULE_TEXT_PATTERN,
    re.IGNORECASE,
)


def make_fallback_diagnosis(target_lo: str, reason: str) -> MisconceptionDiagnosis:
    """Return a valid, low-confidence diagnosis when no stronger rule applies."""
    return MisconceptionDiagnosis(
        target_lo=target_lo or UNKNOWN_TARGET_LO,
        suspected_misconception=HEURISTIC_LABEL_UNCERTAIN_OR_LOW_SIGNAL,
        confidence=FALLBACK_CONFIDENCE,
        rationale=f"{FALLBACK_RATIONALE_PREFIX}{reason}",
        prerequisite_gap_los=[],
        evidence_quotes=[],
    )


class HeuristicDiagnoser:
    """
    Ordered rule list: earlier rules override later ones (explicit precedence).

    Example: power-rule cue is checked before generic confusion so one message
    cannot satisfy both with conflicting labels.
    """

    def __init__(self) -> None:
        self._ordered_rules: List[
            Callable[[str, str], Optional[MisconceptionDiagnosis]]
        ] = [
            self._match_low_signal,
            self._match_power_rule,
            self._match_prerequisite_gap,
            self._match_confusion,
            self._match_uncertain_reasoning,
        ]

    def diagnose(self, target_lo: str, user_input: str) -> MisconceptionDiagnosis:
        """Run rules in order; if none fire, return a generic fallback diagnosis."""
        text = (user_input or "").strip()
        for rule in self._ordered_rules:
            verdict = rule(target_lo, text)
            if verdict is not None:
                return verdict
        return make_fallback_diagnosis(target_lo, FALLBACK_REASON_NO_HEURISTIC_EVIDENCE)

    def _match_low_signal(self, target_lo: str, text: str) -> Optional[MisconceptionDiagnosis]:
        # Too short, empty, or a pure ack line → we should not over-interpret.
        if not text or len(text) < LOW_SIGNAL_MIN_CHARS or _LOW_SIGNAL_RE.match(text):
            return make_fallback_diagnosis(target_lo, FALLBACK_REASON_LOW_SIGNAL)
        return None

    def _match_power_rule(self, target_lo: str, text: str) -> Optional[MisconceptionDiagnosis]:
        if not _POWER_RULE_RE.search(text):
            return None
        return MisconceptionDiagnosis(
            target_lo=target_lo,
            suspected_misconception=HEURISTIC_LABEL_POWER_RULE,
            confidence=HEURISTIC_CONF_POWER_RULE,
            rationale=HEURISTIC_RATIONALE_POWER_RULE,
            prerequisite_gap_los=list(POWER_RULE_PREREQ_LABELS),
            evidence_quotes=_evidence_quotes(text),
        )

    def _match_prerequisite_gap(
        self, target_lo: str, text: str
    ) -> Optional[MisconceptionDiagnosis]:
        if not _PREREQ_GAP_RE.search(text.lower()):
            return None
        # Prereq LO ids/names can be filled later from the knowledge graph; label drives policy.
        return MisconceptionDiagnosis(
            target_lo=target_lo,
            suspected_misconception=HEURISTIC_LABEL_PREREQUISITE_GAP,
            confidence=HEURISTIC_CONF_PREREQ_GAP,
            rationale=HEURISTIC_RATIONALE_PREREQ_GAP,
            prerequisite_gap_los=[],
            evidence_quotes=_evidence_quotes(text),
        )

    def _match_confusion(self, target_lo: str, text: str) -> Optional[MisconceptionDiagnosis]:
        if not _CONFUSION_RE.search(text.lower()):
            return None
        return MisconceptionDiagnosis(
            target_lo=target_lo,
            suspected_misconception=HEURISTIC_LABEL_CONCEPTUAL_CONFUSION,
            confidence=HEURISTIC_CONF_CONFUSION,
            rationale=HEURISTIC_RATIONALE_CONFUSION,
            prerequisite_gap_los=[],
            evidence_quotes=_evidence_quotes(text),
        )

    def _match_uncertain_reasoning(
        self, target_lo: str, text: str
    ) -> Optional[MisconceptionDiagnosis]:
        # Weaker signal than explicit confusion: hedges and questions alone.
        lowered = text.lower()
        has_hedge = any(token in lowered for token in UNCERTAIN_HEDGE_TOKENS)
        has_question = UNCERTAIN_QUESTION_MARK in text
        if not has_question and not has_hedge:
            return None
        return MisconceptionDiagnosis(
            target_lo=target_lo,
            suspected_misconception=HEURISTIC_LABEL_UNCERTAIN_REASONING,
            confidence=HEURISTIC_CONF_UNCERTAIN_REASONING,
            rationale=HEURISTIC_RATIONALE_UNCERTAIN_REASONING,
            prerequisite_gap_los=[],
            evidence_quotes=_evidence_quotes(text),
        )


def _evidence_quotes(text: str) -> List[str]:
    """Truncate student text for evidence_quotes to keep payloads small."""
    snippet = (text or "")[:MAX_EVIDENCE_QUOTE_CHARS]
    if not snippet:
        return []
    return [snippet]
