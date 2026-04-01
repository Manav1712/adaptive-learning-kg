"""
Turn-level progression signals to avoid repeated diagnostic_question loops.

See tutor diagnostic loop plan: explicit gate for suppress_repeat_diagnostic.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .constants import TeachingMoveType
from .diagnosis_config import (
    CONFUSION_LEXICAL_PHRASES,
    LOW_SIGNAL_ACK_PHRASES,
    LOW_SIGNAL_MIN_CHARS,
)

# Student explicitly wants to proceed (substring match, case-insensitive).
_ADVANCE_PHRASES: tuple[str, ...] = (
    "assume i know",
    "assume we know",
    "move on",
    "let's move on",
    "lets move on",
    "let us move on",
    "continue with",
    "let's continue",
    "lets continue",
    "let us continue",
    "go back to",
    "return to",
    "skip this",
    "skip it",
    "i remember",
    "already know",
    "stop asking",
    "don't ask",
    "do not ask",
    "next topic",
    "proceed with",
    "get back to",
    "back to integration",
    "back to the topic",
)

# Reasoning / substance cues for adequate_check_response (any one helps).
_SUBSTANCE_TOKENS: tuple[str, ...] = (
    "because",
    "therefore",
    "horizontal",
    "vertical",
    "base",
    "height",
    "triangle",
    "area",
    "curve",
    "sum",
    "riemann",
    "integral",
    "derivative",
    "limit",
)

_LOW_SIGNAL_ACK_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in LOW_SIGNAL_ACK_PHRASES) + r")[.! ]*$",
    re.IGNORECASE,
)

_CONFUSION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in CONFUSION_LEXICAL_PHRASES) + r")\b",
    re.IGNORECASE,
)

# Minimum length for a message to count as substantive adequate response (beyond short ack).
_ADEQUATE_MIN_CHARS = 40


@dataclass(frozen=True)
class TurnProgressionSignals:
    explicit_advance_intent: bool
    adequate_check_response: bool
    current_confusion_signal: bool
    short_low_signal_ack: bool
    suppress_repeat_diagnostic: bool

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def matches_short_low_signal_ack(user_input: str) -> bool:
    """True for empty, very short, or whole-line ack-only messages (like diagnosis low-signal)."""
    text = (user_input or "").strip()
    if not text:
        return True
    if len(text) < LOW_SIGNAL_MIN_CHARS:
        return True
    if _LOW_SIGNAL_ACK_RE.match(text):
        return True
    return False


def matches_current_confusion_signal(user_input: str) -> bool:
    """Alignment with policy 'stuck' confusion cues on this turn."""
    lower = (user_input or "").lower()
    if _CONFUSION_RE.search(lower):
        return True
    if "confused" in lower:
        return True
    if "don't understand" in lower or "do not understand" in lower:
        return True
    return False


def matches_explicit_advance_intent(user_input: str) -> bool:
    lower = (user_input or "").lower()
    return any(p in lower for p in _ADVANCE_PHRASES)


def matches_adequate_check_response(user_input: str) -> bool:
    """
    Heuristic: substantive enough to treat as passing a check — not correctness proof.
    Requires length, no short-ack, no confusion, and at least one substance signal OR long text.
    """
    text = (user_input or "").strip()
    if matches_short_low_signal_ack(text):
        return False
    if matches_current_confusion_signal(text):
        return False
    if len(text) < _ADEQUATE_MIN_CHARS and not any(
        tok in text.lower() for tok in _SUBSTANCE_TOKENS
    ):
        return False
    lower = text.lower()
    if len(text) >= _ADEQUATE_MIN_CHARS:
        return True
    return any(tok in lower for tok in _SUBSTANCE_TOKENS)


def compute_turn_progression_signals(
    *,
    user_input: str,
    previous_last_selected_move_type: Optional[str],
) -> TurnProgressionSignals:
    """
    Compute flags and the final suppress_repeat_diagnostic gate.

    suppress_repeat_diagnostic is True iff:
      previous_last_selected_move_type == diagnostic_question
      and (explicit_advance_intent or adequate_check_response)
      and not current_confusion_signal
      and not short_low_signal_ack
    """
    text = user_input or ""
    explicit = matches_explicit_advance_intent(text)
    confusion = matches_current_confusion_signal(text)
    short_ack = matches_short_low_signal_ack(text)
    adequate = matches_adequate_check_response(text)

    prior_diag = (previous_last_selected_move_type or "").strip() == TeachingMoveType.DIAGNOSTIC_QUESTION.value

    suppress = (
        prior_diag
        and (explicit or adequate)
        and not confusion
        and not short_ack
    )

    return TurnProgressionSignals(
        explicit_advance_intent=explicit,
        adequate_check_response=adequate,
        current_confusion_signal=confusion,
        short_low_signal_ack=short_ack,
        suppress_repeat_diagnostic=suppress,
    )
