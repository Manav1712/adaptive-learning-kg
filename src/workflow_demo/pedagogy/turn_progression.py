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
    "let's keep going",
    "lets keep going",
    "let's keep moving",
    "lets keep moving",
    "can we continue",
    "can we move on",
    "can we move forward",
    "move forward",
    "lets move forward",
    "let's move forward",
    "keep going",
    "keep moving",
    "okay let's keep",
    "okay, let's keep",
    "i get it",
    "i already know this",
)

# Learner asks for a concrete example or practice (substring, case-insensitive).
_EXAMPLE_REQUEST_PHRASES: tuple[str, ...] = (
    "give me an example",
    "show me an example",
    "can you show me",
    "through an example",
    "walk me through an example",
    "example problem",
    "practice problem",
    "practice problems",
    "practice question",
    "worked example",
    "worked problem",
    "solved example",
    "solved examples",
    "some solved examples",
    "do some solved examples",
    "can i try one",
    "let me try",
    "give me a problem",
    "show me how",
)

_UNDERSTANDING_CONFIDENCE_PHRASES: tuple[str, ...] = (
    "this makes sense",
    "that makes sense",
    "it makes sense",
    "makes sense now",
    "i understand",
    "i understand now",
    "i understand what you mean",
    "i understand that",
    "i get it",
    "i see",
    "i see what you mean",
    "that helps",
    "its clear now",
    "it's clear now",
    "it clicks now",
    "yeah i understand",
    "okay i understand",
    "yes i understand",
    "oh i see",
    "ah i see",
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

# v1 substantive math attempt: equality, expression-shaped tokens, digit + math cue, or numeric-answer phrasing.
_MATH_ISH_CUE_RE = re.compile(r"[\+\-\*/^=()]|\*\*")
_NUMERIC_ANSWER_PHRASE_RE = re.compile(
    r"(?:"
    r"\b(?:the\s+)?(?:answer|result)\s*(?:is|are|=)\s*[-+]?\d"
    r"|\b(?:is|are|equals?)\s+[-+]?\d+\b"
    r")",
    re.IGNORECASE,
)
# Expression-like tokens (not a full math parser): e.g. 2x, x+1, 3/4, ^2, (a+b).
_MATH_EXPRESSION_TOKEN_RE = re.compile(
    r"(?:"
    r"(?<![\w.])\d+[a-zA-Z](?![a-zA-Z])"  # coefficient-variable: 2x, 3n
    r"|[a-zA-Z]\s*[\+\-\*/=^]\s*[\d.a-zA-Z]"  # x=5, n+1
    r"|[\d.a-zA-Z]\s*[\+\-\*/^]\s*[\d.a-zA-Z]"  # 2+3, x*2 (requires op between tokens)
    r"|\^\s*\d|\*\*"  # exponent markers
    r"|\d+\s*/\s*\d+"  # simple fraction
    r"|\([^)]{0,48}[\+\-\*/][^)]{0,48}\)"  # parentheses with an operator inside
    r")",
)


@dataclass(frozen=True)
class TurnProgressionSignals:
    explicit_advance_intent: bool
    adequate_check_response: bool
    current_confusion_signal: bool
    short_low_signal_ack: bool
    learner_requested_example: bool
    substantive_answer_attempt: bool
    suppress_repeat_diagnostic: bool

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def matches_short_low_signal_ack(user_input: str) -> bool:
    """True for empty, very short, or whole-line ack-only messages (like diagnosis low-signal)."""
    text = (user_input or "").strip()
    if not text:
        return True
    lower = text.lower()
    if matches_explicit_advance_intent(text):
        return False
    if matches_learner_requested_example(text):
        return False
    if any(p in lower for p in _UNDERSTANDING_CONFIDENCE_PHRASES):
        return False
    if len(text) < LOW_SIGNAL_MIN_CHARS:
        # Do not treat short math replies (e.g. "x = 7", "2x") as low-signal acks.
        if "=" in text:
            return False
        if _MATH_EXPRESSION_TOKEN_RE.search(text):
            return False
        if _NUMERIC_ANSWER_PHRASE_RE.search(text):
            return False
        if re.search(r"\d", text) and _MATH_ISH_CUE_RE.search(text):
            return False
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


def matches_learner_requested_example(user_input: str) -> bool:
    lower = (user_input or "").lower()
    return any(p in lower for p in _EXAMPLE_REQUEST_PHRASES)


def _looks_like_substantive_math_attempt(text: str) -> bool:
    """Heuristic math substance: not grading, only loop-closure signal."""
    if "=" in text:
        return True
    if _NUMERIC_ANSWER_PHRASE_RE.search(text):
        return True
    if _MATH_EXPRESSION_TOKEN_RE.search(text):
        return True
    if re.search(r"\d", text) and _MATH_ISH_CUE_RE.search(text):
        return True
    return False


_PRIOR_MOVES_SUBSTANTIVE_REPLY: frozenset[str] = frozenset(
    {
        TeachingMoveType.DIAGNOSTIC_QUESTION.value,
        TeachingMoveType.WORKED_EXAMPLE.value,
        TeachingMoveType.GRADUATED_HINT.value,
        TeachingMoveType.EXPLAIN_CONCEPT.value,
    }
)


def matches_substantive_answer_attempt(
    user_input: str,
    previous_last_selected_move_type: Optional[str],
) -> bool:
    """
    True when the prior tutor move was a teaching move and the reply shows real engagement.

    After diagnostic_question: concrete math attempt (existing behavior).
    After worked_example / graduated_hint / explain_concept: math attempt, substance tokens,
    or a non-trivial text reply (so bridge turns still get anti-loop signals).
    Does not encode correctness.
    """
    prior = (previous_last_selected_move_type or "").strip()
    if prior not in _PRIOR_MOVES_SUBSTANTIVE_REPLY:
        return False
    text = (user_input or "").strip()
    if matches_short_low_signal_ack(text):
        return False

    if prior == TeachingMoveType.DIAGNOSTIC_QUESTION.value:
        if len(text) < LOW_SIGNAL_MIN_CHARS and "=" not in text:
            return False
        return _looks_like_substantive_math_attempt(text)

    if _looks_like_substantive_math_attempt(text):
        return True
    lower = text.lower()
    if any(tok in lower for tok in _SUBSTANCE_TOKENS):
        return True
    if len(text) >= 12:
        return True
    return False


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
    lower = text.lower()
    if any(p in lower for p in _UNDERSTANDING_CONFIDENCE_PHRASES):
        return True
    if len(text) < _ADEQUATE_MIN_CHARS and not any(
        tok in lower for tok in _SUBSTANCE_TOKENS
    ):
        return False
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

    suppress_repeat_diagnostic is True when:
      - previous move was diagnostic_question and the reply is not confused and not a
        pure short ack (any engaged reply ends the immediate re-check), or
      - previous move was a bridge/teach move (worked_example, graduated_hint,
        explain_concept) and the learner shows engagement (explicit, adequate,
        substantive, example request, or a non-trivial message).

    Rationale:
      Bridge turns used to drop all anti-loop gates; this keeps the next turn from
      snapping back to another broad check on the same LO subidea.
    """
    text = user_input or ""
    explicit = matches_explicit_advance_intent(text)
    confusion = matches_current_confusion_signal(text)
    short_ack = matches_short_low_signal_ack(text)
    adequate = matches_adequate_check_response(text)
    example_req = matches_learner_requested_example(text)
    substantive = matches_substantive_answer_attempt(text, previous_last_selected_move_type)

    prior = (previous_last_selected_move_type or "").strip()
    prior_diag = prior == TeachingMoveType.DIAGNOSTIC_QUESTION.value
    prior_bridge = prior in {
        TeachingMoveType.WORKED_EXAMPLE.value,
        TeachingMoveType.GRADUATED_HINT.value,
        TeachingMoveType.EXPLAIN_CONCEPT.value,
    }

    suppress_diag = prior_diag and not confusion and not short_ack
    engaged_after_bridge = (
        explicit
        or adequate
        or substantive
        or example_req
        or (prior_bridge and len(text.strip()) >= 12)
    )
    suppress_bridge = prior_bridge and not confusion and not short_ack and engaged_after_bridge

    suppress = suppress_diag or suppress_bridge

    return TurnProgressionSignals(
        explicit_advance_intent=explicit,
        adequate_check_response=adequate,
        current_confusion_signal=confusion,
        short_low_signal_ack=short_ack,
        learner_requested_example=example_req,
        substantive_answer_attempt=substantive,
        suppress_repeat_diagnostic=suppress,
    )
