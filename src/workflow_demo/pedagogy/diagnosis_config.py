"""
Configuration constants for misconception diagnosis.

Heuristic strings and scores live here so rules stay data-driven and auditable.
"""

from __future__ import annotations

import os
from typing import Final, Tuple

# --- Orchestration (MisconceptionDiagnoser) ---
HEURISTIC_ACCEPT_CONFIDENCE: Final[float] = 0.55
FALLBACK_CONFIDENCE: Final[float] = 0.15
LOW_SIGNAL_MIN_CHARS: Final[int] = 6
MAX_EVIDENCE_QUOTE_CHARS: Final[int] = 240

UNKNOWN_TARGET_LO: Final[str] = "unknown"
FALLBACK_RATIONALE_PREFIX: Final[str] = "Fallback diagnosis: "

# --- LLM adapter ---
LLM_MAX_RECENT_MESSAGES: Final[int] = 4
LLM_TEMPERATURE: Final[float] = 0.0
LLM_ENABLE_ENV_VAR: Final[str] = "WORKFLOW_DEMO_ENABLE_DIAGNOSIS_LLM"

LLM_SYSTEM_PROMPT: Final[str] = (
    "Diagnose possible learner misconception. "
    "Return strict JSON with fields: target_lo, suspected_misconception, "
    "confidence (0..1), rationale, prerequisite_gap_los (string list)."
)


def diagnosis_llm_enabled() -> bool:
    """Return True when the diagnosis LLM path is explicitly enabled."""
    flag = os.getenv(LLM_ENABLE_ENV_VAR, "")
    return flag.lower() in {"1", "true", "yes", "on"}


# --- Stable misconception labels (downstream teaching moves / policy) ---
HEURISTIC_LABEL_UNCERTAIN_OR_LOW_SIGNAL: Final[str] = "uncertain_or_low_signal"
HEURISTIC_LABEL_POWER_RULE: Final[str] = "power_rule_exponent_misapplied"
HEURISTIC_LABEL_PREREQUISITE_GAP: Final[str] = "prerequisite_gap"
HEURISTIC_LABEL_CONCEPTUAL_CONFUSION: Final[str] = "conceptual_confusion"
HEURISTIC_LABEL_UNCERTAIN_REASONING: Final[str] = "uncertain_reasoning"

# --- Per-rule confidence scores (within [0.0, 1.0]) ---
HEURISTIC_CONF_POWER_RULE: Final[float] = 0.88
HEURISTIC_CONF_PREREQ_GAP: Final[float] = 0.74
HEURISTIC_CONF_CONFUSION: Final[float] = 0.66
HEURISTIC_CONF_UNCERTAIN_REASONING: Final[float] = 0.41

# --- Human-readable rationales (display / logs; not policy math) ---
HEURISTIC_RATIONALE_POWER_RULE: Final[str] = (
    "Learner likely reduced exponent without multiplying by original exponent."
)
HEURISTIC_RATIONALE_PREREQ_GAP: Final[str] = (
    "Learner asks for foundational refreshers before current topic progression."
)
HEURISTIC_RATIONALE_CONFUSION: Final[str] = (
    "Learner explicitly reports confusion/uncertainty."
)
HEURISTIC_RATIONALE_UNCERTAIN_REASONING: Final[str] = (
    "Learner appears uncertain but without a specific misconception pattern."
)

# --- Evidence snippets bundled with a rule hit ---
POWER_RULE_PREREQ_LABELS: Final[Tuple[str, ...]] = ("Exponent Rules", "Power Rule")

# --- Fallback reason fragments (appended after "Fallback diagnosis: ") ---
FALLBACK_REASON_LOW_SIGNAL: Final[str] = "low-signal learner turn."
FALLBACK_REASON_NO_HEURISTIC_EVIDENCE: Final[str] = "no strong heuristic evidence."

# Whole-line acknowledgments: message must match this set only (plus punctuation).
LOW_SIGNAL_ACK_PHRASES: Final[Tuple[str, ...]] = (
    "ok",
    "okay",
    "thanks",
    "thank you",
    "got it",
    "idk",
    "i don't know",
    "done",
    "yes",
    "no",
)

# Lexical cues for explicit confusion (word-boundary match).
CONFUSION_LEXICAL_PHRASES: Final[Tuple[str, ...]] = (
    "confused",
    "don't understand",
    "do not understand",
    "not sure",
    "i am lost",
    "i'm lost",
    "can't follow",
    "cannot follow",
)

# Lexical cues that the student is asking for foundations / review.
PREREQ_GAP_LEXICAL_PHRASES: Final[Tuple[str, ...]] = (
    "prerequisite",
    "review basics",
    "forgot",
    "what is",
    "before this",
    "foundation",
    "foundations",
)

# Domain-specific typo pattern: claims d/dx(x^2) simplifies to x (demo heuristic).
HEURISTIC_POWER_RULE_TEXT_PATTERN: Final[str] = (
    r"\bderivative\b.*\bx\^?2\b.*\bis\b.*\bx\b"
)

# --- Uncertain-reasoning rule: punctuation + hedging tokens ---
UNCERTAIN_HEDGE_TOKENS: Final[Tuple[str, ...]] = ("maybe", "i think")
UNCERTAIN_QUESTION_MARK: Final[str] = "?"
