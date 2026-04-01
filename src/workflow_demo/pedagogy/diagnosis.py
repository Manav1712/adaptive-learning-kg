"""
Misconception diagnoser used on tutor turns.
"""

from typing import Dict, List, Optional

from .diagnosis_config import HEURISTIC_ACCEPT_CONFIDENCE
from .diagnosis_llm import LLMDiagnosisAdapter, SupportsChatCompletions
from .diagnosis_rules import HeuristicDiagnoser
from .models import LearnerState, MisconceptionDiagnosis


class MisconceptionDiagnoser:
    """
    Hybrid misconception diagnoser:
    - heuristic-first (default path)
    - optional LLM (feature-flagged)
    - guaranteed fallback object
    """

    def __init__(
        self,
        llm_client: Optional[SupportsChatCompletions] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self._heuristics = HeuristicDiagnoser()
        self._llm_adapter = LLMDiagnosisAdapter(llm_client=llm_client, llm_model=llm_model)

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

        heuristic = self._heuristics.diagnose(target_lo=target_lo, user_input=text)
        if heuristic.confidence >= HEURISTIC_ACCEPT_CONFIDENCE:
            return heuristic

        llm_diag = self._llm_adapter.diagnose(
            session_id=session_id,
            target_lo=target_lo,
            user_input=text,
            learner_state=learner_state,
            recent_messages=recent,
        )
        if llm_diag is not None:
            return llm_diag
        return heuristic
