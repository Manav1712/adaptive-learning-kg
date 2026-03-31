"""
Session-scoped pedagogical retrieval policy (Phase 5).

Step-1: pedagogical retrieval intent (why touch retrieval).
Step-2: logical action reuse_pack | augment_pack | refresh_pack.
Physical executor: retrieval_execution_mode (v1 mapping).
"""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    PedagogicalRetrievalIntent,
    RetrievalExecutionMode,
    RetrievalIntent,
    TeachingMoveType,
)
from .models import (
    LearnerState,
    MisconceptionDiagnosis,
    PolicyDecision,
    RetrievalSessionSnapshot,
    TeachingMoveCandidate,
)


class RetrievalPolicyAction(str, Enum):
    """Logical retrieval action."""

    REUSE_PACK = "reuse_pack"
    AUGMENT_PACK = "augment_pack"
    REFRESH_PACK = "refresh_pack"


class PedagogicalRetrievalOutput(BaseModel):
    """One retrieval policy decision and optional updated pack payload."""

    model_config = ConfigDict(extra="forbid")

    action: RetrievalPolicyAction
    pedagogical_retrieval_intent: PedagogicalRetrievalIntent
    retrieval_execution_mode: RetrievalExecutionMode
    legacy_retrieval_intent: Optional[RetrievalIntent] = None
    reason_codes: list[str] = Field(default_factory=list, max_length=24)
    material_triggers: list[str] = Field(default_factory=list, max_length=16)
    state: RetrievalSessionSnapshot
    teaching_pack: Optional[Dict[str, Any]] = None
    pack_delta: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    errors: list[str] = Field(default_factory=list, max_length=8)


def diagnosis_fingerprint(diagnosis: MisconceptionDiagnosis) -> str:
    """Deterministic fingerprint: normalized target, label, sorted prereqs (no rationale)."""
    target_norm = (diagnosis.target_lo or "").strip().lower()
    label = (diagnosis.suspected_misconception or "").strip()
    prereqs = ",".join(sorted((p or "").strip().lower() for p in diagnosis.prerequisite_gap_los if p))
    return f"{target_norm}|{label}|{prereqs}"


def map_action_to_execution_mode(action: RetrievalPolicyAction) -> RetrievalExecutionMode:
    """v1 mapping: logical action -> physical executor."""
    if action == RetrievalPolicyAction.REUSE_PACK:
        return RetrievalExecutionMode.NO_IO
    if action == RetrievalPolicyAction.AUGMENT_PACK:
        return RetrievalExecutionMode.CONSTRAINED_REFRESH
    return RetrievalExecutionMode.FULL_REFRESH


def decide_pedagogical_retrieval_intent(
    *,
    session_target_lo: str,
    instruction_lo: str,
    diagnosis: MisconceptionDiagnosis,
    move_type: TeachingMoveType,
) -> PedagogicalRetrievalIntent:
    """Step-1 pedagogical intent (separate from legacy RetrievalIntent)."""
    if move_type == TeachingMoveType.PREREQ_REMEDIATION:
        return PedagogicalRetrievalIntent.REPAIR_PREREQUISITE
    if move_type == TeachingMoveType.WORKED_EXAMPLE:
        return PedagogicalRetrievalIntent.RETRIEVE_WORKED_EXAMPLE

    label = (diagnosis.suspected_misconception or "").lower()
    misconceptionish = (
        "misconception" in label
        or "confusion" in label
        or label in {"conceptual_confusion", "power_rule_confusion", "prerequisite_gap"}
    )
    if move_type in (TeachingMoveType.GRADUATED_HINT, TeachingMoveType.EXPLAIN_CONCEPT) and misconceptionish:
        return PedagogicalRetrievalIntent.RETRIEVE_MISCONCEPTION_SUPPORT

    return PedagogicalRetrievalIntent.TEACH_CURRENT_CONCEPT


def _normalize_pack(pack: Any) -> Dict[str, Any]:
    if not isinstance(pack, dict):
        return {}
    return dict(pack)


def _is_empty_pack(pack: Dict[str, Any]) -> bool:
    for key in ("key_points", "examples", "practice", "prerequisites", "citations"):
        val = pack.get(key)
        if isinstance(val, list) and len(val) > 0:
            return False
    return True


def _structurally_invalid_pack(pack: Dict[str, Any]) -> bool:
    """Missing expected top-level keys entirely (empty dict is invalid for reuse)."""
    if not pack:
        return True
    return False


def _norm_lo(s: str) -> str:
    return (s or "").strip().lower()


def _pack_covers_instruction_lo(
    pack: Dict[str, Any],
    instruction_lo: str,
    pack_focus_lo: str,
) -> bool:
    """Heuristic: pack content or anchor relates to instruction_lo."""
    ins = _norm_lo(instruction_lo)
    if not ins or ins == "unknown":
        return True
    pf = _norm_lo(pack_focus_lo)
    if ins in pf or pf in ins:
        return True
    parts: List[str] = []
    for kp in pack.get("key_points") or []:
        if isinstance(kp, str):
            parts.append(kp.lower())
    for ex in pack.get("examples") or []:
        if isinstance(ex, dict):
            parts.append(str(ex.get("lo_title") or "").lower())
            parts.append(str(ex.get("snippet") or "").lower())
    blob = " ".join(parts)
    return ins in blob


def _pack_satisfies_pedagogical_intent(
    pack: Dict[str, Any],
    intent: PedagogicalRetrievalIntent,
) -> bool:
    """Artifact expectations for Step-1 intent (v1; no dedicated misconception docs)."""
    key_points = pack.get("key_points") or []
    examples = pack.get("examples") or []
    practice = pack.get("practice") or []
    has_kp = isinstance(key_points, list) and len(key_points) > 0
    has_ex = isinstance(examples, list) and len(examples) > 0
    has_practice = isinstance(practice, list) and len(practice) > 0

    if intent == PedagogicalRetrievalIntent.RETRIEVE_WORKED_EXAMPLE:
        if has_ex:
            return True
        for row in practice:
            if isinstance(row, dict) and row.get("snippet"):
                return True
        return False

    if intent == PedagogicalRetrievalIntent.REPAIR_PREREQUISITE:
        prereqs = pack.get("prerequisites") or []
        return (isinstance(prereqs, list) and len(prereqs) > 0) or has_kp

    if intent == PedagogicalRetrievalIntent.RETRIEVE_MISCONCEPTION_SUPPORT:
        return has_kp or has_ex

    # TEACH_CURRENT_CONCEPT
    return has_kp or has_ex or has_practice


def _map_move_to_intent(move_type: TeachingMoveType) -> RetrievalIntent:
    if move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
        return RetrievalIntent.MISCONCEPTION_REPAIR
    if move_type == TeachingMoveType.GRADUATED_HINT:
        return RetrievalIntent.PRACTICE_ITEM
    if move_type == TeachingMoveType.WORKED_EXAMPLE:
        return RetrievalIntent.WORKED_PARALLEL
    if move_type == TeachingMoveType.PREREQ_REMEDIATION:
        return RetrievalIntent.PREREQUISITE_REFRESH
    return RetrievalIntent.DEFINITION_SNIPPET


def _strict_material_triggers(
    *,
    session_target_lo: str,
    instruction_lo: str,
    prior_session_target_lo: Optional[str],
    prior_instruction_lo: Optional[str],
    pedagogical_intent: PedagogicalRetrievalIntent,
    current_fingerprint: str,
    prior_snapshot: Optional[RetrievalSessionSnapshot],
    pack: Dict[str, Any],
    pack_focus_lo: str,
) -> Set[str]:
    """
    Exactly five OR conditions (plan section 4). Return set of trigger ids.
    """
    triggers: Set[str] = set()
    cur_t = _norm_lo(session_target_lo)
    prev_t = _norm_lo(prior_session_target_lo) if prior_session_target_lo is not None else None
    if prev_t is not None and prev_t != cur_t:
        triggers.add("t1_session_target_changed")

    prev_i = _norm_lo(prior_instruction_lo) if prior_instruction_lo is not None else None
    cur_i = _norm_lo(instruction_lo)
    if prev_i is not None and prev_i != cur_i:
        if not _pack_covers_instruction_lo(pack, instruction_lo, pack_focus_lo):
            triggers.add("t2_instruction_unsupported_by_pack")

    if not _pack_satisfies_pedagogical_intent(pack, pedagogical_intent):
        triggers.add("t3_missing_artifact_for_intent")

    if prior_snapshot and (prior_snapshot.last_diagnosis_fingerprint or ""):
        if prior_snapshot.last_diagnosis_fingerprint != current_fingerprint:
            triggers.add("t4_fingerprint_changed")

    if _structurally_invalid_pack(pack) or _is_empty_pack(pack):
        triggers.add("t5_pack_empty_or_invalid")

    return triggers


def decide_retrieval_action(
    *,
    session_target_lo: str,
    instruction_lo: str,
    prior_session_target_lo: Optional[str],
    prior_instruction_lo: Optional[str],
    pedagogical_intent: PedagogicalRetrievalIntent,
    diagnosis: MisconceptionDiagnosis,
    prior_snapshot: Optional[RetrievalSessionSnapshot],
    teaching_pack: Dict[str, Any],
) -> Tuple[RetrievalPolicyAction, Set[str], List[str]]:
    """Pure decision: logical action + material trigger set."""
    pack = _normalize_pack(teaching_pack)
    fp = diagnosis_fingerprint(diagnosis)
    pf = (prior_snapshot.pack_focus_lo if prior_snapshot else "") or ""

    triggers = _strict_material_triggers(
        session_target_lo=session_target_lo,
        instruction_lo=instruction_lo,
        prior_session_target_lo=prior_session_target_lo,
        prior_instruction_lo=prior_instruction_lo,
        pedagogical_intent=pedagogical_intent,
        current_fingerprint=fp,
        prior_snapshot=prior_snapshot,
        pack=pack,
        pack_focus_lo=pf,
    )

    reasons = sorted(triggers)

    # Reuse only when no material triggers and pack satisfies intent (t3 absent means satisfied)
    if not triggers:
        return RetrievalPolicyAction.REUSE_PACK, triggers, reasons

    if "t5_pack_empty_or_invalid" in triggers or "t1_session_target_changed" in triggers:
        return RetrievalPolicyAction.REFRESH_PACK, triggers, reasons

    if "t2_instruction_unsupported_by_pack" in triggers:
        return RetrievalPolicyAction.REFRESH_PACK, triggers, reasons

    if "t3_missing_artifact_for_intent" in triggers or "t4_fingerprint_changed" in triggers:
        if _is_empty_pack(pack) or _structurally_invalid_pack(pack):
            return RetrievalPolicyAction.REFRESH_PACK, triggers, reasons
        return RetrievalPolicyAction.AUGMENT_PACK, triggers, reasons

    return RetrievalPolicyAction.REUSE_PACK, triggers, reasons


def _merge_augment_into_pack(
    base: Dict[str, Any],
    supplement_rows: List[Dict[str, Any]],
    target_slot: str,
) -> Dict[str, Any]:
    merged = dict(base)
    slot = merged.get(target_slot)
    if not isinstance(slot, list):
        slot = []
    seen = {row.get("content_id") for row in slot if isinstance(row, dict)}
    for row in supplement_rows:
        if not isinstance(row, dict):
            continue
        cid = row.get("content_id")
        if cid and cid in seen:
            continue
        slot.append(row)
        if cid:
            seen.add(cid)
    merged[target_slot] = slot
    return merged


def _augment_target_slot(move_type: TeachingMoveType) -> str:
    if move_type == TeachingMoveType.WORKED_EXAMPLE:
        return "examples"
    if move_type == TeachingMoveType.PREREQ_REMEDIATION:
        return "prerequisites"
    if move_type == TeachingMoveType.DIAGNOSTIC_QUESTION:
        return "citations"
    return "examples"


class PedagogicalRetrievalPolicy:
    """Applies reuse / augment / refresh using TeachingPackRetriever."""

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    def run(
        self,
        *,
        session_target_lo: str,
        instruction_lo: str,
        prior_session_target_lo: Optional[str],
        prior_instruction_lo: Optional[str],
        student_input: str,
        diagnosis: MisconceptionDiagnosis,
        policy_decision: PolicyDecision,
        learner_state: LearnerState,
        session_params: Dict[str, Any],
        prior_snapshot: Optional[RetrievalSessionSnapshot],
        image_path: Optional[str] = None,
        student_profile: Optional[Dict[str, Any]] = None,
    ) -> PedagogicalRetrievalOutput:
        _ = learner_state
        selected = policy_decision.selected_move
        pack = _normalize_pack(session_params.get("teaching_pack"))

        ped_intent = decide_pedagogical_retrieval_intent(
            session_target_lo=session_target_lo,
            instruction_lo=instruction_lo,
            diagnosis=diagnosis,
            move_type=selected.move_type,
        )

        action, material_triggers, reason_codes = decide_retrieval_action(
            session_target_lo=session_target_lo,
            instruction_lo=instruction_lo,
            prior_session_target_lo=prior_session_target_lo,
            prior_instruction_lo=prior_instruction_lo,
            pedagogical_intent=ped_intent,
            diagnosis=diagnosis,
            prior_snapshot=prior_snapshot,
            teaching_pack=pack,
        )

        legacy_intent = selected.retrieval_intent or _map_move_to_intent(selected.move_type)
        execution_mode = map_action_to_execution_mode(action)
        errors: List[str] = []
        fallback_used = False

        base_revision = prior_snapshot.pack_revision if prior_snapshot else 0
        fp = diagnosis_fingerprint(diagnosis)
        focus_for_pack = (instruction_lo or session_target_lo or "").strip()

        new_state = RetrievalSessionSnapshot(
            pack_focus_lo=focus_for_pack or (prior_snapshot.pack_focus_lo if prior_snapshot else ""),
            pack_revision=base_revision,
            last_diagnosis_fingerprint=fp,
            last_selected_move_type=selected.move_type.value,
        )

        if action == RetrievalPolicyAction.REUSE_PACK:
            new_state.pack_revision = base_revision
            return PedagogicalRetrievalOutput(
                action=action,
                pedagogical_retrieval_intent=ped_intent,
                retrieval_execution_mode=execution_mode,
                legacy_retrieval_intent=legacy_intent,
                reason_codes=reason_codes or ["reuse_pack"],
                material_triggers=sorted(material_triggers),
                state=new_state,
                teaching_pack=pack if pack else None,
                fallback_used=False,
            )

        if action == RetrievalPolicyAction.AUGMENT_PACK:
            # v1: constrained_refresh — prefer retrieve_plan over weak candidate merge
            subject = (session_params.get("subject") or "calculus").strip() or "calculus"
            mode = (session_params.get("mode") or "conceptual_review").strip() or "conceptual_review"
            profile = student_profile if isinstance(student_profile, dict) else {}
            query = " ".join(
                p
                for p in (
                    instruction_lo,
                    session_target_lo,
                    student_input[:400],
                    ped_intent.value,
                )
                if p
            ).strip() or (focus_for_pack or "tutoring")

            try:
                session_plan = self._retriever.retrieve_plan(
                    query=query,
                    subject=subject,
                    learning_objective=instruction_lo or session_params.get("learning_objective"),
                    mode=mode,
                    student_profile=profile,
                    top_los=4,
                    top_content=4,
                    enable_rerank=False,
                )
                fresh = asdict(session_plan.teaching_pack)
                new_state.pack_revision = base_revision + 1
                new_state.pack_focus_lo = focus_for_pack or new_state.pack_focus_lo
                return PedagogicalRetrievalOutput(
                    action=action,
                    pedagogical_retrieval_intent=ped_intent,
                    retrieval_execution_mode=execution_mode,
                    legacy_retrieval_intent=legacy_intent,
                    reason_codes=reason_codes or ["augment_constrained_refresh"],
                    material_triggers=sorted(material_triggers),
                    state=new_state,
                    teaching_pack=fresh,
                    fallback_used=False,
                    errors=errors,
                )
            except Exception as exc:
                errors.append(str(exc))
                # Fallback: legacy candidate merge
                target_slot = _augment_target_slot(selected.move_type)
                supplement: List[Dict[str, Any]] = []
                try:
                    result = self._retriever.retrieve_candidates(
                        text_query=query,
                        image_path=image_path,
                        top_k=3,
                        debug=False,
                    )
                    for cand in result.merged_candidates[:3]:
                        supplement.append(
                            {
                                "content_id": f"lo_{cand.lo_id}",
                                "lo_title": cand.title,
                                "content_type": "concept",
                                "snippet": (cand.how_to_teach or cand.title or "")[:400],
                                "score": round(float(cand.score), 4),
                                "source": "pedagogy_augment",
                            }
                        )
                except Exception as exc2:
                    errors.append(str(exc2))

                if supplement:
                    merged = _merge_augment_into_pack(pack, supplement, target_slot)
                    new_state.pack_revision = base_revision + 1
                    return PedagogicalRetrievalOutput(
                        action=action,
                        pedagogical_retrieval_intent=ped_intent,
                        retrieval_execution_mode=execution_mode,
                        legacy_retrieval_intent=legacy_intent,
                        reason_codes=reason_codes + ["augment_merge_fallback"],
                        material_triggers=sorted(material_triggers),
                        state=new_state,
                        teaching_pack=merged,
                        pack_delta={target_slot: supplement},
                        fallback_used=True,
                        errors=errors,
                    )

                new_state.pack_revision = base_revision
                return PedagogicalRetrievalOutput(
                    action=action,
                    pedagogical_retrieval_intent=ped_intent,
                    retrieval_execution_mode=execution_mode,
                    legacy_retrieval_intent=legacy_intent,
                    reason_codes=reason_codes + ["augment_failed"],
                    material_triggers=sorted(material_triggers),
                    state=new_state,
                    teaching_pack=pack if pack else None,
                    fallback_used=True,
                    errors=errors,
                )

        # FULL_REFRESH
        query_parts = [
            session_params.get("student_request") or "",
            student_input,
            session_target_lo,
            instruction_lo,
        ]
        query = " ".join(p for p in query_parts if p).strip() or (session_target_lo or "tutoring")
        subject = (session_params.get("subject") or "calculus").strip() or "calculus"
        mode = (session_params.get("mode") or "conceptual_review").strip() or "conceptual_review"
        profile = student_profile if isinstance(student_profile, dict) else {}

        try:
            session_plan = self._retriever.retrieve_plan(
                query=query,
                subject=subject,
                learning_objective=session_target_lo or session_params.get("learning_objective"),
                mode=mode,
                student_profile=profile,
                top_los=6,
                top_content=6,
                enable_rerank=False,
            )
            fresh_pack = asdict(session_plan.teaching_pack)
            new_state.pack_revision = base_revision + 1
            new_state.pack_focus_lo = (session_target_lo or "").strip() or new_state.pack_focus_lo
            return PedagogicalRetrievalOutput(
                action=action,
                pedagogical_retrieval_intent=ped_intent,
                retrieval_execution_mode=execution_mode,
                legacy_retrieval_intent=legacy_intent,
                reason_codes=reason_codes or ["refresh_pack"],
                material_triggers=sorted(material_triggers),
                state=new_state,
                teaching_pack=fresh_pack,
                fallback_used=False,
                errors=errors,
            )
        except Exception as exc:
            errors.append(str(exc))
            return PedagogicalRetrievalOutput(
                action=action,
                pedagogical_retrieval_intent=ped_intent,
                retrieval_execution_mode=execution_mode,
                legacy_retrieval_intent=legacy_intent,
                reason_codes=reason_codes + ["refresh_failed"],
                material_triggers=sorted(material_triggers),
                state=new_state,
                teaching_pack=pack if pack else None,
                fallback_used=True,
                errors=errors,
            )


def parse_prior_snapshot(raw: Any) -> Optional[RetrievalSessionSnapshot]:
    if raw is None:
        return None
    if isinstance(raw, RetrievalSessionSnapshot):
        return raw
    if isinstance(raw, dict):
        try:
            return RetrievalSessionSnapshot.model_validate(raw)
        except Exception:
            return None
    return None


# Backward-compatible alias
parse_prior_state = parse_prior_snapshot
