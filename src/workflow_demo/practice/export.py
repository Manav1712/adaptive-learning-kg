"""
Optional JSON export hook for completed practice sessions (Round 4).

Feature-flagged and opt-in.  Not required for normal runtime behaviour.

Usage::

    from workflow_demo.practice.export import export_practice_session

    # After session finalization:
    export_practice_session(
        extensions=ext,
        output_path="practice_session_2025_01_15.json",
        session_id="tutor:abc",
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .replay import PracticeReplayArtifact


def export_practice_session(
    extensions: Dict[str, Any],
    *,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    learner_session_id: Optional[str] = None,
    observation_filter: Optional[Any] = None,
) -> Optional[str]:
    """Export a completed practice session to a JSON file.

    Constructs a ``PracticeReplayArtifact`` from the current extensions
    and writes it as indented JSON.

    Args:
        extensions: ``pedagogy_context["extensions"]`` dict.
        output_path: Full path for the output file.  If ``None`` and
            ``output_dir`` is set, a timestamped filename is generated.
        output_dir: Directory for auto-named output.  Ignored when
            ``output_path`` is provided.
        session_id: Optional session id for the artifact metadata.
        learner_session_id: Optional learner session id.
        observation_filter: Optional observation filter to re-derive
            observations during export.

    Returns:
        The path of the written file, or ``None`` if export was skipped
        (e.g. no completed episodes).
    """
    raw_ps = extensions.get("practice_session")
    if not isinstance(raw_ps, dict):
        return None
    completed = raw_ps.get("completed_episodes") or []
    if not completed:
        return None

    artifact = PracticeReplayArtifact.from_extensions(
        extensions,
        session_id=session_id,
        learner_session_id=learner_session_id,
        observation_filter=observation_filter,
    )

    if output_path is None:
        if output_dir is None:
            return None
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_sid = (session_id or "unknown").replace(":", "_").replace("/", "_")
        filename = f"practice_replay_{safe_sid}_{ts}.json"
        output_path = os.path.join(output_dir, filename)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, indent=2, default=str)

    return output_path
