"""
POMDP calibration / initialization stubs.

Provides particle bank construction for the POMDP sequencer:
- ``default_particle_bank``: deterministic 27-particle 3x3x3 grid (fallback)
- ``load_particle_bank_json``: load from a JSON file
- ``particle_bank_from_replay``: heuristic stub for future learned initialization
- ``ParticleBankProvider``: protocol for pluggable bank sources
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Protocol, runtime_checkable

from .particle_filter import LearnerParams

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Default particle bank — deterministic 3x3x3 grid (27 particles)
# ------------------------------------------------------------------

_C1_VALUES = [1.0, 2.5, 4.0]
_C2_VALUES = [1.0, 2.5, 4.0]
_TAU_VALUES = [0.1, 0.3, 0.5]


def default_particle_bank() -> List[LearnerParams]:
    """Return a deterministic 27-particle grid across (c1, c2, tau) domain."""
    return [
        LearnerParams(c1=c1, c2=c2, tau=tau)
        for c1 in _C1_VALUES
        for c2 in _C2_VALUES
        for tau in _TAU_VALUES
    ]


# ------------------------------------------------------------------
# JSON loader
# ------------------------------------------------------------------

def load_particle_bank_json(path: str) -> List[LearnerParams]:
    """Load a particle bank from a JSON file.

    Expected format: ``[{"c1": ..., "c2": ..., "tau": ...}, ...]``
    Returns ``default_particle_bank()`` if loading fails or result is empty.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            logger.warning("Particle bank JSON at %s is empty or not a list; using fallback.", path)
            return default_particle_bank()
        return [
            LearnerParams(c1=float(d["c1"]), c2=float(d["c2"]), tau=float(d["tau"]))
            for d in data
        ]
    except Exception:
        logger.warning("Failed to load particle bank from %s; using fallback.", path, exc_info=True)
        return default_particle_bank()


# ------------------------------------------------------------------
# Replay-based stub
# ------------------------------------------------------------------

def particle_bank_from_replay(artifact: Any) -> List[LearnerParams]:
    """Heuristic stub: extract a rough prior from replay data.

    This is a placeholder for future GBM/learned initialization.
    Currently returns the default bank regardless of input.
    """
    logger.info("particle_bank_from_replay is a stub; returning default bank.")
    return default_particle_bank()


# ------------------------------------------------------------------
# Provider protocol
# ------------------------------------------------------------------

@runtime_checkable
class ParticleBankProvider(Protocol):
    """Pluggable interface for particle bank sources."""

    def get_bank(self) -> List[LearnerParams]: ...
