"""
Particle-filter belief state over learner parameters (c1, c2, tau).

Immutable-style: ``predict`` and ``update`` return new ``BeliefState``
objects so replay is side-effect-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .pomdp_model import (
    POMDPConstants,
    initial_effort,
    poisson_pmf,
    poisson_rate,
    transition,
)

_DEFAULT_CONSTANTS = POMDPConstants()


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LearnerParams:
    """Time-invariant latent parameters for one particle."""

    c1: float
    c2: float
    tau: float


@dataclass
class Particle:
    """One weighted hypothesis about the learner."""

    params: LearnerParams
    weight: float
    se_t: float


@dataclass
class BeliefState:
    """Weighted particle set representing belief over learner parameters."""

    particles: List[Particle]

    # ------------------------------------------------------------------
    # Predict / update
    # ------------------------------------------------------------------

    def predict(
        self,
        served_difficulty: int,
        constants: POMDPConstants = _DEFAULT_CONSTANTS,
    ) -> "BeliefState":
        """Deterministic SE propagation for every particle."""
        new_particles = [
            Particle(
                params=p.params,
                weight=p.weight,
                se_t=transition(
                    p.se_t,
                    p.params.c1,
                    p.params.c2,
                    p.params.tau,
                    served_difficulty,
                    constants,
                ),
            )
            for p in self.particles
        ]
        return BeliefState(particles=new_particles)

    def update(
        self,
        observation_count: int,
        served_difficulty: int,
        constants: POMDPConstants = _DEFAULT_CONSTANTS,
    ) -> "BeliefState":
        """Poisson-likelihood weighting + normalization."""
        weighted: List[Particle] = []
        for p in self.particles:
            lam = poisson_rate(p.se_t, served_difficulty, constants)
            likelihood = poisson_pmf(observation_count, lam)
            weighted.append(
                Particle(params=p.params, weight=p.weight * likelihood, se_t=p.se_t)
            )

        total = sum(pw.weight for pw in weighted)
        if total > 0:
            for pw in weighted:
                pw.weight /= total
        elif weighted:
            uniform = 1.0 / len(weighted)
            for pw in weighted:
                pw.weight = uniform

        return BeliefState(particles=weighted)

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------

    def posterior_expected_effort(self) -> float:
        if not self.particles:
            return 0.0
        return sum(p.weight * p.se_t for p in self.particles)

    def posterior_expected_tau(self) -> float:
        if not self.particles:
            return 0.0
        return sum(p.weight * p.params.tau for p in self.particles)

    def active_particle_count(self, threshold: float = 1e-8) -> int:
        return sum(1 for p in self.particles if p.weight > threshold)

    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w_i^2).  Higher means more informative belief."""
        if not self.particles:
            return 0.0
        sum_sq = sum(p.weight ** 2 for p in self.particles)
        if sum_sq == 0:
            return 0.0
        return 1.0 / sum_sq

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact summary for snapshot (no full particle list)."""
        return {
            "posterior_expected_effort": self.posterior_expected_effort(),
            "posterior_expected_tau": self.posterior_expected_tau(),
            "active_particle_count": self.active_particle_count(),
            "effective_sample_size": self.effective_sample_size(),
        }


# ------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------

def build_uniform_belief(params_list: List[LearnerParams]) -> BeliefState:
    """Create a belief with uniform weights and SE_0 = c1 + c2 per particle."""
    if not params_list:
        return BeliefState(particles=[])
    w = 1.0 / len(params_list)
    particles = [
        Particle(
            params=lp,
            weight=w,
            se_t=initial_effort(lp.c1, lp.c2),
        )
        for lp in params_list
    ]
    return BeliefState(particles=particles)


# ------------------------------------------------------------------
# Serialization helpers (sole path for belief persistence)
# ------------------------------------------------------------------

def serialize_belief(belief: BeliefState) -> List[Dict[str, float]]:
    """Serialize full belief to a list of dicts for storage."""
    return [
        {
            "c1": p.params.c1,
            "c2": p.params.c2,
            "tau": p.params.tau,
            "weight": p.weight,
            "se_t": p.se_t,
        }
        for p in belief.particles
    ]


def deserialize_belief(data: List[Dict[str, float]]) -> BeliefState:
    """Reconstruct a BeliefState from serialized particle dicts."""
    particles = [
        Particle(
            params=LearnerParams(c1=d["c1"], c2=d["c2"], tau=d["tau"]),
            weight=d["weight"],
            se_t=d["se_t"],
        )
        for d in data
    ]
    return BeliefState(particles=particles)
