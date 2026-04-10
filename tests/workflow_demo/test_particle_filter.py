"""Tests for src/workflow_demo/pedagogy/particle_filter.py (~12 tests)."""

from __future__ import annotations

import pytest

from src.workflow_demo.pedagogy.particle_filter import (
    BeliefState,
    LearnerParams,
    Particle,
    build_uniform_belief,
    deserialize_belief,
    serialize_belief,
)
from src.workflow_demo.pedagogy.pomdp_model import POMDPConstants


def _make_params_list() -> list[LearnerParams]:
    return [
        LearnerParams(c1=1.0, c2=1.0, tau=0.1),
        LearnerParams(c1=2.0, c2=2.0, tau=0.3),
        LearnerParams(c1=4.0, c2=3.0, tau=0.5),
    ]


# ------------------------------------------------------------------
# build_uniform_belief
# ------------------------------------------------------------------

class TestBuildUniformBelief:
    def test_weights_sum_to_one(self):
        belief = build_uniform_belief(_make_params_list())
        total = sum(p.weight for p in belief.particles)
        assert total == pytest.approx(1.0)

    def test_se_equals_c1_plus_c2(self):
        belief = build_uniform_belief(_make_params_list())
        for p in belief.particles:
            assert p.se_t == pytest.approx(p.params.c1 + p.params.c2)

    def test_empty_params_list(self):
        belief = build_uniform_belief([])
        assert belief.particles == []


# ------------------------------------------------------------------
# predict
# ------------------------------------------------------------------

class TestPredict:
    def test_weights_unchanged(self):
        belief = build_uniform_belief(_make_params_list())
        predicted = belief.predict(served_difficulty=1)
        for orig, pred in zip(belief.particles, predicted.particles):
            assert pred.weight == pytest.approx(orig.weight)

    def test_se_changes_with_served_difficulty(self):
        belief = build_uniform_belief(_make_params_list())
        pred_a0 = belief.predict(served_difficulty=0)
        pred_a3 = belief.predict(served_difficulty=3)
        for p0, p3 in zip(pred_a0.particles, pred_a3.particles):
            assert p0.se_t != pytest.approx(p3.se_t), (
                "Different served_difficulty should produce different SE"
            )


# ------------------------------------------------------------------
# update
# ------------------------------------------------------------------

class TestUpdate:
    def test_weights_sum_to_one_after_update(self):
        belief = build_uniform_belief(_make_params_list())
        belief = belief.predict(served_difficulty=1)
        updated = belief.update(observation_count=2, served_difficulty=1)
        total = sum(p.weight for p in updated.particles)
        assert total == pytest.approx(1.0)

    def test_different_served_difficulties_produce_different_weights(self):
        params = _make_params_list()
        b1 = build_uniform_belief(params).predict(1).update(2, served_difficulty=1)
        b2 = build_uniform_belief(params).predict(1).update(2, served_difficulty=3)
        w1 = [p.weight for p in b1.particles]
        w2 = [p.weight for p in b2.particles]
        assert w1 != w2

    def test_low_observation_shifts_toward_lower_effort(self):
        params = [
            LearnerParams(c1=1.0, c2=1.0, tau=0.5),  # low effort
            LearnerParams(c1=4.0, c2=4.0, tau=0.1),  # high effort
        ]
        belief = build_uniform_belief(params)
        belief = belief.predict(served_difficulty=1)
        updated = belief.update(observation_count=1, served_difficulty=1)
        assert updated.particles[0].weight > updated.particles[1].weight

    def test_high_observation_shifts_toward_higher_effort(self):
        params = [
            LearnerParams(c1=1.0, c2=1.0, tau=0.5),  # low effort
            LearnerParams(c1=4.0, c2=4.0, tau=0.1),  # high effort
        ]
        belief = build_uniform_belief(params)
        belief = belief.predict(served_difficulty=1)
        updated = belief.update(observation_count=10, served_difficulty=1)
        assert updated.particles[1].weight > updated.particles[0].weight


# ------------------------------------------------------------------
# Posterior summaries
# ------------------------------------------------------------------

class TestPosteriorSummaries:
    def test_expected_effort_is_weighted_mean(self):
        belief = BeliefState(particles=[
            Particle(params=LearnerParams(1, 1, 0.1), weight=0.25, se_t=2.0),
            Particle(params=LearnerParams(2, 2, 0.3), weight=0.75, se_t=6.0),
        ])
        expected = 0.25 * 2.0 + 0.75 * 6.0
        assert belief.posterior_expected_effort() == pytest.approx(expected)

    def test_active_particle_count(self):
        belief = BeliefState(particles=[
            Particle(params=LearnerParams(1, 1, 0.1), weight=0.99, se_t=2.0),
            Particle(params=LearnerParams(2, 2, 0.3), weight=1e-12, se_t=6.0),
        ])
        assert belief.active_particle_count() == 1

    def test_ess_in_range(self):
        belief = build_uniform_belief(_make_params_list())
        ess = belief.effective_sample_size()
        assert 1.0 <= ess <= len(belief.particles)

    def test_single_particle(self):
        belief = build_uniform_belief([LearnerParams(2.0, 1.5, 0.3)])
        assert belief.posterior_expected_effort() == pytest.approx(3.5)
        assert belief.active_particle_count() == 1


# ------------------------------------------------------------------
# Serialization round-trip
# ------------------------------------------------------------------

class TestSerialization:
    def test_round_trip_preserves_data(self):
        belief = build_uniform_belief(_make_params_list())
        belief = belief.predict(served_difficulty=2)
        belief = belief.update(observation_count=3, served_difficulty=2)

        data = serialize_belief(belief)
        restored = deserialize_belief(data)

        assert len(restored.particles) == len(belief.particles)
        for orig, rest in zip(belief.particles, restored.particles):
            assert rest.params.c1 == pytest.approx(orig.params.c1)
            assert rest.params.c2 == pytest.approx(orig.params.c2)
            assert rest.params.tau == pytest.approx(orig.params.tau)
            assert rest.weight == pytest.approx(orig.weight)
            assert rest.se_t == pytest.approx(orig.se_t)
