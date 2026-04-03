"""Tests for RUC search procedure (Algorithm 1, Section 3.3)."""

import numpy as np
import pytest

from ruc_ts.search.annealing import AnnealingSchedule
from ruc_ts.search.perturbation import PerturbationEngine
from ruc_ts.grammar.operators import OperatorGrammar
from ruc_ts.grammar.expression_tree import generate_random_tree


class TestAnnealingSchedule:
    def test_initial_temperature(self):
        """tau_0 = 1.0."""
        schedule = AnnealingSchedule(tau_0=1.0, total_iterations=200)
        assert schedule.temperature(0) == pytest.approx(1.0, abs=0.01)

    def test_final_temperature_zero(self):
        """tau_I -> 0 at last iteration."""
        schedule = AnnealingSchedule(tau_0=1.0, total_iterations=200)
        assert schedule.temperature(200) <= 0.01

    def test_linear_decay(self):
        """tau_t = tau_0 * (1 - t/I)."""
        schedule = AnnealingSchedule(tau_0=1.0, total_iterations=200)
        assert schedule.temperature(100) == pytest.approx(0.5, abs=0.01)

    def test_metropolis_always_accepts_improvement(self):
        schedule = AnnealingSchedule(tau_0=1.0, total_iterations=200)
        assert schedule.metropolis_accept(0.1, schedule.temperature(0))

    def test_metropolis_rejects_at_zero_temp(self):
        """At tau~0, never accept deterioration."""
        schedule = AnnealingSchedule(tau_0=1.0, total_iterations=200)
        accepted = sum(
            schedule.metropolis_accept(-0.5, 0.001) for _ in range(100)
        )
        assert accepted < 10


class TestPerturbation:
    @pytest.fixture
    def grammar(self):
        return OperatorGrammar()

    @pytest.fixture
    def tree(self, grammar):
        return generate_random_tree(grammar, ["close", "volume", "high"], seed=42)

    def test_perturb_produces_tree(self, grammar, tree):
        engine = PerturbationEngine()
        mutated = engine.perturb(tree, grammar)
        assert mutated is not None
        assert hasattr(mutated, "depth")

    def test_perturb_preserves_depth_bound(self, grammar, tree):
        engine = PerturbationEngine()
        for _ in range(20):
            mutated = engine.perturb(tree, grammar)
            assert mutated.depth() <= 4
