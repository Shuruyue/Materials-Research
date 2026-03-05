"""Tests for atlas.discovery.alchemy.optimizer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from atlas.discovery.alchemy.optimizer import CompositionOptimizer


class _FakeManager:
    def __init__(self):
        self.alchemical_pairs = [[(0, 38)], [(0, 20)]]
        self.alchemical_weights = torch.nn.Parameter(
            torch.tensor([0.7, 0.3], dtype=torch.float32)
        )


class _FakeAtoms:
    def __init__(self, calc):
        self.calc = calc

    def get_potential_energy(self):
        self.calc.results["alchemical_grad"] = np.array([0.5, -0.5], dtype=np.float32)
        return 1.5


class _FakeCalculator:
    def __init__(self):
        self.alchemy_manager = _FakeManager()
        self.results: dict[str, object] = {}
        self.calculate_alchemical_grad = False
        self.atoms = _FakeAtoms(self)

    def set_alchemical_weights(self, new_weights):
        tensor = torch.tensor(new_weights, dtype=torch.float32)
        with torch.no_grad():
            self.alchemy_manager.alchemical_weights.copy_(tensor)


def test_optimizer_rejects_boolean_learning_rate():
    with pytest.raises(ValueError, match="learning_rate"):
        CompositionOptimizer(_FakeCalculator(), learning_rate=True)  # type: ignore[arg-type]


def test_optimizer_step_updates_weights_with_simplex_projection():
    calc = _FakeCalculator()
    opt = CompositionOptimizer(calc, learning_rate=0.2)
    energy, new_weights = opt.step()
    assert energy == pytest.approx(1.5)
    assert np.all(np.isfinite(new_weights))
    assert np.all(new_weights >= 0.0)
    assert np.all(new_weights <= 1.0)
    assert float(np.sum(new_weights)) == pytest.approx(1.0)


def test_optimizer_run_rejects_non_integral_steps():
    calc = _FakeCalculator()
    opt = CompositionOptimizer(calc, learning_rate=0.1)
    with pytest.raises(ValueError, match="steps"):
        opt.run(steps=2.5)  # type: ignore[arg-type]
