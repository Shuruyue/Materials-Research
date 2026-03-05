"""Unit tests for phase6 active-learning helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.phase5_active_learning.active_learning import (
    ActiveLearningLoop,
    acquisition_expected_improvement,
    acquisition_random,
    acquisition_uncertainty,
    multi_objective_screening,
    pareto_frontier,
)


class _DummyModel:
    def __init__(self, **_kwargs):
        pass


def test_acquisition_uncertainty_handles_non_finite_std():
    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([[1.0, np.nan], [np.inf, 0.2], [0.1, 0.3]])
    selected = acquisition_uncertainty(mean, std, batch_size=2)
    assert selected.tolist() == [0, 2]


def test_acquisition_uncertainty_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="leading dimension"):
        acquisition_uncertainty(
            mean=np.array([0.1, 0.2]),
            std=np.array([0.1, 0.2, 0.3]),
            batch_size=1,
        )
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        acquisition_uncertainty(
            mean=np.array([0.1, 0.2]),
            std=np.array([0.1, 0.2]),
            batch_size=1.5,
        )


def test_acquisition_expected_improvement_handles_non_finite_std():
    mean = np.array([-0.2, -0.1, -0.05, 0.3])
    std = np.array([0.1, np.nan, np.inf, 0.0])
    selected = acquisition_expected_improvement(mean, std, best_so_far=-0.15, batch_size=2, maximize=False)
    assert selected.shape == (2,)
    assert np.all((selected >= 0) & (selected < len(mean)))


def test_acquisition_expected_improvement_rejects_non_finite_best():
    with pytest.raises(ValueError, match="best_so_far"):
        acquisition_expected_improvement(
            mean=np.array([0.1, 0.2]),
            std=np.array([0.1, 0.2]),
            best_so_far=float("nan"),
        )


def test_acquisition_random_validates_bounds():
    assert acquisition_random(0, batch_size=3).size == 0
    assert acquisition_random(5, batch_size=0).size == 0
    with pytest.raises(ValueError, match="n_pool"):
        acquisition_random(-1, batch_size=1)
    with pytest.raises(ValueError, match="n_pool"):
        acquisition_random(3.5, batch_size=1)
    with pytest.raises(ValueError, match="n_pool"):
        acquisition_random(True, batch_size=1)
    with pytest.raises(ValueError, match="batch_size"):
        acquisition_random(5, batch_size=-1)
    with pytest.raises(ValueError, match="seed"):
        acquisition_random(5, batch_size=1, seed=-3)
    with pytest.raises(ValueError, match="seed"):
        acquisition_random(5, batch_size=1, seed=1.25)
    with pytest.raises(ValueError, match="seed"):
        acquisition_random(5, batch_size=1, seed=True)


def test_pareto_frontier_filters_invalid_rows_and_preserves_indices():
    objectives = np.array(
        [
            [1.0, 1.0],
            [np.nan, 5.0],
            [2.0, 0.5],
            [0.5, 2.0],
        ]
    )
    idx = pareto_frontier(objectives, maximize=[True, True])
    assert set(idx.tolist()) == {0, 2, 3}


def test_pareto_frontier_rejects_invalid_maximize_shape():
    objectives = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="maximize length"):
        pareto_frontier(objectives, maximize=[True])


def test_active_learning_loop_validates_strategy_and_budgets():
    with pytest.raises(ValueError, match="Unsupported strategy"):
        ActiveLearningLoop(_DummyModel, {}, strategy="bad")
    with pytest.raises(ValueError, match="query_budget"):
        ActiveLearningLoop(_DummyModel, {}, query_budget=0)
    with pytest.raises(ValueError, match="initial_budget"):
        ActiveLearningLoop(_DummyModel, {}, initial_budget=0)
    with pytest.raises(ValueError, match="positive integer"):
        ActiveLearningLoop(_DummyModel, {}, query_budget=1.2)


def test_active_learning_loop_run_resets_history_on_empty_dataset():
    loop = ActiveLearningLoop(_DummyModel, {}, initial_budget=10, query_budget=5, n_iterations=2)
    loop.history["iteration"].append(99)
    history = loop.run([], None, None, device="cpu")
    assert history["iteration"] == []
    assert history["n_train"] == []
    assert history["selected_indices"] == []


def test_multi_objective_screening_rejects_non_integral_top_k():
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        multi_objective_screening(
            model=None,
            loader=[],
            objectives={"bulk_modulus": "high"},
            device="cpu",
            top_k=2.5,
        )
