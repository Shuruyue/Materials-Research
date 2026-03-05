"""Tests for objective-space conversion and feasibility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from atlas.active_learning.objective_space import (
    clip01,
    collect_history_objective_points,
    feasibility_mask_from_points,
    infer_objective_dimension,
    objective_points_from_map,
    objective_points_from_terms,
)


def test_clip01_nonfinite_returns_zero():
    assert clip01(float("nan")) == 0.0
    assert clip01(float("inf")) == 0.0
    assert clip01(np.array([0.6])) == 0.6


def test_objective_points_from_map_sanitizes_and_clips_values():
    c1 = SimpleNamespace()
    c2 = SimpleNamespace()
    candidates = [c1, c2]
    objective_map = {
        id(c1): np.array([1.2, np.nan, -0.7], dtype=float),
        id(c2): np.array([0.4, 0.6, np.inf], dtype=float),
    }
    out = objective_points_from_map(candidates, objective_map, obj_dim=3)
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.4, 0.6, 1.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)


def test_collect_history_objective_points_with_joint_synthesis_and_padding():
    history = [
        SimpleNamespace(topo_probability=0.9, stability_score=0.8, synthesis_score=0.3),
        SimpleNamespace(topo_probability=0.7, stability_score=0.6, synthesis_score=0.05),
        SimpleNamespace(topo_probability=0.1, stability_score=0.9, synthesis_score=0.9),
    ]
    out = collect_history_objective_points(
        history,
        obj_dim=4,
        topo_threshold=0.2,
        use_joint_synthesis=True,
        synthesis_threshold=0.1,
    )
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out[0], np.array([0.9, 0.8, 0.3, 0.0], dtype=float), rtol=0.0, atol=1e-12)


def test_feasibility_mask_from_points_handles_1d_and_nonfinite_rows():
    mask_single = feasibility_mask_from_points(np.array([0.9, 0.2, 0.3], dtype=float), topo_threshold=0.5)
    assert mask_single.shape == (1,)
    assert bool(mask_single[0]) is True

    points = np.array(
        [
            [0.8, 0.2, 0.4],
            [0.7, 0.1, np.nan],
            [0.1, 0.9, 0.9],
        ],
        dtype=float,
    )
    mask = feasibility_mask_from_points(
        points,
        topo_threshold=0.2,
        use_joint_synthesis=True,
        synthesis_threshold=0.2,
    )
    np.testing.assert_array_equal(mask, np.array([True, False, False], dtype=bool))


def test_feasibility_mask_joint_synthesis_requires_third_objective():
    points = np.array([[0.9, 0.8], [0.2, 0.9]], dtype=float)
    mask = feasibility_mask_from_points(
        points,
        topo_threshold=0.1,
        use_joint_synthesis=True,
        synthesis_threshold=0.0,
    )
    np.testing.assert_array_equal(mask, np.array([False, False], dtype=bool))


def test_infer_objective_dimension_falls_back_to_safe_default_on_bad_map():
    assert infer_objective_dimension(None, default=3) == 3
    assert infer_objective_dimension({1: "bad"}, default=2) == 2


def test_objective_points_from_terms_clips_and_truncates_to_shortest_length():
    out = objective_points_from_terms(
        topo_terms=[1.2, 0.5, 0.3],
        stability_terms=[0.4, float("nan")],
        synthesis_terms=[0.9, -0.5, 0.2],
    )
    expected = np.array([[1.0, 0.4, 0.9], [0.5, 0.0, 0.0]], dtype=float)
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-12)


def test_objective_points_from_map_non_integral_obj_dim_falls_back_to_one():
    c1 = SimpleNamespace()
    out = objective_points_from_map([c1], {id(c1): np.array([0.6, 0.4])}, obj_dim=2.5)
    assert out.shape == (1, 1)
    np.testing.assert_allclose(out[0], np.array([0.6]), rtol=0.0, atol=1e-12)


def test_collect_history_thresholds_are_clipped_to_unit_interval():
    history = [
        SimpleNamespace(topo_probability=0.2, stability_score=0.4, synthesis_score=0.5),
        SimpleNamespace(topo_probability=0.8, stability_score=0.9, synthesis_score=1.0),
    ]
    out = collect_history_objective_points(
        history,
        obj_dim=3,
        topo_threshold=-5.0,
        use_joint_synthesis=True,
        synthesis_threshold=2.0,
    )
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], np.array([0.8, 0.9, 1.0], dtype=float), rtol=0.0, atol=1e-12)


def test_feasibility_mask_thresholds_are_clipped_to_unit_interval():
    points = np.array(
        [
            [0.9, 0.4, 0.2],
            [0.7, 0.5, 0.8],
        ],
        dtype=float,
    )
    mask = feasibility_mask_from_points(
        points,
        topo_threshold=2.0,
        use_joint_synthesis=True,
        synthesis_threshold=-3.0,
    )
    np.testing.assert_array_equal(mask, np.array([False, False], dtype=bool))
