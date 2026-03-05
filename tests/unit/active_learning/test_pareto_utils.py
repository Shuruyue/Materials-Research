"""Tests for Pareto/hypervolume utilities used by active-learning selection."""

import numpy as np

from atlas.active_learning.pareto_utils import (
    hypervolume,
    hypervolume_2d,
    mc_hv_improvements_shared,
    non_dominated_sort,
    pareto_front,
)


def _bruteforce_front_ranks(points: np.ndarray) -> dict[int, int]:
    remaining = list(range(points.shape[0]))
    ranks: dict[int, int] = {}
    rank = 0
    while remaining:
        front: list[int] = []
        for idx in remaining:
            p = points[idx]
            dominated = False
            for jdx in remaining:
                if jdx == idx:
                    continue
                q = points[jdx]
                if np.all(q >= p) and np.any(q > p):
                    dominated = True
                    break
            if not dominated:
                front.append(idx)
        for idx in front:
            ranks[idx] = rank
        remaining = [idx for idx in remaining if idx not in front]
        rank += 1
    return ranks


def test_non_dominated_sort_matches_bruteforce_ranking():
    rng = np.random.default_rng(11)
    points = rng.random((28, 3))

    fronts = non_dominated_sort(points)
    got: dict[int, int] = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            got[int(idx)] = rank

    expected = _bruteforce_front_ranks(points)
    assert got == expected


def test_hypervolume_2d_exact_on_known_configuration():
    # Two complementary points above reference create a known union area.
    points = np.array(
        [
            [1.0, 0.2],
            [0.2, 1.0],
        ],
        dtype=float,
    )
    ref = np.array([0.0, 0.0], dtype=float)
    hv = hypervolume_2d(points, ref)
    np.testing.assert_allclose(hv, 0.36, atol=1e-10, rtol=0.0)


def test_shared_hv_improvement_matches_exact_2d_deltas():
    hist = np.array([[0.3, 0.3]], dtype=float)
    cand = np.array(
        [
            [0.9, 0.4],
            [0.4, 0.9],
            [0.2, 0.2],
        ],
        dtype=float,
    )
    ref = np.array([0.0, 0.0], dtype=float)

    shared = mc_hv_improvements_shared(
        hist,
        cand,
        ref,
        feas_threshold=0.0,
        hv_mc_samples=4000,
        hv_mc_seed=5,
        iteration=1,
        hv_chunk_size=128,
    )

    baseline = hypervolume(hist, ref, hv_mc_samples=4000, hv_mc_seed=5, iteration=1, hv_chunk_size=128)
    direct = np.zeros(cand.shape[0], dtype=float)
    for idx, point in enumerate(cand):
        augmented = np.vstack([hist, point.reshape(1, -1)])
        hv = hypervolume(augmented, ref, hv_mc_samples=4000, hv_mc_seed=5, iteration=1, hv_chunk_size=128)
        direct[idx] = max(0.0, hv - baseline)

    np.testing.assert_allclose(shared, direct, rtol=0.0, atol=1e-12)


def test_shared_hv_improvement_3d_respects_feasibility_threshold():
    hist = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.3],
        ],
        dtype=float,
    )
    cand = np.array(
        [
            [0.9, 0.4, 0.4],
            [0.4, 0.9, 0.5],
            [0.1, 0.9, 0.9],  # infeasible by topo threshold
        ],
        dtype=float,
    )
    ref = np.array([0.0, 0.0, 0.0], dtype=float)

    improvements = mc_hv_improvements_shared(
        hist,
        cand,
        ref,
        feas_threshold=0.2,
        hv_mc_samples=12000,
        hv_mc_seed=17,
        iteration=2,
        hv_chunk_size=256,
    )

    assert improvements[0] > 0.0
    assert improvements[1] > 0.0
    assert improvements[2] == 0.0


def test_pareto_front_empty_preserves_dimension():
    points = np.zeros((0, 3), dtype=float)
    front = pareto_front(points)
    assert front.shape == (0, 3)


def test_non_dominated_sort_treats_nonfinite_points_as_worst():
    points = np.array(
        [
            [0.9, 0.9],
            [0.8, 0.8],
            [np.nan, 1.0],
        ],
        dtype=float,
    )
    fronts = non_dominated_sort(points)
    rank_map: dict[int, int] = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            rank_map[int(idx)] = rank
    assert rank_map[2] > rank_map[0]


def test_hypervolume_ignores_nonfinite_rows():
    clean = np.array([[0.8, 0.4], [0.4, 0.8]], dtype=float)
    dirty = np.vstack([clean, np.array([[np.nan, 0.9], [np.inf, 0.2]], dtype=float)])
    ref = np.array([0.0, 0.0], dtype=float)
    clean_hv = hypervolume(clean, ref)
    dirty_hv = hypervolume(dirty, ref)
    np.testing.assert_allclose(dirty_hv, clean_hv, rtol=0.0, atol=1e-12)


def test_non_dominated_sort_all_nonfinite_returns_single_terminal_front():
    points = np.array([[np.nan, 1.0], [np.inf, -np.inf]], dtype=float)
    fronts = non_dominated_sort(points)
    assert len(fronts) == 1
    np.testing.assert_array_equal(np.sort(fronts[0]), np.array([0, 1], dtype=int))


def test_hypervolume_single_point_3d_matches_exact_box_volume():
    point = np.array([[0.9, 0.8, 0.7]], dtype=float)
    ref = np.array([0.1, 0.2, 0.3], dtype=float)
    hv = hypervolume(point, ref, hv_mc_samples=256, hv_mc_seed=123, iteration=1)
    expected = float((0.9 - 0.1) * (0.8 - 0.2) * (0.7 - 0.3))
    np.testing.assert_allclose(hv, expected, rtol=0.0, atol=1e-12)
