"""
Pareto and hypervolume helpers for active-learning selection.

This module intentionally contains pure NumPy functions so that:
- the controller keeps orchestration concerns only
- ranking / HV math can be unit-tested in isolation
- hot loops can be optimized without touching workflow code
"""

from __future__ import annotations

import numpy as np


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset under maximization."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    ge = np.all(arr[:, None, :] >= arr[None, :, :], axis=2)
    gt = np.any(arr[:, None, :] > arr[None, :, :], axis=2)
    dominates = ge & gt
    np.fill_diagonal(dominates, False)
    dominated = np.any(dominates, axis=0)
    return arr[~dominated]


def non_dominated_sort(points: np.ndarray) -> list[np.ndarray]:
    """
    Fast non-dominated sorting (maximization) using a dominance matrix.

    Compared with pairwise point-by-point elimination in each front,
    this keeps the overall complexity close to O(n^2) for fixed objective
    dimension, matching practical fast-sort implementations used by NSGA-II.
    """
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return []
    n = int(arr.shape[0])
    if n == 1:
        return [np.array([0], dtype=int)]

    ge = np.all(arr[:, None, :] >= arr[None, :, :], axis=2)
    gt = np.any(arr[:, None, :] > arr[None, :, :], axis=2)
    dominates = ge & gt
    np.fill_diagonal(dominates, False)

    domination_count = dominates.sum(axis=0).astype(np.int32, copy=False)
    remaining = np.ones(n, dtype=bool)
    fronts: list[np.ndarray] = []

    while np.any(remaining):
        front_mask = remaining & (domination_count == 0)
        if not np.any(front_mask):
            # Defensive fallback for numerical/pathological ties.
            front = np.flatnonzero(remaining)
            fronts.append(front)
            break
        front = np.flatnonzero(front_mask)
        fronts.append(front)
        remaining[front] = False
        if not np.any(remaining):
            break
        domination_count -= dominates[front].sum(axis=0).astype(np.int32, copy=False)
        domination_count[~remaining] = -1
    return fronts


def crowding_distance(points: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance for a single front."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        return np.zeros(0, dtype=float)
    n, m = arr.shape
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)

    dist = np.zeros(n, dtype=float)
    for dim in range(m):
        order = np.argsort(arr[:, dim], kind="mergesort")
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        lo = float(arr[order[0], dim])
        hi = float(arr[order[-1], dim])
        denom = max(1e-12, hi - lo)
        if n > 2:
            inner = order[1:-1]
            prev_v = arr[order[:-2], dim]
            next_v = arr[order[2:], dim]
            contrib = (next_v - prev_v) / denom
            finite_mask = ~np.isinf(dist[inner])
            dist[inner[finite_mask]] += contrib[finite_mask]
    return dist


def pareto_rank_score(points: np.ndarray) -> np.ndarray:
    """Convert Pareto front rank + crowding into a normalized [0, 1] score."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=float)

    scores = np.zeros(arr.shape[0], dtype=float)
    fronts = non_dominated_sort(arr)
    if not fronts:
        return scores

    for rank, front_idx in enumerate(fronts):
        front_points = arr[front_idx]
        crowd = crowding_distance(front_points)
        finite = crowd[np.isfinite(crowd)]
        if finite.size > 0:
            cmin = float(np.min(finite))
            cmax = float(np.max(finite))
            denom = max(1e-12, cmax - cmin)
            crowd_norm = np.ones_like(crowd, dtype=float)
            finite_mask = np.isfinite(crowd)
            crowd_norm[finite_mask] = (crowd[finite_mask] - cmin) / denom
        else:
            crowd_norm = np.ones_like(crowd, dtype=float)
        rank_term = 1.0 / (1.0 + float(rank))
        scores[front_idx] = rank_term * (0.75 + 0.25 * crowd_norm)

    max_score = float(np.max(scores))
    if max_score > 0.0:
        scores /= max_score
    return scores


def hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    """
    Exact 2D dominated hypervolume for maximization objectives.

    The implementation sweeps sorted Pareto points once and accumulates
    rectangle strips, avoiding repeated sub-filtering per x-slice.
    """
    arr = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if arr.size == 0 or ref.size != 2:
        return 0.0

    pts = pareto_front(arr)
    rx, ry = float(ref[0]), float(ref[1])
    valid = pts[(pts[:, 0] > rx) & (pts[:, 1] > ry)]
    if valid.size == 0:
        return 0.0

    order = np.argsort(valid[:, 0], kind="mergesort")
    x = valid[order, 0]
    y = valid[order, 1]
    running_max_y = np.maximum.accumulate(y[::-1])[::-1]

    left = np.concatenate([[rx], x[:-1]])
    width = np.maximum(0.0, x - np.maximum(left, rx))
    height = np.maximum(0.0, running_max_y - ry)
    hv = float(np.sum(width * height))
    return max(0.0, hv)


def _sample_hv_probes(
    reference: np.ndarray,
    upper: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.random((samples, reference.shape[0]))
    side = (upper - reference).reshape(1, -1)
    return reference.reshape(1, -1) + u * side


def hypervolume(
    points: np.ndarray,
    reference: np.ndarray,
    *,
    hv_mc_samples: int = 4096,
    hv_mc_seed: int = 12345,
    iteration: int = 1,
    hv_chunk_size: int = 512,
) -> float:
    """
    Dominated hypervolume for maximization objectives.

    - Exact for 1D and 2D.
    - Monte Carlo estimate for >=3D.
    """
    arr = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if arr.size == 0 or arr.ndim != 2:
        return 0.0
    dim = int(arr.shape[1])
    if dim <= 0 or ref.size != dim:
        return 0.0

    valid = arr[np.all(arr > ref.reshape(1, -1), axis=1)]
    if valid.size == 0:
        return 0.0
    if dim == 1:
        return float(max(0.0, np.max(valid[:, 0]) - ref[0]))
    if dim == 2:
        return hypervolume_2d(valid, ref)

    upper = np.max(valid, axis=0)
    side = upper - ref
    if np.any(side <= 1e-12):
        return 0.0
    box_volume = float(np.prod(side))
    if box_volume <= 0.0:
        return 0.0

    samples = max(256, int(hv_mc_samples))
    seed = int(hv_mc_seed) + max(1, int(iteration))
    probes = _sample_hv_probes(ref, upper, samples=samples, seed=seed)
    chunk = max(64, int(hv_chunk_size))

    dominated_count = 0
    for start in range(0, samples, chunk):
        batch = probes[start : start + chunk]
        dominates = np.all(valid[None, :, :] >= batch[:, None, :], axis=2)
        dominated_count += int(np.any(dominates, axis=1).sum())
    return float(box_volume * (dominated_count / float(samples)))


def mc_hv_improvements_shared(
    hist_arr: np.ndarray,
    cand_points: np.ndarray,
    reference: np.ndarray,
    *,
    feas_threshold: float,
    hv_mc_samples: int = 4096,
    hv_mc_seed: int = 12345,
    iteration: int = 1,
    hv_chunk_size: int = 512,
) -> np.ndarray:
    """
    Shared-sample Monte Carlo HV-improvement estimate for candidate pools.

    For dim>=3 this computes all candidate gains in vectorized chunks:
    gain_i = box_volume * P(probe not dominated by history and dominated by i)
    """
    cand = np.asarray(cand_points, dtype=float)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    hist = np.asarray(hist_arr, dtype=float)
    if cand.size == 0:
        return np.zeros(0, dtype=float)
    if cand.ndim != 2:
        return np.zeros(cand.shape[0] if cand.ndim > 0 else 0, dtype=float)

    n_cand, dim = cand.shape
    out = np.zeros(n_cand, dtype=float)
    if ref.size != dim:
        return out

    if dim <= 2:
        baseline = hypervolume(
            hist,
            ref,
            hv_mc_samples=hv_mc_samples,
            hv_mc_seed=hv_mc_seed,
            iteration=iteration,
            hv_chunk_size=hv_chunk_size,
        )
        for idx, p in enumerate(cand):
            if p[0] < feas_threshold:
                continue
            augmented = p.reshape(1, dim) if hist.size == 0 else np.vstack([hist, p.reshape(1, dim)])
            out[idx] = max(
                0.0,
                hypervolume(
                    augmented,
                    ref,
                    hv_mc_samples=hv_mc_samples,
                    hv_mc_seed=hv_mc_seed,
                    iteration=iteration,
                    hv_chunk_size=hv_chunk_size,
                )
                - baseline,
            )
        return out

    valid_hist = (
        hist[np.all(hist > ref.reshape(1, -1), axis=1)]
        if hist.size
        else np.zeros((0, dim), dtype=float)
    )
    valid_mask = (cand[:, 0] >= float(feas_threshold)) & np.all(cand > ref.reshape(1, -1), axis=1)
    if not np.any(valid_mask) and valid_hist.size == 0:
        return out

    merged = cand[valid_mask] if valid_hist.size == 0 else np.vstack([valid_hist, cand[valid_mask]])
    if merged.size == 0:
        return out
    upper = np.max(merged, axis=0)
    side = upper - ref
    if np.any(side <= 1e-12):
        return out

    samples = max(256, int(hv_mc_samples))
    seed = int(hv_mc_seed) + max(1, int(iteration))
    probes = _sample_hv_probes(ref, upper, samples=samples, seed=seed)
    box_volume = float(np.prod(side))
    if box_volume <= 0.0:
        return out

    if valid_hist.size == 0:
        dominated_hist = np.zeros(samples, dtype=bool)
    else:
        chunk = max(64, int(hv_chunk_size))
        dominated_hist = np.zeros(samples, dtype=bool)
        for start in range(0, samples, chunk):
            batch = probes[start : start + chunk]
            dominates_hist = np.all(valid_hist[None, :, :] >= batch[:, None, :], axis=2)
            dominated_hist[start : start + chunk] = np.any(dominates_hist, axis=1)
    uncovered = ~dominated_hist
    if not np.any(uncovered):
        return out

    valid_indices = np.flatnonzero(valid_mask)
    cand_valid = cand[valid_indices]
    chunk = max(8, int(hv_chunk_size // max(1, dim)))
    for start in range(0, cand_valid.shape[0], chunk):
        sub = cand_valid[start : start + chunk]
        dominates_sub = np.all(sub[:, None, :] >= probes[None, :, :], axis=2)
        newly_dominated = dominates_sub[:, uncovered]
        gain_ratio = newly_dominated.mean(axis=1)
        out[valid_indices[start : start + chunk]] = box_volume * gain_ratio
    return out
