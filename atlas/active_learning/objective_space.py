"""
Objective-space helpers for active-learning scoring and Pareto utilities.

The controller and surrogate code frequently need:
- objective clipping to [0, 1]
- conversion of candidate fields into objective matrices
- consistent feasibility masking for topology/synthesis constraints
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np


def clip01(value: float) -> float:
    scalar = safe_float(value, 0.0)
    return float(max(0.0, min(1.0, scalar)))


def safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1 and np.isfinite(arr[0]):
            return float(arr[0])
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _coerce_obj_dim(obj_dim: int, *, default: int = 1) -> int:
    if isinstance(obj_dim, bool):
        return max(1, int(default))
    if isinstance(obj_dim, Integral):
        return max(1, int(obj_dim))
    if isinstance(obj_dim, Real):
        value = float(obj_dim)
        if not np.isfinite(value) or not value.is_integer():
            return max(1, int(default))
        return max(1, int(value))
    try:
        dim = int(obj_dim)
    except Exception:
        return max(1, int(default))
    return max(1, dim)


def _coerce_threshold(value: float, default: float) -> float:
    return clip01(safe_float(value, default))


def _sanitize_objective_row(raw: object, *, obj_dim: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=float).reshape(-1)
    out = np.zeros(obj_dim, dtype=float)
    if arr.size == 0:
        return out
    dim = min(obj_dim, int(arr.shape[0]))
    cleaned = np.nan_to_num(arr[:dim], nan=0.0, posinf=1.0, neginf=0.0)
    out[:dim] = np.clip(cleaned, 0.0, 1.0)
    return out


def infer_objective_dimension(
    objective_map: dict[int, np.ndarray] | None,
    *,
    default: int = 2,
) -> int:
    fallback = _coerce_obj_dim(default)
    if not objective_map:
        return fallback
    try:
        first_obj = next(iter(objective_map.values()))
        dim = int(np.asarray(first_obj, dtype=float).reshape(-1).shape[0])
        return _coerce_obj_dim(dim)
    except Exception:
        return fallback


def objective_points_from_terms(
    topo_terms: Sequence[float],
    stability_terms: Sequence[float],
    synthesis_terms: Sequence[float] | None = None,
) -> np.ndarray:
    dim = 2 if synthesis_terms is None else 3
    count = min(len(topo_terms), len(stability_terms))
    if synthesis_terms is not None:
        count = min(count, len(synthesis_terms))
    if count <= 0:
        return np.zeros((0, dim), dtype=float)

    out = np.zeros((count, dim), dtype=float)
    if synthesis_terms is None:
        for idx, (t, s) in enumerate(zip(topo_terms, stability_terms, strict=False)):
            if idx >= count:
                break
            out[idx, 0] = clip01(safe_float(t, 0.0))
            out[idx, 1] = clip01(safe_float(s, 0.0))
    else:
        for idx, (t, s, sy) in enumerate(zip(topo_terms, stability_terms, synthesis_terms, strict=False)):
            if idx >= count:
                break
            out[idx, 0] = clip01(safe_float(t, 0.0))
            out[idx, 1] = clip01(safe_float(s, 0.0))
            out[idx, 2] = clip01(safe_float(sy, 0.0))
    return out


def objective_points_from_map(
    candidates: Sequence[object],
    objective_map: dict[int, np.ndarray] | None,
    *,
    obj_dim: int,
) -> np.ndarray:
    dim = _coerce_obj_dim(obj_dim)
    if not candidates:
        return np.zeros((0, dim), dtype=float)
    out = np.zeros((len(candidates), dim), dtype=float)
    if not objective_map:
        return out
    for idx, cand in enumerate(candidates):
        raw = objective_map.get(id(cand))
        if raw is None:
            continue
        out[idx] = _sanitize_objective_row(raw, obj_dim=dim)
    return out


def collect_history_objective_points(
    candidates: Sequence[object],
    *,
    obj_dim: int,
    topo_threshold: float = 0.0,
    use_joint_synthesis: bool = False,
    synthesis_threshold: float = 0.1,
) -> np.ndarray:
    dim = _coerce_obj_dim(obj_dim)
    if not candidates:
        return np.zeros((0, dim), dtype=float)

    topo_cut = _coerce_threshold(topo_threshold, 0.0)
    syn_cut = _coerce_threshold(synthesis_threshold, 0.1)
    points: list[list[float]] = []
    for cand in candidates:
        topo = clip01(safe_float(getattr(cand, "topo_probability", 0.0), 0.0))
        if topo < topo_cut:
            continue
        stab = clip01(safe_float(getattr(cand, "stability_score", 0.0), 0.0))
        row = [topo]
        if dim >= 2:
            row.append(stab)
        if dim >= 3:
            syn = clip01(safe_float(getattr(cand, "synthesis_score", 0.0), 0.0))
            if use_joint_synthesis and syn < syn_cut:
                continue
            row.append(syn)
        if len(row) < dim:
            row.extend([0.0] * (dim - len(row)))
        points.append(row[:dim])

    if not points:
        return np.zeros((0, dim), dtype=float)
    return np.asarray(points, dtype=float)


def feasibility_mask_from_points(
    cand_points: np.ndarray,
    *,
    topo_threshold: float,
    use_joint_synthesis: bool = False,
    synthesis_threshold: float = 0.1,
) -> np.ndarray:
    arr = np.asarray(cand_points, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 1:
        return np.zeros(0, dtype=bool)

    topo_cut = _coerce_threshold(topo_threshold, 0.0)
    syn_cut = _coerce_threshold(synthesis_threshold, 0.1)
    finite_rows = np.all(np.isfinite(arr), axis=1)
    mask = finite_rows & (arr[:, 0] >= topo_cut)
    if use_joint_synthesis:
        if arr.shape[1] >= 3:
            mask = mask & (arr[:, 2] >= syn_cut)
        else:
            mask = np.zeros(arr.shape[0], dtype=bool)
    return mask
