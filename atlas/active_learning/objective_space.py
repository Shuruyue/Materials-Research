"""
Objective-space helpers for active-learning scoring and Pareto utilities.

The controller and surrogate code frequently need:
- objective clipping to [0, 1]
- conversion of candidate fields into objective matrices
- consistent feasibility masking for topology/synthesis constraints
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def infer_objective_dimension(
    objective_map: dict[int, np.ndarray] | None,
    *,
    default: int = 2,
) -> int:
    if not objective_map:
        return int(default)
    try:
        first_obj = next(iter(objective_map.values()))
        return int(np.asarray(first_obj, dtype=float).reshape(-1).shape[0])
    except Exception:
        return int(default)


def objective_points_from_terms(
    topo_terms: Sequence[float],
    stability_terms: Sequence[float],
    synthesis_terms: Sequence[float] | None = None,
) -> np.ndarray:
    if synthesis_terms is None:
        points = [
            [clip01(safe_float(t, 0.0)), clip01(safe_float(s, 0.0))]
            for t, s in zip(topo_terms, stability_terms, strict=False)
        ]
    else:
        points = [
            [clip01(safe_float(t, 0.0)), clip01(safe_float(s, 0.0)), clip01(safe_float(sy, 0.0))]
            for t, s, sy in zip(topo_terms, stability_terms, synthesis_terms, strict=False)
        ]
    return np.asarray(points, dtype=float)


def objective_points_from_map(
    candidates: Sequence[object],
    objective_map: dict[int, np.ndarray] | None,
    *,
    obj_dim: int,
) -> np.ndarray:
    if not candidates:
        return np.zeros((0, obj_dim), dtype=float)
    out = np.zeros((len(candidates), obj_dim), dtype=float)
    if not objective_map:
        return out
    for idx, cand in enumerate(candidates):
        raw = objective_map.get(id(cand))
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=float).reshape(-1)
        dim = min(obj_dim, arr.shape[0])
        out[idx, :dim] = arr[:dim]
    return out


def collect_history_objective_points(
    candidates: Sequence[object],
    *,
    obj_dim: int,
    topo_threshold: float = 0.0,
    use_joint_synthesis: bool = False,
    synthesis_threshold: float = 0.1,
) -> np.ndarray:
    if not candidates:
        return np.zeros((0, obj_dim), dtype=float)

    points: list[list[float]] = []
    for cand in candidates:
        topo = clip01(safe_float(getattr(cand, "topo_probability", 0.0), 0.0))
        if topo < topo_threshold:
            continue
        stab = clip01(safe_float(getattr(cand, "stability_score", 0.0), 0.0))
        if obj_dim >= 3:
            syn = clip01(safe_float(getattr(cand, "synthesis_score", 0.0), 0.0))
            if use_joint_synthesis and syn < synthesis_threshold:
                continue
            points.append([topo, stab, syn])
        else:
            points.append([topo, stab])

    if not points:
        return np.zeros((0, obj_dim), dtype=float)
    return np.asarray(points, dtype=float)


def feasibility_mask_from_points(
    cand_points: np.ndarray,
    *,
    topo_threshold: float,
    use_joint_synthesis: bool = False,
    synthesis_threshold: float = 0.1,
) -> np.ndarray:
    if cand_points.size == 0:
        return np.zeros(0, dtype=bool)

    mask = cand_points[:, 0] >= float(topo_threshold)
    if use_joint_synthesis and cand_points.shape[1] >= 3:
        mask = mask & (cand_points[:, 2] >= float(synthesis_threshold))
    return mask

