"""
Split Governance for ATLAS.

Provides deterministic IID, compositional, and prototype splits
with SHA-256 manifest generation.

CLI entrypoint: ``make-splits``

Extends existing split logic in :class:`CrystalPropertyDataset` with
OOD-aware splitting strategies.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SPLIT_STRATEGIES = ("iid", "compositional", "prototype", "all")
SPLIT_SCHEMA_VERSION = "2.0"
_SPLIT_NAMES = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ids_hash(ids: list[str]) -> str:
    """Deterministic hash of a sorted ID list."""
    return _sha256_hex("\n".join(sorted(ids)))


def _extract_elements(formula: str) -> frozenset[str]:
    """Extract element symbols from a chemical formula string.

    Simple parser: splits on uppercase letters.
    'Li2Fe3O4' -> {'Li', 'Fe', 'O'}
    """
    import re
    elems = re.findall(r"[A-Z][a-z]?", formula)
    return frozenset(elems)


def _chemical_system(formula: str) -> str:
    """Return sorted chemical system string, e.g. 'Fe-Li-O'."""
    return "-".join(sorted(_extract_elements(formula)))


def _normalized_ratios(
    ratios: tuple[float, float, float],
) -> np.ndarray:
    arr = np.asarray(ratios, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"ratios must have length 3, got {ratios}")
    if np.any(arr < 0):
        raise ValueError(f"ratios must be non-negative, got {ratios}")
    s = float(arr.sum())
    if s <= 0.0:
        raise ValueError(f"ratios must sum to positive value, got {ratios}")
    return arr / s


def _deterministic_split_counts(
    n: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    """
    Deterministic integer split counts with residual balancing.

    Ensures each positive-ratio split receives at least one sample when feasible.
    """
    r = _normalized_ratios(ratios)
    raw = r * float(n)
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            counts[idx] += 1

    positive = [i for i, rv in enumerate(r) if rv > 0]
    if n >= len(positive):
        for idx in positive:
            if counts[idx] > 0:
                continue
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _require_parallel_lengths(
    sample_ids: list[str],
    values: list[Any],
    value_name: str,
):
    if len(sample_ids) != len(values):
        raise ValueError(
            f"sample_ids and {value_name} length mismatch: "
            f"{len(sample_ids)} != {len(values)}"
        )


def _require_unique_ids(sample_ids: list[str]):
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique for deterministic split assignment")


def _partition_objective(
    sample_counts: np.ndarray,
    group_counts: np.ndarray,
    target_samples: np.ndarray,
    target_groups: np.ndarray,
    *,
    size_weight: float,
    group_weight: float,
) -> float:
    sample_denom = np.maximum(target_samples, 1.0)
    group_denom = np.maximum(target_groups, 1.0)
    size_err = np.sum(((sample_counts - target_samples) / sample_denom) ** 2)
    group_err = np.sum(((group_counts - target_groups) / group_denom) ** 2)
    return float(size_weight * size_err + group_weight * group_err)


def _label_balance_objective(
    label_sums: np.ndarray,
    label_counts: np.ndarray,
    *,
    global_mean: float,
    global_scale: float,
    positive_splits: list[int],
) -> float:
    if not positive_splits:
        return 0.0
    err = 0.0
    for idx in positive_splits:
        if label_counts[idx] <= 0.0:
            # Missing labels in a positive split is penalized to avoid extreme drift.
            err += 1.0
            continue
        mean_i = label_sums[idx] / label_counts[idx]
        err += float(((mean_i - global_mean) / global_scale) ** 2)
    return err / float(len(positive_splits))


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _normalize_rowwise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return vectors / norms


def _build_group_label_stats(
    group_to_indices: dict[str, list[int]],
    targets: list[float | int | None],
) -> dict[str, tuple[float, float, float]]:
    stats: dict[str, tuple[float, float, float]] = {}
    for group, indices in group_to_indices.items():
        s = 0.0
        c = 0.0
        ss = 0.0
        for idx in indices:
            v = _as_finite_float(targets[idx])
            if v is None:
                continue
            s += v
            c += 1.0
            ss += v * v
        if c > 0:
            stats[group] = (s, c, ss)
    return stats


def _build_group_vectors_from_samples(
    group_to_indices: dict[str, list[int]],
    feature_vectors: list[Sequence[float] | None],
) -> dict[str, np.ndarray]:
    group_vectors: dict[str, np.ndarray] = {}
    expected_dim: int | None = None
    for group, indices in group_to_indices.items():
        acc: np.ndarray | None = None
        cnt = 0.0
        for idx in indices:
            raw = feature_vectors[idx]
            if raw is None:
                continue
            vec = np.asarray(raw, dtype=float).reshape(-1)
            if vec.size == 0 or not np.all(np.isfinite(vec)):
                continue
            if expected_dim is None:
                expected_dim = int(vec.size)
            elif int(vec.size) != expected_dim:
                raise ValueError(
                    f"feature_vectors dimension mismatch: expected {expected_dim}, got {vec.size}"
                )
            if acc is None:
                acc = np.zeros(expected_dim, dtype=float)
            acc += vec
            cnt += 1.0
        if acc is not None and cnt > 0:
            group_vectors[group] = acc / cnt
    return group_vectors


def _cosine_similarity_matrix_from_group_vectors(
    groups: list[str],
    group_vectors: dict[str, np.ndarray],
) -> np.ndarray | None:
    if not groups:
        return None
    if not all(g in group_vectors for g in groups):
        return None
    vecs = [group_vectors[g] for g in groups]
    dim = int(vecs[0].size)
    if any(int(v.size) != dim for v in vecs):
        return None
    mat = np.vstack(vecs).astype(float)
    if mat.size == 0:
        return None
    mat = _normalize_rowwise(mat)
    sim = mat @ mat.T
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 0.0)
    return 0.5 * (sim + sim.T)


def _chemical_system_similarity_matrix(groups: list[str]) -> np.ndarray:
    if not groups:
        return np.zeros((0, 0), dtype=float)
    elem_sets = []
    for g in groups:
        if not g or g == "unknown":
            elem_sets.append(set())
        else:
            elem_sets.append(set(g.split("-")))
    n = len(groups)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            u = elem_sets[i] | elem_sets[j]
            if not u:
                s = 0.0
            else:
                s = float(len(elem_sets[i] & elem_sets[j]) / len(u))
            sim[i, j] = s
            sim[j, i] = s
    return sim


def _spacegroup_number(value: str | int | None) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= out <= 230:
        return out
    return None


def _spacegroup_crystal_system(sg_number: int) -> int:
    # 0..6 = triclinic, monoclinic, orthorhombic, tetragonal, trigonal,
    # hexagonal, cubic.
    if sg_number <= 2:
        return 0
    if sg_number <= 15:
        return 1
    if sg_number <= 74:
        return 2
    if sg_number <= 142:
        return 3
    if sg_number <= 167:
        return 4
    if sg_number <= 194:
        return 5
    return 6


def _prototype_similarity_matrix(groups: list[str]) -> np.ndarray:
    n = len(groups)
    sim = np.zeros((n, n), dtype=float)
    sg_nums = [_spacegroup_number(g) for g in groups]
    for i in range(n):
        for j in range(i + 1, n):
            ni = sg_nums[i]
            nj = sg_nums[j]
            if ni is None or nj is None:
                s = 0.0
            else:
                family_i = _spacegroup_crystal_system(ni)
                family_j = _spacegroup_crystal_system(nj)
                family_sim = 1.0 if family_i == family_j else 0.0
                number_sim = max(0.0, 1.0 - abs(float(ni - nj)) / 230.0)
                s = 0.6 * family_sim + 0.4 * number_sim
            sim[i, j] = s
            sim[j, i] = s
    return sim


def _balanced_group_split(
    group_to_ids: dict[str, list[str]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
    n_restarts: int = 8,
    size_weight: float = 1.0,
    group_weight: float = 0.15,
    label_stats_by_group: dict[str, tuple[float, float, float]] | None = None,
    label_weight: float = 0.0,
    similarity_matrix: np.ndarray | None = None,
    leakage_weight: float = 0.0,
    pair_weights: tuple[float, float, float] = (0.5, 1.0, 0.5),
    local_moves: int = 64,
) -> dict[str, list[str]]:
    """
    Balanced split by whole groups with ratio-aware objective minimization.

    This is a heuristic for NP-hard partitioning. The objective follows the
    "balance + leakage constraints" spirit of modern split optimization
    literature:
    - Ben-David et al. (2010), domain-shift-aware evaluation.
    - DataSAIL (Nat. Commun. 2025), leakage-aware split optimization.
    - Gretton et al. (JMLR 2012), kernel two-sample discrepancy (MMD).
    """
    if not group_to_ids:
        return {"train": [], "val": [], "test": []}

    ratios_arr = _normalized_ratios(ratios)
    groups = sorted(group_to_ids.keys())
    n_groups = len(groups)
    g2i = {g: i for i, g in enumerate(groups)}
    positive_splits = [i for i, rv in enumerate(ratios_arr) if rv > 0]

    group_sizes = {g: len(group_to_ids[g]) for g in groups}
    total_samples = float(sum(group_sizes[g] for g in groups))
    total_groups = float(len(groups))
    target_samples = ratios_arr * total_samples
    target_groups = ratios_arr * total_groups

    group_sizes_arr = np.array([group_sizes[g] for g in groups], dtype=float)

    # Optional label balance terms.
    label_enabled = bool(label_stats_by_group) and float(label_weight) > 0.0
    group_label_sum = np.zeros(n_groups, dtype=float)
    group_label_count = np.zeros(n_groups, dtype=float)
    global_label_sum = 0.0
    global_label_count = 0.0
    global_label_sumsq = 0.0
    if label_enabled and label_stats_by_group is not None:
        for g, (s, c, ss) in label_stats_by_group.items():
            if g not in g2i or c <= 0:
                continue
            idx = g2i[g]
            group_label_sum[idx] = float(s)
            group_label_count[idx] = float(c)
            global_label_sum += float(s)
            global_label_count += float(c)
            global_label_sumsq += float(ss)
        if global_label_count <= 0:
            label_enabled = False
        else:
            global_label_mean = global_label_sum / global_label_count
            mean_sq = global_label_sumsq / global_label_count
            var = max(mean_sq - global_label_mean * global_label_mean, 1e-12)
            global_label_scale = max(float(np.sqrt(var)), 1e-6)

    # Optional cross-split similarity leakage term.
    leakage_enabled = (
        similarity_matrix is not None
        and float(leakage_weight) > 0.0
        and n_groups == similarity_matrix.shape[0]
        and similarity_matrix.shape[0] == similarity_matrix.shape[1]
    )
    if leakage_enabled:
        sim = np.asarray(similarity_matrix, dtype=float)
        sim = np.clip(sim, 0.0, 1.0)
        sim = 0.5 * (sim + sim.T)
        np.fill_diagonal(sim, 0.0)
        tv_w, tt_w, vt_w = pair_weights
        wmat = np.array(
            [[0.0, float(tv_w), float(tt_w)],
             [float(tv_w), 0.0, float(vt_w)],
             [float(tt_w), float(vt_w), 0.0]],
            dtype=float,
        )
        leakage_norm = float(max(n_groups * max(n_groups - 1, 1) / 2.0, 1.0))

    def total_score(
        sample_counts: np.ndarray,
        group_counts: np.ndarray,
        label_sums: np.ndarray,
        label_counts: np.ndarray,
        leakage_num: float,
    ) -> float:
        base = _partition_objective(
            sample_counts,
            group_counts,
            target_samples,
            target_groups,
            size_weight=size_weight,
            group_weight=group_weight,
        )
        score = base
        if label_enabled:
            score += float(label_weight) * _label_balance_objective(
                label_sums,
                label_counts,
                global_mean=global_label_mean,
                global_scale=global_label_scale,
                positive_splits=positive_splits,
            )
        if leakage_enabled:
            score += float(leakage_weight) * (leakage_num / leakage_norm)
        return float(score)

    def leakage_delta_for_new(
        group_idx: int,
        split_idx: int,
        assignment_idx: np.ndarray,
        assigned_indices: list[int],
    ) -> float:
        if not leakage_enabled:
            return 0.0
        delta = 0.0
        for h in assigned_indices:
            delta += sim[group_idx, h] * wmat[split_idx, int(assignment_idx[h])]
        return delta

    def leakage_delta_for_move(
        group_idx: int,
        src_idx: int,
        dst_idx: int,
        assignment_idx: np.ndarray,
    ) -> float:
        if not leakage_enabled:
            return 0.0
        delta = 0.0
        for h in range(n_groups):
            if h == group_idx:
                continue
            sh = int(assignment_idx[h])
            delta += sim[group_idx, h] * (wmat[dst_idx, sh] - wmat[src_idx, sh])
        return delta

    def build_with_rng(rng: np.random.RandomState) -> tuple[dict[str, int], float]:
        assignment: dict[str, int] = {}
        assignment_idx = np.full(n_groups, -1, dtype=int)
        assigned_indices: list[int] = []
        sample_counts = np.zeros(3, dtype=float)
        group_counts = np.zeros(3, dtype=float)
        label_sums = np.zeros(3, dtype=float)
        label_counts = np.zeros(3, dtype=float)
        leakage_num = 0.0

        # Largest-first improves partition quality for cardinality-constrained
        # load balancing (classical approximation intuition from bin packing).
        order = sorted(groups, key=lambda g: (-group_sizes[g], rng.rand()))
        for g in order:
            gi = g2i[g]
            best_split = 0
            best_score = float("inf")
            ties: list[int] = []
            for s in range(3):
                trial_samples = sample_counts.copy()
                trial_groups = group_counts.copy()
                trial_label_sums = label_sums.copy()
                trial_label_counts = label_counts.copy()
                trial_samples[s] += group_sizes_arr[gi]
                trial_groups[s] += 1.0
                trial_leakage = leakage_num
                if label_enabled:
                    trial_label_sums[s] += group_label_sum[gi]
                    trial_label_counts[s] += group_label_count[gi]
                if leakage_enabled:
                    trial_leakage += leakage_delta_for_new(
                        gi, s, assignment_idx, assigned_indices
                    )
                score = total_score(
                    trial_samples,
                    trial_groups,
                    trial_label_sums,
                    trial_label_counts,
                    trial_leakage,
                )
                if score + 1e-12 < best_score:
                    best_score = score
                    best_split = s
                    ties = [s]
                elif abs(score - best_score) <= 1e-12:
                    ties.append(s)
            if len(ties) > 1:
                best_split = int(ties[rng.randint(len(ties))])
            assignment[g] = best_split
            assignment_idx[gi] = best_split
            sample_counts[best_split] += group_sizes[g]
            group_counts[best_split] += 1.0
            if label_enabled:
                label_sums[best_split] += group_label_sum[gi]
                label_counts[best_split] += group_label_count[gi]
            if leakage_enabled:
                leakage_num += leakage_delta_for_new(
                    gi, best_split, assignment_idx, assigned_indices
                )
                assigned_indices.append(gi)

        # Local one-move improvement.
        for _ in range(local_moves):
            improved = False
            for g in order:
                gi = g2i[g]
                src = assignment[g]
                best_split = src
                best_score = total_score(
                    sample_counts,
                    group_counts,
                    label_sums,
                    label_counts,
                    leakage_num,
                )
                for dst in range(3):
                    if dst == src:
                        continue
                    # Keep at least one group in split if feasible.
                    if (
                        src in positive_splits
                        and group_counts[src] <= 1
                        and len(groups) >= len(positive_splits)
                    ):
                        continue
                    trial_samples = sample_counts.copy()
                    trial_groups = group_counts.copy()
                    trial_label_sums = label_sums.copy()
                    trial_label_counts = label_counts.copy()
                    trial_samples[src] -= group_sizes_arr[gi]
                    trial_groups[src] -= 1.0
                    trial_samples[dst] += group_sizes_arr[gi]
                    trial_groups[dst] += 1.0
                    if label_enabled:
                        trial_label_sums[src] -= group_label_sum[gi]
                        trial_label_counts[src] -= group_label_count[gi]
                        trial_label_sums[dst] += group_label_sum[gi]
                        trial_label_counts[dst] += group_label_count[gi]
                    trial_leakage = leakage_num
                    if leakage_enabled:
                        trial_leakage += leakage_delta_for_move(
                            gi, src, dst, assignment_idx
                        )
                    score = total_score(
                        trial_samples,
                        trial_groups,
                        trial_label_sums,
                        trial_label_counts,
                        trial_leakage,
                    )
                    if score + 1e-12 < best_score:
                        best_score = score
                        best_split = dst
                if best_split != src:
                    leak_delta = 0.0
                    if leakage_enabled:
                        leak_delta = leakage_delta_for_move(
                            gi, src, best_split, assignment_idx
                        )
                    assignment[g] = best_split
                    assignment_idx[gi] = best_split
                    sample_counts[src] -= group_sizes_arr[gi]
                    group_counts[src] -= 1.0
                    sample_counts[best_split] += group_sizes_arr[gi]
                    group_counts[best_split] += 1.0
                    if label_enabled:
                        label_sums[src] -= group_label_sum[gi]
                        label_counts[src] -= group_label_count[gi]
                        label_sums[best_split] += group_label_sum[gi]
                        label_counts[best_split] += group_label_count[gi]
                    if leakage_enabled:
                        leakage_num += leak_delta
                    improved = True
            if not improved:
                break

        # Guarantee non-empty positive-ratio splits when possible.
        if len(groups) >= len(positive_splits):
            for split_idx in positive_splits:
                if group_counts[split_idx] > 0:
                    continue
                donor_candidates = [i for i in positive_splits if group_counts[i] > 1]
                if not donor_candidates:
                    continue
                best_move = None
                current_score = total_score(
                    sample_counts,
                    group_counts,
                    label_sums,
                    label_counts,
                    leakage_num,
                )
                for donor in donor_candidates:
                    donor_groups = [g for g, s in assignment.items() if s == donor]
                    donor_groups.sort(key=lambda x: group_sizes[x])  # smallest perturbation
                    for g in donor_groups:
                        gi = g2i[g]
                        trial_samples = sample_counts.copy()
                        trial_groups = group_counts.copy()
                        trial_label_sums = label_sums.copy()
                        trial_label_counts = label_counts.copy()
                        trial_samples[donor] -= group_sizes_arr[gi]
                        trial_groups[donor] -= 1.0
                        trial_samples[split_idx] += group_sizes_arr[gi]
                        trial_groups[split_idx] += 1.0
                        if label_enabled:
                            trial_label_sums[donor] -= group_label_sum[gi]
                            trial_label_counts[donor] -= group_label_count[gi]
                            trial_label_sums[split_idx] += group_label_sum[gi]
                            trial_label_counts[split_idx] += group_label_count[gi]
                        trial_leakage = leakage_num
                        if leakage_enabled:
                            trial_leakage += leakage_delta_for_move(
                                gi, donor, split_idx, assignment_idx
                            )
                        score = total_score(
                            trial_samples,
                            trial_groups,
                            trial_label_sums,
                            trial_label_counts,
                            trial_leakage,
                        )
                        delta = score - current_score
                        if best_move is None or delta < best_move[0]:
                            best_move = (delta, g, donor)
                        break
                if best_move is not None:
                    _, g_move, donor = best_move
                    gi = g2i[g_move]
                    leak_delta = 0.0
                    if leakage_enabled:
                        leak_delta = leakage_delta_for_move(
                            gi, donor, split_idx, assignment_idx
                        )
                    assignment[g_move] = split_idx
                    assignment_idx[gi] = split_idx
                    sample_counts[donor] -= group_sizes_arr[gi]
                    group_counts[donor] -= 1.0
                    sample_counts[split_idx] += group_sizes_arr[gi]
                    group_counts[split_idx] += 1.0
                    if label_enabled:
                        label_sums[donor] -= group_label_sum[gi]
                        label_counts[donor] -= group_label_count[gi]
                        label_sums[split_idx] += group_label_sum[gi]
                        label_counts[split_idx] += group_label_count[gi]
                    if leakage_enabled:
                        leakage_num += leak_delta

        final_score = total_score(
            sample_counts,
            group_counts,
            label_sums,
            label_counts,
            leakage_num,
        )
        return assignment, final_score

    base_rng = np.random.RandomState(seed)
    best_assignment: dict[str, int] | None = None
    best_score = float("inf")
    restarts = max(int(n_restarts), 1)
    for _ in range(restarts):
        local_seed = int(base_rng.randint(0, 2**31 - 1))
        local_rng = np.random.RandomState(local_seed)
        assignment, score = build_with_rng(local_rng)
        if score + 1e-12 < best_score:
            best_score = score
            best_assignment = assignment

    assert best_assignment is not None
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for g, split_idx in best_assignment.items():
        result[_SPLIT_NAMES[split_idx]].extend(group_to_ids[g])
    return result


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------


@dataclass
class SplitManifestV2:
    """Deterministic split manifest with auditable split assignment metadata."""

    schema_version: str = SPLIT_SCHEMA_VERSION
    strategy: str = "iid"
    split_id: str = ""
    seed: int = 42
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    split_hash: str = ""
    assignment_hash: str = ""
    dataset_fingerprint: str = ""
    group_definition_version: str = "1"
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_strategy": self.strategy,
            "split_id": self.split_id,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "split_hash": self.split_hash,
            "assignment_hash": self.assignment_hash,
            "dataset_fingerprint": self.dataset_fingerprint,
            "group_definition_version": self.group_definition_version,
            "splits": self.splits,
            "metadata": self.metadata,
        }

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def compute_hash(self) -> str:
        """Compute overall split hash from per-split ID hashes."""
        parts = [f"schema:{self.schema_version}", f"strategy:{self.strategy}", f"seed:{self.seed}"]
        for name in sorted(self.splits.keys()):
            h = self.splits[name].get("sample_ids_hash", "")
            parts.append(f"{name}:{h}")
        self.split_hash = f"sha256:{_sha256_hex('|'.join(parts))}"
        return self.split_hash


# Backward-compatible alias
SplitManifest = SplitManifestV2


# ---------------------------------------------------------------------------
# Splitting functions
# ---------------------------------------------------------------------------


def iid_split(
    sample_ids: list[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Deterministic random IID split.

    Parameters
    ----------
    sample_ids : list[str]
        Unique sample identifiers.
    seed : int
        Random seed for reproducibility.
    ratios : tuple
        (train, val, test) ratios.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``
    """
    _require_unique_ids(sample_ids)
    n_train, n_val, _ = _deterministic_split_counts(len(sample_ids), ratios)

    rng = np.random.RandomState(seed)
    indices = np.arange(len(sample_ids))
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": [sample_ids[i] for i in train_idx],
        "val": [sample_ids[i] for i in val_idx],
        "test": [sample_ids[i] for i in test_idx],
    }


def compositional_split(
    sample_ids: list[str],
    formulas: list[str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    targets: list[float | int | None] | None = None,
    feature_vectors: list[Sequence[float] | None] | None = None,
    label_weight: float = 0.20,
    leakage_weight: float = 0.20,
    n_restarts: int = 8,
) -> dict[str, list[str]]:
    """Compositional OOD split: groups by chemical system.

    Entire chemical systems (e.g., all Li-Fe-O compounds) are held out
    for validation/test so that test compositions are never seen during training.

    Parameters
    ----------
    sample_ids : list[str]
        Unique sample identifiers (parallel with ``formulas``).
    formulas : list[str]
        Chemical formula per sample.
    seed : int
        Random seed.
    ratios : tuple
        (train, val, test) ratios (by group count, approximately).
    targets : list, optional
        Optional numeric property targets used for stratified label-balance
        regularization across splits.
    feature_vectors : list, optional
        Optional per-sample feature vectors used to build a cross-group
        similarity matrix for leakage minimization. If omitted, a
        chemical-system Jaccard similarity is used.
    label_weight : float
        Weight for label-balance term.
    leakage_weight : float
        Weight for cross-split similarity leakage term.
    n_restarts : int
        Number of random restarts for the group partition heuristic.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``
    """
    _require_unique_ids(sample_ids)
    _require_parallel_lengths(sample_ids, formulas, "formulas")
    if targets is not None:
        _require_parallel_lengths(sample_ids, targets, "targets")
    if feature_vectors is not None:
        _require_parallel_lengths(sample_ids, feature_vectors, "feature_vectors")

    # Group samples by chemical system.
    system_map: dict[str, list[str]] = defaultdict(list)
    system_to_indices: dict[str, list[int]] = defaultdict(list)
    for sid, formula in zip(sample_ids, formulas):
        system = _chemical_system(formula) if formula else "unknown"
        system_map[system].append(sid)
    for idx, formula in enumerate(formulas):
        system = _chemical_system(formula) if formula else "unknown"
        system_to_indices[system].append(idx)

    label_stats = None
    if targets is not None:
        label_stats = _build_group_label_stats(system_to_indices, targets)

    groups = sorted(system_map.keys())
    if feature_vectors is not None:
        group_vectors = _build_group_vectors_from_samples(system_to_indices, feature_vectors)
        similarity_matrix = _cosine_similarity_matrix_from_group_vectors(groups, group_vectors)
        if similarity_matrix is None:
            similarity_matrix = _chemical_system_similarity_matrix(groups)
    else:
        similarity_matrix = _chemical_system_similarity_matrix(groups)

    return _balanced_group_split(
        system_map,
        seed=seed,
        ratios=ratios,
        n_restarts=n_restarts,
        label_stats_by_group=label_stats,
        label_weight=label_weight,
        similarity_matrix=similarity_matrix,
        leakage_weight=leakage_weight,
    )


def prototype_split(
    sample_ids: list[str],
    spacegroups: list[int | str],
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    targets: list[float | int | None] | None = None,
    feature_vectors: list[Sequence[float] | None] | None = None,
    label_weight: float = 0.20,
    leakage_weight: float = 0.15,
    n_restarts: int = 8,
) -> dict[str, list[str]]:
    """Prototype OOD split: groups by crystal structure prototype (spacegroup).

    Entire spacegroups are held out for test so that test structures
    are never seen during training.

    Parameters
    ----------
    sample_ids : list[str]
        Unique sample identifiers (parallel with ``spacegroups``).
    spacegroups : list
        Spacegroup number or label per sample.
    seed : int
        Random seed.
    ratios : tuple
        (train, val, test) ratios (by group count, approximately).
    targets : list, optional
        Optional numeric property targets used for stratified label-balance
        regularization across splits.
    feature_vectors : list, optional
        Optional per-sample feature vectors used to build cross-group
        similarity for leakage minimization. If omitted, a spacegroup-family
        similarity kernel is used.
    label_weight : float
        Weight for label-balance term.
    leakage_weight : float
        Weight for cross-split similarity leakage term.
    n_restarts : int
        Number of random restarts for the group partition heuristic.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``
    """
    _require_unique_ids(sample_ids)
    _require_parallel_lengths(sample_ids, spacegroups, "spacegroups")
    if targets is not None:
        _require_parallel_lengths(sample_ids, targets, "targets")
    if feature_vectors is not None:
        _require_parallel_lengths(sample_ids, feature_vectors, "feature_vectors")

    # Group by spacegroup.
    sg_map: dict[str, list[str]] = defaultdict(list)
    sg_to_indices: dict[str, list[int]] = defaultdict(list)
    for sid, sg in zip(sample_ids, spacegroups):
        sg_key = str(sg) if sg else "unknown"
        sg_map[sg_key].append(sid)
    for idx, sg in enumerate(spacegroups):
        sg_key = str(sg) if sg else "unknown"
        sg_to_indices[sg_key].append(idx)

    label_stats = None
    if targets is not None:
        label_stats = _build_group_label_stats(sg_to_indices, targets)

    groups = sorted(sg_map.keys())
    if feature_vectors is not None:
        group_vectors = _build_group_vectors_from_samples(sg_to_indices, feature_vectors)
        similarity_matrix = _cosine_similarity_matrix_from_group_vectors(groups, group_vectors)
        if similarity_matrix is None:
            similarity_matrix = _prototype_similarity_matrix(groups)
    else:
        similarity_matrix = _prototype_similarity_matrix(groups)

    return _balanced_group_split(
        sg_map,
        seed=seed,
        ratios=ratios,
        n_restarts=n_restarts,
        label_stats_by_group=label_stats,
        label_weight=label_weight,
        similarity_matrix=similarity_matrix,
        leakage_weight=leakage_weight,
    )


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------


def build_assignment_records(
    splits: dict[str, list[str]],
    *,
    group_by_id: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build a deterministic row-wise split assignment list."""
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        for sample_id in sorted(splits.get(split, [])):
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "split": split,
                    "group": str(group_by_id.get(sample_id, "")) if group_by_id else "",
                }
            )
    return rows


def compute_split_overlap_counts(splits: dict[str, list[str]]) -> dict[str, int]:
    """Compute pairwise overlap counts for split integrity checks."""
    train = set(splits.get("train", []))
    val = set(splits.get("val", []))
    test = set(splits.get("test", []))
    return {
        "train__val": len(train & val),
        "train__test": len(train & test),
        "val__test": len(val & test),
    }


def _assignment_hash(assignment_rows: list[dict[str, str]]) -> str:
    payload = json.dumps(assignment_rows, sort_keys=True, separators=(",", ":"))
    return f"sha256:{_sha256_hex(payload)}"


def generate_manifest(
    strategy: str,
    splits: dict[str, list[str]],
    *,
    seed: int = 42,
    split_id: str = "",
    dataset_fingerprint: str = "",
    group_definition_version: str = "1",
    assignment_rows: list[dict[str, str]] | None = None,
    group_by_id: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> SplitManifestV2:
    """Build a split manifest from computed splits."""
    if assignment_rows is None:
        assignment_rows = build_assignment_records(splits, group_by_id=group_by_id)

    manifest = SplitManifestV2(
        strategy=strategy,
        seed=seed,
        split_id=split_id,
        dataset_fingerprint=dataset_fingerprint,
        group_definition_version=group_definition_version,
        assignment_hash=_assignment_hash(assignment_rows),
    )

    for name, ids in splits.items():
        group_count = 0
        if group_by_id:
            group_count = len({group_by_id.get(sample_id, "") for sample_id in ids})
        manifest.splits[name] = {
            "n_samples": len(ids),
            "sample_ids_hash": f"sha256:{_ids_hash(ids)}",
            "n_groups": group_count,
        }
    if extra_metadata:
        manifest.metadata = extra_metadata
    manifest.compute_hash()
    if not manifest.split_id:
        short_hash = manifest.split_hash.split(":", 1)[-1][:12]
        manifest.split_id = f"{strategy}_s{seed}_{short_hash}"
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make-splits",
        description="ATLAS split governance: deterministic IID, compositional, and prototype splits.",
    )
    parser.add_argument(
        "--strategy",
        choices=SPLIT_STRATEGIES,
        default="iid",
        help="Split strategy (default: iid)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)",
    )
    parser.add_argument(
        "--property-group",
        type=str,
        default="priority7",
        help="Property group for dataset loading",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Max samples (default: 3000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for split manifests (default: artifacts/splits/)",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.8,0.1,0.1",
        help="Train,val,test ratios (default: 0.8,0.1,0.1)",
    )
    parser.add_argument(
        "--emit-assignment",
        action="store_true",
        help="Emit per-sample assignment files (JSON + CSV)",
    )
    parser.add_argument(
        "--split-id",
        type=str,
        default="",
        help="Optional explicit split identifier (default: auto-generated)",
    )
    parser.add_argument(
        "--group-definition-version",
        type=str,
        default="1",
        help="Version string for grouping logic definition",
    )
    return parser


def _write_assignment_json(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return path


def _write_assignment_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for make-splits."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Parse ratios
    try:
        ratios = tuple(float(x) for x in args.ratios.split(","))
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 0.01:
            raise ValueError
    except (ValueError, TypeError):
        print("[ERROR] --ratios must be 3 comma-separated floats summing to 1.0")
        return 1

    # Determine strategies to run
    strategies = (
        ["iid", "compositional", "prototype"]
        if args.strategy == "all"
        else [args.strategy]
    )

    try:
        from atlas.data.crystal_dataset import (
            PROPERTY_MAP,
            resolve_phase2_property_group,
        )
        from atlas.data.jarvis_client import JARVISClient
    except ImportError as exc:
        print(f"[ERROR] Cannot import dataset module: {exc}")
        return 1

    # Resolve properties
    try:
        properties = resolve_phase2_property_group(args.property_group)
    except (KeyError, ValueError):
        from atlas.data.crystal_dataset import DEFAULT_PROPERTIES
        properties = list(DEFAULT_PROPERTIES)

    # Load raw DataFrame (no graph building needed for splits)
    try:
        import pandas as pd

        client = JARVISClient()
        df = client.load_dft_3d()

        # Filter valid rows (same logic as CrystalPropertyDataset)
        rev_map = {v: k for k, v in PROPERTY_MAP.items()}
        jarvis_cols = [rev_map[p] for p in properties if p in rev_map]
        valid_mask = pd.Series(False, index=df.index)
        for col in jarvis_cols:
            if col in df.columns:
                valid_mask |= df[col].notna() & (df[col] != "na")
        df = df[valid_mask].copy()

        # Stability filter
        if "ehull" in df.columns:
            df = df[df["ehull"].notna() & (df["ehull"] <= 0.1)]

        df = df[df["atoms"].notna()].reset_index(drop=True)

        if args.max_samples and len(df) > args.max_samples:
            df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)

    except Exception as exc:
        print(f"[ERROR] Could not load dataset: {exc}")
        return 1

    if len(df) == 0:
        print("[ERROR] Dataset is empty after filtering")
        return 1

    print(f"[make-splits] Loaded {len(df)} samples for splitting")

    sample_ids = [str(x) for x in df["jid"].tolist()] if "jid" in df.columns else [str(i) for i in range(len(df))]
    formulas = [str(x) for x in df["formula"].tolist()] if "formula" in df.columns else [""] * len(df)
    spacegroups = (
        [str(x) for x in df["spg_number"].tolist()]
        if "spg_number" in df.columns
        else ["unknown"] * len(df)
    )
    dataset_fingerprint = f"sha256:{_ids_hash(sample_ids)}"

    # Resolve output dir
    from atlas.config import get_config
    cfg = get_config()
    output_dir = args.output_dir or (cfg.paths.artifacts_dir / "splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run each strategy
    for strategy in strategies:
        group_by_id: dict[str, str] = {}
        if strategy == "iid":
            splits = iid_split(sample_ids, seed=args.seed, ratios=ratios)
            group_by_id = {sid: "iid" for sid in sample_ids}
        elif strategy == "compositional":
            splits = compositional_split(
                sample_ids, formulas, seed=args.seed, ratios=ratios
            )
            group_by_id = {
                sid: (_chemical_system(formula) if formula else "unknown")
                for sid, formula in zip(sample_ids, formulas)
            }
        elif strategy == "prototype":
            splits = prototype_split(
                sample_ids, spacegroups, seed=args.seed, ratios=ratios
            )
            group_by_id = {
                sid: (str(spacegroup) if spacegroup else "unknown")
                for sid, spacegroup in zip(sample_ids, spacegroups)
            }
        else:
            print(f"[WARN] Unknown strategy: {strategy}")
            continue

        assignment_rows = build_assignment_records(splits, group_by_id=group_by_id)
        overlap_counts = compute_split_overlap_counts(splits)
        if sum(overlap_counts.values()) > 0:
            print(f"[ERROR] Split overlap detected for strategy='{strategy}': {overlap_counts}")
            return 2
        split_id = args.split_id
        if split_id and args.strategy == "all":
            split_id = f"{split_id}_{strategy}"
        manifest = generate_manifest(
            strategy,
            splits,
            seed=args.seed,
            split_id=split_id,
            dataset_fingerprint=dataset_fingerprint,
            group_definition_version=args.group_definition_version,
            assignment_rows=assignment_rows,
            group_by_id=group_by_id,
            extra_metadata={
                "property_group": args.property_group,
                "max_samples": args.max_samples,
                "ratios": list(ratios),
                "n_total": len(sample_ids),
                "assignment_json": f"split_assignment_{strategy}.json",
                "assignment_csv": f"split_assignment_{strategy}.csv",
                "overlap_counts": overlap_counts,
            },
        )

        out_path = output_dir / f"split_manifest_{strategy}.json"
        manifest.to_json(out_path)
        if args.emit_assignment:
            assignment_json = output_dir / f"split_assignment_{strategy}.json"
            assignment_csv = output_dir / f"split_assignment_{strategy}.csv"
            _write_assignment_json(assignment_json, assignment_rows)
            _write_assignment_csv(assignment_csv, assignment_rows)

        print(f"[make-splits] {strategy} manifest saved: {out_path}")
        for name, info in manifest.splits.items():
            print(f"  {name}: n={info['n_samples']}")
        print(
            "  overlap: "
            f"train__val={overlap_counts['train__val']}, "
            f"train__test={overlap_counts['train__test']}, "
            f"val__test={overlap_counts['val__test']}"
        )
        print(f"  hash: {manifest.split_hash}")
        if args.emit_assignment:
            print(
                f"  assignment: json={output_dir / f'split_assignment_{strategy}.json'}, "
                f"csv={output_dir / f'split_assignment_{strategy}.csv'}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
