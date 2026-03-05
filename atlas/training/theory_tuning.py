"""
Theory-backed tuning profiles and round-to-round adaptation helpers.

The goal is to keep training defaults explainable and reproducible:
- round-1 starts from literature-aligned priors
- rounds 2/3 adapt deterministically based on observed metrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

ProfileParams = dict[str, Any]
ParamBounds = dict[str, tuple[float, float]]

_VALID_OBJECTIVE_MODES = {"min", "max"}


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool_", "bool"}


@dataclass(frozen=True)
class MetricObjective:
    """Metric extraction and optimization direction."""

    keys: tuple[str, ...]
    mode: str  # "min" or "max"
    min_relative_improvement: float = 0.015

    def __post_init__(self) -> None:
        normalized_keys = tuple(str(key).strip() for key in self.keys if str(key).strip())
        if not normalized_keys:
            raise ValueError("MetricObjective.keys must contain at least one non-empty key")
        mode = str(self.mode).strip().lower()
        if mode not in _VALID_OBJECTIVE_MODES:
            raise ValueError(f"MetricObjective.mode must be one of {_VALID_OBJECTIVE_MODES!r}")
        threshold = _coerce_non_negative_float(
            self.min_relative_improvement,
            "MetricObjective.min_relative_improvement",
        )

        object.__setattr__(self, "keys", normalized_keys)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "min_relative_improvement", threshold)

    def improvement(self, prev_score: float, curr_score: float) -> float:
        """Return relative improvement (>0 means better)."""
        prev = _coerce_finite_float(prev_score, "prev_score")
        curr = _coerce_finite_float(curr_score, "curr_score")
        denom = max(abs(prev), 1e-12)
        if self.mode == "min":
            return (prev - curr) / denom
        return (curr - prev) / denom


@dataclass(frozen=True)
class TheoryProfile:
    """Round-1 prior + adaptation boundaries."""

    phase: str
    algorithm: str
    stage: str
    params: ProfileParams
    bounds: ParamBounds
    objective: MetricObjective
    references: tuple[str, ...]

    def __post_init__(self) -> None:
        phase = _normalize_profile_token(self.phase, "phase")
        algorithm = _normalize_profile_token(self.algorithm, "algorithm")
        stage = _normalize_profile_token(self.stage, "stage")
        if not isinstance(self.params, dict):
            raise TypeError("TheoryProfile.params must be a dictionary")
        if not isinstance(self.bounds, dict):
            raise TypeError("TheoryProfile.bounds must be a dictionary")
        if not isinstance(self.references, tuple) or not self.references:
            raise ValueError("TheoryProfile.references must be a non-empty tuple")

        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "stage", stage)


_REG_BOUNDS: ParamBounds = {
    "epochs": (5, 3000),
    "batch-size": (4, 256),
    "lr": (1e-5, 5e-3),
    "max-samples": (200, 120000),
    "acc-steps": (1, 16),
    "r-max": (3.5, 8.0),
    "hidden-dim": (32, 1024),
    "n-conv": (1, 8),
}

_TOPO_GNN_BOUNDS: ParamBounds = {
    "epochs": (10, 600),
    "batch-size": (8, 128),
    "lr": (1e-5, 3e-3),
    "max-samples": (500, 30000),
    "hidden": (32, 384),
}

_RF_BOUNDS: ParamBounds = {
    "max-samples": (500, 30000),
    "n-estimators": (80, 3000),
    "max-depth": (4, 48),
    "min-samples-leaf": (1, 12),
}


def _obj(keys: tuple[str, ...], mode: str, thr: float = 0.015) -> MetricObjective:
    return MetricObjective(keys=keys, mode=mode, min_relative_improvement=thr)


def _normalize_profile_token(value: Any, name: str) -> str:
    token = str(value).strip().lower()
    if not token:
        raise ValueError(f"{name} must be a non-empty string")
    return token


def _coerce_finite_float(value: Any, name: str) -> float:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be a finite real number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _coerce_non_negative_float(value: Any, name: str) -> float:
    scalar = _coerce_finite_float(value, name)
    if scalar < 0:
        raise ValueError(f"{name} must be >= 0")
    return scalar


# IDs are documented in docs/THEORY_BACKED_TUNING.md
_REFS_NEURAL = (
    "cgcnn_2018",
    "smith_2018_disciplined",
    "smith_2017_superconvergence",
    "loshchilov_2019_adamw",
    "bergstra_bengio_2012_random_search",
    "li_2018_hyperband",
)
_REFS_E3NN = (
    "e3nn_2022",
    "kendall_2018_uncertainty_weighting",
    "smith_2018_disciplined",
    "loshchilov_2019_adamw",
    "li_2018_hyperband",
)
_REFS_MACE = (
    "mace_2022",
    "mace_mp_2023",
    "smith_2018_disciplined",
    "li_2018_hyperband",
)
_REFS_RF = (
    "breiman_2001_rf",
    "probst_2019_rf_tuning",
    "li_2018_hyperband",
)


PROFILES: dict[tuple[str, str, str], TheoryProfile] = {
    # ------------------------------------------------------------------
    # Phase 1 (CGCNN)
    # ------------------------------------------------------------------
    ("phase1", "cgcnn", "lite"): TheoryProfile(
        phase="phase1",
        algorithm="cgcnn",
        stage="lite",
        params={
            "level": "lite",
            "epochs": 40,
            "batch-size": 32,
            "lr": 0.003,
            "max-samples": 4000,
            "hidden-dim": 64,
            "n-conv": 2,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae", "test_mae"), "min", 0.02),
        references=_REFS_NEURAL,
    ),
    ("phase1", "cgcnn", "std"): TheoryProfile(
        phase="phase1",
        algorithm="cgcnn",
        stage="std",
        params={
            "level": "std",
            "epochs": 300,
            "batch-size": 64,
            "lr": 0.001,
            "max-samples": 30000,
            "hidden-dim": 128,
            "n-conv": 3,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae", "test_mae"), "min", 0.015),
        references=_REFS_NEURAL,
    ),
    ("phase1", "cgcnn", "competition"): TheoryProfile(
        phase="phase1",
        algorithm="cgcnn",
        stage="competition",
        params={
            "competition": True,
            "epochs": 450,
            "batch-size": 48,
            "lr": 8e-4,
            "max-samples": 40000,
            "hidden-dim": 192,
            "n-conv": 4,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae", "test_mae"), "min", 0.012),
        references=_REFS_NEURAL,
    ),
    ("phase1", "cgcnn", "pro"): TheoryProfile(
        phase="phase1",
        algorithm="cgcnn",
        stage="pro",
        params={
            "level": "pro",
            "epochs": 900,
            "batch-size": 48,
            "lr": 6e-4,
            "max-samples": 50000,
            "hidden-dim": 384,
            "n-conv": 5,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae", "test_mae"), "min", 0.01),
        references=_REFS_NEURAL,
    ),
    ("phase1", "cgcnn", "max"): TheoryProfile(
        phase="phase1",
        algorithm="cgcnn",
        stage="max",
        params={
            "level": "max",
            "epochs": 1400,
            "batch-size": 32,
            "lr": 4e-4,
            "max-samples": 60000,
            "hidden-dim": 512,
            "n-conv": 6,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae", "test_mae"), "min", 0.008),
        references=_REFS_NEURAL,
    ),
    # ------------------------------------------------------------------
    # Phase 2 (E3NN multitask)
    # ------------------------------------------------------------------
    ("phase2", "e3nn", "lite"): TheoryProfile(
        phase="phase2",
        algorithm="e3nn",
        stage="lite",
        params={"level": "lite", "epochs": 30, "batch-size": 8, "lr": 0.002},
        bounds=_REG_BOUNDS,
        objective=_obj(("best_train_loss",), "min", 0.02),
        references=_REFS_E3NN,
    ),
    ("phase2", "e3nn", "std"): TheoryProfile(
        phase="phase2",
        algorithm="e3nn",
        stage="std",
        params={"level": "std", "epochs": 120, "batch-size": 16, "lr": 0.001},
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.015),
        references=_REFS_E3NN,
    ),
    ("phase2", "e3nn", "competition"): TheoryProfile(
        phase="phase2",
        algorithm="e3nn",
        stage="competition",
        params={"competition": True, "epochs": 260, "batch-size": 6, "lr": 4.5e-4},
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.012),
        references=_REFS_E3NN,
    ),
    ("phase2", "e3nn", "pro"): TheoryProfile(
        phase="phase2",
        algorithm="e3nn",
        stage="pro",
        params={"level": "pro", "epochs": 500, "batch-size": 4, "lr": 5e-4},
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.01),
        references=_REFS_E3NN,
    ),
    ("phase2", "e3nn", "max"): TheoryProfile(
        phase="phase2",
        algorithm="e3nn",
        stage="max",
        params={"level": "max", "epochs": 800, "batch-size": 4, "lr": 3e-4},
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.008),
        references=_REFS_E3NN,
    ),
    # ------------------------------------------------------------------
    # Phase 2 (CGCNN multitask)
    # ------------------------------------------------------------------
    ("phase2", "cgcnn", "lite"): TheoryProfile(
        phase="phase2",
        algorithm="cgcnn",
        stage="lite",
        params={
            "level": "lite",
            "preset": "small",
            "epochs": 60,
            "batch-size": 96,
            "lr": 0.0015,
            "max-samples": 6000,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("avg_test_mae",), "min", 0.02),
        references=_REFS_NEURAL,
    ),
    ("phase2", "cgcnn", "std"): TheoryProfile(
        phase="phase2",
        algorithm="cgcnn",
        stage="std",
        params={
            "level": "std",
            "preset": "medium",
            "epochs": 220,
            "batch-size": 128,
            "lr": 0.001,
            "max-samples": 30000,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("avg_test_mae",), "min", 0.015),
        references=_REFS_NEURAL,
    ),
    ("phase2", "cgcnn", "competition"): TheoryProfile(
        phase="phase2",
        algorithm="cgcnn",
        stage="competition",
        params={
            "competition": True,
            "preset": "medium",
            "epochs": 280,
            "batch-size": 128,
            "lr": 9e-4,
            "max-samples": 40000,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("avg_test_mae",), "min", 0.012),
        references=_REFS_NEURAL,
    ),
    ("phase2", "cgcnn", "pro"): TheoryProfile(
        phase="phase2",
        algorithm="cgcnn",
        stage="pro",
        params={
            "level": "pro",
            "preset": "large",
            "epochs": 320,
            "batch-size": 128,
            "lr": 8e-4,
            "max-samples": 50000,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("avg_test_mae",), "min", 0.01),
        references=_REFS_NEURAL,
    ),
    ("phase2", "cgcnn", "max"): TheoryProfile(
        phase="phase2",
        algorithm="cgcnn",
        stage="max",
        params={
            "level": "max",
            "preset": "large",
            "epochs": 500,
            "batch-size": 160,
            "lr": 7e-4,
            "max-samples": 60000,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("avg_test_mae",), "min", 0.008),
        references=_REFS_NEURAL,
    ),
    # ------------------------------------------------------------------
    # Phase 3 (Equivariant specialist)
    # ------------------------------------------------------------------
    ("phase3", "equivariant", "lite"): TheoryProfile(
        phase="phase3",
        algorithm="equivariant",
        stage="lite",
        params={
            "level": "lite",
            "epochs": 300,
            "batch-size": 12,
            "acc-steps": 2,
            "lr": 3e-4,
            "max-samples": 12000,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.015),
        references=_REFS_E3NN,
    ),
    ("phase3", "equivariant", "std"): TheoryProfile(
        phase="phase3",
        algorithm="equivariant",
        stage="std",
        params={
            "level": "std",
            "epochs": 800,
            "batch-size": 16,
            "acc-steps": 4,
            "lr": 2e-4,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.012),
        references=_REFS_E3NN,
    ),
    ("phase3", "equivariant", "competition"): TheoryProfile(
        phase="phase3",
        algorithm="equivariant",
        stage="competition",
        params={
            "competition": True,
            "epochs": 900,
            "batch-size": 16,
            "acc-steps": 3,
            "lr": 2e-4,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.01),
        references=_REFS_E3NN,
    ),
    ("phase3", "equivariant", "pro"): TheoryProfile(
        phase="phase3",
        algorithm="equivariant",
        stage="pro",
        params={
            "level": "pro",
            "epochs": 1500,
            "batch-size": 16,
            "acc-steps": 4,
            "lr": 2e-4,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.008),
        references=_REFS_E3NN,
    ),
    ("phase3", "equivariant", "max"): TheoryProfile(
        phase="phase3",
        algorithm="equivariant",
        stage="max",
        params={
            "level": "max",
            "epochs": 2200,
            "batch-size": 16,
            "acc-steps": 6,
            "lr": 1.5e-4,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("best_val_mae",), "min", 0.006),
        references=_REFS_E3NN,
    ),
    # ------------------------------------------------------------------
    # Phase 3 (MACE potentials)
    # ------------------------------------------------------------------
    ("phase3", "mace", "lite"): TheoryProfile(
        phase="phase3",
        algorithm="mace",
        stage="lite",
        params={
            "level": "lite",
            "epochs": 120,
            "batch-size": 16,
            "lr": 5e-4,
            "r-max": 5.0,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("return_code",), "min", 0.0),
        references=_REFS_MACE,
    ),
    ("phase3", "mace", "std"): TheoryProfile(
        phase="phase3",
        algorithm="mace",
        stage="std",
        params={
            "level": "std",
            "epochs": 350,
            "batch-size": 32,
            "lr": 3e-4,
            "r-max": 5.0,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("return_code",), "min", 0.0),
        references=_REFS_MACE,
    ),
    ("phase3", "mace", "competition"): TheoryProfile(
        phase="phase3",
        algorithm="mace",
        stage="competition",
        params={
            "competition": True,
            "epochs": 450,
            "batch-size": 32,
            "lr": 2.5e-4,
            "r-max": 5.5,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("return_code",), "min", 0.0),
        references=_REFS_MACE,
    ),
    ("phase3", "mace", "pro"): TheoryProfile(
        phase="phase3",
        algorithm="mace",
        stage="pro",
        params={
            "level": "pro",
            "epochs": 700,
            "batch-size": 32,
            "lr": 2e-4,
            "r-max": 5.5,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("return_code",), "min", 0.0),
        references=_REFS_MACE,
    ),
    ("phase3", "mace", "max"): TheoryProfile(
        phase="phase3",
        algorithm="mace",
        stage="max",
        params={
            "level": "max",
            "epochs": 1000,
            "batch-size": 48,
            "lr": 1e-4,
            "r-max": 6.0,
        },
        bounds=_REG_BOUNDS,
        objective=_obj(("return_code",), "min", 0.0),
        references=_REFS_MACE,
    ),
    # ------------------------------------------------------------------
    # Phase 4 (TopoGNN)
    # ------------------------------------------------------------------
    ("phase4", "topognn", "lite"): TheoryProfile(
        phase="phase4",
        algorithm="topognn",
        stage="lite",
        params={
            "level": "lite",
            "epochs": 40,
            "batch-size": 24,
            "lr": 0.001,
            "hidden": 96,
            "max-samples": 2500,
        },
        bounds=_TOPO_GNN_BOUNDS,
        objective=_obj(("best_val_acc",), "max", 0.01),
        references=_REFS_NEURAL,
    ),
    ("phase4", "topognn", "std"): TheoryProfile(
        phase="phase4",
        algorithm="topognn",
        stage="std",
        params={
            "level": "std",
            "epochs": 120,
            "batch-size": 32,
            "lr": 8e-4,
            "hidden": 128,
            "max-samples": 5000,
        },
        bounds=_TOPO_GNN_BOUNDS,
        objective=_obj(("best_val_acc",), "max", 0.008),
        references=_REFS_NEURAL,
    ),
    ("phase4", "topognn", "competition"): TheoryProfile(
        phase="phase4",
        algorithm="topognn",
        stage="competition",
        params={
            "competition": True,
            "epochs": 160,
            "batch-size": 40,
            "lr": 7e-4,
            "hidden": 144,
            "max-samples": 6000,
        },
        bounds=_TOPO_GNN_BOUNDS,
        objective=_obj(("best_val_acc",), "max", 0.008),
        references=_REFS_NEURAL,
    ),
    ("phase4", "topognn", "pro"): TheoryProfile(
        phase="phase4",
        algorithm="topognn",
        stage="pro",
        params={
            "level": "pro",
            "epochs": 220,
            "batch-size": 48,
            "lr": 6e-4,
            "hidden": 160,
            "max-samples": 8000,
        },
        bounds=_TOPO_GNN_BOUNDS,
        objective=_obj(("best_val_acc",), "max", 0.006),
        references=_REFS_NEURAL,
    ),
    ("phase4", "topognn", "max"): TheoryProfile(
        phase="phase4",
        algorithm="topognn",
        stage="max",
        params={
            "level": "max",
            "epochs": 320,
            "batch-size": 64,
            "lr": 5e-4,
            "hidden": 192,
            "max-samples": 12000,
        },
        bounds=_TOPO_GNN_BOUNDS,
        objective=_obj(("best_val_acc",), "max", 0.005),
        references=_REFS_NEURAL,
    ),
    # ------------------------------------------------------------------
    # Phase 4 (Random Forest)
    # ------------------------------------------------------------------
    ("phase4", "rf", "lite"): TheoryProfile(
        phase="phase4",
        algorithm="rf",
        stage="lite",
        params={
            "level": "lite",
            "max-samples": 2500,
            "n-estimators": 300,
            "max-depth": 16,
            "min-samples-leaf": 2,
        },
        bounds=_RF_BOUNDS,
        objective=_obj(("validation_f1",), "max", 0.01),
        references=_REFS_RF,
    ),
    ("phase4", "rf", "std"): TheoryProfile(
        phase="phase4",
        algorithm="rf",
        stage="std",
        params={
            "level": "std",
            "max-samples": 5000,
            "n-estimators": 700,
            "max-depth": 24,
            "min-samples-leaf": 2,
        },
        bounds=_RF_BOUNDS,
        objective=_obj(("validation_f1",), "max", 0.008),
        references=_REFS_RF,
    ),
    ("phase4", "rf", "competition"): TheoryProfile(
        phase="phase4",
        algorithm="rf",
        stage="competition",
        params={
            "competition": True,
            "max-samples": 7000,
            "n-estimators": 900,
            "max-depth": 22,
            "min-samples-leaf": 2,
        },
        bounds=_RF_BOUNDS,
        objective=_obj(("validation_f1",), "max", 0.008),
        references=_REFS_RF,
    ),
    ("phase4", "rf", "pro"): TheoryProfile(
        phase="phase4",
        algorithm="rf",
        stage="pro",
        params={
            "level": "pro",
            "max-samples": 9000,
            "n-estimators": 1200,
            "max-depth": 28,
            "min-samples-leaf": 1,
        },
        bounds=_RF_BOUNDS,
        objective=_obj(("validation_f1",), "max", 0.006),
        references=_REFS_RF,
    ),
    ("phase4", "rf", "max"): TheoryProfile(
        phase="phase4",
        algorithm="rf",
        stage="max",
        params={
            "level": "max",
            "max-samples": 12000,
            "n-estimators": 1500,
            "max-depth": 32,
            "min-samples-leaf": 1,
        },
        bounds=_RF_BOUNDS,
        objective=_obj(("validation_f1",), "max", 0.005),
        references=_REFS_RF,
    ),
}


DEFAULT_STAGE_ORDER = ["lite", "std", "competition", "pro", "max"]

DEFAULT_PHASE_ALGORITHMS = {
    "phase1": ["cgcnn"],
    "phase2": ["e3nn", "cgcnn"],
    "phase3": ["equivariant", "mace"],
    "phase4": ["topognn", "rf"],
}


def get_profile(phase: str, algorithm: str, stage: str) -> TheoryProfile:
    key = (
        _normalize_profile_token(phase, "phase"),
        _normalize_profile_token(algorithm, "algorithm"),
        _normalize_profile_token(stage, "stage"),
    )
    if key not in PROFILES:
        raise KeyError(f"No theory profile for phase={phase}, algorithm={algorithm}, stage={stage}")
    return PROFILES[key]


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _coerce(param: str, value: float, template: Any) -> Any:
    if _is_boolean_like(template):
        return bool(round(value))
    if isinstance(template, int):
        return int(round(value))
    if isinstance(template, float):
        return float(value)
    _INT_PARAMS = {
        "epochs", "batch-size", "max-samples", "acc-steps",
        "hidden-dim", "n-conv", "hidden", "n-estimators",
        "max-depth", "min-samples-leaf",
    }
    if param in _INT_PARAMS:
        return int(round(value))
    return value


def _adjust_numeric(
    params: ProfileParams,
    bounds: ParamBounds,
    key: str,
    *,
    factor: float | None = None,
    delta: float | None = None,
) -> None:
    if factor is None and delta is None:
        return
    if key not in params:
        return
    cur = params[key]
    if _is_boolean_like(cur) or not isinstance(cur, (int, float)):
        return
    new_val = float(cur)
    if factor is not None:
        new_val *= factor
    if delta is not None:
        new_val += delta
    lo, hi = bounds.get(key, (-float("inf"), float("inf")))
    new_val = _clip(new_val, lo, hi)
    params[key] = _coerce(key, new_val, cur)


def adapt_params_for_next_round(
    *,
    profile: TheoryProfile,
    current_params: ProfileParams,
    previous_score: float | None,
    current_score: float | None,
    failed: bool,
) -> tuple[ProfileParams, str, float | None]:
    """
    Return (next_params, reason, relative_improvement).

    Design notes:
    - first transition (round1 -> round2) gets a deterministic warm-up ramp
    - if failed / metric missing, use a conservative recovery move
    - otherwise adapt by relative improvement against threshold
    """
    if not isinstance(profile, TheoryProfile):
        raise TypeError(f"profile must be TheoryProfile, got {type(profile)!r}")
    if not isinstance(current_params, dict):
        raise TypeError("current_params must be a dictionary")

    next_params: ProfileParams = dict(current_params)
    if failed:
        _adjust_numeric(next_params, profile.bounds, "lr", factor=0.6)
        _adjust_numeric(next_params, profile.bounds, "batch-size", factor=0.75)
        _adjust_numeric(next_params, profile.bounds, "epochs", factor=1.35)
        _adjust_numeric(next_params, profile.bounds, "max-samples", factor=1.15)
        _adjust_numeric(next_params, profile.bounds, "n-estimators", factor=1.3)
        _adjust_numeric(next_params, profile.bounds, "max-depth", delta=2)
        return next_params, "metric_missing_or_run_failed_recovery", None

    prev_score = (
        _coerce_finite_float(previous_score, "previous_score")
        if previous_score is not None
        else None
    )
    curr_score = (
        _coerce_finite_float(current_score, "current_score")
        if current_score is not None
        else None
    )

    obj = profile.objective
    improvement: float | None = None

    if curr_score is None:
        # Recovery: lower LR, slightly smaller batch, more epochs.
        _adjust_numeric(next_params, profile.bounds, "lr", factor=0.6)
        _adjust_numeric(next_params, profile.bounds, "batch-size", factor=0.75)
        _adjust_numeric(next_params, profile.bounds, "epochs", factor=1.35)
        _adjust_numeric(next_params, profile.bounds, "max-samples", factor=1.15)
        _adjust_numeric(next_params, profile.bounds, "n-estimators", factor=1.3)
        _adjust_numeric(next_params, profile.bounds, "max-depth", delta=2)
        return next_params, "metric_missing_or_run_failed_recovery", None

    if prev_score is None:
        # Round-1 -> Round-2: deterministic resource ramp (successive-halving style).
        _adjust_numeric(next_params, profile.bounds, "lr", factor=0.9)
        _adjust_numeric(next_params, profile.bounds, "epochs", factor=1.2)
        _adjust_numeric(next_params, profile.bounds, "max-samples", factor=1.15)
        _adjust_numeric(next_params, profile.bounds, "n-estimators", factor=1.15)
        return next_params, "round1_to_round2_resource_ramp", None

    improvement = obj.improvement(prev_score, curr_score)
    if improvement < obj.min_relative_improvement:
        # Improvement too small: stronger optimization move.
        _adjust_numeric(next_params, profile.bounds, "lr", factor=0.7)
        _adjust_numeric(next_params, profile.bounds, "epochs", factor=1.25)
        _adjust_numeric(next_params, profile.bounds, "max-samples", factor=1.2)
        _adjust_numeric(next_params, profile.bounds, "batch-size", factor=0.8)
        _adjust_numeric(next_params, profile.bounds, "n-estimators", factor=1.25)
        _adjust_numeric(next_params, profile.bounds, "max-depth", delta=2)
        _adjust_numeric(next_params, profile.bounds, "min-samples-leaf", delta=-1)
        reason = (
            f"improvement_below_threshold"
            f"(relative={improvement:.4f},threshold={obj.min_relative_improvement:.4f})"
        )
        return next_params, reason, improvement

    # Good improvement: continue with mild annealing.
    _adjust_numeric(next_params, profile.bounds, "lr", factor=0.9)
    _adjust_numeric(next_params, profile.bounds, "epochs", factor=1.1)
    _adjust_numeric(next_params, profile.bounds, "max-samples", factor=1.1)
    _adjust_numeric(next_params, profile.bounds, "n-estimators", factor=1.1)
    reason = (
        f"improvement_good_anneal"
        f"(relative={improvement:.4f},threshold={obj.min_relative_improvement:.4f})"
    )
    return next_params, reason, improvement


def extract_score_from_manifest(
    manifest: dict[str, Any] | None,
    objective: MetricObjective,
) -> float | None:
    if not manifest:
        return None
    result = manifest.get("result", {})
    if not isinstance(result, dict):
        result = {}
    for key in objective.keys:
        if key in result:
            value = result[key]
            if _is_boolean_like(value):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        if key in manifest:
            value = manifest[key]
            if _is_boolean_like(value):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    return None
