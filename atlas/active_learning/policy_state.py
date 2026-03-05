"""Policy state and typed config for active-learning decision engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "y", "on"}:
            return True
        if key in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        out_f = float(value)
        if not np.isfinite(out_f) or not out_f.is_integer():
            return int(default)
        return int(out_f)
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _as_finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _coerce_text(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    text = str(value).strip().lower()
    if not text:
        return str(default)
    return text


def _conformal_quantile_level(n_points: int, alpha: float) -> float:
    """Finite-sample split-conformal quantile level."""
    n = max(1, _coerce_int(n_points, 1))
    a = float(np.clip(_coerce_float(alpha, 0.05), 1e-9, 1.0 - 1e-9))
    level = float(np.ceil((n + 1) * (1.0 - a)) / n)
    return float(np.clip(level, 0.0, 1.0))


@dataclass
class ActiveLearningPolicyConfig:
    """Typed policy-level configuration replacing ad-hoc getattr access."""

    policy_name: str = "legacy"
    risk_mode: str = "soft"  # soft | hard | hybrid
    cost_aware: bool = False
    calibration_window: int = 128
    ood_combination: str = "and"  # and | or
    ood_gate_threshold: float = 0.85
    conformal_gate_threshold: float = 0.85
    conformal_alpha: float = 0.05
    conformal_penalty_weight: float = 0.25
    ood_penalty_weight: float = 0.20
    max_conformal_radius: float = 2.5
    diversity_novelty_boost: float = 0.10
    cost_eps: float = 1e-6
    relax_cost_base: float = 1.0
    classify_cost_base: float = 0.2
    calibration_min_points: int = 12

    # Runtime stabilizers
    relax_timeout_sec: float = 0.0
    relax_max_retries: int = 1
    relax_circuit_breaker_failures: int = 8
    relax_circuit_breaker_cooldown_iters: int = 2
    relax_retry_backoff_sec: float = 0.0
    relax_retry_backoff_max_sec: float = 0.0
    relax_retry_jitter: float = 0.0

    @classmethod
    def from_profile(
        cls,
        profile: object,
        *,
        policy_name: str = "legacy",
        risk_mode: str = "soft",
        cost_aware: bool = False,
        calibration_window: int = 128,
    ) -> ActiveLearningPolicyConfig:
        return cls(
            policy_name=_coerce_text(getattr(profile, "policy_name", policy_name), policy_name),
            risk_mode=_coerce_text(getattr(profile, "risk_mode", risk_mode), risk_mode),
            cost_aware=_coerce_bool(getattr(profile, "cost_aware", cost_aware), cost_aware),
            calibration_window=_coerce_int(getattr(profile, "calibration_window", calibration_window), calibration_window),
            ood_combination=_coerce_text(getattr(profile, "ood_combination", "and"), "and"),
            ood_gate_threshold=_coerce_float(getattr(profile, "ood_gate_threshold", 0.85), 0.85),
            conformal_gate_threshold=_coerce_float(getattr(profile, "conformal_gate_threshold", 0.85), 0.85),
            conformal_alpha=_coerce_float(getattr(profile, "conformal_alpha", 0.05), 0.05),
            conformal_penalty_weight=_coerce_float(getattr(profile, "conformal_penalty_weight", 0.25), 0.25),
            ood_penalty_weight=_coerce_float(getattr(profile, "policy_ood_penalty_weight", 0.20), 0.20),
            max_conformal_radius=_coerce_float(getattr(profile, "max_conformal_radius", 2.5), 2.5),
            diversity_novelty_boost=_coerce_float(getattr(profile, "diversity_novelty_boost", 0.10), 0.10),
            cost_eps=_coerce_float(getattr(profile, "cost_eps", 1e-6), 1e-6),
            relax_cost_base=_coerce_float(getattr(profile, "relax_cost_base", 1.0), 1.0),
            classify_cost_base=_coerce_float(getattr(profile, "classify_cost_base", 0.2), 0.2),
            calibration_min_points=_coerce_int(getattr(profile, "calibration_min_points", 12), 12),
            relax_timeout_sec=_coerce_float(getattr(profile, "relax_timeout_sec", 0.0), 0.0),
            relax_max_retries=_coerce_int(getattr(profile, "relax_max_retries", 1), 1),
            relax_circuit_breaker_failures=_coerce_int(getattr(profile, "relax_circuit_breaker_failures", 8), 8),
            relax_circuit_breaker_cooldown_iters=_coerce_int(
                getattr(profile, "relax_circuit_breaker_cooldown_iters", 2),
                2,
            ),
            relax_retry_backoff_sec=_coerce_float(getattr(profile, "relax_retry_backoff_sec", 0.0), 0.0),
            relax_retry_backoff_max_sec=_coerce_float(getattr(profile, "relax_retry_backoff_max_sec", 0.0), 0.0),
            relax_retry_jitter=_coerce_float(getattr(profile, "relax_retry_jitter", 0.0), 0.0),
        ).validated()

    def validated(self) -> ActiveLearningPolicyConfig:
        self.policy_name = _coerce_text(self.policy_name, "legacy")
        if self.policy_name not in {"legacy", "cmoeic"}:
            raise ValueError(f"Unsupported policy_name: {self.policy_name!r}")

        self.risk_mode = _coerce_text(self.risk_mode, "soft")
        if self.risk_mode not in {"soft", "hard", "hybrid"}:
            raise ValueError(f"Unsupported risk_mode: {self.risk_mode!r}")

        self.ood_combination = _coerce_text(self.ood_combination, "and")
        if self.ood_combination not in {"and", "or"}:
            raise ValueError(f"Unsupported ood_combination: {self.ood_combination!r}")

        self.cost_aware = _coerce_bool(self.cost_aware, False)
        self.calibration_window = max(1, _coerce_int(self.calibration_window, 128))
        self.ood_gate_threshold = float(np.clip(_coerce_float(self.ood_gate_threshold, 0.85), 0.0, 1.0))
        self.conformal_gate_threshold = float(
            np.clip(_coerce_float(self.conformal_gate_threshold, 0.85), 0.0, 1.0)
        )
        self.conformal_alpha = float(np.clip(_coerce_float(self.conformal_alpha, 0.05), 1e-6, 0.25))
        self.conformal_penalty_weight = max(0.0, _coerce_float(self.conformal_penalty_weight, 0.25))
        self.ood_penalty_weight = max(0.0, _coerce_float(self.ood_penalty_weight, 0.20))
        self.max_conformal_radius = max(1e-6, _coerce_float(self.max_conformal_radius, 2.5))
        self.diversity_novelty_boost = max(0.0, _coerce_float(self.diversity_novelty_boost, 0.10))
        self.cost_eps = max(1e-9, _coerce_float(self.cost_eps, 1e-6))
        self.relax_cost_base = max(1e-6, _coerce_float(self.relax_cost_base, 1.0))
        self.classify_cost_base = max(1e-6, _coerce_float(self.classify_cost_base, 0.2))
        self.calibration_min_points = max(4, _coerce_int(self.calibration_min_points, 12))

        self.relax_timeout_sec = max(0.0, _coerce_float(self.relax_timeout_sec, 0.0))
        self.relax_max_retries = max(0, _coerce_int(self.relax_max_retries, 1))
        self.relax_circuit_breaker_failures = max(
            1,
            _coerce_int(self.relax_circuit_breaker_failures, 8),
        )
        self.relax_circuit_breaker_cooldown_iters = max(
            1,
            _coerce_int(self.relax_circuit_breaker_cooldown_iters, 2),
        )
        self.relax_retry_backoff_sec = max(0.0, _coerce_float(self.relax_retry_backoff_sec, 0.0))
        self.relax_retry_backoff_max_sec = max(
            self.relax_retry_backoff_sec,
            _coerce_float(self.relax_retry_backoff_max_sec, self.relax_retry_backoff_sec),
        )
        self.relax_retry_jitter = float(np.clip(_coerce_float(self.relax_retry_jitter, 0.0), 0.0, 1.0))
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyState:
    """Mutable policy runtime state that can be persisted across resume."""

    iteration: int = 0
    calibration_points: int = 0
    std_scale: float = 1.0
    conformal_scale: float = 1.0
    last_calibration_error: float = 0.0

    relax_circuit_open_until_iter: int = 0
    relax_consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> PolicyState:
        if not isinstance(payload, dict):
            return cls()
        return cls(
            iteration=_coerce_int(payload.get("iteration", 0), 0),
            calibration_points=_coerce_int(payload.get("calibration_points", 0), 0),
            std_scale=_coerce_float(payload.get("std_scale", 1.0), 1.0),
            conformal_scale=_coerce_float(payload.get("conformal_scale", 1.0), 1.0),
            last_calibration_error=_coerce_float(payload.get("last_calibration_error", 0.0), 0.0),
            relax_circuit_open_until_iter=_coerce_int(payload.get("relax_circuit_open_until_iter", 0), 0),
            relax_consecutive_failures=_coerce_int(payload.get("relax_consecutive_failures", 0), 0),
        ).validated()

    def validated(self) -> PolicyState:
        self.iteration = max(0, _coerce_int(self.iteration, 0))
        self.calibration_points = max(0, _coerce_int(self.calibration_points, 0))
        self.std_scale = max(1e-6, _coerce_float(self.std_scale, 1.0))
        self.conformal_scale = max(1e-6, _coerce_float(self.conformal_scale, 1.0))
        self.last_calibration_error = max(0.0, _coerce_float(self.last_calibration_error, 0.0))
        self.relax_circuit_open_until_iter = max(0, _coerce_int(self.relax_circuit_open_until_iter, 0))
        self.relax_consecutive_failures = max(0, _coerce_int(self.relax_consecutive_failures, 0))
        return self

    def update_calibration(self, history_candidates: list[object], *, cfg: ActiveLearningPolicyConfig) -> None:
        """
        Estimate robust calibration factors from recent prediction residuals.

        Uses recent candidates with both predicted and relaxed energies:
        - std_scale: RMS normalized residual scale
        - conformal_scale: quantile scale for interval radius
        """
        window = max(1, int(cfg.calibration_window))
        recent = history_candidates[-window:] if len(history_candidates) > window else history_candidates
        residuals: list[float] = []
        normalized: list[float] = []
        for cand in recent:
            pred = _as_finite_float(getattr(cand, "energy_mean", None))
            obs = _as_finite_float(getattr(cand, "energy_per_atom", None))
            std = _as_finite_float(getattr(cand, "energy_std", None))
            if pred is None or obs is None:
                continue
            err = abs(float(obs) - float(pred))
            residuals.append(err)
            if std is not None:
                denom = max(1e-6, abs(float(std)))
                normalized.append(err / denom)

        n = len(residuals)
        self.calibration_points = n
        if n < int(cfg.calibration_min_points):
            self.std_scale = 1.0
            self.conformal_scale = 1.0
            self.last_calibration_error = float(np.mean(residuals)) if residuals else 0.0
            return

        if normalized:
            z = np.asarray(normalized, dtype=float)
            z = z[np.isfinite(z)]
            if z.size > 0:
                self.std_scale = float(max(1.0, np.sqrt(np.mean(z * z))))
                q = float(np.quantile(z, _conformal_quantile_level(int(z.size), float(cfg.conformal_alpha))))
                self.conformal_scale = max(1.0, q)
            else:
                self.std_scale = 1.0
                self.conformal_scale = 1.0
        else:
            self.std_scale = 1.0
            self.conformal_scale = 1.0

        self.last_calibration_error = float(np.mean(residuals)) if residuals else 0.0
