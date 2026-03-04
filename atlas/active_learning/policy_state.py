"""Policy state and typed config for active-learning decision engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


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
            policy_name=str(getattr(profile, "policy_name", policy_name)).strip().lower(),
            risk_mode=str(getattr(profile, "risk_mode", risk_mode)).strip().lower(),
            cost_aware=bool(getattr(profile, "cost_aware", cost_aware)),
            calibration_window=int(getattr(profile, "calibration_window", calibration_window)),
            ood_combination=str(getattr(profile, "ood_combination", "and")).strip().lower(),
            ood_gate_threshold=float(getattr(profile, "ood_gate_threshold", 0.85)),
            conformal_gate_threshold=float(getattr(profile, "conformal_gate_threshold", 0.85)),
            conformal_alpha=float(getattr(profile, "conformal_alpha", 0.05)),
            conformal_penalty_weight=float(getattr(profile, "conformal_penalty_weight", 0.25)),
            ood_penalty_weight=float(getattr(profile, "policy_ood_penalty_weight", 0.20)),
            max_conformal_radius=float(getattr(profile, "max_conformal_radius", 2.5)),
            diversity_novelty_boost=float(getattr(profile, "diversity_novelty_boost", 0.10)),
            cost_eps=float(getattr(profile, "cost_eps", 1e-6)),
            relax_cost_base=float(getattr(profile, "relax_cost_base", 1.0)),
            classify_cost_base=float(getattr(profile, "classify_cost_base", 0.2)),
            calibration_min_points=int(getattr(profile, "calibration_min_points", 12)),
            relax_timeout_sec=float(getattr(profile, "relax_timeout_sec", 0.0)),
            relax_max_retries=int(getattr(profile, "relax_max_retries", 1)),
            relax_circuit_breaker_failures=int(getattr(profile, "relax_circuit_breaker_failures", 8)),
            relax_circuit_breaker_cooldown_iters=int(getattr(profile, "relax_circuit_breaker_cooldown_iters", 2)),
        ).validated()

    def validated(self) -> ActiveLearningPolicyConfig:
        self.policy_name = str(self.policy_name).strip().lower()
        if self.policy_name not in {"legacy", "cmoeic"}:
            raise ValueError(f"Unsupported policy_name: {self.policy_name!r}")

        self.risk_mode = str(self.risk_mode).strip().lower()
        if self.risk_mode not in {"soft", "hard", "hybrid"}:
            raise ValueError(f"Unsupported risk_mode: {self.risk_mode!r}")

        self.ood_combination = str(self.ood_combination).strip().lower()
        if self.ood_combination not in {"and", "or"}:
            raise ValueError(f"Unsupported ood_combination: {self.ood_combination!r}")

        self.calibration_window = max(1, int(self.calibration_window))
        self.ood_gate_threshold = float(np.clip(self.ood_gate_threshold, 0.0, 1.0))
        self.conformal_gate_threshold = float(np.clip(self.conformal_gate_threshold, 0.0, 1.0))
        self.conformal_alpha = float(np.clip(self.conformal_alpha, 1e-6, 0.25))
        self.conformal_penalty_weight = max(0.0, float(self.conformal_penalty_weight))
        self.ood_penalty_weight = max(0.0, float(self.ood_penalty_weight))
        self.max_conformal_radius = max(1e-6, float(self.max_conformal_radius))
        self.diversity_novelty_boost = max(0.0, float(self.diversity_novelty_boost))
        self.cost_eps = max(1e-9, float(self.cost_eps))
        self.relax_cost_base = max(1e-6, float(self.relax_cost_base))
        self.classify_cost_base = max(1e-6, float(self.classify_cost_base))
        self.calibration_min_points = max(4, int(self.calibration_min_points))

        self.relax_timeout_sec = max(0.0, float(self.relax_timeout_sec))
        self.relax_max_retries = max(0, int(self.relax_max_retries))
        self.relax_circuit_breaker_failures = max(1, int(self.relax_circuit_breaker_failures))
        self.relax_circuit_breaker_cooldown_iters = max(1, int(self.relax_circuit_breaker_cooldown_iters))
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
            iteration=int(payload.get("iteration", 0)),
            calibration_points=int(payload.get("calibration_points", 0)),
            std_scale=float(payload.get("std_scale", 1.0)),
            conformal_scale=float(payload.get("conformal_scale", 1.0)),
            last_calibration_error=float(payload.get("last_calibration_error", 0.0)),
            relax_circuit_open_until_iter=int(payload.get("relax_circuit_open_until_iter", 0)),
            relax_consecutive_failures=int(payload.get("relax_consecutive_failures", 0)),
        )

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
            pred = getattr(cand, "energy_mean", None)
            obs = getattr(cand, "energy_per_atom", None)
            std = getattr(cand, "energy_std", None)
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
                q = float(np.quantile(z, 1.0 - float(cfg.conformal_alpha)))
                self.conformal_scale = max(1.0, q)
            else:
                self.std_scale = 1.0
                self.conformal_scale = 1.0
        else:
            self.std_scale = 1.0
            self.conformal_scale = 1.0

        self.last_calibration_error = float(np.mean(residuals)) if residuals else 0.0
