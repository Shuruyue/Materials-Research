"""
Gaussian-process surrogate helper for active-learning acquisition.

References:
- Gardner et al. (2014), constrained Bayesian optimization:
  https://proceedings.mlr.press/v32/gardner14.html
- Srinivas et al. (2010), GP-UCB:
  https://arxiv.org/abs/0912.3995
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional runtime
    HAS_SKLEARN = False
    GaussianProcessRegressor = None  # type: ignore[assignment]
    RationalQuadratic = None  # type: ignore[assignment]
    WhiteKernel = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class GPSurrogateConfig:
    min_points: int = 8
    kappa: float = 1.5
    alpha: float = 1e-6
    random_state: int = 42
    blend_ratio: float = 0.5
    feasibility_power: float = 1.0


class GPSurrogateAcquirer:
    """
    Small constrained-GP wrapper using candidate-level scalar descriptors.

    Feature vector:
    [topo_probability, stability_score, heuristic_topo_score, novelty_score,
     energy_mean_or_e_pa, energy_std]
    """

    def __init__(self, config: GPSurrogateConfig | None = None):
        self.config = config or GPSurrogateConfig()
        self._x_hist: list[np.ndarray] = []
        self._y_obj_hist: list[float] = []
        self._y_feas_hist: list[float] = []
        self._objective_model = None
        self._feasibility_model = None
        if not HAS_SKLEARN:
            logger.warning("scikit-learn unavailable; GP surrogate branch disabled.")

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def candidate_to_features(self, candidate) -> np.ndarray:
        energy_mean = getattr(candidate, "energy_mean", None)
        if energy_mean is None:
            energy_mean = getattr(candidate, "energy_per_atom", None)
        energy_std = getattr(candidate, "energy_std", 0.0)
        return np.array(
            [
                self._safe_float(getattr(candidate, "topo_probability", 0.0)),
                self._safe_float(getattr(candidate, "stability_score", 0.0)),
                self._safe_float(getattr(candidate, "heuristic_topo_score", 0.0)),
                self._safe_float(getattr(candidate, "novelty_score", 0.0)),
                self._safe_float(energy_mean, 0.0),
                self._safe_float(energy_std, 0.0),
            ],
            dtype=float,
        )

    def _objective_target(self, candidate) -> float:
        # Use true optimization target proxy instead of acquisition_value to
        # avoid score-on-score feedback loops.
        # For formation-energy minimization, larger utility should be better.
        energy_mean = getattr(candidate, "energy_mean", None)
        if energy_mean is None:
            energy_mean = getattr(candidate, "energy_per_atom", None)
        if energy_mean is not None:
            return -self._safe_float(energy_mean, default=0.0)
        return self._safe_float(getattr(candidate, "stability_score", 0.0), default=0.0)

    def _feasibility_target(self, candidate) -> float:
        topo = self._safe_float(getattr(candidate, "topo_probability", 0.0), default=0.0)
        return float(np.clip(topo, 0.0, 1.0))

    def update(self, candidates: Sequence[object]):
        for c in candidates:
            x = self.candidate_to_features(c)
            y_obj = self._objective_target(c)
            y_feas = self._feasibility_target(c)
            if not (np.isfinite(y_obj) and np.isfinite(y_feas)):
                continue
            self._x_hist.append(x)
            self._y_obj_hist.append(y_obj)
            self._y_feas_hist.append(y_feas)
        self._fit_if_ready()

    def _fit_if_ready(self):
        if not HAS_SKLEARN:
            return
        if len(self._y_obj_hist) < self.config.min_points:
            return
        x = np.vstack(self._x_hist)
        y_obj = np.asarray(self._y_obj_hist, dtype=float)
        y_feas = np.asarray(self._y_feas_hist, dtype=float)

        obj_kernel = RationalQuadratic(length_scale=1.0, alpha=0.5) + WhiteKernel(noise_level=1e-6)
        obj_model = GaussianProcessRegressor(
            kernel=obj_kernel,
            alpha=self.config.alpha,
            normalize_y=True,
            random_state=self.config.random_state,
        )
        feas_kernel = RationalQuadratic(length_scale=1.0, alpha=0.5) + WhiteKernel(noise_level=1e-5)
        feas_model = GaussianProcessRegressor(
            kernel=feas_kernel,
            alpha=max(self.config.alpha, 1e-5),
            normalize_y=True,
            random_state=self.config.random_state,
        )
        try:
            obj_model.fit(x, y_obj)
            self._objective_model = obj_model
            # Fit a separate feasibility surrogate when there is signal.
            if np.std(y_feas) > 1e-8:
                feas_model.fit(x, y_feas)
                self._feasibility_model = feas_model
            else:
                self._feasibility_model = None
        except Exception as exc:
            logger.warning(f"GP fit failed; fallback to base acquisition. Error: {exc}")
            self._objective_model = None
            self._feasibility_model = None

    def _predict_feasibility(self, x: np.ndarray) -> np.ndarray:
        if self._feasibility_model is None:
            # Fallback to current classifier estimates embedded in features.
            # Feature index 0 is topo_probability.
            return np.clip(x[:, 0], 0.0, 1.0)
        mean_feas, _ = self._feasibility_model.predict(x, return_std=True)
        return np.clip(mean_feas, 0.0, 1.0)

    def suggest_constrained_utility(self, candidates: Sequence[object]) -> np.ndarray | None:
        if self._objective_model is None:
            return None
        x = np.vstack([self.candidate_to_features(c) for c in candidates])
        mean_obj, std_obj = self._objective_model.predict(x, return_std=True)
        obj_ucb = mean_obj + self.config.kappa * std_obj
        feas_prob = self._predict_feasibility(x)
        return obj_ucb * np.power(feas_prob, self.config.feasibility_power)

    def suggest_ucb(self, candidates: Sequence[object]) -> np.ndarray | None:
        # Backward-compatible alias.
        return self.suggest_constrained_utility(candidates)

    @property
    def history_size(self) -> int:
        return len(self._y_obj_hist)

    @property
    def objective_history_size(self) -> int:
        return len(self._y_obj_hist)

    @property
    def feasibility_history_size(self) -> int:
        return len(self._y_feas_hist)

