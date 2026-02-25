"""
Gaussian-process surrogate helper for active-learning acquisition.
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


class GPSurrogateAcquirer:
    """
    Small GP wrapper using candidate-level scalar descriptors.

    Feature vector:
    [topo_probability, stability_score, heuristic_topo_score, novelty_score,
     energy_mean_or_e_pa, energy_std]
    """

    def __init__(self, config: GPSurrogateConfig | None = None):
        self.config = config or GPSurrogateConfig()
        self._x_hist: list[np.ndarray] = []
        self._y_hist: list[float] = []
        self._model = None
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

    def update(self, candidates: Sequence[object]):
        for c in candidates:
            y = self._safe_float(getattr(c, "acquisition_value", None), default=np.nan)
            if np.isnan(y):
                continue
            self._x_hist.append(self.candidate_to_features(c))
            self._y_hist.append(y)
        self._fit_if_ready()

    def _fit_if_ready(self):
        if not HAS_SKLEARN:
            return
        if len(self._y_hist) < self.config.min_points:
            return
        x = np.vstack(self._x_hist)
        y = np.asarray(self._y_hist, dtype=float)
        kernel = RationalQuadratic(length_scale=1.0, alpha=0.5) + WhiteKernel(noise_level=1e-6)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.alpha,
            normalize_y=True,
            random_state=self.config.random_state,
        )
        try:
            model.fit(x, y)
            self._model = model
        except Exception as exc:
            logger.warning(f"GP fit failed; fallback to base acquisition. Error: {exc}")
            self._model = None

    def suggest_ucb(self, candidates: Sequence[object]) -> np.ndarray | None:
        if self._model is None:
            return None
        x = np.vstack([self.candidate_to_features(c) for c in candidates])
        mean, std = self._model.predict(x, return_std=True)
        return mean + self.config.kappa * std

    @property
    def history_size(self) -> int:
        return len(self._y_hist)

