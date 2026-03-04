"""
Gaussian-process surrogate helper for active-learning acquisition.
高斯過程代理模型：用於主動學習採集分數的理論化與穩健化。

References / 參考文獻:
- Srinivas et al. (2010), GP-UCB regret bounds:
  https://arxiv.org/abs/0912.3995
- Gardner et al. (2014), constrained Bayesian optimization:
  https://proceedings.mlr.press/v32/gardner14.html
- Jones et al. (1998), Efficient Global Optimization (EI):
  https://link.springer.com/article/10.1023/A:1008306431147
- Ament et al. (2023), LogEI (numerically stable EI):
  https://arxiv.org/abs/2310.20708
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from atlas.active_learning.acquisition import schedule_ucb_kappa

try:
    from scipy.special import log_ndtr as _scipy_log_ndtr
    from scipy.special import ndtr as _scipy_ndtr

    HAS_SCIPY_SPECIAL = True
except Exception:  # pragma: no cover - optional runtime
    HAS_SCIPY_SPECIAL = False
    _scipy_ndtr = None  # type: ignore[assignment]
    _scipy_log_ndtr = None  # type: ignore[assignment]

try:
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional runtime
    HAS_SKLEARN = False
    GaussianProcessClassifier = None  # type: ignore[assignment]
    GaussianProcessRegressor = None  # type: ignore[assignment]
    ConstantKernel = None  # type: ignore[assignment]
    Matern = None  # type: ignore[assignment]
    WhiteKernel = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _ndtr(z: np.ndarray) -> np.ndarray:
    if HAS_SCIPY_SPECIAL:
        return _scipy_ndtr(z)
    # Fallback for environments without scipy.special.
    erf_vec = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(z / _SQRT_2))


def _log_ndtr(z: np.ndarray) -> np.ndarray:
    if HAS_SCIPY_SPECIAL:
        return _scipy_log_ndtr(z)
    out = np.empty_like(z, dtype=float)
    direct = z > -8.0
    out[direct] = np.log(np.maximum(_ndtr(z[direct]), 1e-300))
    if np.any(~direct):
        # Mills-ratio asymptotic: log Phi(z) for large negative z.
        z_tail = np.minimum(z[~direct], -1e-12)
        out[~direct] = -0.5 * z_tail * z_tail - np.log(-z_tail) - 0.5 * _LOG_2PI
    return out


@dataclass
class GPSurrogateConfig:
    min_points: int = 8
    kappa: float = 1.5
    alpha: float = 1e-6
    random_state: int = 42
    blend_ratio: float = 0.5
    feasibility_power: float = 1.0
    acquisition_mode: str = "logei"  # {"ucb", "ei", "logei"}
    kappa_schedule: str = "gp_ucb"  # {"fixed", "anneal", "gp_ucb"}
    kappa_min: float = 0.75
    kappa_decay: float = 0.08
    ucb_delta: float = 0.1
    ucb_dimension: int = 5
    ei_jitter: float = 0.0
    feasibility_mode: str = "chance"  # {"chance", "mean", "lcb"}
    feasibility_threshold: float = 0.5
    feasibility_kappa: float = 0.5
    use_feasibility_classifier: bool = True
    feasibility_label_threshold: float = 0.5
    coupling_strength: float = 0.35
    min_coupling_points: int = 12
    normalize_output: bool = True
    include_energy_feature: bool = False


class GPSurrogateAcquirer:
    """
    Constrained-GP wrapper using candidate-level scalar descriptors.
    使用候選者標量描述子的 constrained GP 包裝器。

    Default feature vector / 預設特徵向量:
    [topo_probability, stability_score, heuristic_topo_score, novelty_score, energy_std]
    """

    def __init__(self, config: GPSurrogateConfig | None = None):
        self.config = config or GPSurrogateConfig()
        self._x_hist: list[np.ndarray] = []
        self._y_obj_hist: list[float] = []
        self._y_obj_noise_hist: list[float] = []
        self._y_feas_hist: list[float] = []
        self._objective_model = None
        self._feasibility_model = None
        self._feasibility_is_classifier = False
        self._obj_feas_corr = 0.0
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
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
        energy_std = self._safe_float(getattr(candidate, "energy_std", 0.0), 0.0)
        topo_prob = self._safe_float(getattr(candidate, "topo_probability", 0.0), 0.0)
        stability = self._safe_float(getattr(candidate, "stability_score", 0.0), 0.0)
        heuristic = self._safe_float(getattr(candidate, "heuristic_topo_score", 0.0), 0.0)
        novelty = self._safe_float(getattr(candidate, "novelty_score", 0.0), 0.0)

        features = [topo_prob, stability, heuristic, novelty, max(0.0, energy_std)]
        if self.config.include_energy_feature:
            # Optional leakage-prone feature kept as explicit opt-in only.
            # 可選的能量均值特徵，預設關閉以避免 target leakage。
            features.append(self._safe_float(energy_mean, 0.0))
        return np.asarray(features, dtype=float)

    def _objective_target(self, candidate) -> float:
        # Avoid score-on-score feedback loops by using physical proxy target.
        # 使用物理 proxy（形成能）避免「分數學分數」回授回路。
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
            if not (np.isfinite(y_obj) and np.isfinite(y_feas) and np.all(np.isfinite(x))):
                continue
            obs_std = self._safe_float(getattr(c, "energy_std", 0.0), 0.0)
            obs_var = max(0.0, obs_std) ** 2
            self._x_hist.append(x)
            self._y_obj_hist.append(y_obj)
            self._y_obj_noise_hist.append(obs_var)
            self._y_feas_hist.append(y_feas)
        self._fit_if_ready()

    @staticmethod
    def _normalize_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        x_norm = (x - mean) / std
        return x_norm, mean.reshape(-1), std.reshape(-1)

    def _transform_features(self, x: np.ndarray) -> np.ndarray:
        if self._x_mean is None or self._x_std is None:
            return x
        return (x - self._x_mean.reshape(1, -1)) / self._x_std.reshape(1, -1)

    @staticmethod
    def _schedule_ucb_kappa(
        t: int,
        *,
        base_kappa: float,
        mode: str,
        dimension: int,
        delta: float,
        kappa_min: float,
        decay: float,
    ) -> float:
        return schedule_ucb_kappa(
            iteration=max(1, int(t)),
            base_kappa=base_kappa,
            mode=mode,
            dimension=dimension,
            delta=delta,
            kappa_min=kappa_min,
            decay=decay,
        )

    def _current_kappa(self) -> float:
        t = max(1, len(self._y_obj_hist))
        return self._schedule_ucb_kappa(
            t,
            base_kappa=self.config.kappa,
            mode=self.config.kappa_schedule,
            dimension=max(int(self.config.ucb_dimension), self.feature_dimension),
            delta=self.config.ucb_delta,
            kappa_min=self.config.kappa_min,
            decay=self.config.kappa_decay,
        )

    @staticmethod
    def _rank_normalize(scores: np.ndarray) -> np.ndarray:
        out = np.zeros_like(scores, dtype=float)
        finite_mask = np.isfinite(scores)
        if not np.any(finite_mask):
            return out
        finite_vals = scores[finite_mask]
        if finite_vals.size == 1:
            out[finite_mask] = 1.0
            return out
        ranks = np.argsort(np.argsort(finite_vals))
        out[finite_mask] = ranks / float(finite_vals.size - 1)
        return out

    @staticmethod
    def _rank_array(x: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(x))
        return ranks.astype(float)

    @classmethod
    def _spearman_corr(cls, a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 3 or b.size < 3:
            return 0.0
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        ra = cls._rank_array(a)
        rb = cls._rank_array(b)
        corr = np.corrcoef(ra, rb)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(np.clip(corr, -0.95, 0.95))

    def _estimate_obj_feas_correlation(self, y_obj: np.ndarray, y_feas: np.ndarray) -> float:
        # Correlation proxy for objective-feasibility coupling.
        # 用 Spearman 相關係數作為 objective/feasibility 耦合強度估計。
        return self._spearman_corr(y_obj, y_feas)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        q = np.clip(p, 1e-12, 1.0 - 1e-12)
        return np.log(q / (1.0 - q))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        z = np.clip(x, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _apply_objective_feasibility_coupling(
        self,
        feas_prob: np.ndarray,
        mean_obj: np.ndarray,
    ) -> np.ndarray:
        if not np.any(np.isfinite(feas_prob)):
            return feas_prob
        if len(self._y_obj_hist) < int(self.config.min_coupling_points):
            return feas_prob
        rho = float(getattr(self, "_obj_feas_corr", 0.0))
        lam = max(0.0, float(self.config.coupling_strength))
        if lam <= 0.0 or abs(rho) < 1e-6:
            return feas_prob

        # Logistic coupling:
        # logit(P_feas_joint) = logit(P_feas) + lam * rho * standardized_obj_rank
        # 用邏輯回歸型耦合把 objective 訊息注入 feasibility。
        obj_rank = self._rank_normalize(mean_obj)
        centered = (obj_rank - 0.5) * 2.0  # [-1, 1]
        coupled_logit = self._logit(np.clip(feas_prob, 1e-12, 1.0 - 1e-12)) + lam * rho * centered
        return np.clip(self._sigmoid(coupled_logit), 0.0, 1.0)

    @staticmethod
    def _phi(z: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * z * z) / _SQRT_2PI

    @classmethod
    def _expected_improvement(cls, mean: np.ndarray, std: np.ndarray, best: float, jitter: float) -> np.ndarray:
        sigma = np.maximum(std, 1e-12)
        improve = mean - float(best) - float(jitter)
        z = improve / sigma
        ei = improve * _ndtr(z) + sigma * cls._phi(z)
        deterministic = std <= 1e-12
        if np.any(deterministic):
            ei[deterministic] = np.maximum(improve[deterministic], 0.0)
        return np.maximum(ei, 0.0)

    @classmethod
    def _log_expected_improvement(
        cls,
        mean: np.ndarray,
        std: np.ndarray,
        best: float,
        jitter: float,
    ) -> np.ndarray:
        """
        Numerically stable log(EI) computation.
        數值穩定的 log(EI) 計算。

        Ref:
        - Ament et al. (2023), LogEI: https://arxiv.org/abs/2310.20708
        """
        sigma = np.maximum(std, 1e-12)
        improve = mean - float(best) - float(jitter)
        z = improve / sigma

        # Main branch: direct EI then log for moderate z.
        ei = improve * _ndtr(z) + sigma * cls._phi(z)
        log_ei = np.log(np.maximum(ei, 1e-300))

        # Tail branch: asymptotic stabilization for very negative z.
        tail = z < -8.0
        if np.any(tail):
            z_tail = np.minimum(z[tail], -1e-12)
            log_sigma = np.log(sigma[tail])
            log_phi = -0.5 * z_tail * z_tail - 0.5 * _LOG_2PI
            log_ei[tail] = log_sigma + log_phi - 2.0 * np.log(-z_tail)

        deterministic = std <= 1e-12
        if np.any(deterministic):
            det_imp = np.maximum(improve[deterministic], 0.0)
            log_ei[deterministic] = np.log(np.maximum(det_imp, 1e-300))
        return log_ei

    def _fit_if_ready(self):
        if not HAS_SKLEARN:
            return
        if len(self._y_obj_hist) < self.config.min_points:
            return

        x = np.vstack(self._x_hist)
        y_obj = np.asarray(self._y_obj_hist, dtype=float)
        y_obj_noise = np.asarray(self._y_obj_noise_hist, dtype=float)
        y_feas = np.asarray(self._y_feas_hist, dtype=float)
        x_norm, x_mean, x_std = self._normalize_features(x)
        if y_obj_noise.shape[0] != y_obj.shape[0]:
            y_obj_noise = np.zeros_like(y_obj)
        y_obj_noise = np.clip(y_obj_noise, 0.0, None)
        alpha_vec = np.maximum(float(self.config.alpha), 1e-12) + y_obj_noise

        dim = x_norm.shape[1]
        obj_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dim, dtype=float),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e1))
        obj_model = GaussianProcessRegressor(
            kernel=obj_kernel,
            alpha=alpha_vec,
            normalize_y=True,
            random_state=self.config.random_state,
        )

        feas_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dim, dtype=float),
            length_scale_bounds=(1e-3, 1e3),
            nu=1.5,
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e1))
        feas_model = GaussianProcessRegressor(
            kernel=feas_kernel,
            alpha=max(self.config.alpha, 1e-5),
            normalize_y=True,
            random_state=self.config.random_state,
        )
        feas_class_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dim, dtype=float),
            length_scale_bounds=(1e-3, 1e3),
            nu=1.5,
        )
        feas_classifier = GaussianProcessClassifier(
            kernel=feas_class_kernel,
            random_state=self.config.random_state,
        )

        try:
            obj_model.fit(x_norm, y_obj)
            self._objective_model = obj_model
            self._x_mean = x_mean
            self._x_std = x_std
            self._obj_feas_corr = self._estimate_obj_feas_correlation(y_obj, y_feas)

            # Fit a separate feasibility surrogate when there is signal.
            # 當 feasibility 有變異時才擬合，以避免退化模型。
            if np.std(y_feas) > 1e-8:
                use_classifier = bool(self.config.use_feasibility_classifier)
                if use_classifier:
                    label_threshold = float(self.config.feasibility_label_threshold)
                    y_labels = (y_feas >= label_threshold).astype(int)
                    if np.unique(y_labels).size >= 2:
                        feas_classifier.fit(x_norm, y_labels)
                        self._feasibility_model = feas_classifier
                        self._feasibility_is_classifier = True
                    else:
                        use_classifier = False
                if not use_classifier:
                    feas_model.fit(x_norm, y_feas)
                    self._feasibility_model = feas_model
                    self._feasibility_is_classifier = False
            else:
                self._feasibility_model = None
                self._feasibility_is_classifier = False
        except Exception as exc:
            logger.warning(f"GP fit failed; fallback to base acquisition. Error: {exc}")
            self._objective_model = None
            self._feasibility_model = None
            self._feasibility_is_classifier = False
            self._obj_feas_corr = 0.0
            self._x_mean = None
            self._x_std = None

    def _predict_feasibility(self, x_raw: np.ndarray) -> np.ndarray:
        if self._feasibility_model is None:
            # Fallback to classifier estimates embedded in features.
            # 後備方案：使用 feature[0] 的 topo_probability。
            return np.clip(x_raw[:, 0], 0.0, 1.0)

        x = self._transform_features(x_raw)
        mode = str(self.config.feasibility_mode).strip().lower()
        if self._feasibility_is_classifier:
            proba = self._feasibility_model.predict_proba(x)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                base_prob = proba[:, 1]
            else:
                base_prob = np.asarray(proba).reshape(-1)
            if mode == "lcb":
                std_proxy = np.sqrt(np.clip(base_prob * (1.0 - base_prob), 0.0, 0.25))
                feas_prob = np.clip(
                    base_prob - float(self.config.feasibility_kappa) * std_proxy,
                    0.0,
                    1.0,
                )
            else:
                # Classifier output is already a calibrated chance-like feasibility.
                # 分類器輸出已是機率語義，可直接作為 chance feasibility。
                feas_prob = base_prob
        else:
            mean_feas, std_feas = self._feasibility_model.predict(x, return_std=True)
            mean_feas = np.clip(mean_feas, 0.0, 1.0)
            std_feas = np.maximum(std_feas, 1e-9)
            if mode == "mean":
                feas_prob = mean_feas
            elif mode == "lcb":
                feas_prob = np.clip(mean_feas - float(self.config.feasibility_kappa) * std_feas, 0.0, 1.0)
            else:
                # Chance-constrained feasibility P(f >= tau) under Gaussian surrogate.
                # 高斯代理下的機率約束：P(f >= tau)。
                z = (mean_feas - float(self.config.feasibility_threshold)) / std_feas
                feas_prob = _ndtr(z)
        return np.clip(feas_prob, 0.0, 1.0)

    def _objective_utility(self, mean_obj: np.ndarray, std_obj: np.ndarray) -> np.ndarray:
        mode = str(self.config.acquisition_mode).strip().lower()
        if not self._y_obj_hist:
            return mean_obj

        if mode == "ucb":
            return mean_obj + self._current_kappa() * std_obj

        incumbent = float(np.max(np.asarray(self._y_obj_hist, dtype=float)))
        if mode == "ei":
            return self._expected_improvement(mean_obj, std_obj, incumbent, self.config.ei_jitter)
        if mode == "logei":
            return self._log_expected_improvement(mean_obj, std_obj, incumbent, self.config.ei_jitter)
        return mean_obj + self._current_kappa() * std_obj

    def suggest_constrained_utility(self, candidates: Sequence[object]) -> np.ndarray | None:
        if self._objective_model is None or not candidates:
            return None

        x_raw = np.vstack([self.candidate_to_features(c) for c in candidates])
        x = self._transform_features(x_raw)
        mean_obj, std_obj = self._objective_model.predict(x, return_std=True)
        std_obj = np.maximum(std_obj, 0.0)

        obj_utility = self._objective_utility(mean_obj, std_obj)
        feas_prob = self._predict_feasibility(x_raw)
        feas_prob = self._apply_objective_feasibility_coupling(feas_prob, mean_obj)
        mode = str(self.config.acquisition_mode).strip().lower()

        if mode == "logei":
            # In log-domain: log U = logEI + p * log P(feasible).
            # 在對數域組合，避免極小 EI/feasibility 產生下溢。
            utility = obj_utility + float(self.config.feasibility_power) * np.log(
                np.clip(feas_prob, 1e-12, 1.0)
            )
        else:
            utility = obj_utility * np.power(feas_prob, float(self.config.feasibility_power))

        if self.config.normalize_output:
            return self._rank_normalize(utility)
        return utility

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

    @property
    def feature_dimension(self) -> int:
        if self._x_hist:
            return int(self._x_hist[0].shape[0])
        return 6 if self.config.include_energy_feature else 5

    @property
    def current_kappa(self) -> float:
        return self._current_kappa()

