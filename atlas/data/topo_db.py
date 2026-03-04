"""
Topological materials database utility.

This module keeps a lightweight CSV-backed store and optional SQLite mirror.
It intentionally preserves a legacy API used by existing tests/scripts.
"""

from __future__ import annotations

import difflib
import logging
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import get_config

logger = logging.getLogger(__name__)

_DEFAULT_COLUMNS = [
    "jid",
    "formula",
    "space_group",
    "topo_class",
    "band_gap",
    "source",
    # Optional evidence / probabilistic fields.
    "spillage",
    "si_score",
    "ml_probability",
    "ml_uncertainty",
    "topological_probability",
    "topology_ci_low",
    "topology_ci_high",
    "ood_score",
]

_SEED_ROWS = [
    {"jid": "seed-ti-001", "formula": "Bi2Se3", "space_group": 166, "topo_class": "TI", "band_gap": 0.30, "source": "seed"},
    {"jid": "seed-ti-002", "formula": "Bi2Te3", "space_group": 166, "topo_class": "TI", "band_gap": 0.15, "source": "seed"},
    {"jid": "seed-tsm-001", "formula": "TaAs", "space_group": 109, "topo_class": "TSM", "band_gap": 0.00, "source": "seed"},
    {"jid": "seed-tsm-002", "formula": "Cd3As2", "space_group": 137, "topo_class": "TSM", "band_gap": 0.00, "source": "seed"},
    {"jid": "seed-tri-001", "formula": "Si", "space_group": 227, "topo_class": "TRIVIAL", "band_gap": 1.10, "source": "seed"},
]

TOPO_CLASSES = (
    "TRIVIAL",
    "TI",
    "TSM",
    "TCI",
    "WEYL",
    "DIRAC",
    "LINENODE",
    "MAGNETIC_TI",
)

_TOPO_POSITIVE = set(TOPO_CLASSES) - {"TRIVIAL"}

_EVIDENCE_CHANNELS = ("si", "spillage", "ml")

# Light-weight periodic table index for composition vectors.
_ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
]
_ELEMENT_TO_INDEX = {el: i for i, el in enumerate(_ELEMENTS)}
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


@dataclass
class ReliabilityState:
    """
    Beta posterior reliability state for one evidence channel.

    References:
    - Dawid & Skene (1979): latent annotator reliability.
    """

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return float(self.alpha / max(self.alpha + self.beta, 1e-12))

    @property
    def variance(self) -> float:
        denom = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1.0)
        return float((self.alpha * self.beta) / max(denom, 1e-12))


@dataclass
class TopoMaterial:
    # Legacy positional order used in tests:
    # TopoMaterial(jid, formula, space_group, topo_class, band_gap)
    jid: str
    formula: str
    space_group: int
    topo_class: str
    band_gap: float = 0.0
    source: str = "custom"
    spillage: float | None = None
    si_score: float | None = None
    ml_probability: float | None = None
    ml_uncertainty: float | None = None
    topological_probability: float | None = None
    topology_ci_low: float | None = None
    topology_ci_high: float | None = None
    ood_score: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Normalize class label casing on write.
        d["topo_class"] = str(self.topo_class).upper()
        return d

    def is_topological(self) -> bool:
        return str(self.topo_class).upper() in _TOPO_POSITIVE


class TopoDB:
    """
    CSV-first DB wrapper with optional SQLite export/query.
    """

    def __init__(self, use_sql: bool = False):
        cfg = get_config()
        self.db_dir: Path = cfg.paths.raw_dir
        self.db_file: Path = self.db_dir / "topological_materials.csv"
        self.sql_path: Path = self.db_dir / "topological_materials.db"
        self.use_sql = use_sql
        self._df: pd.DataFrame | None = None
        self._channel_reliability: dict[str, ReliabilityState] = {
            # Mildly uninformative priors.
            "si": ReliabilityState(alpha=2.0, beta=2.0),
            "spillage": ReliabilityState(alpha=2.0, beta=2.0),
            "ml": ReliabilityState(alpha=2.0, beta=2.0),
        }
        self._channel_corr = np.eye(len(_EVIDENCE_CHANNELS), dtype=float)
        self._last_inference_diagnostics: dict[str, Any] = {}

        if self.use_sql and self.sql_path.exists():
            self._df = self._load_from_sql()
        elif self.db_file.exists():
            self._df = self._load_from_csv()

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if self.db_file.exists():
                self._df = self._load_from_csv()
            else:
                self._df = self._empty_df()
        return self._df

    def _empty_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=_DEFAULT_COLUMNS)

    def _normalize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in _DEFAULT_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        out["jid"] = out["jid"].astype("string")
        out["formula"] = out["formula"].astype("string")
        out["source"] = out["source"].astype("string").fillna("custom")
        out["topo_class"] = out["topo_class"].astype("string").str.upper()

        numeric_cols = [
            "space_group",
            "band_gap",
            "spillage",
            "si_score",
            "ml_probability",
            "ml_uncertainty",
            "topological_probability",
            "topology_ci_low",
            "topology_ci_high",
            "ood_score",
        ]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        return out[_DEFAULT_COLUMNS]

    def _load_from_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.db_file)
        except Exception as exc:
            logger.warning(f"Failed to read {self.db_file}: {exc}. Using empty DB.")
            return self._empty_df()
        return self._normalize_frame(df)

    def _load_from_sql(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.sql_path)
            df = pd.read_sql("SELECT * FROM materials", conn)
            conn.close()
            return self._normalize_frame(df)
        except Exception as exc:
            logger.warning(f"Failed to read SQL DB {self.sql_path}: {exc}")
            return self._empty_df()

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    @staticmethod
    def _clip_prob(x: np.ndarray | float, eps: float = 1e-9):
        return np.clip(x, eps, 1.0 - eps)

    @staticmethod
    def _betacf(a: float, b: float, x: float) -> float:
        # Continued fraction for incomplete beta function (Numerical Recipes).
        max_iter = 200
        eps = 3e-14
        fpmin = 1e-300
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    @classmethod
    def _regularized_beta_inc(cls, a: float, b: float, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        log_bt = (
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x)
            + b * math.log(1.0 - x)
        )
        bt = math.exp(log_bt)
        if x < (a + 1.0) / (a + b + 2.0):
            return float(bt * cls._betacf(a, b, x) / a)
        return float(1.0 - bt * cls._betacf(b, a, 1.0 - x) / b)

    @classmethod
    def _beta_ppf(cls, q: float, a: float, b: float) -> float:
        q = float(np.clip(q, 0.0, 1.0))
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return 1.0
        lo = 0.0
        hi = 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            cdf = cls._regularized_beta_inc(a, b, mid)
            if cdf < q:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < 1e-8:
                break
        return float(0.5 * (lo + hi))

    @classmethod
    def _beta_interval(
        cls,
        alpha: float,
        beta: float,
        *,
        z: float = 1.96,
        method: str = "normal",
        level: float | None = None,
    ) -> tuple[float, float]:
        canon = str(method).strip().lower()
        if canon not in {"normal", "exact"}:
            raise ValueError("method must be one of {'normal','exact'}")

        if level is None:
            # Two-sided normal-equivalent central coverage from z.
            level = math.erf(float(z) / math.sqrt(2.0))
        level = float(np.clip(level, 1e-6, 1.0 - 1e-6))

        if canon == "normal":
            m = alpha / max(alpha + beta, 1e-12)
            var = (alpha * beta) / max((alpha + beta) ** 2 * (alpha + beta + 1.0), 1e-12)
            s = math.sqrt(max(var, 1e-18))
            lo = float(np.clip(m - float(z) * s, 0.0, 1.0))
            hi = float(np.clip(m + float(z) * s, 0.0, 1.0))
            return lo, hi

        q_lo = 0.5 * (1.0 - level)
        q_hi = 1.0 - q_lo
        lo = cls._beta_ppf(q_lo, alpha, beta)
        hi = cls._beta_ppf(q_hi, alpha, beta)
        return float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))

    @staticmethod
    def _nearest_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        sym = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.clip(eigvals, eps, None)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @classmethod
    def _normalize_correlation_matrix(cls, corr: np.ndarray) -> np.ndarray:
        corr = np.asarray(corr, dtype=float)
        corr = cls._nearest_psd(corr, eps=1e-10)
        d = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
        corr = corr / np.outer(d, d)
        corr = np.clip(corr, -0.99, 0.99)
        np.fill_diagonal(corr, 1.0)
        return 0.5 * (corr + corr.T)

    @classmethod
    def _calibration_metrics(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        *,
        n_bins: int = 10,
    ) -> tuple[float, float, float]:
        """
        Binary calibration metrics: ECE, Brier score, and NLL.

        Reference:
        - Guo et al. (2017), temperature scaling for calibrated probabilities.
        """
        p = cls._clip_prob(np.asarray(probs, dtype=float), eps=1e-9)
        y = np.asarray(labels, dtype=float)
        if p.size == 0:
            return 0.0, 0.0, 0.0
        brier = float(np.mean((p - y) ** 2))
        nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

        bins = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
        ece = 0.0
        n = float(p.size)
        for i in range(len(bins) - 1):
            lo = bins[i]
            hi = bins[i + 1]
            if i == len(bins) - 2:
                mask = (p >= lo) & (p <= hi)
            else:
                mask = (p >= lo) & (p < hi)
            if not np.any(mask):
                continue
            conf = float(np.mean(p[mask]))
            acc = float(np.mean(y[mask]))
            ece += abs(conf - acc) * (float(np.sum(mask)) / n)
        return float(ece), brier, nll

    @classmethod
    def _fit_temperature(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        *,
        min_temp: float = 0.25,
        max_temp: float = 8.0,
        steps: int = 161,
    ) -> float:
        """
        Fit scalar temperature by minimizing binary NLL in logit space.
        """
        p = cls._clip_prob(np.asarray(probs, dtype=float), eps=1e-9)
        y = np.asarray(labels, dtype=float)
        if p.size == 0:
            return 1.0
        lo = max(float(min_temp), 1e-3)
        hi = max(float(max_temp), lo + 1e-6)
        temps = np.exp(np.linspace(math.log(lo), math.log(hi), int(max(8, steps))))
        logits = np.log(p) - np.log(1.0 - p)

        best_t = 1.0
        best_nll = float("inf")
        for t in temps:
            q = cls._sigmoid(logits / float(t))
            q = cls._clip_prob(q, eps=1e-9)
            nll = float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))
            if nll + 1e-12 < best_nll:
                best_nll = nll
                best_t = float(t)
        return float(best_t)

    def last_inference_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the latest infer_topology_probabilities call."""
        return dict(self._last_inference_diagnostics)

    @staticmethod
    def _truth_binary_from_series(series: pd.Series) -> np.ndarray:
        arr = np.full(len(series), np.nan, dtype=float)
        if pd.api.types.is_numeric_dtype(series):
            vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            vals = np.clip(vals, 0.0, 1.0)
            arr[np.isfinite(vals)] = vals[np.isfinite(vals)]
            return arr
        vals = series.astype("string").str.upper().to_numpy()
        for i, v in enumerate(vals):
            if v is pd.NA or v is None:
                continue
            s = str(v)
            if s in _TOPO_POSITIVE:
                arr[i] = 1.0
            elif s == "TRIVIAL":
                arr[i] = 0.0
        return arr

    @staticmethod
    def _parse_formula_counts(formula: str) -> dict[str, float]:
        if formula is None:
            return {}
        s = str(formula).strip()
        if not s or s.lower() in {"none", "nan", "<na>"}:
            return {}
        out: dict[str, float] = {}
        for el, n in _FORMULA_TOKEN.findall(s):
            count = float(n) if n else 1.0
            out[el] = out.get(el, 0.0) + count
        return out

    @staticmethod
    def _formula_vector(formula: str) -> np.ndarray:
        vec = np.zeros(len(_ELEMENTS), dtype=float)
        counts = TopoDB._parse_formula_counts(formula)
        total = float(sum(counts.values()))
        if total <= 0.0:
            return vec
        for el, c in counts.items():
            idx = _ELEMENT_TO_INDEX.get(el)
            if idx is not None:
                vec[idx] = c / total
        return vec

    def _channel_probability_table(
        self,
        df: pd.DataFrame,
        *,
        si_center: float,
        si_temperature: float,
        spillage_threshold: float,
        spillage_temperature: float,
        ml_uncertainty_scale: float,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        n = len(df)
        probs = {k: np.full(n, np.nan, dtype=float) for k in _EVIDENCE_CHANNELS}
        confs = {k: np.zeros(n, dtype=float) for k in _EVIDENCE_CHANNELS}

        si = pd.to_numeric(df.get("si_score"), errors="coerce").to_numpy(dtype=float)
        si_mask = np.isfinite(si)
        if np.any(si_mask):
            temp = max(float(si_temperature), 1e-8)
            probs["si"][si_mask] = self._sigmoid((si[si_mask] - float(si_center)) / temp)
            confs["si"][si_mask] = 1.0

        sp = pd.to_numeric(df.get("spillage"), errors="coerce").to_numpy(dtype=float)
        sp_mask = np.isfinite(sp)
        if np.any(sp_mask):
            temp = max(float(spillage_temperature), 1e-8)
            probs["spillage"][sp_mask] = self._sigmoid(
                (sp[sp_mask] - float(spillage_threshold)) / temp
            )
            confs["spillage"][sp_mask] = 1.0

        ml = pd.to_numeric(df.get("ml_probability"), errors="coerce").to_numpy(dtype=float)
        ml_mask = np.isfinite(ml)
        if np.any(ml_mask):
            probs["ml"][ml_mask] = np.clip(ml[ml_mask], 0.0, 1.0)
            unc = pd.to_numeric(df.get("ml_uncertainty"), errors="coerce").to_numpy(dtype=float)
            unc = np.where(np.isfinite(unc), np.clip(unc, 0.0, None), 0.5)
            conf = 1.0 / (1.0 + max(float(ml_uncertainty_scale), 1e-8) * unc)
            confs["ml"][ml_mask] = np.clip(conf[ml_mask], 0.05, 1.0)

        return probs, confs

    def channel_reliability(self) -> dict[str, float]:
        return {k: float(v.mean) for k, v in self._channel_reliability.items()}

    def channel_correlation(self) -> dict[str, dict[str, float]]:
        labels = list(_EVIDENCE_CHANNELS)
        out: dict[str, dict[str, float]] = {}
        for i, li in enumerate(labels):
            out[li] = {}
            for j, lj in enumerate(labels):
                out[li][lj] = float(self._channel_corr[i, j])
        return out

    def reset_channel_reliability(self):
        self._channel_reliability = {
            "si": ReliabilityState(alpha=2.0, beta=2.0),
            "spillage": ReliabilityState(alpha=2.0, beta=2.0),
            "ml": ReliabilityState(alpha=2.0, beta=2.0),
        }
        self._channel_corr = np.eye(len(_EVIDENCE_CHANNELS), dtype=float)

    def calibrate_channel_reliability(
        self,
        *,
        truth_col: str = "topo_class",
        min_samples: int = 8,
        si_center: float = 0.0,
        si_temperature: float = 0.35,
        spillage_threshold: float = 0.5,
        spillage_temperature: float = 0.10,
        ml_uncertainty_scale: float = 1.5,
        corr_shrinkage: float = 0.25,
    ) -> dict[str, float]:
        """
        Update channel reliability posteriors from agreement with known labels.

        References:
        - Dawid & Skene (1979), soft reliability estimation.
        """
        if truth_col not in self.df.columns or len(self.df) == 0:
            return self.channel_reliability()

        work = self.df.copy()
        y_true = self._truth_binary_from_series(work[truth_col])
        probs, _ = self._channel_probability_table(
            work,
            si_center=si_center,
            si_temperature=si_temperature,
            spillage_threshold=spillage_threshold,
            spillage_temperature=spillage_temperature,
            ml_uncertainty_scale=ml_uncertainty_scale,
        )

        residuals = np.full((len(work), len(_EVIDENCE_CHANNELS)), np.nan, dtype=float)
        for channel in _EVIDENCE_CHANNELS:
            p = probs[channel]
            mask = np.isfinite(y_true) & np.isfinite(p)
            if int(np.sum(mask)) < int(min_samples):
                continue
            err = np.abs(p[mask] - y_true[mask])
            soft_success = float(np.sum(1.0 - err))
            soft_failure = float(np.sum(err))
            state = self._channel_reliability[channel]
            self._channel_reliability[channel] = ReliabilityState(
                alpha=state.alpha + soft_success,
                beta=state.beta + soft_failure,
            )
            ci = _EVIDENCE_CHANNELS.index(channel)
            residuals[mask, ci] = p[mask] - y_true[mask]

        corr = np.eye(len(_EVIDENCE_CHANNELS), dtype=float)
        for i in range(len(_EVIDENCE_CHANNELS)):
            for j in range(i + 1, len(_EVIDENCE_CHANNELS)):
                mask = np.isfinite(residuals[:, i]) & np.isfinite(residuals[:, j])
                if int(np.sum(mask)) < int(min_samples):
                    c = 0.0
                else:
                    ri = residuals[mask, i]
                    rj = residuals[mask, j]
                    ri = ri - float(np.mean(ri))
                    rj = rj - float(np.mean(rj))
                    denom = math.sqrt(float(np.dot(ri, ri) * np.dot(rj, rj)))
                    c = float(np.dot(ri, rj) / denom) if denom > 1e-12 else 0.0
                corr[i, j] = c
                corr[j, i] = c
        lam = float(np.clip(corr_shrinkage, 0.0, 1.0))
        corr = (1.0 - lam) * corr + lam * np.eye(len(_EVIDENCE_CHANNELS), dtype=float)
        self._channel_corr = self._normalize_correlation_matrix(corr)
        return self.channel_reliability()

    def infer_topology_probabilities(  # noqa: C901
        self,
        *,
        base_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        evidence_strength: float = 24.0,
        si_center: float = 0.0,
        si_temperature: float = 0.35,
        spillage_threshold: float = 0.5,
        spillage_temperature: float = 0.10,
        ml_uncertainty_scale: float = 1.5,
        ci_z: float = 1.96,
        ci_method: str = "exact",
        ci_level: float | None = None,
        calibrate_reliability: bool = True,
        min_calibration_samples: int = 8,
        correlation_mode: str = "correlated",
        corr_shrinkage: float = 0.25,
        correlation_penalty: float = 1.0,
        weight_constraint: str = "nonnegative",
        channel_reliability_override: dict[str, Any] | None = None,
        score_calibration: str = "none",
        calibration_scheme: str = "cross_fit",
        calibration_min_samples: int = 12,
        calibration_bins: int = 10,
        calibration_folds: int = 3,
        calibration_seed: int = 0,
        update_topo_class: bool = False,
        decision_threshold: float = 0.55,
        semimetal_gap_threshold: float = 0.05,
        ood_scale: float = 0.35,
        reference_formulas: list[str] | None = None,
        persist: bool = False,
    ) -> pd.DataFrame:
        """
        Reliability-aware probabilistic topology inference with OOD scoring.

        Mathematics:
        1) Correlation-aware precision pooling in logit space.
        2) Channel weights are reliability-tempered (Beta posterior means).
        3) Beta posterior interval via exact quantile inversion (or normal approx).
        4) OOD score from composition-space nearest-neighbor distance.
        5) Optional post-hoc temperature scaling on labeled subset.
        6) Cross-fit calibration diagnostics to reduce in-sample optimism.

        References:
        - Genest & Zidek (1986): pooling probability opinions.
        - Dawid & Skene (1979): reliability modeling.
        - Aitken (1935): generalized least squares for correlated estimators.
        - Ledoit & Wolf (2004): shrinkage covariance stabilization.
        - Ben-David et al. (2010): distribution-shift risk.
        """
        if len(self.df) == 0:
            return self.df.copy()

        work = self._normalize_frame(self.df.copy())
        if calibrate_reliability:
            self.calibrate_channel_reliability(
                min_samples=min_calibration_samples,
                si_center=si_center,
                si_temperature=si_temperature,
                spillage_threshold=spillage_threshold,
                spillage_temperature=spillage_temperature,
                ml_uncertainty_scale=ml_uncertainty_scale,
                corr_shrinkage=corr_shrinkage,
            )

        probs, confs = self._channel_probability_table(
            work,
            si_center=si_center,
            si_temperature=si_temperature,
            spillage_threshold=spillage_threshold,
            spillage_temperature=spillage_temperature,
            ml_uncertainty_scale=ml_uncertainty_scale,
        )

        ws = np.asarray(base_weights, dtype=float)
        if ws.shape != (3,):
            raise ValueError("base_weights must contain three values (si, spillage, ml)")
        ws = np.clip(ws, 0.0, None)
        if float(ws.sum()) <= 0:
            ws = np.asarray([0.45, 0.35, 0.20], dtype=float)
        ws = ws / ws.sum()
        channel_weight = {ch: float(ws[i]) for i, ch in enumerate(_EVIDENCE_CHANNELS)}
        if weight_constraint not in {"nonnegative", "unconstrained"}:
            raise ValueError("weight_constraint must be one of {'nonnegative','unconstrained'}")
        calibration_mode = str(score_calibration).strip().lower()
        if calibration_mode not in {"none", "temperature"}:
            raise ValueError("score_calibration must be one of {'none','temperature'}")
        calibration_scheme_canon = str(calibration_scheme).strip().lower()
        if calibration_scheme_canon not in {"in_sample", "cross_fit"}:
            raise ValueError("calibration_scheme must be one of {'in_sample','cross_fit'}")
        mode = str(correlation_mode).strip().lower()
        if mode not in {"correlated", "independent"}:
            raise ValueError("correlation_mode must be one of {'correlated','independent'}")
        corr_full = (
            self._channel_corr.copy()
            if mode == "correlated"
            else np.eye(len(_EVIDENCE_CHANNELS), dtype=float)
        )
        corr_full = self._normalize_correlation_matrix(corr_full)
        override_reliability: dict[str, float] = {}
        if channel_reliability_override:
            for ch, raw in channel_reliability_override.items():
                ch_key = str(ch).strip().lower()
                if ch_key not in _EVIDENCE_CHANNELS:
                    continue
                rel: float | None = None
                if isinstance(raw, ReliabilityState):
                    rel = float(raw.mean)
                elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
                    with np.errstate(invalid="ignore"):
                        a = float(raw[0])
                        b = float(raw[1])
                    if np.isfinite(a) and np.isfinite(b) and (a + b) > 0:
                        rel = float(a / (a + b))
                else:
                    with np.errstate(invalid="ignore"):
                        rv = float(raw)
                    if np.isfinite(rv):
                        rel = rv
                if rel is not None:
                    override_reliability[ch_key] = float(np.clip(rel, 1e-6, 1.0))

        n = len(work)
        p_out = np.zeros(n, dtype=float)
        ci_lo = np.zeros(n, dtype=float)
        ci_hi = np.zeros(n, dtype=float)
        n_eff_out = np.zeros(n, dtype=float)
        y_prior = self._truth_binary_from_series(work["topo_class"])

        for i in range(n):
            p_terms: list[float] = []
            info_terms: list[float] = []
            idx_terms: list[int] = []
            for ch in _EVIDENCE_CHANNELS:
                p_ch = probs[ch][i]
                if not np.isfinite(p_ch):
                    continue
                rel = override_reliability.get(ch, self._channel_reliability[ch].mean)
                conf = float(np.clip(confs[ch][i], 0.05, 1.0))
                info = channel_weight[ch] * max(rel, 1e-6) * conf
                p_terms.append(float(self._clip_prob(p_ch)))
                info_terms.append(float(max(info, 1e-6)))
                idx_terms.append(_EVIDENCE_CHANNELS.index(ch))

            if not p_terms:
                # Fall back to known class label or uninformative prior.
                if np.isfinite(y_prior[i]):
                    p = float(np.clip(y_prior[i], 0.0, 1.0))
                else:
                    p = 0.5
                n_eff = max(0.5 * float(evidence_strength), 1.0)
            else:
                logits = np.log(np.asarray(p_terms, dtype=float)) - np.log(
                    1.0 - np.asarray(p_terms, dtype=float)
                )
                info = np.asarray(info_terms, dtype=float)
                std = np.sqrt(1.0 / np.clip(info, 1e-9, None))
                corr_sub = corr_full[np.ix_(idx_terms, idx_terms)]
                corr_sub = self._normalize_correlation_matrix(corr_sub)
                cov = np.outer(std, std) * corr_sub
                cov = self._nearest_psd(cov, eps=1e-12)

                ones = np.ones(len(logits), dtype=float)
                inv_cov = np.linalg.pinv(cov, rcond=1e-12)
                denom = float(ones @ inv_cov @ ones)
                if denom <= 0.0:
                    wv = info / max(float(np.sum(info)), 1e-12)
                else:
                    wv = (inv_cov @ ones) / denom
                if weight_constraint == "nonnegative":
                    wv = np.clip(wv, 0.0, None)
                    if float(np.sum(wv)) <= 0.0:
                        wv = np.ones_like(wv)
                wv = wv / max(float(np.sum(wv)), 1e-12)

                fused_logit = float(np.sum(wv * logits))
                p = float(self._sigmoid(fused_logit))

                if len(logits) > 1:
                    off = corr_sub[np.triu_indices(len(logits), k=1)]
                    mean_corr = float(np.mean(np.abs(off))) if off.size else 0.0
                else:
                    mean_corr = 0.0
                corr_inflation = 1.0 + float(correlation_penalty) * mean_corr * max(
                    len(logits) - 1, 0
                )
                n_eff = max(
                    float(evidence_strength) * float(np.sum(info)) / max(corr_inflation, 1e-6),
                    1.0,
                )

            alpha = float(prior_alpha) + p * n_eff
            beta = float(prior_beta) + (1.0 - p) * n_eff
            lo, hi = self._beta_interval(
                alpha,
                beta,
                z=float(ci_z),
                method=ci_method,
                level=ci_level,
            )
            p_out[i] = p
            ci_lo[i] = lo
            ci_hi[i] = hi
            n_eff_out[i] = float(n_eff)

        labeled_mask = np.isfinite(y_prior)
        labeled_idx = np.where(labeled_mask)[0]
        n_labeled = int(labeled_idx.size)
        p_base = p_out.copy()
        temp = 1.0
        ece_before = 0.0
        brier_before = 0.0
        nll_before = 0.0
        ece_after = 0.0
        brier_after = 0.0
        nll_after = 0.0
        eval_scheme = "none"
        n_eval = 0
        folds_used = 0
        fold_temps: list[float] = []

        if n_labeled > 0:
            ece_before, brier_before, nll_before = self._calibration_metrics(
                p_base[labeled_mask],
                y_prior[labeled_mask],
                n_bins=int(max(2, calibration_bins)),
            )

            p_cal = p_base.copy()
            if (
                calibration_mode == "temperature"
                and n_labeled >= int(max(1, calibration_min_samples))
            ):
                temp = self._fit_temperature(
                    p_base[labeled_mask],
                    y_prior[labeled_mask],
                )
                logits_all = np.log(self._clip_prob(p_base, eps=1e-9)) - np.log(
                    1.0 - self._clip_prob(p_base, eps=1e-9)
                )
                p_cal = self._sigmoid(logits_all / max(float(temp), 1e-9))
                p_cal = self._clip_prob(p_cal, eps=1e-9)
                for i in range(n):
                    p_i = float(p_cal[i])
                    n_eff = max(float(n_eff_out[i]), 1.0)
                    alpha = float(prior_alpha) + p_i * n_eff
                    beta = float(prior_beta) + (1.0 - p_i) * n_eff
                    lo, hi = self._beta_interval(
                        alpha,
                        beta,
                        z=float(ci_z),
                        method=ci_method,
                        level=ci_level,
                    )
                    ci_lo[i] = lo
                    ci_hi[i] = hi
                p_out = p_cal

            # Calibration diagnostics: prefer cross-fit when requested and feasible.
            use_cross_fit = (
                calibration_mode == "temperature"
                and calibration_scheme_canon == "cross_fit"
                and n_labeled >= int(max(4, calibration_min_samples))
            )
            if use_cross_fit:
                k = int(max(2, calibration_folds))
                perm = labeled_idx.copy()
                rng = np.random.RandomState(int(calibration_seed))
                rng.shuffle(perm)
                fold_ids = np.mod(np.arange(perm.size), k)
                p_cv = np.full(n, np.nan, dtype=float)
                for fold in range(k):
                    val_mask_fold = fold_ids == fold
                    if not np.any(val_mask_fold):
                        continue
                    train_idx = perm[~val_mask_fold]
                    val_idx = perm[val_mask_fold]
                    if train_idx.size < int(max(2, calibration_min_samples)):
                        continue
                    t_fold = self._fit_temperature(
                        p_base[train_idx],
                        y_prior[train_idx],
                    )
                    fold_temps.append(float(t_fold))
                    logits_val = np.log(self._clip_prob(p_base[val_idx], eps=1e-9)) - np.log(
                        1.0 - self._clip_prob(p_base[val_idx], eps=1e-9)
                    )
                    q_val = self._sigmoid(logits_val / max(float(t_fold), 1e-9))
                    p_cv[val_idx] = self._clip_prob(q_val, eps=1e-9)
                    folds_used += 1
                cv_mask = np.isfinite(p_cv[labeled_idx])
                if np.any(cv_mask):
                    eval_scheme = "cross_fit"
                    n_eval = int(np.sum(cv_mask))
                    eval_probs = p_cv[labeled_idx][cv_mask]
                    eval_labels = y_prior[labeled_idx][cv_mask]
                    ece_after, brier_after, nll_after = self._calibration_metrics(
                        eval_probs,
                        eval_labels,
                        n_bins=int(max(2, calibration_bins)),
                    )
                else:
                    eval_scheme = "in_sample_fallback"
                    n_eval = int(n_labeled)
                    ece_after, brier_after, nll_after = self._calibration_metrics(
                        p_out[labeled_mask],
                        y_prior[labeled_mask],
                        n_bins=int(max(2, calibration_bins)),
                    )
            else:
                eval_scheme = "in_sample"
                n_eval = int(n_labeled)
                ece_after, brier_after, nll_after = self._calibration_metrics(
                    p_out[labeled_mask],
                    y_prior[labeled_mask],
                    n_bins=int(max(2, calibration_bins)),
                )

        self._last_inference_diagnostics = {
            "mode": calibration_mode,
            "scheme": calibration_scheme_canon,
            "eval_scheme": eval_scheme,
            "temperature": float(temp),
            "temperature_cv_mean": float(np.mean(fold_temps)) if fold_temps else float(temp),
            "temperature_cv_std": float(np.std(fold_temps)) if fold_temps else 0.0,
            "folds_used": int(folds_used),
            "n_labeled": int(n_labeled),
            "n_eval": int(n_eval),
            "ece_before": float(ece_before),
            "brier_before": float(brier_before),
            "nll_before": float(nll_before),
            "ece_after": float(ece_after),
            "brier_after": float(brier_after),
            "nll_after": float(nll_after),
            "channel_reliability_used": {
                ch: float(override_reliability.get(ch, self._channel_reliability[ch].mean))
                for ch in _EVIDENCE_CHANNELS
            },
        }

        work["topological_probability"] = p_out
        work["topology_ci_low"] = ci_lo
        work["topology_ci_high"] = ci_hi

        formula_series = work["formula"].astype("string").fillna("")
        vectors = np.vstack([self._formula_vector(str(f)) for f in formula_series.to_list()])
        if reference_formulas is None:
            ref_formulas = [
                str(f)
                for f in formula_series.to_list()
                if str(f).strip() and str(f).lower() not in {"nan", "<na>"}
            ]
        else:
            ref_formulas = [str(f) for f in reference_formulas]

        if ref_formulas:
            ref = np.vstack([self._formula_vector(f) for f in ref_formulas])
            v_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            r_norm = np.linalg.norm(ref, axis=1, keepdims=True)
            v = vectors / np.where(v_norm > 1e-12, v_norm, 1.0)
            r = ref / np.where(r_norm > 1e-12, r_norm, 1.0)
            sims = v @ r.T
            max_sim = np.max(sims, axis=1)
            dist = np.clip(1.0 - max_sim, 0.0, 1.0)
            scale = max(float(ood_scale), 1e-6)
            ood = self._sigmoid((dist - scale) / scale)
        else:
            ood = np.full(n, 0.5, dtype=float)
        work["ood_score"] = np.clip(ood, 0.0, 1.0)

        if update_topo_class:
            prob = work["topological_probability"].to_numpy(dtype=float)
            gap = pd.to_numeric(work["band_gap"], errors="coerce").to_numpy(dtype=float)
            gap = np.where(np.isfinite(gap), gap, 0.0)
            is_topo = prob >= float(decision_threshold)
            cls = np.where(
                is_topo,
                np.where(gap <= float(semimetal_gap_threshold), "TSM", "TI"),
                "TRIVIAL",
            )
            work["topo_class"] = cls

        work = self._normalize_frame(work)
        if persist:
            self._df = work
            self.save()
            if self.use_sql:
                self.to_sql()
        return work

    def rank_topological_candidates(
        self,
        *,
        min_probability: float = 0.55,
        max_ood: float = 1.0,
        uncertainty_weight: float = 0.35,
        ood_penalty: float = 0.50,
        top_k: int | None = None,
        recompute_if_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Rank candidates by exploitation-exploration score with OOD penalty.

        score = p_topo + lambda_u * uncertainty_width - lambda_ood * ood_score
        """
        work = self.df.copy()
        if recompute_if_missing and (
            "topological_probability" not in work.columns
            or work["topological_probability"].isna().all()
        ):
            work = self.infer_topology_probabilities(persist=False)

        prob = pd.to_numeric(work.get("topological_probability"), errors="coerce").fillna(0.5)
        ci_lo = pd.to_numeric(work.get("topology_ci_low"), errors="coerce").fillna(0.0)
        ci_hi = pd.to_numeric(work.get("topology_ci_high"), errors="coerce").fillna(1.0)
        ood = pd.to_numeric(work.get("ood_score"), errors="coerce").fillna(0.5)
        width = np.clip(ci_hi - ci_lo, 0.0, 1.0)

        score = prob + float(uncertainty_weight) * width - float(ood_penalty) * ood
        ranked = work.copy()
        ranked["acquisition_score"] = score
        ranked = ranked[
            (prob >= float(min_probability)) & (ood <= float(max_ood))
        ].copy()
        ranked = ranked.sort_values("acquisition_score", ascending=False).reset_index(drop=True)
        if top_k is not None:
            ranked = ranked.head(int(top_k)).reset_index(drop=True)
        return ranked

    def load_seed_data(self):
        seed_df = pd.DataFrame(_SEED_ROWS)
        if self.df.empty:
            merged = seed_df.copy()
        else:
            merged = pd.concat([self.df, seed_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["jid"], keep="last")
        self._df = self._normalize_frame(merged).reset_index(drop=True)
        self.save()
        if self.use_sql:
            self.to_sql()

    def save(self):
        self.save_csv()

    def save_csv(self, df: pd.DataFrame | None = None):
        out = df if df is not None else self.df
        self.db_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.db_file, index=False)

    def to_sql(self, db_path: Path | None = None):
        path = db_path or self.sql_path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        self.df.to_sql("materials", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_formula ON materials (formula)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_topo ON materials (topo_class)")
        conn.commit()
        conn.close()

    def query_sql(self, sql_query: str) -> pd.DataFrame:
        if not self.sql_path.exists():
            self.to_sql()
        conn = sqlite3.connect(self.sql_path)
        try:
            return pd.read_sql(sql_query, conn)
        finally:
            conn.close()

    def fuzzy_search(self, query: str, cutoff: float = 0.6) -> pd.DataFrame:
        formulas = self.df["formula"].astype(str).tolist()
        matches = difflib.get_close_matches(query, formulas, n=5, cutoff=cutoff)
        if not matches:
            return self._empty_df()
        return self.df[self.df["formula"].isin(matches)].copy()

    def query(
        self,
        topo_class: str | None = None,
        elements: list[str] | None = None,
        band_gap_range: tuple | None = None,
        exact_formula: str | None = None,
    ) -> pd.DataFrame:
        df = self.df.copy()

        if exact_formula:
            df = df[df["formula"] == exact_formula]

        if topo_class:
            df = df[df["topo_class"] == str(topo_class).upper()]

        if band_gap_range:
            lo, hi = band_gap_range
            if lo is not None:
                df = df[df["band_gap"] >= lo]
            if hi is not None:
                df = df[df["band_gap"] <= hi]

        if elements:
            required = {str(e) for e in elements}

            def has_elements(formula: str) -> bool:
                present = set(self._parse_formula_counts(formula).keys())
                return required.issubset(present)

            df = df[df["formula"].apply(has_elements)]

        return df.reset_index(drop=True)

    def add_material(self, formula: str, topo_class: str, properties: dict):
        jid = properties.get("jid", f"mat-{len(self.df):06d}")
        mat = TopoMaterial(
            jid=jid,
            formula=formula,
            space_group=int(properties.get("space_group", 0)),
            topo_class=str(topo_class).upper(),
            band_gap=float(properties.get("band_gap", 0.0)),
            source=str(properties.get("source", "custom")),
            spillage=_as_float(properties.get("spillage")),
            si_score=_as_float(properties.get("si_score")),
            ml_probability=_as_float(properties.get("ml_probability")),
            ml_uncertainty=_as_float(properties.get("ml_uncertainty")),
            topological_probability=_as_float(properties.get("topological_probability")),
            topology_ci_low=_as_float(properties.get("topology_ci_low")),
            topology_ci_high=_as_float(properties.get("topology_ci_high")),
            ood_score=_as_float(properties.get("ood_score")),
        )
        self.add_materials([mat])

    def add_materials(self, materials: list[TopoMaterial]):
        if not materials:
            return
        new_df = pd.DataFrame([m.to_dict() for m in materials])
        if self.df.empty:
            merged = new_df.copy()
        else:
            merged = pd.concat([self.df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["jid"], keep="last")
        self._df = self._normalize_frame(merged).reset_index(drop=True)
        self.save()
        if self.use_sql:
            self.to_sql()

    def stats(self) -> dict:
        df = self.df
        by_class = df["topo_class"].value_counts().to_dict() if len(df) else {}
        by_source = df["source"].value_counts().to_dict() if len(df) else {}
        return {
            "total": int(len(df)),
            "by_class": by_class,
            "by_source": by_source,
            "channel_reliability": self.channel_reliability(),
            "channel_correlation": self.channel_correlation(),
            "last_inference_diagnostics": self.last_inference_diagnostics(),
        }
