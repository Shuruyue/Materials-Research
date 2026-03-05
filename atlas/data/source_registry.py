"""
Data source registry for materials datasets.

This module centralizes dataset metadata so experiments can be reproduced
with explicit source names and citations.

Algorithmic extensions in this file are intentionally lightweight and depend
only on NumPy. They implement:
1) Bayesian source-reliability updates via Beta-Binomial posteriors.
2) Drift-aware source scoring for target-dependent routing.
3) Correlation-aware scalar fusion via generalized least squares (GLS).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace as dataclass_replace
from numbers import Real

import numpy as np


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _normalize_text(value: object, *, field_name: str) -> str:
    if _is_boolean_like(value) or not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


@dataclass(frozen=True)
class DataSourceSpec:
    """Dataset descriptor used across training/inference pipelines."""

    key: str
    name: str
    domain: str
    primary_targets: list[str] = field(default_factory=list)
    url: str = ""
    citation: str = ""
    reliability_prior_alpha: float = 1.0
    reliability_prior_beta: float = 1.0

    def __post_init__(self) -> None:
        key = _normalize_text(self.key, field_name="DataSourceSpec.key")
        name = _normalize_text(self.name, field_name="DataSourceSpec.name")
        domain = _normalize_text(self.domain, field_name="DataSourceSpec.domain")

        targets = self.primary_targets
        if isinstance(targets, str):
            targets = [targets]
        elif targets is None:
            targets = []
        elif not isinstance(targets, Iterable):
            raise TypeError("DataSourceSpec.primary_targets must be an iterable of strings")

        cleaned_targets: list[str] = []
        seen: set[str] = set()
        for target in targets:
            if _is_boolean_like(target) or not isinstance(target, str):
                raise TypeError("DataSourceSpec.primary_targets entries must be strings")
            token = target.strip()
            if not token or token in seen:
                continue
            cleaned_targets.append(token)
            seen.add(token)

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "primary_targets", cleaned_targets)
        if self.url is None:
            object.__setattr__(self, "url", "")
        else:
            object.__setattr__(self, "url", str(self.url).strip())
        if self.citation is None:
            object.__setattr__(self, "citation", "")
        else:
            object.__setattr__(self, "citation", str(self.citation).strip())


@dataclass(frozen=True)
class SourceReliability:
    """Posterior reliability state p(success | source) ~ Beta(alpha, beta)."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        a = float(self.alpha) if math.isfinite(float(self.alpha)) else 0.0
        b = float(self.beta) if math.isfinite(float(self.beta)) else 0.0
        denom = a + b
        if denom <= 1e-12:
            return 0.5
        return float(np.clip(a / denom, 0.0, 1.0))

    @property
    def variance(self) -> float:
        a = max(float(self.alpha), 0.0) if math.isfinite(float(self.alpha)) else 0.0
        b = max(float(self.beta), 0.0) if math.isfinite(float(self.beta)) else 0.0
        s = a + b
        denom = s * s * (s + 1.0)
        if denom <= 1e-12:
            return 0.0
        out = (a * b) / denom
        return float(max(out, 0.0))


@dataclass(frozen=True)
class SourceEstimate:
    """
    Scalar estimate contributed by one source.

    `drift_distance` is an optional non-negative shift metric between the
    source distribution and current task distribution.
    """

    source_key: str
    value: float
    std: float
    drift_distance: float = 0.0


@dataclass(frozen=True)
class FusedEstimate:
    """Result of reliability-aware and correlation-aware scalar fusion."""

    mean: float
    std: float
    weights: dict[str, float]
    effective_sample_size: float


@dataclass(frozen=True)
class ReliabilitySnapshot:
    """Serializable snapshot of registry reliability posteriors."""

    state: dict[str, tuple[float, float]]


class DataSourceRegistry:
    """
    In-memory registry with probabilistic source weighting utilities.

    References:
    - Dawid, A. P. & Skene, A. M. (1979), "Maximum Likelihood Estimation
      of Observer Error-Rates Using the EM Algorithm".
    - Aitken, A. C. (1935), "On Least Squares and Linear Combinations
      of Observations".
    - Ben-David, S. et al. (2010), "A theory of learning from different
      domains".
    - Ledoit, O. & Wolf, M. (2004), "A well-conditioned estimator for
      large-dimensional covariance matrices".
    - Jagannathan, R. & Ma, T. (2003), "Risk reduction in large portfolios:
      Why imposing the wrong constraints helps".
    """

    def __init__(self):
        self._sources: dict[str, DataSourceSpec] = {}
        self._reliability: dict[str, SourceReliability] = {}
        self._reliability_priors: dict[str, SourceReliability] = {}

    @staticmethod
    def _normalize_source_key(value: str, *, field_name: str = "key") -> str:
        return _normalize_text(value, field_name=field_name)

    @staticmethod
    def _coerce_nonnegative_finite(value: float, default: float = 0.0) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return float(max(out, 0.0))

    @staticmethod
    def _coerce_nonnegative_finite_strict(value: float, *, field_name: str) -> float:
        if _is_boolean_like(value):
            raise ValueError(f"{field_name} must be finite and non-negative, not boolean")
        if not isinstance(value, Real):
            raise ValueError(f"{field_name} must be finite and non-negative")
        out = float(value)
        if not math.isfinite(out) or out < 0.0:
            raise ValueError(f"{field_name} must be finite and non-negative")
        return out

    @classmethod
    def _coerce_positive_finite(cls, value: float, default: float = 1.0) -> float:
        out = cls._coerce_nonnegative_finite(value, default=default)
        if out <= 0.0:
            return float(default)
        return out

    def register(self, spec: DataSourceSpec, *, replace: bool = False):
        if not isinstance(spec, DataSourceSpec):
            raise TypeError(f"spec must be DataSourceSpec, got {type(spec).__name__}")
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean")
        key = self._normalize_source_key(spec.key, field_name="DataSourceSpec.key")
        if key in self._sources and not replace:
            raise ValueError(
                f"Data source '{key}' already registered. Use replace=True to overwrite."
            )
        spec_norm = spec if spec.key == key else dataclass_replace(spec, key=key)
        self._sources[key] = spec_norm
        prior = SourceReliability(
            alpha=self._coerce_positive_finite(spec_norm.reliability_prior_alpha, default=1.0),
            beta=self._coerce_positive_finite(spec_norm.reliability_prior_beta, default=1.0),
        )
        self._reliability[key] = prior
        self._reliability_priors[key] = prior

    def get(self, key: str) -> DataSourceSpec:
        key = self._normalize_source_key(key)
        if key not in self._sources:
            available = ", ".join(sorted(self._sources.keys()))
            raise KeyError(f"Unknown data source '{key}'. Available: {available}")
        return self._sources[key]

    def get_reliability(self, key: str) -> SourceReliability:
        key = self._normalize_source_key(key)
        if key not in self._reliability:
            available = ", ".join(sorted(self._sources.keys()))
            raise KeyError(f"Unknown data source '{key}'. Available: {available}")
        return self._reliability[key]

    def update_reliability(
        self, key: str, *, successes: float = 0.0, failures: float = 0.0
    ):
        """
        Update source reliability posterior via Beta-Binomial conjugacy.

        For each source:
          prior  ~ Beta(alpha, beta)
          data   ~ Binomial(successes, failures)
          post   ~ Beta(alpha + successes, beta + failures)
        """
        succ = self._coerce_nonnegative_finite_strict(successes, field_name="successes")
        fail = self._coerce_nonnegative_finite_strict(failures, field_name="failures")
        state = self.get_reliability(key)
        self._reliability[key] = SourceReliability(
            alpha=state.alpha + succ,
            beta=state.beta + fail,
        )

    def snapshot_reliability(self) -> ReliabilitySnapshot:
        """Capture reliability state for reproducible experimentation."""
        return ReliabilitySnapshot(
            state={k: (v.alpha, v.beta) for k, v in self._reliability.items()}
        )

    def restore_reliability(self, snapshot: ReliabilitySnapshot):
        """Restore reliability posteriors from a previously captured snapshot."""
        restored: dict[str, SourceReliability] = {}
        for key in self.list_keys():
            if key not in snapshot.state:
                raise KeyError(f"Snapshot missing source key '{key}'")
            alpha, beta = snapshot.state[key]
            restored[key] = SourceReliability(
                alpha=self._coerce_positive_finite(alpha, default=1.0),
                beta=self._coerce_positive_finite(beta, default=1.0),
            )
        self._reliability = restored

    def reset_reliability(self, keys: Iterable[str] | None = None):
        """Reset reliability posteriors to per-source priors."""
        if keys is None:
            keys = self.list_keys()
        for key in keys:
            normalized_key = self._normalize_source_key(key)
            if normalized_key not in self._reliability_priors:
                available = ", ".join(sorted(self._sources.keys()))
                raise KeyError(f"Unknown data source '{normalized_key}'. Available: {available}")
            prior = self._reliability_priors[normalized_key]
            self._reliability[normalized_key] = SourceReliability(alpha=prior.alpha, beta=prior.beta)

    @contextmanager
    def reliability_scope(self):
        """
        Context-managed reliability isolation.

        Any posterior updates inside the scope are rolled back on exit.
        """
        snapshot = self.snapshot_reliability()
        try:
            yield
        finally:
            self.restore_reliability(snapshot)

    def source_score(
        self,
        key: str,
        *,
        target: str | None = None,
        drift_distance: float = 0.0,
        drift_lambda: float = 1.0,
    ) -> float:
        """
        Drift-aware reliability score used for source routing.

        score = coverage(target) * reliability_mean * exp(-lambda * drift)
        """
        spec = self.get(key)
        rel = self.get_reliability(key).mean
        if not math.isfinite(rel):
            rel = 0.5
        coverage = 1.0
        if target:
            if target in spec.primary_targets:
                coverage = 1.0
            elif "task_specific" in spec.primary_targets:
                coverage = 0.6
            else:
                coverage = 0.2
        drift = self._coerce_nonnegative_finite(drift_distance, default=0.0)
        lam = self._coerce_nonnegative_finite(drift_lambda, default=1.0)
        drift_penalty = math.exp(-lam * drift)
        return coverage * rel * drift_penalty

    def rank_sources(
        self,
        *,
        target: str,
        drift_by_source: dict[str, float] | None = None,
        drift_lambda: float = 1.0,
    ) -> list[tuple[str, float]]:
        """Rank all sources by drift-aware reliability score."""
        drift_map = drift_by_source or {}
        ranked = []
        for key in self.list_keys():
            score = self.source_score(
                key,
                target=target,
                drift_distance=drift_map.get(key, 0.0),
                drift_lambda=drift_lambda,
            )
            ranked.append((key, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    @staticmethod
    def _nearest_psd(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Project a symmetric matrix to the positive semi-definite cone."""
        sym = np.asarray(matrix, dtype=float)
        sym = np.nan_to_num(sym, nan=0.0, posinf=0.0, neginf=0.0)
        sym = 0.5 * (sym + sym.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.clip(eigvals, eps, None)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @classmethod
    def _normalize_correlation_matrix(cls, corr: np.ndarray) -> np.ndarray:
        """Return a numerically stable PSD correlation matrix."""
        corr = np.asarray(corr, dtype=float)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = cls._nearest_psd(corr, eps=1e-10)
        d = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
        corr = corr / np.outer(d, d)
        corr = np.clip(corr, -0.99, 0.99)
        np.fill_diagonal(corr, 1.0)
        return 0.5 * (corr + corr.T)

    def estimate_correlation_matrix(
        self,
        source_keys: list[str],
        residuals_by_source: dict[str, Iterable[float]],
        *,
        shrinkage: float = 0.2,
    ) -> np.ndarray:
        """
        Estimate source correlation matrix from residual traces.

        Uses pairwise empirical correlation on overlapping tails, then applies
        Ledoit-Wolf style shrinkage toward identity for stability.
        """
        if not source_keys:
            raise ValueError("source_keys must be non-empty")
        normalized_keys = [self._normalize_source_key(key, field_name="source key") for key in source_keys]
        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("source_keys must be unique for correlation estimation")
        n = len(source_keys)
        raw = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                key_i = normalized_keys[i]
                key_j = normalized_keys[j]
                r_i = np.asarray(list(residuals_by_source.get(key_i, [])), dtype=float)
                r_j = np.asarray(list(residuals_by_source.get(key_j, [])), dtype=float)
                r_i = r_i[np.isfinite(r_i)]
                r_j = r_j[np.isfinite(r_j)]
                overlap = min(len(r_i), len(r_j))
                if overlap < 3:
                    corr_ij = 0.0
                else:
                    a = r_i[-overlap:] - np.mean(r_i[-overlap:])
                    b = r_j[-overlap:] - np.mean(r_j[-overlap:])
                    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
                    corr_ij = float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0
                    corr_ij = float(np.clip(corr_ij, -0.99, 0.99))
                raw[i, j] = corr_ij
                raw[j, i] = corr_ij
        lam = self._coerce_nonnegative_finite(shrinkage, default=0.2)
        lam = float(np.clip(lam, 0.0, 1.0))
        shrunk = (1.0 - lam) * raw + lam * np.eye(n, dtype=float)
        return self._normalize_correlation_matrix(shrunk)

    @staticmethod
    def _weight_map(keys: list[str], weights: np.ndarray) -> dict[str, float]:
        totals = {key: keys.count(key) for key in set(keys)}
        emitted: dict[str, int] = {}
        out: dict[str, float] = {}
        for key, weight in zip(keys, weights, strict=True):
            if totals[key] == 1:
                out[key] = float(weight)
                continue
            idx = emitted.get(key, 0)
            emitted[key] = idx + 1
            out[f"{key}#{idx}"] = float(weight)
        return out

    @staticmethod
    def _gls_weights_unconstrained(cov: np.ndarray) -> np.ndarray:
        ones = np.ones(cov.shape[0], dtype=float)
        cov = np.asarray(cov, dtype=float)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        inv_cov = np.linalg.pinv(cov, rcond=1e-12)
        denom = float(ones @ inv_cov @ ones)
        if denom <= 0.0 or not math.isfinite(denom):
            raise ValueError("invalid covariance in fusion (non-positive denominator)")
        return (inv_cov @ ones) / denom

    @classmethod
    def _gls_weights_nonnegative(cls, cov: np.ndarray) -> np.ndarray:
        """
        Active-set solver for non-negative minimum-variance GLS weights.

        Solves:
            min_w w^T Sigma w
            s.t. 1^T w = 1, w_i >= 0
        """
        n = cov.shape[0]
        active = list(range(n))
        weights = np.zeros(n, dtype=float)
        while active:
            sub_cov = cov[np.ix_(active, active)]
            sub_w = cls._gls_weights_unconstrained(sub_cov)
            if np.all(sub_w >= -1e-12):
                weights[:] = 0.0
                weights[active] = np.clip(sub_w, 0.0, None)
                total = float(weights.sum())
                if total <= 0.0:
                    raise ValueError("invalid non-negative fusion weights (zero sum)")
                return weights / total
            drop_local = int(np.argmin(sub_w))
            active.pop(drop_local)
        raise ValueError("failed to compute non-negative fusion weights")

    def fuse_scalar_estimates(
        self,
        estimates: Iterable[SourceEstimate],
        *,
        pairwise_correlation: float = 0.0,
        correlation_matrix: np.ndarray | None = None,
        residuals_by_source: dict[str, Iterable[float]] | None = None,
        correlation_shrinkage: float = 0.2,
        reliability_temperature: float = 1.0,
        drift_lambda: float = 1.0,
        min_std: float = 1e-6,
        weight_constraint: str = "nonnegative",
    ) -> FusedEstimate:
        """
        Fuse correlated scalar estimates using reliability-aware GLS.

        Implementation notes:
        - Base variance is sigma_i^2.
        - Low reliability / high drift inflates variance:
            var_i <- var_i / (reliability_i^tau * exp(-lambda * drift_i))
        - Off-diagonal covariance uses constant correlation rho:
            cov_ij = rho * sqrt(var_i * var_j)
        - GLS weights:
            w = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)
        """
        est_list = list(estimates)
        if not est_list:
            raise ValueError("at least one estimate is required")
        if weight_constraint not in {"nonnegative", "unconstrained"}:
            raise ValueError("weight_constraint must be 'nonnegative' or 'unconstrained'")
        tau = self._coerce_nonnegative_finite(reliability_temperature, default=1.0)
        min_std = self._coerce_positive_finite(min_std, default=1e-6)

        valid_estimates: list[SourceEstimate] = []
        for e in est_list:
            if not math.isfinite(float(e.value)):
                continue
            if not math.isfinite(float(e.std)) or float(e.std) <= 0.0:
                continue
            valid_estimates.append(e)
        if not valid_estimates:
            raise ValueError("all estimates were invalid (non-finite value/std)")

        keys = [self._normalize_source_key(e.source_key, field_name="SourceEstimate.source_key") for e in valid_estimates]
        values = np.array([float(e.value) for e in valid_estimates], dtype=float)
        variances = np.zeros(len(valid_estimates), dtype=float)
        for i, e in enumerate(valid_estimates):
            rel = max(self.get_reliability(keys[i]).mean, 1e-6) ** tau
            drift = self._coerce_nonnegative_finite(e.drift_distance, default=0.0)
            lam = self._coerce_nonnegative_finite(drift_lambda, default=1.0)
            drift_factor = math.exp(-lam * drift)
            reliability_factor = max(rel * drift_factor, 1e-6)
            sigma = max(float(e.std), min_std)
            variances[i] = (sigma * sigma) / reliability_factor

        n = len(valid_estimates)
        if correlation_matrix is not None:
            corr = np.asarray(correlation_matrix, dtype=float)
            if corr.shape != (n, n):
                raise ValueError(
                    f"correlation_matrix shape must be {(n, n)}, got {corr.shape}"
                )
            corr = self._normalize_correlation_matrix(corr)
        elif residuals_by_source is not None:
            corr = self.estimate_correlation_matrix(
                keys,
                residuals_by_source,
                shrinkage=correlation_shrinkage,
            )
        else:
            try:
                rho_raw = float(pairwise_correlation)
            except Exception:
                rho_raw = 0.0
            if not math.isfinite(rho_raw):
                rho_raw = 0.0
            rho = float(np.clip(rho_raw, -0.99, 0.99))
            corr = np.full((n, n), rho, dtype=float)
            np.fill_diagonal(corr, 1.0)
            corr = self._normalize_correlation_matrix(corr)

        stds = np.sqrt(np.clip(variances, 1e-18, None))
        cov = np.outer(stds, stds) * corr
        cov = self._nearest_psd(cov, eps=1e-12)

        if weight_constraint == "nonnegative":
            w = self._gls_weights_nonnegative(cov)
        else:
            w = self._gls_weights_unconstrained(cov)

        mean = float(w @ values)
        var = max(float(w @ cov @ w), 1e-18)
        n_eff = float((w.sum() ** 2) / max(np.sum(w * w), 1e-18))
        weights = self._weight_map(keys, w)
        return FusedEstimate(
            mean=mean, std=math.sqrt(var), weights=weights, effective_sample_size=n_eff
        )

    def list_keys(self) -> list[str]:
        return sorted(self._sources.keys())

    def list_all(self) -> list[DataSourceSpec]:
        return [self._sources[k] for k in self.list_keys()]


DATA_SOURCES = DataSourceRegistry()

# Inorganic/metal/semiconductor-focused defaults for this project.
DATA_SOURCES.register(
    DataSourceSpec(
        key="jarvis_dft",
        name="JARVIS-DFT",
        domain="inorganic_crystals",
        primary_targets=[
            "formation_energy",
            "band_gap",
            "bulk_modulus",
            "shear_modulus",
        ],
        url="https://jarvis.nist.gov/",
        citation="Choudhary et al., NPJ Comput Mater (2020).",
    )
)
DATA_SOURCES.register(
    DataSourceSpec(
        key="materials_project",
        name="Materials Project",
        domain="inorganic_crystals",
        primary_targets=[
            "formation_energy",
            "band_gap",
            "elasticity",
            "dielectric",
        ],
        url="https://materialsproject.org/",
        citation="Jain et al., APL Materials (2013).",
    )
)
DATA_SOURCES.register(
    DataSourceSpec(
        key="matbench",
        name="Matbench",
        domain="benchmark_suite",
        primary_targets=["task_specific"],
        url="https://matbench.materialsproject.org/",
        citation="Dunn et al., NPJ Comput Mater (2020).",
    )
)
DATA_SOURCES.register(
    DataSourceSpec(
        key="oqmd",
        name="OQMD",
        domain="inorganic_crystals",
        primary_targets=["formation_energy"],
        url="https://oqmd.org/",
        citation="Saal et al., JOM (2013).",
    )
)
