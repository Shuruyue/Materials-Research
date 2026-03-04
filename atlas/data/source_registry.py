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
from dataclasses import dataclass, field

import numpy as np


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


@dataclass(frozen=True)
class SourceReliability:
    """Posterior reliability state p(success | source) ~ Beta(alpha, beta)."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        denom = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1.0)
        return (self.alpha * self.beta) / max(denom, 1e-12)


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

    def register(self, spec: DataSourceSpec):
        self._sources[spec.key] = spec
        prior = SourceReliability(
            alpha=float(spec.reliability_prior_alpha),
            beta=float(spec.reliability_prior_beta),
        )
        self._reliability[spec.key] = prior
        self._reliability_priors[spec.key] = prior

    def get(self, key: str) -> DataSourceSpec:
        if key not in self._sources:
            available = ", ".join(sorted(self._sources.keys()))
            raise KeyError(f"Unknown data source '{key}'. Available: {available}")
        return self._sources[key]

    def get_reliability(self, key: str) -> SourceReliability:
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
        if successes < 0 or failures < 0:
            raise ValueError("successes/failures must be non-negative")
        state = self.get_reliability(key)
        self._reliability[key] = SourceReliability(
            alpha=state.alpha + float(successes),
            beta=state.beta + float(failures),
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
            restored[key] = SourceReliability(alpha=float(alpha), beta=float(beta))
        self._reliability = restored

    def reset_reliability(self, keys: Iterable[str] | None = None):
        """Reset reliability posteriors to per-source priors."""
        if keys is None:
            keys = self.list_keys()
        for key in keys:
            if key not in self._reliability_priors:
                available = ", ".join(sorted(self._sources.keys()))
                raise KeyError(f"Unknown data source '{key}'. Available: {available}")
            prior = self._reliability_priors[key]
            self._reliability[key] = SourceReliability(alpha=prior.alpha, beta=prior.beta)

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
        coverage = 1.0
        if target:
            if target in spec.primary_targets:
                coverage = 1.0
            elif "task_specific" in spec.primary_targets:
                coverage = 0.6
            else:
                coverage = 0.2
        drift = max(float(drift_distance), 0.0)
        drift_penalty = math.exp(-max(float(drift_lambda), 0.0) * drift)
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
        sym = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.clip(eigvals, eps, None)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    @classmethod
    def _normalize_correlation_matrix(cls, corr: np.ndarray) -> np.ndarray:
        """Return a numerically stable PSD correlation matrix."""
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
        n = len(source_keys)
        raw = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                key_i = source_keys[i]
                key_j = source_keys[j]
                r_i = np.asarray(list(residuals_by_source.get(key_i, [])), dtype=float)
                r_j = np.asarray(list(residuals_by_source.get(key_j, [])), dtype=float)
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
        lam = float(np.clip(shrinkage, 0.0, 1.0))
        shrunk = (1.0 - lam) * raw + lam * np.eye(n, dtype=float)
        return self._normalize_correlation_matrix(shrunk)

    @staticmethod
    def _gls_weights_unconstrained(cov: np.ndarray) -> np.ndarray:
        ones = np.ones(cov.shape[0], dtype=float)
        inv_cov = np.linalg.pinv(cov, rcond=1e-12)
        denom = float(ones @ inv_cov @ ones)
        if denom <= 0.0:
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
        tau = max(float(reliability_temperature), 0.0)
        min_std = max(float(min_std), 1e-12)

        keys = [e.source_key for e in est_list]
        values = np.array([float(e.value) for e in est_list], dtype=float)
        variances = np.zeros(len(est_list), dtype=float)
        for i, e in enumerate(est_list):
            rel = max(self.get_reliability(e.source_key).mean, 1e-6) ** tau
            drift = max(float(e.drift_distance), 0.0)
            drift_factor = math.exp(-max(float(drift_lambda), 0.0) * drift)
            reliability_factor = max(rel * drift_factor, 1e-6)
            sigma = max(float(e.std), min_std)
            variances[i] = (sigma * sigma) / reliability_factor

        n = len(est_list)
        if correlation_matrix is not None:
            corr = np.asarray(correlation_matrix, dtype=float)
            if corr.shape != (n, n):
                raise ValueError(
                    f"correlation_matrix shape must be {(n, n)}, got {corr.shape}"
                )
            corr = self._normalize_correlation_matrix(corr)
        elif residuals_by_source is not None:
            corr = self.estimate_correlation_matrix(
                keys, residuals_by_source, shrinkage=correlation_shrinkage
            )
        else:
            rho = float(np.clip(pairwise_correlation, -0.99, 0.99))
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
        weights = {k: float(v) for k, v in zip(keys, w)}
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
