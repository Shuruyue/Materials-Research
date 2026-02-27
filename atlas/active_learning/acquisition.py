"""
Acquisition functions for Bayesian active learning.

This module provides:
1. Raw acquisition primitives (EI/UCB/PI/Thompson).
2. Strategy normalization (aliases -> canonical names).
3. A unified scorer that always returns "higher is better" utility.
"""

from __future__ import annotations

import torch


ACQUISITION_ALIASES = {
    "ei": "ei",
    "expected_improvement": "ei",
    "expected-improvement": "ei",
    "pi": "pi",
    "probability_of_improvement": "pi",
    "probability-of-improvement": "pi",
    "ucb": "ucb",
    "upper_confidence_bound": "ucb",
    "upper-confidence-bound": "ucb",
    "lcb": "lcb",
    "lower_confidence_bound": "lcb",
    "lower-confidence-bound": "lcb",
    "thompson": "thompson",
    "thompson_sampling": "thompson",
    "thompson-sampling": "thompson",
    "mean": "mean",
    "greedy": "mean",
    "uncertainty": "uncertainty",
    "std": "uncertainty",
}

BASE_ACQUISITION_STRATEGIES = (
    "ei",
    "pi",
    "ucb",
    "lcb",
    "thompson",
    "mean",
    "uncertainty",
)

DISCOVERY_ACQUISITION_STRATEGIES = (
    "hybrid",      # legacy behavior in controller (EI blend when available)
    "stability",   # direct stability_score in controller
    *BASE_ACQUISITION_STRATEGIES,
)


def _as_tensor(x: torch.Tensor | float, ref: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if ref is not None:
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(x, dtype=torch.float32)


def normalize_acquisition_strategy(strategy: str) -> str:
    key = str(strategy).strip().lower().replace(" ", "_")
    if key not in ACQUISITION_ALIASES:
        available = ", ".join(sorted(DISCOVERY_ACQUISITION_STRATEGIES))
        raise ValueError(f"Unknown acquisition strategy: {strategy!r}. Available: {available}")
    return ACQUISITION_ALIASES[key]


def expected_improvement(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    best_f: float,
    maximize: bool = True,
    jitter: float = 0.01,
) -> torch.Tensor:
    """
    Compute expected improvement (EI).

    EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
    where Z = (mu - f_best - xi) / sigma
    """
    mean = _as_tensor(mean)
    std = _as_tensor(std, ref=mean).clamp(min=1e-9)

    if not maximize:
        # Transform minimization into maximization.
        return expected_improvement(-mean, std, -best_f, maximize=True, jitter=jitter)

    normal = torch.distributions.Normal(0.0, 1.0)
    delta = mean - best_f - jitter
    z = delta / std
    cdf = normal.cdf(z)
    pdf = normal.log_prob(z).exp()
    ei = delta * cdf + std * pdf
    return torch.clamp(ei, min=0.0)


def probability_of_improvement(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    best_f: float,
    maximize: bool = True,
    jitter: float = 0.01,
) -> torch.Tensor:
    """Compute probability of improvement (PI)."""
    mean = _as_tensor(mean)
    std = _as_tensor(std, ref=mean).clamp(min=1e-9)

    if not maximize:
        # Transform minimization into maximization.
        return probability_of_improvement(-mean, std, -best_f, maximize=True, jitter=jitter)

    normal = torch.distributions.Normal(0.0, 1.0)
    z = (mean - best_f - jitter) / std
    return normal.cdf(z)


def upper_confidence_bound(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    kappa: float = 2.0,
    maximize: bool = True,
) -> torch.Tensor:
    """
    Compute confidence bound.

    maximize=True  -> UCB = mu + kappa * sigma
    maximize=False -> LCB = mu - kappa * sigma
    """
    mean = _as_tensor(mean)
    std = _as_tensor(std, ref=mean).clamp(min=1e-9)
    if maximize:
        return mean + kappa * std
    return mean - kappa * std


def thompson_sampling(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one Thompson sample from Normal(mean, std)."""
    mean = _as_tensor(mean)
    std = _as_tensor(std, ref=mean).clamp(min=1e-9)
    noise = torch.randn(
        mean.shape,
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    return mean + std * noise


def score_acquisition(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    *,
    strategy: str = "ei",
    best_f: float = 0.0,
    maximize: bool = True,
    kappa: float = 2.0,
    jitter: float = 0.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Unified acquisition scoring.

    Returns a utility tensor where larger values are always preferred.
    """
    canon = normalize_acquisition_strategy(strategy)
    mean = _as_tensor(mean)
    std = _as_tensor(std, ref=mean).clamp(min=1e-9)

    if canon == "ei":
        return expected_improvement(mean, std, best_f=best_f, maximize=maximize, jitter=jitter)
    if canon == "pi":
        return probability_of_improvement(mean, std, best_f=best_f, maximize=maximize, jitter=jitter)
    if canon in {"ucb", "lcb"}:
        bound = upper_confidence_bound(mean, std, kappa=kappa, maximize=maximize)
        return bound if maximize else -bound
    if canon == "thompson":
        sample = thompson_sampling(mean, std, generator=generator)
        return sample if maximize else -sample
    if canon == "mean":
        return mean if maximize else -mean
    if canon == "uncertainty":
        return std

    # Defensive fallback; should never execute due to normalization.
    raise ValueError(f"Unsupported acquisition strategy: {strategy!r}")
