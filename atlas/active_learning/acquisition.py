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

_CONTROLLER_ONLY_ACQUISITION_STRATEGIES = ("hybrid", "stability")
_MIN_STD = 1e-9
_INV_SQRT_2PI = 0.3989422804014327


def _as_tensor(x: torch.Tensor | float, ref: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if ref is not None and (x.dtype != ref.dtype or x.device != ref.device):
            return x.to(dtype=ref.dtype, device=ref.device)
        return x
    if ref is not None:
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(x, dtype=torch.float32)


def _sanitize_std(std: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(std):
        std = std.to(torch.float32)
    floor = max(_MIN_STD, torch.finfo(std.dtype).tiny)
    std = std.abs()
    std = torch.nan_to_num(
        std,
        nan=floor,
        posinf=torch.finfo(std.dtype).max,
        neginf=floor,
    )
    return std.clamp(min=floor)


def _prepare_mean_std(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_t = _as_tensor(mean)
    std_t = _as_tensor(std, ref=mean_t)
    if not torch.is_floating_point(mean_t):
        mean_t = mean_t.to(torch.float32)
        std_t = std_t.to(dtype=mean_t.dtype, device=mean_t.device)
    std_t = _sanitize_std(std_t)
    return mean_t, std_t


def _expected_improvement_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    best_f: float,
    jitter: float,
) -> torch.Tensor:
    delta = mean - best_f - jitter
    z = delta / std
    cdf = torch.special.ndtr(z)
    pdf = _INV_SQRT_2PI * torch.exp(-0.5 * z * z)
    ei = delta * cdf + std * pdf
    return torch.clamp(ei, min=0.0)


def _probability_of_improvement_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    best_f: float,
    jitter: float,
) -> torch.Tensor:
    z = (mean - best_f - jitter) / std
    return torch.special.ndtr(z)


def _upper_confidence_bound_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    kappa: float,
    maximize: bool,
) -> torch.Tensor:
    if maximize:
        return mean + kappa * std
    return mean - kappa * std


def _thompson_sampling_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    noise = torch.randn(
        mean.shape,
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )
    return mean + std * noise


def normalize_acquisition_strategy(strategy: str) -> str:
    key = str(strategy).strip().lower().replace(" ", "_")
    if key not in ACQUISITION_ALIASES:
        available = ", ".join(BASE_ACQUISITION_STRATEGIES)
        controller_only = ", ".join(_CONTROLLER_ONLY_ACQUISITION_STRATEGIES)
        raise ValueError(
            f"Unknown acquisition strategy: {strategy!r}. "
            f"Supported in score_acquisition: {available}. "
            f"Controller-only strategies: {controller_only}."
        )
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
    mean_t, std_t = _prepare_mean_std(mean, std)
    if not maximize:
        # Transform minimization into maximization.
        mean_t = -mean_t
        best_f = -best_f
    return _expected_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)


def probability_of_improvement(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    best_f: float,
    maximize: bool = True,
    jitter: float = 0.01,
) -> torch.Tensor:
    """Compute probability of improvement (PI)."""
    mean_t, std_t = _prepare_mean_std(mean, std)
    if not maximize:
        # Transform minimization into maximization.
        mean_t = -mean_t
        best_f = -best_f
    return _probability_of_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)


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
    mean_t, std_t = _prepare_mean_std(mean, std)
    return _upper_confidence_bound_prepared(mean_t, std_t, kappa=kappa, maximize=maximize)


def thompson_sampling(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one Thompson sample from Normal(mean, std)."""
    mean_t, std_t = _prepare_mean_std(mean, std)
    return _thompson_sampling_prepared(mean_t, std_t, generator=generator)


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
    mean_t, std_t = _prepare_mean_std(mean, std)

    if canon == "ei":
        if not maximize:
            mean_t = -mean_t
            best_f = -best_f
        return _expected_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)
    if canon == "pi":
        if not maximize:
            mean_t = -mean_t
            best_f = -best_f
        return _probability_of_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)
    if canon == "ucb":
        bound = _upper_confidence_bound_prepared(mean_t, std_t, kappa=kappa, maximize=maximize)
        return bound if maximize else -bound
    if canon == "lcb":
        bound = _upper_confidence_bound_prepared(mean_t, std_t, kappa=kappa, maximize=not maximize)
        return bound if maximize else -bound
    if canon == "thompson":
        sample = _thompson_sampling_prepared(mean_t, std_t, generator=generator)
        return sample if maximize else -sample
    if canon == "mean":
        return mean_t if maximize else -mean_t
    if canon == "uncertainty":
        return std_t

    # Defensive fallback; should never execute due to normalization.
    raise ValueError(f"Unsupported acquisition strategy: {strategy!r}")
