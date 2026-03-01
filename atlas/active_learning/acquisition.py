"""
Acquisition functions for Bayesian active learning.

This module provides:
1. Raw acquisition primitives (EI/UCB/PI/Thompson).
2. Strategy normalization (aliases -> canonical names).
3. A unified scorer that always returns "higher is better" utility.

References:
- Jones et al. (1998), Efficient Global Optimization (EI)
  https://link.springer.com/article/10.1023/A:1008306431147
- Srinivas et al. (2010), GP-UCB regret bounds
  https://arxiv.org/abs/0912.3995
- Ament et al. (2023), LogEI numerical stabilization
  https://arxiv.org/abs/2310.20708
- Letham et al. (2019), Noisy Expected Improvement (NEI)
  https://arxiv.org/abs/1706.07094
"""

from __future__ import annotations

import math

import torch

ACQUISITION_ALIASES = {
    "ei": "ei",
    "expected_improvement": "ei",
    "expected-improvement": "ei",
    "pi": "pi",
    "probability_of_improvement": "pi",
    "probability-of-improvement": "pi",
    "log_pi": "log_pi",
    "log-pi": "log_pi",
    "log_probability_of_improvement": "log_pi",
    "log-probability-of-improvement": "log_pi",
    "nei": "nei",
    "noisy_ei": "nei",
    "noisy-expected-improvement": "nei",
    "noisy_expected_improvement": "nei",
    "log_nei": "log_nei",
    "log-nei": "log_nei",
    "log_noisy_ei": "log_nei",
    "log-noisy-expected-improvement": "log_nei",
    "log_noisy_expected_improvement": "log_nei",
    "ucb": "ucb",
    "upper_confidence_bound": "ucb",
    "upper-confidence-bound": "ucb",
    "lcb": "lcb",
    "lower_confidence_bound": "lcb",
    "lower-confidence-bound": "lcb",
    "thompson": "thompson",
    "thompson_sampling": "thompson",
    "thompson-sampling": "thompson",
    "log_ei": "log_ei",
    "log-ei": "log_ei",
    "log_expected_improvement": "log_ei",
    "log-expected-improvement": "log_ei",
    "mean": "mean",
    "greedy": "mean",
    "uncertainty": "uncertainty",
    "std": "uncertainty",
}

BASE_ACQUISITION_STRATEGIES = (
    "ei",
    "log_ei",
    "pi",
    "log_pi",
    "nei",
    "log_nei",
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
_ASYMPTOTIC_LOG_EI_SWITCH = -5.0


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


def _log_clamped(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).tiny
    return torch.log(torch.clamp(x, min=eps))


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


def _log_expected_improvement_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    best_f: float,
    jitter: float,
) -> torch.Tensor:
    """
    Numerically robust log-EI implementation.

    Reference:
    - Ament et al. (2023): https://arxiv.org/abs/2310.20708
    """
    delta = mean - best_f - jitter
    z = delta / std

    cdf = torch.special.ndtr(z)
    pdf = _INV_SQRT_2PI * torch.exp(-0.5 * z * z)
    naive = z * cdf + pdf
    log_naive = _log_clamped(naive)

    # Tail stabilization for very negative z:
    # z * Phi(z) + phi(z) ~ phi(z) / z^2, as z -> -inf.
    # This preserves usable gradients far in the negative tail.
    neg_z = torch.clamp(-z, min=1.0)
    log_phi = -0.5 * z * z + math.log(_INV_SQRT_2PI)
    log_tail = log_phi - 2.0 * torch.log(neg_z)

    use_tail = z < _ASYMPTOTIC_LOG_EI_SWITCH
    log_h = torch.where(use_tail, log_tail, log_naive)
    return _log_clamped(std) + log_h


def _probability_of_improvement_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    best_f: float,
    jitter: float,
) -> torch.Tensor:
    z = (mean - best_f - jitter) / std
    return torch.special.ndtr(z)


def _log_probability_of_improvement_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    best_f: float,
    jitter: float,
) -> torch.Tensor:
    """
    Numerically robust log-PI implementation using log_ndtr.

    Reference:
    - BoTorch analytic acquisition docs:
      https://botorch.readthedocs.io/en/stable/acquisition.html
    """
    z = (mean - best_f - jitter) / std
    log_pi = torch.special.log_ndtr(z)
    floor = math.log(torch.finfo(z.dtype).tiny)
    return torch.clamp(log_pi, min=floor)


def _prepare_observed(
    observed_mean: torch.Tensor | list[float] | tuple[float, ...] | None,
    observed_std: torch.Tensor | list[float] | tuple[float, ...] | None,
    *,
    ref: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if observed_mean is None:
        return None, None
    obs_mean_t = _as_tensor(observed_mean, ref=ref).reshape(-1)
    obs_mean_t = torch.nan_to_num(obs_mean_t, nan=0.0, posinf=0.0, neginf=0.0)
    if obs_mean_t.numel() == 0:
        return None, None

    if observed_std is None:
        obs_std_t = torch.zeros_like(obs_mean_t)
    else:
        obs_std_t = _as_tensor(observed_std, ref=obs_mean_t).reshape(-1)
        if obs_std_t.numel() != obs_mean_t.numel():
            raise ValueError(
                "observed_std must have the same number of elements as observed_mean."
            )
        obs_std_t = _sanitize_std(obs_std_t)
    return obs_mean_t, obs_std_t


def _noisy_expected_improvement_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    observed_mean: torch.Tensor | None,
    observed_std: torch.Tensor | None,
    best_f_fallback: float,
    jitter: float,
    mc_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Monte Carlo approximation of NEI using sampled noisy incumbent values.

    Reference:
    - Letham et al. (2019): https://arxiv.org/abs/1706.07094
    """
    if observed_mean is None or observed_std is None or observed_mean.numel() == 0:
        return _expected_improvement_prepared(mean, std, best_f=best_f_fallback, jitter=jitter)

    sample_count = max(8, int(mc_samples))
    eps = torch.randn(
        (sample_count, observed_mean.numel()),
        dtype=observed_mean.dtype,
        device=observed_mean.device,
        generator=generator,
    )
    noisy_obs = observed_mean.unsqueeze(0) + observed_std.unsqueeze(0) * eps
    noisy_best = noisy_obs.max(dim=1).values

    flat_mean = mean.reshape(-1, 1)
    flat_std = std.reshape(-1, 1)
    delta = flat_mean - noisy_best.unsqueeze(0) - jitter
    z = delta / flat_std
    cdf = torch.special.ndtr(z)
    pdf = _INV_SQRT_2PI * torch.exp(-0.5 * z * z)
    ei_samples = torch.clamp(delta * cdf + flat_std * pdf, min=0.0)
    nei = ei_samples.mean(dim=1)
    return nei.reshape(mean.shape)


def _log_noisy_expected_improvement_prepared(
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    observed_mean: torch.Tensor | None,
    observed_std: torch.Tensor | None,
    best_f_fallback: float,
    jitter: float,
    mc_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    nei = _noisy_expected_improvement_prepared(
        mean,
        std,
        observed_mean=observed_mean,
        observed_std=observed_std,
        best_f_fallback=best_f_fallback,
        jitter=jitter,
        mc_samples=mc_samples,
        generator=generator,
    )
    return _log_clamped(nei)


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


def schedule_ucb_kappa(
    iteration: int,
    *,
    base_kappa: float = 2.0,
    mode: str = "fixed",
    dimension: int = 1,
    delta: float = 0.1,
    kappa_min: float = 1.0,
    decay: float = 0.08,
) -> float:
    """
    Time-varying UCB exploration parameter.

    References:
    - Srinivas et al. (2010) for GP-UCB beta_t schedule:
      https://arxiv.org/abs/0912.3995
    """
    t = max(1, int(iteration))
    mode_key = str(mode).strip().lower().replace("-", "_")

    if mode_key in {"fixed", "constant"}:
        return float(base_kappa)
    if mode_key in {"anneal", "exp_decay"}:
        k0 = max(float(base_kappa), float(kappa_min))
        return float(kappa_min + (k0 - kappa_min) * math.exp(-float(decay) * (t - 1)))
    if mode_key in {"gp_ucb", "srinivas"}:
        d = max(1, int(dimension))
        delta_eff = min(max(float(delta), 1e-12), 1.0 - 1e-12)
        beta_t = 2.0 * math.log((t ** (d / 2.0 + 2.0)) * (math.pi**2) / (3.0 * delta_eff))
        return float(max(float(kappa_min), math.sqrt(max(beta_t, 1e-12))))
    raise ValueError(f"Unknown kappa schedule mode: {mode!r}")


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


def log_expected_improvement(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    best_f: float,
    maximize: bool = True,
    jitter: float = 0.01,
) -> torch.Tensor:
    """Compute logarithm of expected improvement (LogEI)."""
    mean_t, std_t = _prepare_mean_std(mean, std)
    if not maximize:
        mean_t = -mean_t
        best_f = -best_f
    return _log_expected_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)


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


def log_probability_of_improvement(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    best_f: float,
    maximize: bool = True,
    jitter: float = 0.01,
) -> torch.Tensor:
    """Compute logarithm of probability of improvement (LogPI)."""
    mean_t, std_t = _prepare_mean_std(mean, std)
    if not maximize:
        mean_t = -mean_t
        best_f = -best_f
    return _log_probability_of_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)


def noisy_expected_improvement(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    *,
    observed_mean: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    observed_std: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    best_f_fallback: float = 0.0,
    maximize: bool = True,
    jitter: float = 0.01,
    mc_samples: int = 128,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute Monte Carlo Noisy Expected Improvement (NEI)."""
    mean_t, std_t = _prepare_mean_std(mean, std)
    obs_mean_t, obs_std_t = _prepare_observed(observed_mean, observed_std, ref=mean_t)

    if not maximize:
        mean_t = -mean_t
        best_f_fallback = -best_f_fallback
        if obs_mean_t is not None:
            obs_mean_t = -obs_mean_t

    return _noisy_expected_improvement_prepared(
        mean_t,
        std_t,
        observed_mean=obs_mean_t,
        observed_std=obs_std_t,
        best_f_fallback=best_f_fallback,
        jitter=jitter,
        mc_samples=mc_samples,
        generator=generator,
    )


def log_noisy_expected_improvement(
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    *,
    observed_mean: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    observed_std: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    best_f_fallback: float = 0.0,
    maximize: bool = True,
    jitter: float = 0.01,
    mc_samples: int = 128,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute logarithm of Monte Carlo Noisy Expected Improvement (log-NEI)."""
    nei = noisy_expected_improvement(
        mean,
        std,
        observed_mean=observed_mean,
        observed_std=observed_std,
        best_f_fallback=best_f_fallback,
        maximize=maximize,
        jitter=jitter,
        mc_samples=mc_samples,
        generator=generator,
    )
    return _log_clamped(nei)


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
    kappa_schedule: str = "fixed",
    iteration: int | None = None,
    ucb_dimension: int = 1,
    ucb_delta: float = 0.1,
    kappa_min: float = 1.0,
    kappa_decay: float = 0.08,
    observed_mean: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    observed_std: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    nei_mc_samples: int = 128,
    jitter: float = 0.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Unified acquisition scoring.

    Returns a utility tensor where larger values are always preferred.
    """
    canon = normalize_acquisition_strategy(strategy)
    mean_t, std_t = _prepare_mean_std(mean, std)
    obs_mean_t, obs_std_t = _prepare_observed(observed_mean, observed_std, ref=mean_t)

    if canon == "ei":
        if not maximize:
            mean_t = -mean_t
            best_f = -best_f
        return _expected_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)
    if canon == "log_ei":
        if not maximize:
            mean_t = -mean_t
            best_f = -best_f
        return _log_expected_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)
    if canon == "pi":
        if not maximize:
            mean_t = -mean_t
            best_f = -best_f
        return _probability_of_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)
    if canon == "log_pi":
        if not maximize:
            mean_t = -mean_t
            best_f = -best_f
        return _log_probability_of_improvement_prepared(mean_t, std_t, best_f=best_f, jitter=jitter)
    if canon == "nei":
        if not maximize:
            mean_t = -mean_t
            best_f = -best_f
            if obs_mean_t is not None:
                obs_mean_t = -obs_mean_t
        return _noisy_expected_improvement_prepared(
            mean_t,
            std_t,
            observed_mean=obs_mean_t,
            observed_std=obs_std_t,
            best_f_fallback=best_f,
            jitter=jitter,
            mc_samples=nei_mc_samples,
            generator=generator,
        )
    if canon == "log_nei":
        if not maximize:
            mean_t = -mean_t
            best_f = -best_f
            if obs_mean_t is not None:
                obs_mean_t = -obs_mean_t
        return _log_noisy_expected_improvement_prepared(
            mean_t,
            std_t,
            observed_mean=obs_mean_t,
            observed_std=obs_std_t,
            best_f_fallback=best_f,
            jitter=jitter,
            mc_samples=nei_mc_samples,
            generator=generator,
        )
    if canon == "ucb":
        kappa_t = schedule_ucb_kappa(
            iteration=1 if iteration is None else int(iteration),
            base_kappa=kappa,
            mode=kappa_schedule,
            dimension=ucb_dimension,
            delta=ucb_delta,
            kappa_min=kappa_min,
            decay=kappa_decay,
        )
        bound = _upper_confidence_bound_prepared(mean_t, std_t, kappa=kappa_t, maximize=maximize)
        return bound if maximize else -bound
    if canon == "lcb":
        kappa_t = schedule_ucb_kappa(
            iteration=1 if iteration is None else int(iteration),
            base_kappa=kappa,
            mode=kappa_schedule,
            dimension=ucb_dimension,
            delta=ucb_delta,
            kappa_min=kappa_min,
            decay=kappa_decay,
        )
        bound = _upper_confidence_bound_prepared(mean_t, std_t, kappa=kappa_t, maximize=not maximize)
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
