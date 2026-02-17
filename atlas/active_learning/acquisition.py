"""
Acquisition Functions for Bayesian Active Learning

Optimizes the trade-off between exploration (high uncertainty) and exploitation (high predicted value).
"""

import torch
import numpy as np
from typing import Union

def expected_improvement(
    mean: torch.Tensor,
    std: torch.Tensor,
    best_f: float,
    maximize: bool = True,
    jitter: float = 0.01
) -> torch.Tensor:
    """
    Computes Expected Improvement (EI).
    
    EI(x) = (μ(x) - f_best - ξ)Φ(Z) + σ(x)φ(Z)  if maximize
    EI(x) = (f_best - μ(x) - ξ)Φ(Z) + σ(x)φ(Z)  if minimize
    
    where Z = (μ(x) - f_best - ξ) / σ(x)
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
        std = torch.tensor(std)

    if not maximize:
        # Transform minimize problem to maximize (-mean)
        return expected_improvement(-mean, std, -best_f, maximize=True, jitter=jitter)

    # Standard Normal
    normal = torch.distributions.Normal(0.0, 1.0)
    
    # Z-score
    delta = mean - best_f - jitter
    z = delta / (std + 1e-9)
    
    cdf = normal.cdf(z)
    pdf = normal.log_prob(z).exp()
    
    ei = delta * cdf + std * pdf
    return torch.clamp(ei, min=0.0)


def upper_confidence_bound(
    mean: torch.Tensor,
    std: torch.Tensor,
    kappa: float = 2.0,
    maximize: bool = True
) -> torch.Tensor:
    """
    Computes Upper Confidence Bound (UCB).
    
    UCB(x) = μ(x) + κ * σ(x)  if maximize
    LCB(x) = μ(x) - κ * σ(x)  if minimize
    """
    if maximize:
        return mean + kappa * std
    else:
        return mean - kappa * std
