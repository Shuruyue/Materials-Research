"""
Evaluation Metrics

Metrics for evaluating crystal property predictions:
- Scalar: MAE, RMSE, R², MaxAE
- Tensor: Frobenius norm error, eigenvalue agreement, symmetry violation
"""

import torch
import numpy as np
from typing import Dict, Optional


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return (pred - target).abs().mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return ((pred - target) ** 2).mean().sqrt().item()


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination (R²)."""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def max_ae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Maximum Absolute Error."""
    return (pred - target).abs().max().item()


def scalar_metrics(
    pred: torch.Tensor, target: torch.Tensor, prefix: str = ""
) -> Dict[str, float]:
    """
    Compute all scalar metrics.

    Returns:
        Dict with MAE, RMSE, R², MaxAE
    """
    p = prefix + "_" if prefix else ""
    return {
        f"{p}MAE": mae(pred, target),
        f"{p}RMSE": rmse(pred, target),
        f"{p}R2": r2_score(pred, target),
        f"{p}MaxAE": max_ae(pred, target),
    }


def frobenius_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Frobenius norm of tensor prediction error.

    Args:
        pred: (B, M, N) predicted tensor
        target: (B, M, N) target tensor

    Returns:
        Average Frobenius norm error
    """
    diff = pred - target
    frob = torch.linalg.norm(diff.reshape(diff.size(0), -1), dim=-1)
    return frob.mean().item()


def symmetry_violation(C: torch.Tensor) -> float:
    """
    Measure how much a predicted tensor violates symmetry (Cij = Cji).

    Args:
        C: (B, N, N) predicted tensor

    Returns:
        Average asymmetry magnitude
    """
    asym = C - C.transpose(-1, -2)
    return asym.abs().mean().item()


def eigenvalue_agreement(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compare eigenvalue ordering between predicted and target tensors.

    Args:
        pred: (B, N, N)
        target: (B, N, N)

    Returns:
        Spearman rank correlation of eigenvalues (averaged over batch)
    """
    from scipy.stats import spearmanr

    scores = []
    for i in range(pred.size(0)):
        ev_pred = torch.linalg.eigvalsh(pred[i]).cpu().numpy()
        ev_target = torch.linalg.eigvalsh(target[i]).cpu().numpy()
        corr, _ = spearmanr(ev_pred, ev_target)
        if not np.isnan(corr):
            scores.append(corr)

    return np.mean(scores) if scores else 0.0


def tensor_metrics(
    pred: torch.Tensor, target: torch.Tensor, prefix: str = ""
) -> Dict[str, float]:
    """
    Compute all tensor metrics.

    Returns:
        Dict with Frobenius error, symmetry violation, eigenvalue agreement
    """
    p = prefix + "_" if prefix else ""
    return {
        f"{p}Frobenius": frobenius_error(pred, target),
        f"{p}SymViolation": symmetry_violation(pred),
        f"{p}EigAgreement": eigenvalue_agreement(pred, target),
    }
