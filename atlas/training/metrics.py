"""
Evaluation Metrics

Metrics for evaluating crystal property predictions:
- Scalar: MAE, RMSE, R², MaxAE
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Tensor: Frobenius norm error, eigenvalue agreement, symmetry violation
"""


import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return (pred - target).abs().mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return ((pred - target) ** 2).mean().sqrt().item()


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination (R²)."""
    if len(target) < 2: return 0.0
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum() + 1e-8
    return (1 - ss_res / ss_tot).item()


def max_ae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Maximum Absolute Error."""
    return (pred - target).abs().max().item()


def scalar_metrics(
    pred: torch.Tensor, target: torch.Tensor, prefix: str = ""
) -> dict[str, float]:
    """
    Compute all scalar metrics.
    """
    p = prefix + "_" if prefix else ""
    return {
        f"{p}MAE": mae(pred, target),
        f"{p}RMSE": rmse(pred, target),
        f"{p}R2": r2_score(pred, target),
        f"{p}MaxAE": max_ae(pred, target),
    }


def classification_metrics(
    logits: torch.Tensor, target: torch.Tensor, prefix: str = ""
) -> dict[str, float]:
    """
    Compute classification metrics.

    Args:
        logits: (N,) raw logits
        target: (N,) binary targets (0 or 1)
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    y_true = target.cpu().numpy().astype(int)

    p = prefix + "_" if prefix else ""

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.5 # If only one class present

    return {
        f"{p}Accuracy": accuracy_score(y_true, preds),
        f"{p}Precision": precision_score(y_true, preds, zero_division=0),
        f"{p}Recall": recall_score(y_true, preds, zero_division=0),
        f"{p}F1": f1_score(y_true, preds, zero_division=0),
        f"{p}AUC": auc
    }


def frobenius_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Frobenius norm of tensor prediction error.
    """
    diff = pred - target
    frob = torch.linalg.norm(diff.reshape(diff.size(0), -1), dim=-1)
    return frob.mean().item()


def symmetry_violation(C: torch.Tensor) -> float:
    """
    Measure how much a predicted tensor violates symmetry (Cij = Cji).
    """
    asym = C - C.transpose(-1, -2)
    return asym.abs().mean().item()


def eigenvalue_agreement(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compare eigenvalue ordering between predicted and target tensors.
    """
    from scipy.stats import spearmanr

    scores = []
    for i in range(pred.size(0)):
        try:
            ev_pred = torch.linalg.eigvalsh(pred[i]).cpu().numpy()
            ev_target = torch.linalg.eigvalsh(target[i]).cpu().numpy()
            corr, _ = spearmanr(ev_pred, ev_target)
            if not np.isnan(corr):
                scores.append(corr)
        except RuntimeError:
            pass # Eigendecomposition failed

    return np.mean(scores) if scores else 0.0


def tensor_metrics(
    pred: torch.Tensor, target: torch.Tensor, prefix: str = ""
) -> dict[str, float]:
    """
    Compute all tensor metrics.
    """
    p = prefix + "_" if prefix else ""
    return {
        f"{p}Frobenius": frobenius_error(pred, target),
        f"{p}SymViolation": symmetry_violation(pred),
        f"{p}EigAgreement": eigenvalue_agreement(pred, target),
    }
