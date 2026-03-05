"""
Evaluation Metrics

Metrics for evaluating crystal property predictions:
- Scalar: MAE, RMSE, R², MaxAE
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Tensor: Frobenius norm error, eigenvalue agreement, symmetry violation
"""


import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:  # pragma: no cover - exercised via monkeypatch in tests
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:  # pragma: no cover - SciPy is optional
    _scipy_spearmanr = None


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    """Cast inputs to float tensor for metric computation."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.detach().to(dtype=torch.float32)


def _finite_pair_1d(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten and keep only finite paired values."""
    p = _as_float_tensor(pred).reshape(-1)
    t = _as_float_tensor(target).reshape(-1)
    n = min(p.numel(), t.numel())
    if n == 0:
        return p.new_zeros(0), t.new_zeros(0)
    p = p[:n]
    t = t[:n]
    mask = torch.isfinite(p) & torch.isfinite(t)
    return p[mask], t[mask]


def _safe_item(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    scalar = float(value.item())
    if not math.isfinite(scalar):
        return 0.0
    return scalar


def _safe_float(value: float, default: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(default)
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(scalar):
        return float(default)
    return scalar


def _normalize_prefix(prefix: str) -> str:
    token = str(prefix).strip()
    return f"{token}_" if token else ""


def _as_square_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor-like input to (B, N, N) square batches."""
    tensor = _as_float_tensor(x)
    if tensor.ndim == 2 and tensor.size(0) == tensor.size(1):
        tensor = tensor.unsqueeze(0)
    if tensor.ndim < 3 or tensor.size(-1) != tensor.size(-2):
        return tensor.new_zeros((0, 0, 0))
    return tensor.reshape(-1, tensor.size(-2), tensor.size(-1))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation for Spearman fallback when SciPy is unavailable."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < sorted_values.size:
        j = i + 1
        while j < sorted_values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if _scipy_spearmanr is not None:
        corr, _ = _scipy_spearmanr(x, y)
        return float(corr)
    rx = _rankdata(x)
    ry = _rankdata(y)
    std_x = float(rx.std())
    std_y = float(ry.std())
    if std_x <= 0.0 or std_y <= 0.0:
        return float("nan")
    corr = float(np.corrcoef(rx, ry)[0, 1])
    return corr


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    p, t = _finite_pair_1d(pred, target)
    if p.numel() == 0:
        return 0.0
    return _safe_item((p - t).abs().mean())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    p, t = _finite_pair_1d(pred, target)
    if p.numel() == 0:
        return 0.0
    return _safe_item(torch.sqrt(((p - t) ** 2).mean()))


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Coefficient of determination (R²)."""
    p, t = _finite_pair_1d(pred, target)
    if t.numel() < 2:
        return 0.0
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    if ss_tot <= 1e-12:
        return 0.0
    return _safe_item(1 - ss_res / ss_tot)


def max_ae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Maximum Absolute Error."""
    p, t = _finite_pair_1d(pred, target)
    if p.numel() == 0:
        return 0.0
    return _safe_item((p - t).abs().max())


def scalar_metrics(
    pred: torch.Tensor, target: torch.Tensor, prefix: str = ""
) -> dict[str, float]:
    """
    Compute all scalar metrics.
    """
    p = _normalize_prefix(prefix)
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
    p_logits, t_labels = _finite_pair_1d(logits, target)
    p = _normalize_prefix(prefix)
    if p_logits.numel() == 0:
        return {
            f"{p}Accuracy": 0.0,
            f"{p}Precision": 0.0,
            f"{p}Recall": 0.0,
            f"{p}F1": 0.0,
            f"{p}AUC": 0.5,
        }

    probs = torch.sigmoid(p_logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_true = (t_labels > 0.5).cpu().numpy().astype(int)

    if np.unique(y_true).size < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = 0.5
        if not np.isfinite(auc):
            auc = 0.5

    return {
        f"{p}Accuracy": _safe_float(accuracy_score(y_true, preds), default=0.0),
        f"{p}Precision": _safe_float(precision_score(y_true, preds, zero_division=0), default=0.0),
        f"{p}Recall": _safe_float(recall_score(y_true, preds, zero_division=0), default=0.0),
        f"{p}F1": _safe_float(f1_score(y_true, preds, zero_division=0), default=0.0),
        f"{p}AUC": _safe_float(auc, default=0.5),
    }


def frobenius_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Frobenius norm of tensor prediction error.
    """
    p = _as_square_batch(pred)
    t = _as_square_batch(target)
    n = min(p.size(0), t.size(0))
    if n == 0:
        return 0.0
    p = p[:n]
    t = t[:n]
    row_mask = torch.isfinite(p.reshape(n, -1)).all(dim=1) & torch.isfinite(t.reshape(n, -1)).all(dim=1)
    if not row_mask.any():
        return 0.0
    diff = p[row_mask] - t[row_mask]
    frob = torch.linalg.norm(diff.reshape(diff.size(0), -1), dim=-1)
    return _safe_item(frob.mean())


def symmetry_violation(C: torch.Tensor) -> float:
    """
    Measure how much a predicted tensor violates symmetry (Cij = Cji).
    """
    tensor = _as_square_batch(C)
    if tensor.numel() == 0:
        return 0.0
    finite_mask = torch.isfinite(tensor.reshape(tensor.shape[0], -1)).all(dim=1)
    tensor = tensor[finite_mask]
    if tensor.numel() == 0:
        return 0.0
    asym = tensor - tensor.transpose(-1, -2)
    return _safe_item(asym.abs().mean())


def eigenvalue_agreement(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compare eigenvalue ordering between predicted and target tensors.
    """
    p = _as_square_batch(pred)
    t = _as_square_batch(target)
    n = min(p.size(0), t.size(0))
    if n == 0:
        return 0.0
    scores = []
    for i in range(n):
        try:
            pi = p[i]
            ti = t[i]
            if not torch.isfinite(pi).all() or not torch.isfinite(ti).all():
                continue
            ev_pred = torch.linalg.eigvalsh(pi).cpu().numpy()
            ev_target = torch.linalg.eigvalsh(ti).cpu().numpy()
            pred_const = np.allclose(ev_pred, ev_pred[0])
            target_const = np.allclose(ev_target, ev_target[0])
            if pred_const or target_const:
                scores.append(1.0 if np.allclose(ev_pred, ev_target) else 0.0)
                continue
            corr = _spearman_corr(ev_pred, ev_target)
            if np.isfinite(corr):
                scores.append(float(np.clip(corr, -1.0, 1.0)))
        except (RuntimeError, ValueError):
            pass  # Eigendecomposition failed

    return float(np.mean(scores)) if scores else 0.0


def tensor_metrics(
    pred: torch.Tensor, target: torch.Tensor, prefix: str = ""
) -> dict[str, float]:
    """
    Compute all tensor metrics.
    """
    p = _normalize_prefix(prefix)
    return {
        f"{p}Frobenius": frobenius_error(pred, target),
        f"{p}SymViolation": symmetry_violation(pred),
        f"{p}EigAgreement": eigenvalue_agreement(pred, target),
    }
