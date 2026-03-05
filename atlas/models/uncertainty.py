"""
Uncertainty Quantification for GNN Predictions

Methods:
- EnsembleUQ: train M models, use prediction variance as uncertainty
- MCDropoutUQ: single model with dropout at inference time
"""

import math
from numbers import Integral, Real

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_prediction_payload(prediction):
    if isinstance(prediction, dict):
        if not prediction:
            raise ValueError("Model prediction dict cannot be empty")
        normalized = {}
        for task, value in prediction.items():
            if not isinstance(task, str):
                raise TypeError(f"Prediction task name must be str, got {type(task)!r}")
            if not torch.is_tensor(value):
                raise TypeError(f"Prediction for task {task!r} must be Tensor, got {type(value)!r}")
            normalized[task] = value
        return normalized
    if torch.is_tensor(prediction):
        return {"prediction": prediction}
    raise TypeError(
        "Model prediction must be dict[str, Tensor] or Tensor, "
        f"got {type(prediction)!r}"
    )


def _validate_prediction_keys(
    all_predictions: list[dict[str, torch.Tensor]],
) -> tuple[str, ...]:
    task_names = tuple(all_predictions[0].keys())
    expected = set(task_names)
    for idx, payload in enumerate(all_predictions[1:], start=1):
        current = set(payload.keys())
        if current != expected:
            raise ValueError(
                f"Inconsistent prediction keys across models/samples at index {idx}: "
                f"expected {sorted(expected)}, got {sorted(current)}"
            )
    return task_names


def _validate_prediction_shapes(
    all_predictions: list[dict[str, torch.Tensor]],
    task_names: tuple[str, ...],
) -> None:
    expected_shapes = {
        task: tuple(all_predictions[0][task].shape)
        for task in task_names
    }
    for idx, payload in enumerate(all_predictions[1:], start=1):
        for task in task_names:
            current_shape = tuple(payload[task].shape)
            if current_shape != expected_shapes[task]:
                raise ValueError(
                    f"Inconsistent prediction shape for task {task!r} at index {idx}: "
                    f"expected {expected_shapes[task]}, got {current_shape}"
                )


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _coerce_positive_int(value: object, name: str) -> int:
    """Convert scalar count parameters to strict positive integers."""
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be integer-valued, not boolean")

    if isinstance(value, Integral):
        integer = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        rounded = round(scalar)
        if abs(scalar - rounded) > 1e-9:
            raise ValueError(f"{name} must be integer-valued")
        integer = int(rounded)
    else:
        raise ValueError(f"{name} must be integer-valued")

    if integer <= 0:
        raise ValueError(f"{name} must be > 0")
    return integer


def _coerce_non_negative_finite_float(
    value: object,
    name: str,
    *,
    default: float,
) -> float:
    if _is_boolean_like(value):
        return float(default)
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(scalar):
        return float(default)
    return max(scalar, 0.0)


class EnsembleUQ(nn.Module):
    """
    Ensemble-based uncertainty quantification.

    Trains M independent models and uses prediction variance as uncertainty.
    Most reliable UQ method but M× computational cost.

    Args:
        model_factory: callable that creates a fresh model instance
        n_models: number of ensemble members
    """

    def __init__(self, model_factory, n_models: int = 5):
        super().__init__()
        n_models_int = _coerce_positive_int(n_models, "n_models")
        self.models = nn.ModuleList([model_factory() for _ in range(n_models_int)])
        self.n_models = n_models_int

    def forward(self, *args, **kwargs) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through all ensemble members.

        Returns:
            Dict[task_name → (mean, std)] for each predicted property
        """
        all_predictions = [_normalize_prediction_payload(model(*args, **kwargs)) for model in self.models]

        # Aggregate predictions
        result = {}
        task_names = _validate_prediction_keys(all_predictions)
        _validate_prediction_shapes(all_predictions, task_names)
        for task in task_names:
            preds = torch.stack([p[task] for p in all_predictions], dim=0)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
            mean = preds.mean(dim=0)
            std = preds.std(dim=0, unbiased=False).clamp_min(0.0)
            result[task] = (mean, std)

        return result

    def predict_with_uncertainty(
        self, *args, **kwargs
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Returns predictions with uncertainty estimates.

        Returns:
            Dict[task_name → {"mean": Tensor, "std": Tensor, "all": Tensor}]
        """
        all_predictions = [_normalize_prediction_payload(model(*args, **kwargs)) for model in self.models]

        result = {}
        task_names = _validate_prediction_keys(all_predictions)
        _validate_prediction_shapes(all_predictions, task_names)
        for task in task_names:
            preds = torch.stack([p[task] for p in all_predictions], dim=0)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
            result[task] = {
                "mean": preds.mean(dim=0),
                "std": preds.std(dim=0, unbiased=False).clamp_min(0.0),
                "all": preds,
            }

        return result


class MCDropoutUQ(nn.Module):
    """
    Monte Carlo Dropout uncertainty quantification.

    Uses dropout at inference time (model.train() during inference)
    to estimate prediction uncertainty.

    Cheaper than ensemble but less reliable.

    Args:
        model: trained model with dropout layers
        n_samples: number of forward passes at inference
    """

    def __init__(self, model: nn.Module, n_samples: int = 30):
        super().__init__()
        n_samples_int = _coerce_positive_int(n_samples, "n_samples")
        self.model = model
        self.n_samples = n_samples_int

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for m in self.model.modules():
            if isinstance(m, nn.modules.dropout._DropoutNd):
                m.train()

    def predict_with_uncertainty(
        self, *args, **kwargs
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Run N stochastic forward passes and compute statistics.

        Returns:
            Dict[task_name → {"mean": Tensor, "std": Tensor}]
        """
        was_training = self.model.training
        self.model.eval()
        self._enable_dropout()

        all_predictions = []
        try:
            with torch.no_grad():
                for _ in range(self.n_samples):
                    pred = self.model(*args, **kwargs)
                    all_predictions.append(_normalize_prediction_payload(pred))
        finally:
            self.model.train(was_training)

        result = {}
        task_names = _validate_prediction_keys(all_predictions)
        _validate_prediction_shapes(all_predictions, task_names)
        for task in task_names:
            preds = torch.stack([p[task] for p in all_predictions], dim=0)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
            result[task] = {
                "mean": preds.mean(dim=0),
                "std": preds.std(dim=0, unbiased=False).clamp_min(0.0),
            }

        return result


class EvidentialRegression(nn.Module):
    """
    Evidential deep learning for single-pass uncertainty (Amini et al., NeurIPS 2020).

    Predicts Normal-Inverse-Gamma distribution parameters: (γ, ν, α, β)
    providing both aleatoric and epistemic uncertainty in one forward pass.

    Aleatoric (data noise): σ²_a = β / (ν * (α - 1))
    Epistemic (model uncertainty): σ²_e = β / (ν * α * (α - 1))

    Args:
        input_dim: dimension of graph embedding
        output_dim: number of target properties
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, output_dim * 4),
        )

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            embedding: (B, input_dim) graph-level embedding

        Returns:
            Dict with "mean", "aleatoric", "epistemic", "total_std"
        """
        raw = self.head(embedding).reshape(-1, self.output_dim, 4)

        gamma = raw[..., 0]                            # mean
        nu = F.softplus(raw[..., 1]) + 1e-6            # > 0
        alpha = F.softplus(raw[..., 2]) + 1.0 + 1e-6   # > 1
        beta = F.softplus(raw[..., 3]) + 1e-6          # > 0

        aleatoric = torch.nan_to_num(
            beta / (nu * (alpha - 1)),
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        ).clamp_min(0.0)
        epistemic = torch.nan_to_num(
            beta / (nu * alpha * (alpha - 1)),
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        ).clamp_min(0.0)
        total_var = torch.nan_to_num(
            aleatoric + epistemic,
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        ).clamp_min(0.0)

        return {
            "mean": gamma,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_std": total_var.sqrt(),
            "_gamma": gamma, "_nu": nu, "_alpha": alpha, "_beta": beta,
        }

    @staticmethod
    def evidential_loss(pred: dict, target: torch.Tensor, coeff: float = 0.01) -> torch.Tensor:
        """NIG negative log-likelihood + evidence regularizer."""
        gamma, nu, alpha, beta = pred["_gamma"], pred["_nu"], pred["_alpha"], pred["_beta"]
        coeff_v = _coerce_non_negative_finite_float(coeff, "coeff", default=0.01)
        target = target.to(device=gamma.device, dtype=gamma.dtype)

        finite_mask = (
            torch.isfinite(target)
            & torch.isfinite(gamma)
            & torch.isfinite(nu)
            & torch.isfinite(alpha)
            & torch.isfinite(beta)
        )
        if not torch.any(finite_mask):
            return torch.zeros((), dtype=target.dtype, device=target.device)

        error = (target - gamma).abs()
        error = torch.where(finite_mask, error, torch.zeros_like(error))

        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=target.device, dtype=target.dtype) / nu.clamp_min(1e-8))
            - alpha * torch.log((2 * beta).clamp_min(1e-8))
            + (alpha + 0.5) * torch.log((nu * error**2 + 2 * beta).clamp_min(1e-8))
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )
        reg = error * (2 * nu + alpha)
        loss = torch.nan_to_num(nll + coeff_v * reg, nan=0.0, posinf=1e6, neginf=0.0)
        return loss[finite_mask].mean()
