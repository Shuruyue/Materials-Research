"""
Physics-Constrained Loss Functions

Loss functions that embed physical constraints:
- Positivity: K >= 0, G >= 0
- Bounds: ε >= 1, Voigt/Reuss bounds
- Stability: Born stability criteria for elastic tensors
- Multi-task weighting: uncertainty-based or GradNorm
"""

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_SUPPORTED_PROPERTY_LOSSES = {"mse", "l1", "huber", "bce"}
_SUPPORTED_TASK_TYPES = {"regression", "classification", "evidential"}
_SUPPORTED_STRATEGIES = {"fixed", "uncertainty"}


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool_", "bool"}


def _require_finite_scalar(value: float, name: str, *, min_value: float | None = None) -> float:
    """Validate scalar config values used by loss modules."""
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be numeric, got bool")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if min_value is not None and scalar < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return scalar


def _zero_loss(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return a scalar zero loss on the requested device/dtype."""
    return torch.zeros((), device=device, dtype=dtype, requires_grad=True)


def _normalize_named_mapping(values: Mapping[str, object] | None, *, name: str) -> dict[str, object]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, object] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{name} contains empty task name")
        if key in normalized:
            raise ValueError(f"{name} contains duplicate task name after normalization: {key!r}")
        normalized[key] = raw_value
    return normalized


def _validate_known_task_keys(
    values: Mapping[str, object],
    *,
    valid_task_names: set[str],
    name: str,
) -> None:
    unknown = sorted(set(values) - valid_task_names)
    if unknown:
        listed = ", ".join(unknown)
        raise KeyError(f"{name} contains unknown task(s): {listed}")


class PropertyLoss(nn.Module):
    """
    Single-property regression loss with optional physical constraints.
    """
    def __init__(
        self,
        property_name: str = "band_gap",
        constraint: str | None = None,
        constraint_weight: float = 0.1,
        loss_type: str = "mse"
    ):
        super().__init__()
        self.property_name = property_name
        if constraint not in {None, "positive", "greater_than_one"}:
            raise ValueError(
                f"Unsupported constraint={constraint!r}. Supported: None, 'positive', 'greater_than_one'"
            )
        self.constraint = constraint
        self.constraint_weight = _require_finite_scalar(constraint_weight, "constraint_weight", min_value=0.0)
        if loss_type not in _SUPPORTED_PROPERTY_LOSSES:
            supported = ", ".join(sorted(_SUPPORTED_PROPERTY_LOSSES))
            raise ValueError(f"Unknown loss_type='{loss_type}'. Supported: {supported}")
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("pred and target must be torch.Tensor")
        if pred.shape != target.shape:
            if pred.numel() != target.numel():
                raise ValueError(
                    f"pred and target must have same shape or same number of elements, "
                    f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
                )
            target = target.reshape_as(pred)

        # Filter invalid rows in both prediction/target tensors to avoid propagating NaN/Inf.
        mask = torch.isfinite(pred) & torch.isfinite(target)
        if not mask.any():
            return _zero_loss(pred.device, pred.dtype)

        pred = pred[mask]
        target = target[mask]

        if self.loss_type == "mse":
            base_loss = F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            base_loss = F.l1_loss(pred, target)
        elif self.loss_type == "huber":
            base_loss = F.huber_loss(pred, target, delta=1.0)
        elif self.loss_type == "bce":
            # For classification tasks like stability
            # Target should be 0 or 1
            target = torch.clamp(target, 0.0, 1.0)
            base_loss = F.binary_cross_entropy_with_logits(pred, target)

        if self.constraint == "positive":
            penalty = torch.mean(F.relu(-pred))
            return base_loss + self.constraint_weight * penalty
        if self.constraint == "greater_than_one":
            penalty = torch.mean(F.relu(1.0 - pred))
            return base_loss + self.constraint_weight * penalty
        if self.constraint is not None:
            raise ValueError(f"Unsupported constraint='{self.constraint}'")

        return base_loss



class EvidentialLoss(nn.Module):
    """
    Loss for Evidential Deep Learning (Regression).
    Combines NLL of Normal-Inverse-Gamma distribution with evidence regularization.
    """
    def __init__(self, coeff: float = 0.05):
        super().__init__()
        self.coeff = _require_finite_scalar(coeff, "coeff", min_value=0.0)

    def forward(self, pred: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if not isinstance(pred, Mapping):
            raise TypeError("pred must be a mapping of evidential tensors")
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be a torch.Tensor")
        dtype = target.dtype if target.is_floating_point() else torch.float32
        target = target.to(dtype=dtype)
        required = {"gamma", "nu", "alpha", "beta"}
        missing = required.difference(pred)
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"Evidential prediction missing required keys: {names}")

        gamma = torch.as_tensor(pred["gamma"], dtype=dtype, device=target.device).reshape(-1)
        nu = torch.as_tensor(pred["nu"], dtype=dtype, device=target.device).reshape(-1)
        alpha = torch.as_tensor(pred["alpha"], dtype=dtype, device=target.device).reshape(-1)
        beta = torch.as_tensor(pred["beta"], dtype=dtype, device=target.device).reshape(-1)
        target = target.reshape(-1)
        n = min(target.numel(), gamma.numel(), nu.numel(), alpha.numel(), beta.numel())
        if n == 0:
            return _zero_loss(target.device, target.dtype)
        target = target[:n]
        gamma = gamma[:n]
        nu = nu[:n]
        alpha = alpha[:n]
        beta = beta[:n]
        mask = (
            torch.isfinite(target)
            & torch.isfinite(gamma)
            & torch.isfinite(nu)
            & torch.isfinite(alpha)
            & torch.isfinite(beta)
            & (nu > 0.0)
            & (alpha > 0.0)
            & (beta > 0.0)
        )
        if not mask.any():
            return _zero_loss(target.device, target.dtype)

        target = target[mask]
        gamma = gamma[mask]
        nu = nu[mask]
        alpha = alpha[mask]
        beta = beta[mask]

        error = (target - gamma).abs()
        eps = torch.finfo(target.dtype).eps
        nu = torch.clamp(nu, min=eps)
        alpha = torch.clamp(alpha, min=eps)
        beta = torch.clamp(beta, min=eps)
        pi = target.new_tensor(torch.pi)

        # Negative Log Likelihood
        nll = (
            0.5 * torch.log(pi / nu)
            - alpha * torch.log(2 * beta)
            + (alpha + 0.5) * torch.log(nu * error.square() + 2 * beta)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Evidence Regularization (penalize high evidence for incorrect predictions)
        reg = error * (2 * nu + alpha)
        total = nll + self.coeff * reg
        finite = torch.isfinite(total)
        if not finite.any():
            return _zero_loss(target.device, target.dtype)
        return total[finite].mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with adaptive task weighting.
    Supports "fixed" (manual) or "uncertainty" (Kendall 2018) weighting.
    """
    def __init__(
        self,
        task_names: list[str],
        task_types: dict[str, str] | None = None, # "regression" or "classification"
        task_weights: dict[str, float] | None = None,
        strategy: str = "uncertainty",
        constraints: dict[str, str] | None = None,
    ):
        super().__init__()
        if not task_names:
            raise ValueError("task_names must not be empty")
        normalized_names = [str(name).strip() for name in task_names]
        if any(not name for name in normalized_names):
            raise ValueError("task_names must not contain empty names")
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("task_names must be unique")
        self.task_names = normalized_names
        valid_task_names = set(normalized_names)
        strategy = str(strategy).strip().lower()
        if strategy not in _SUPPORTED_STRATEGIES:
            supported = ", ".join(sorted(_SUPPORTED_STRATEGIES))
            raise ValueError(f"Unknown strategy='{strategy}'. Supported: {supported}")
        self.strategy = strategy
        normalized_constraints = _normalize_named_mapping(constraints, name="constraints")
        _validate_known_task_keys(
            normalized_constraints, valid_task_names=valid_task_names, name="constraints"
        )
        self.constraints = {k: str(v).strip() for k, v in normalized_constraints.items()}

        normalized_task_types = _normalize_named_mapping(task_types, name="task_types")
        _validate_known_task_keys(
            normalized_task_types, valid_task_names=valid_task_names, name="task_types"
        )
        raw_types = normalized_task_types or dict.fromkeys(normalized_names, "regression")
        self.task_types = {}
        for name in normalized_names:
            task_type = str(raw_types.get(name, "regression")).strip().lower()
            if task_type not in _SUPPORTED_TASK_TYPES:
                supported = ", ".join(sorted(_SUPPORTED_TASK_TYPES))
                raise ValueError(f"Unknown task type '{task_type}' for task '{name}'. Supported: {supported}")
            self.task_types[name] = task_type

        normalized_weights = _normalize_named_mapping(task_weights, name="task_weights")
        _validate_known_task_keys(
            normalized_weights, valid_task_names=valid_task_names, name="task_weights"
        )
        if strategy == "fixed":
            provided = normalized_weights
            self.weights = {}
            for name in normalized_names:
                raw_weight = provided.get(name, 1.0)
                self.weights[name] = _require_finite_scalar(raw_weight, f"task_weights[{name}]", min_value=0.0)
        else:
            # Learned log-variance per task
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in normalized_names
            })

        self.task_losses = nn.ModuleDict()
        for name in normalized_names:
            t_type = self.task_types.get(name, "regression")

            if t_type == "evidential":
                self.task_losses[name] = EvidentialLoss()
            else:
                l_type = "bce" if t_type == "classification" else "mse"
                self.task_losses[name] = PropertyLoss(
                    property_name=name,
                    constraint=self.constraints.get(name),
                    loss_type=l_type
                )

    @staticmethod
    def _infer_device(
        predictions: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor],
    ) -> torch.device:
        for collection in (predictions, targets):
            for value in collection.values():
                if isinstance(value, torch.Tensor):
                    return value.device
                if isinstance(value, dict):
                    for nested in value.values():
                        if isinstance(nested, torch.Tensor):
                            return nested.device
        return torch.device("cpu")

    def forward(
        self,
        predictions: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        losses: dict[str, torch.Tensor] = {}
        total = _zero_loss(self._infer_device(predictions, targets))

        for name in self.task_names:
            if name not in predictions or name not in targets:
                continue

            # Ensure target shape matches pred
            tgt = targets[name]
            pred = predictions[name]
            if (
                isinstance(pred, torch.Tensor)
                and isinstance(tgt, torch.Tensor)
                and tgt.shape != pred.shape
            ):
                if tgt.numel() != pred.numel():
                    raise ValueError(
                        f"Shape mismatch for task '{name}': "
                        f"prediction {tuple(pred.shape)} vs target {tuple(tgt.shape)}"
                    )
                tgt = tgt.reshape_as(pred)

            task_loss = self.task_losses[name](pred, tgt)
            if not torch.isfinite(task_loss).item():
                continue
            losses[name] = task_loss

            if self.strategy == "fixed":
                total = total + self.weights.get(name, 1.0) * task_loss
            else:
                # Classification (BCE) doesn't have the 0.5 factor typically in the uncertainty formulation
                # But regression does: 0.5 * exp(-s) * loss + 0.5 * s
                dev = task_loss.device
                if total.device != dev:
                    total = total.to(dev)

                # Ensure log_var/precision on correct device
                if self.task_types.get(name) == "evidential":
                    # Evidential loss already includes uncertainty (aleatoric)
                    # We just sum it up (maybe weighted by strategy if needed, but usually fixed)
                    # For now, treat it as a term in the total loss
                    total = total + task_loss
                else:
                    # Homoscedastic uncertainty weighting for standard regression/classification
                    log_var = torch.clamp(torch.nan_to_num(self.log_vars[name].to(dev), nan=0.0), min=-20.0, max=20.0)
                    precision = torch.exp(-log_var)
                    total = total + precision * task_loss + 0.5 * log_var

        losses["total"] = total
        return losses
