"""
Physics-constrained loss functions.

Extends base loss functions with:
- Voigt-Reuss-Hill bounds for elastic moduli
- Born stability criteria
- Multi-objective physics guidance
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_ELASTIC_DIM = 6
_EPS = 1e-6
_BOUND_TOL = 1e-3


def _is_boolean_like(value) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool_", "bool"}


def _as_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _as_elastic_batch(value) -> torch.Tensor:
    tensor = _as_float_tensor(value)
    if tensor.ndim == 2 and tensor.size(0) == _ELASTIC_DIM and tensor.size(1) == _ELASTIC_DIM:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim < 3 or tensor.size(-1) != _ELASTIC_DIM or tensor.size(-2) != _ELASTIC_DIM:
        return tensor.new_zeros((0, _ELASTIC_DIM, _ELASTIC_DIM))
    tensor = tensor.reshape(-1, _ELASTIC_DIM, _ELASTIC_DIM)
    finite_mask = torch.isfinite(tensor.reshape(tensor.size(0), -1)).all(dim=1)
    if not finite_mask.any():
        return tensor.new_zeros((0, _ELASTIC_DIM, _ELASTIC_DIM))
    return tensor[finite_mask]


def _flatten_finite(tensor: torch.Tensor) -> torch.Tensor:
    values = _as_float_tensor(tensor).reshape(-1)
    return values[torch.isfinite(values)]


def _zero_scalar(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


def _zero_loss_from_inputs(*values) -> torch.Tensor:
    """
    Build a zero scalar that stays on the same device/dtype and keeps grad path when possible.
    """
    for value in values:
        if isinstance(value, torch.Tensor):
            sanitized = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            return sanitized.reshape(-1).sum() * 0.0
    return _zero_scalar()


def _coerce_non_negative_float(value: float, *, name: str) -> float:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be finite and >= 0")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return scalar


def _aligned_finite_vectors(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Align multiple vectors by index and keep only rows where all entries are finite.
    """
    if not values:
        return tuple()
    vectors = [_as_float_tensor(value).reshape(-1) for value in values]
    n = min(vector.numel() for vector in vectors)
    if n == 0:
        return tuple(vector.new_zeros((0,)) for vector in vectors)
    trimmed = [vector[:n] for vector in vectors]
    stacked = torch.stack(trimmed, dim=0)
    finite_mask = torch.isfinite(stacked).all(dim=0)
    if not finite_mask.any():
        return tuple(vector.new_zeros((0,)) for vector in trimmed)
    return tuple(vector[finite_mask] for vector in trimmed)


class VoigtReussBoundsLoss(nn.Module):
    """Enforce Voigt-Reuss bounds on predicted elastic moduli."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = _coerce_non_negative_float(weight, name="weight")

    @staticmethod
    def voigt_average(C: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute Voigt averages from elastic tensor (upper bound)."""
        batch = _as_elastic_batch(C)
        if batch.numel() == 0:
            empty = _as_float_tensor(C).new_zeros((0,))
            return {"K_V": empty, "G_V": empty}

        k_voigt = (
            batch[:, 0, 0]
            + batch[:, 1, 1]
            + batch[:, 2, 2]
            + 2.0 * (batch[:, 0, 1] + batch[:, 0, 2] + batch[:, 1, 2])
        ) / 9.0
        g_voigt = (
            batch[:, 0, 0]
            + batch[:, 1, 1]
            + batch[:, 2, 2]
            - batch[:, 0, 1]
            - batch[:, 0, 2]
            - batch[:, 1, 2]
            + 3.0 * (batch[:, 3, 3] + batch[:, 4, 4] + batch[:, 5, 5])
        ) / 15.0
        return {"K_V": torch.nan_to_num(k_voigt, nan=0.0), "G_V": torch.nan_to_num(g_voigt, nan=0.0)}

    @staticmethod
    def reuss_average(C: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute Reuss averages from compliance tensor (lower bound)."""
        batch = _as_elastic_batch(C)
        if batch.numel() == 0:
            empty = _as_float_tensor(C).new_zeros((0,))
            return {"K_R": empty, "G_R": empty}

        eye = torch.eye(_ELASTIC_DIM, device=batch.device, dtype=batch.dtype).unsqueeze(0)
        compliance = torch.linalg.pinv(batch + eye * _EPS)

        k_den = (
            compliance[:, 0, 0]
            + compliance[:, 1, 1]
            + compliance[:, 2, 2]
            + 2.0 * (compliance[:, 0, 1] + compliance[:, 0, 2] + compliance[:, 1, 2])
        )
        g_den = (
            4.0 * (compliance[:, 0, 0] + compliance[:, 1, 1] + compliance[:, 2, 2])
            - 4.0 * (compliance[:, 0, 1] + compliance[:, 0, 2] + compliance[:, 1, 2])
            + 3.0 * (compliance[:, 3, 3] + compliance[:, 4, 4] + compliance[:, 5, 5])
        )

        k_reuss = torch.where(k_den.abs() > _EPS, 1.0 / k_den, torch.zeros_like(k_den))
        g_reuss = torch.where(g_den.abs() > _EPS, 15.0 / g_den, torch.zeros_like(g_den))
        return {
            "K_R": torch.nan_to_num(k_reuss, nan=0.0, posinf=0.0, neginf=0.0),
            "G_R": torch.nan_to_num(g_reuss, nan=0.0, posinf=0.0, neginf=0.0),
        }

    def forward(
        self,
        K_pred: torch.Tensor,
        G_pred: torch.Tensor,
        C_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize scalar moduli that violate Voigt-Reuss bounds."""
        voigt = self.voigt_average(C_pred)
        reuss = self.reuss_average(C_pred)

        (
            k_values,
            g_values,
            k_reuss,
            k_voigt,
            g_reuss,
            g_voigt,
        ) = _aligned_finite_vectors(
            K_pred,
            G_pred,
            reuss["K_R"],
            voigt["K_V"],
            reuss["G_R"],
            voigt["G_V"],
        )
        if k_values.numel() == 0:
            return _zero_loss_from_inputs(K_pred, G_pred, C_pred)

        penalty_k = (F.relu(k_reuss - k_values - _BOUND_TOL) + F.relu(k_values - k_voigt - _BOUND_TOL)).mean()
        penalty_g = (F.relu(g_reuss - g_values - _BOUND_TOL) + F.relu(g_values - g_voigt - _BOUND_TOL)).mean()
        return self.weight * (penalty_k + penalty_g)


class PhysicsConstraintLoss(nn.Module):
    """Combined physics constraints for all predicted properties."""

    DEFAULTS = {
        "positivity": 0.1,
        "dielectric_lower": 0.1,
        "born_stability": 0.1,
        "voigt_reuss": 0.05,
    }

    def __init__(self, alpha: dict[str, float] | None = None):
        super().__init__()
        base = dict(self.DEFAULTS)
        if alpha:
            for key, value in alpha.items():
                if key not in base:
                    raise KeyError(f"unsupported alpha key: {key}")
                base[key] = _coerce_non_negative_float(value, name=f"alpha[{key}]")
        self.alpha = base
        self.voigt_reuss = VoigtReussBoundsLoss(weight=1.0)

    def forward(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(predictions, dict):
            raise TypeError("predictions must be a dict")

        device = self._get_device(predictions)
        loss = _zero_scalar(device=device)

        bulk = predictions.get("bulk_modulus")
        if isinstance(bulk, torch.Tensor):
            values = _flatten_finite(bulk)
            if values.numel() > 0:
                loss = loss + self.alpha["positivity"] * F.relu(-values).mean()

        shear = predictions.get("shear_modulus")
        if isinstance(shear, torch.Tensor):
            values = _flatten_finite(shear)
            if values.numel() > 0:
                loss = loss + self.alpha["positivity"] * F.relu(-values).mean()

        dielectric = predictions.get("dielectric")
        if isinstance(dielectric, torch.Tensor):
            values = _flatten_finite(dielectric)
            if values.numel() > 0:
                loss = loss + self.alpha["dielectric_lower"] * F.relu(1.0 - values).mean()

        elastic_tensor = predictions.get("elastic_tensor")
        if isinstance(elastic_tensor, torch.Tensor):
            c_batch = _as_elastic_batch(elastic_tensor)
            if c_batch.numel() > 0:
                c_sym = 0.5 * (c_batch + c_batch.transpose(-1, -2))
                try:
                    eigenvalues = torch.linalg.eigvalsh(c_sym)
                except RuntimeError:
                    eigenvalues = None
                if eigenvalues is not None:
                    eig_values = _flatten_finite(eigenvalues)
                    if eig_values.numel() > 0:
                        loss = loss + self.alpha["born_stability"] * F.relu(-eig_values + 1e-5).mean()

        if isinstance(bulk, torch.Tensor) and isinstance(shear, torch.Tensor) and isinstance(elastic_tensor, torch.Tensor):
            loss = loss + self.alpha["voigt_reuss"] * self.voigt_reuss(bulk, shear, elastic_tensor)

        if loss.requires_grad:
            return loss
        # Keep gradient path alive when predictions are present but constraints are inactive.
        for value in predictions.values():
            if isinstance(value, torch.Tensor):
                sanitized = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                return loss + (sanitized.reshape(-1).sum() * 0.0)
        return loss

    @staticmethod
    def _get_device(values: dict[str, torch.Tensor]) -> torch.device:
        for tensor in values.values():
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        return torch.device("cpu")
