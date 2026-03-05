"""
Angle basis helpers used by M3GNet-style three-body interaction code.

The original implementation references MatGL-specific expansions; this module
provides compact, dependency-light replacements with matching call signatures.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _coerce_positive_int(value: object, name: str) -> int:
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


class SimpleMLPAngleExpansion(nn.Module):
    """
    Learnable MLP-based angle feature expansion.

    Input: `(r_ij, r_ik, cos(theta))`
    Output: `(n_triplets, n_basis)`
    """

    def __init__(self, n_basis: int = 20):
        super().__init__()
        basis = _coerce_positive_int(n_basis, "n_basis")
        self.net = nn.Sequential(
            nn.Linear(3, max(16, basis)),
            nn.SiLU(),
            nn.Linear(max(16, basis), basis),
        )

    def forward(
        self,
        r_ij: torch.Tensor,
        r_ik: torch.Tensor,
        cos_theta: torch.Tensor,
    ) -> torch.Tensor:
        if r_ij.ndim != 2 or r_ik.ndim != 2 or cos_theta.ndim != 2:
            raise ValueError("r_ij, r_ik, and cos_theta must be rank-2 tensors")
        if r_ij.shape != r_ik.shape or r_ij.shape != cos_theta.shape:
            raise ValueError("r_ij, r_ik, and cos_theta must have identical shapes")
        if r_ij.size(1) != 1:
            raise ValueError("r_ij, r_ik, and cos_theta must have shape (T, 1)")
        x = torch.cat([r_ij, r_ik, cos_theta], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(self.net(x), nan=0.0, posinf=0.0, neginf=0.0)


class SphericalBesselHarmonicsExpansion(nn.Module):
    """
    Compact deterministic basis expansion for three-body terms.

    This is a practical approximation that combines:
    - radial sine basis for `r_ij`, `r_ik`
    - cosine angular basis for `theta`

    Output dimension is `max_n * max_l`.
    """

    def __init__(self, max_n: int = 4, max_l: int = 4):
        super().__init__()
        self.max_n = _coerce_positive_int(max_n, "max_n")
        self.max_l = _coerce_positive_int(max_l, "max_l")
        n_idx = torch.arange(1, self.max_n + 1, dtype=torch.float32).view(1, -1)
        l_idx = torch.arange(1, self.max_l + 1, dtype=torch.float32).view(1, -1)
        self.register_buffer("n_idx", n_idx)
        self.register_buffer("l_idx", l_idx)

    def forward(
        self,
        r_ij: torch.Tensor,
        r_ik: torch.Tensor,
        cos_theta: torch.Tensor,
    ) -> torch.Tensor:
        if r_ij.ndim != 2 or r_ik.ndim != 2 or cos_theta.ndim != 2:
            raise ValueError("r_ij, r_ik, and cos_theta must be rank-2 tensors")
        if r_ij.shape != r_ik.shape or r_ij.shape != cos_theta.shape:
            raise ValueError("r_ij, r_ik, and cos_theta must have identical shapes")
        if r_ij.size(1) != 1:
            raise ValueError("r_ij, r_ik, and cos_theta must have shape (T, 1)")
        # Shape normalize to (T, 1)
        r_ij = torch.nan_to_num(r_ij, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1)
        r_ik = torch.nan_to_num(r_ik, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1)
        cos_theta = torch.nan_to_num(cos_theta, nan=1.0, posinf=1.0, neginf=-1.0).reshape(-1, 1).clamp(-1.0, 1.0)

        # Radial basis (normalized distance proxy; stable and bounded)
        radial = torch.sin(self.n_idx * (r_ij + r_ik) / 2.0)

        # Angular basis
        theta = torch.acos(cos_theta)
        angular = torch.cos(self.l_idx * theta)

        # Outer product -> flatten
        basis = radial.unsqueeze(-1) * angular.unsqueeze(-2)
        return torch.nan_to_num(
            basis.reshape(r_ij.size(0), self.max_n * self.max_l),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

