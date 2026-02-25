"""
Angle basis helpers used by M3GNet-style three-body interaction code.

The original implementation references MatGL-specific expansions; this module
provides compact, dependency-light replacements with matching call signatures.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleMLPAngleExpansion(nn.Module):
    """
    Learnable MLP-based angle feature expansion.

    Input: `(r_ij, r_ik, cos(theta))`
    Output: `(n_triplets, n_basis)`
    """

    def __init__(self, n_basis: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, max(16, n_basis)),
            nn.SiLU(),
            nn.Linear(max(16, n_basis), n_basis),
        )

    def forward(
        self,
        r_ij: torch.Tensor,
        r_ik: torch.Tensor,
        cos_theta: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([r_ij, r_ik, cos_theta], dim=-1)
        return self.net(x)


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
        self.max_n = int(max_n)
        self.max_l = int(max_l)
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
        # Shape normalize to (T, 1)
        r_ij = r_ij.view(-1, 1)
        r_ik = r_ik.view(-1, 1)
        cos_theta = cos_theta.view(-1, 1).clamp(-1.0, 1.0)

        # Radial basis (normalized distance proxy; stable and bounded)
        radial = torch.sin(self.n_idx * (r_ij + r_ik) / 2.0)

        # Angular basis
        theta = torch.acos(cos_theta)
        angular = torch.cos(self.l_idx * theta)

        # Outer product -> flatten
        basis = radial.unsqueeze(-1) * angular.unsqueeze(-2)
        return basis.reshape(r_ij.size(0), self.max_n * self.max_l)

