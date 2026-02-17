"""
Physics-Constrained Loss Functions (Enhanced)

Extends the base loss functions with:
- Voigt-Reuss-Hill bounds for elastic moduli
- Born stability criteria
- Multi-objective physics guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

# ─────────────────────────────────────────────────────────
#  Voigt-Reuss-Hill Bounds Loss
# ─────────────────────────────────────────────────────────

class VoigtReussBoundsLoss(nn.Module):
    """
    Enforces Voigt-Reuss bounds on predicted elastic moduli.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    @staticmethod
    def voigt_average(C: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Voigt averages from elastic tensor (upper bound).
        """
        # Voigt bulk modulus
        K_V = (
            C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2]
            + 2 * (C[:, 0, 1] + C[:, 0, 2] + C[:, 1, 2])
        ) / 9.0

        # Voigt shear modulus
        G_V = (
            C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2]
            - C[:, 0, 1] - C[:, 0, 2] - C[:, 1, 2]
            + 3 * (C[:, 3, 3] + C[:, 4, 4] + C[:, 5, 5])
        ) / 15.0

        return {"K_V": K_V, "G_V": G_V}

    @staticmethod
    def reuss_average(C: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Reuss averages from compliance tensor (lower bound).
        """
        # Compliance tensor S = C^{-1}
        # Add small epsilon to diagonal for stability
        eps = 1e-6
        C_stable = C + torch.eye(6, device=C.device).unsqueeze(0) * eps
        try:
            S = torch.linalg.inv(C_stable)
        except RuntimeError:
            # Fallback if singular
            return {"K_R": torch.zeros_like(C[:,0,0]), "G_R": torch.zeros_like(C[:,0,0])}

        K_R = 1.0 / (
            S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2]
            + 2 * (S[:, 0, 1] + S[:, 0, 2] + S[:, 1, 2])
        )

        G_R = 15.0 / (
            4 * (S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2])
            - 4 * (S[:, 0, 1] + S[:, 0, 2] + S[:, 1, 2])
            + 3 * (S[:, 3, 3] + S[:, 4, 4] + S[:, 5, 5])
        )

        return {"K_R": K_R, "G_R": G_R}

    def forward(
        self,
        K_pred: torch.Tensor,
        G_pred: torch.Tensor,
        C_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize scalar moduli that violate Voigt-Reuss bounds.
        """
        voigt = self.voigt_average(C_pred)
        reuss = self.reuss_average(C_pred)

        K = K_pred.view(-1)
        G = G_pred.view(-1)
        
        K_R, K_V = reuss["K_R"], voigt["K_V"]
        G_R, G_V = reuss["G_R"], voigt["G_V"]

        # Penalty: K must be in [K_R, K_V]
        # Allow small tolerance
        penalty_K = (
            F.relu(K_R - K - 1e-3)     # K < K_Reuss
            + F.relu(K - K_V - 1e-3)    # K > K_Voigt
        ).mean()

        penalty_G = (
            F.relu(G_R - G - 1e-3)
            + F.relu(G - G_V - 1e-3)
        ).mean()

        return self.weight * (penalty_K + penalty_G)


# ─────────────────────────────────────────────────────────
#  Full Physics Constraint Loss
# ─────────────────────────────────────────────────────────

class PhysicsConstraintLoss(nn.Module):
    """
    Combined physics constraints for all predicted properties.

    L_physics = α₁·ReLU(-K̂) + α₂·ReLU(-Ĝ) + α₃·ReLU(1-ε̂)
              + α₄·BornStability(Ĉij) + α₅·VoigtReuss(K̂,Ĝ,Ĉij)
    """

    DEFAULTS = {
        "positivity": 0.1,
        "dielectric_lower": 0.1,
        "born_stability": 0.1,
        "voigt_reuss": 0.05,
    }

    def __init__(self, alpha: Optional[Dict[str, float]] = None):
        super().__init__()
        self.alpha = {**self.DEFAULTS, **(alpha or {})}
        self.voigt_reuss = VoigtReussBoundsLoss(weight=1.0)

    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self._get_device(predictions))

        # Positivity: K ≥ 0, G ≥ 0
        if "bulk_modulus" in predictions:
            loss = loss + self.alpha["positivity"] * F.relu(-predictions["bulk_modulus"]).mean()
        if "shear_modulus" in predictions:
            loss = loss + self.alpha["positivity"] * F.relu(-predictions["shear_modulus"]).mean()

        # Dielectric: ε ≥ 1
        if "dielectric" in predictions:
            loss = loss + self.alpha["dielectric_lower"] * F.relu(1.0 - predictions["dielectric"]).mean()

        # Born stability: eigenvalues of Cij > 0
        if "elastic_tensor" in predictions:
            C = predictions["elastic_tensor"]
            # Symmetrize
            C_sym = 0.5 * (C + C.transpose(-1, -2))
            try:
                eigenvalues = torch.linalg.eigvalsh(C_sym)
                # Penalize only negative eigenvalues slightly below zero to allow numerical noise
                loss = loss + self.alpha["born_stability"] * F.relu(-eigenvalues + 1e-5).mean()
            except RuntimeError:
                pass # Eigendecomposition failed

        # Voigt-Reuss bounds (Consistency check)
        if all(k in predictions for k in ["bulk_modulus", "shear_modulus", "elastic_tensor"]):
            loss = loss + self.alpha["voigt_reuss"] * self.voigt_reuss(
                predictions["bulk_modulus"],
                predictions["shear_modulus"],
                predictions["elastic_tensor"],
            )

        return loss

    @staticmethod
    def _get_device(d):
        for v in d.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return "cpu"
