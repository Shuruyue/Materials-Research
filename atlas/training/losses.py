"""
Physics-Constrained Loss Functions

Loss functions that embed physical constraints:
- Positivity: K >= 0, G >= 0
- Bounds: ε >= 1, Voigt/Reuss bounds
- Stability: Born stability criteria for elastic tensors
- Multi-task weighting: uncertainty-based or GradNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PropertyLoss(nn.Module):
    """
    Single-property regression loss with optional physical constraints.

    Args:
        property_name: name of the property (for constraint lookup)
        constraint: type of physical constraint
            - None: plain MSE
            - "positive": property must be >= 0
            - "greater_than_one": property must be >= 1
        constraint_weight: penalty weight for constraint violation
    """

    def __init__(
        self,
        property_name: str = "band_gap",
        constraint: Optional[str] = None,
        constraint_weight: float = 0.1,
    ):
        super().__init__()
        self.property_name = property_name
        self.constraint = constraint
        self.constraint_weight = constraint_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1) predicted values
            target: (B, 1) target values

        Returns:
            loss scalar
        """
        mse = F.mse_loss(pred, target)

        if self.constraint == "positive":
            penalty = torch.mean(F.relu(-pred))
            return mse + self.constraint_weight * penalty

        elif self.constraint == "greater_than_one":
            penalty = torch.mean(F.relu(1.0 - pred))
            return mse + self.constraint_weight * penalty

        return mse


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with adaptive task weighting.

    Supports multiple strategies:
    - "fixed": manually set weights
    - "uncertainty": learned uncertainty (Kendall et al. 2018)
        L = Σ (1/(2σᵢ²)) · Lᵢ + log(σᵢ)

    Args:
        task_names: list of task names
        task_weights: dict of fixed weights (for "fixed" strategy)
        strategy: "fixed" or "uncertainty"
        constraints: dict of {task_name: constraint_type}
    """

    def __init__(
        self,
        task_names: list,
        task_weights: Optional[Dict[str, float]] = None,
        strategy: str = "uncertainty",
        constraints: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.task_names = task_names
        self.strategy = strategy
        self.constraints = constraints or {}

        if strategy == "fixed":
            self.weights = task_weights or {t: 1.0 for t in task_names}
        elif strategy == "uncertainty":
            # Learned log-variance per task (Kendall 2018)
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in task_names
            })

        # Per-task constraint losses
        self.task_losses = nn.ModuleDict({
            name: PropertyLoss(
                property_name=name,
                constraint=self.constraints.get(name),
            )
            for name in task_names
        })

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: {task_name: (B, d) predicted}
            targets: {task_name: (B, d) target}

        Returns:
            Dict with "total" loss and per-task losses
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for name in self.task_names:
            if name not in predictions or name not in targets:
                continue

            task_loss = self.task_losses[name](predictions[name], targets[name])
            losses[name] = task_loss

            if self.strategy == "fixed":
                total = total + self.weights[name] * task_loss
            elif self.strategy == "uncertainty":
                log_var = self.log_vars[name]
                precision = torch.exp(-log_var)
                total = total + precision * task_loss + log_var

        losses["total"] = total
        return losses

    def get_task_weights(self) -> Dict[str, float]:
        """Get current effective task weights."""
        if self.strategy == "fixed":
            return self.weights
        elif self.strategy == "uncertainty":
            return {
                name: torch.exp(-self.log_vars[name]).item()
                for name in self.task_names
            }


class BornStabilityLoss(nn.Module):
    """
    Born stability constraint for elastic tensors.

    A crystal is mechanically stable if all eigenvalues of Cij are positive.
    This loss penalizes negative eigenvalues.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, C_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C_pred: (B, 6, 6) predicted elastic tensor

        Returns:
            Penalty for negative eigenvalues
        """
        # Symmetrize
        C_sym = 0.5 * (C_pred + C_pred.transpose(-1, -2))

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(C_sym)

        # Penalize negative eigenvalues
        penalty = F.relu(-eigenvalues).mean()

        return self.weight * penalty
