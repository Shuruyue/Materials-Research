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
import torch.nn.functional as F
import sys
from typing import Dict, Optional, List

class PropertyLoss(nn.Module):
    """
    Single-property regression loss with optional physical constraints.
    """
    def __init__(
        self,
        property_name: str = "band_gap",
        constraint: Optional[str] = None,
        constraint_weight: float = 0.1,
        loss_type: str = "mse"
    ):
        super().__init__()
        self.property_name = property_name
        self.constraint = constraint
        self.constraint_weight = constraint_weight
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Filter NaNs in target
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
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
            base_loss = F.binary_cross_entropy_with_logits(pred, target)
        else:
            base_loss = F.mse_loss(pred, target)

        if self.constraint == "positive":
            penalty = torch.mean(F.relu(-pred))
            return base_loss + self.constraint_weight * penalty
        elif self.constraint == "greater_than_one":
            penalty = torch.mean(F.relu(1.0 - pred))
            return base_loss + self.constraint_weight * penalty
        
        return base_loss



class EvidentialLoss(nn.Module):
    """
    Loss for Evidential Deep Learning (Regression).
    Combines NLL of Normal-Inverse-Gamma distribution with evidence regularization.
    """
    def __init__(self, coeff: float = 0.05):
        super().__init__()
        self.coeff = coeff

    def forward(self, pred: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        # Filter NaNs
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=target.device, requires_grad=True)
            
        target = target[mask]
        gamma = pred["gamma"][mask]
        nu = pred["nu"][mask]
        alpha = pred["alpha"][mask]
        beta = pred["beta"][mask]

        error = (target - gamma).abs()
        
        # Negative Log Likelihood
        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=target.device) / nu)
            - alpha * torch.log(2 * beta)
            + (alpha + 0.5) * torch.log(nu * error**2 + 2 * beta)
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )
        
        # Evidence Regularization (penalize high evidence for incorrect predictions)
        reg = error * (2 * nu + alpha)
        
        return (nll + self.coeff * reg).mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with adaptive task weighting.
    Supports "fixed" (manual) or "uncertainty" (Kendall 2018) weighting.
    """
    def __init__(
        self,
        task_names: list,
        task_types: Optional[Dict[str, str]] = None, # "regression" or "classification"
        task_weights: Optional[Dict[str, float]] = None,
        strategy: str = "uncertainty",
        constraints: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.task_names = task_names
        self.strategy = strategy
        self.constraints = constraints or {}
        self.task_types = task_types or {t: "regression" for t in task_names}

        if strategy == "fixed":
            self.weights = task_weights or {t: 1.0 for t in task_names}
        elif strategy == "uncertainty":
            # Learned log-variance per task
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in task_names
            })

        self.task_losses = nn.ModuleDict()
        self.task_losses = nn.ModuleDict()
        for name in task_names:
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

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for name in self.task_names:
            if name not in predictions or name not in targets:
                continue

            # Ensure target shape matches pred
            tgt = targets[name]
            pred = predictions[name]
            if isinstance(pred, torch.Tensor) and tgt.shape != pred.shape:
                tgt = tgt.view_as(pred)

            task_loss = self.task_losses[name](pred, tgt)
            losses[name] = task_loss

            if self.strategy == "fixed":
                total = total + self.weights.get(name, 1.0) * task_loss
            elif self.strategy == "uncertainty":
                log_var = self.log_vars[name]
                precision = torch.exp(-log_var)
                # Classification (BCE) doesn't have the 0.5 factor typically in the uncertainty formulation
                # But regression does: 0.5 * exp(-s) * loss + 0.5 * s
                
                
                # Device check
                dev = pred.device
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
                    log_var = self.log_vars[name].to(dev)
                    precision = torch.exp(-log_var)
                    
                    if task_loss.device != dev:
                        task_loss = task_loss.to(dev)
                    
                    total = total + precision * task_loss + 0.5 * log_var

        losses["total"] = total
        return losses
