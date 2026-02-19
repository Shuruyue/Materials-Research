"""
Uncertainty Quantification for GNN Predictions

Methods:
- EnsembleUQ: train M models, use prediction variance as uncertainty
- MCDropoutUQ: single model with dropout at inference time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


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
        self.models = nn.ModuleList([model_factory() for _ in range(n_models)])
        self.n_models = n_models

    def forward(self, *args, **kwargs) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through all ensemble members.

        Returns:
            Dict[task_name → (mean, std)] for each predicted property
        """
        all_predictions = [model(*args, **kwargs) for model in self.models]

        # Aggregate predictions
        result = {}
        task_names = all_predictions[0].keys()
        for task in task_names:
            preds = torch.stack([p[task] for p in all_predictions], dim=0)
            mean = preds.mean(dim=0)
            std = preds.std(dim=0)
            result[task] = (mean, std)

        return result

    def predict_with_uncertainty(
        self, *args, **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns predictions with uncertainty estimates.

        Returns:
            Dict[task_name → {"mean": Tensor, "std": Tensor, "all": Tensor}]
        """
        all_predictions = [model(*args, **kwargs) for model in self.models]

        result = {}
        task_names = all_predictions[0].keys()
        for task in task_names:
            preds = torch.stack([p[task] for p in all_predictions], dim=0)
            result[task] = {
                "mean": preds.mean(dim=0),
                "std": preds.std(dim=0),
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
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_with_uncertainty(
        self, *args, **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Run N stochastic forward passes and compute statistics.

        Returns:
            Dict[task_name → {"mean": Tensor, "std": Tensor}]
        """
        self.model.eval()
        self._enable_dropout()

        all_predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(*args, **kwargs)
                all_predictions.append(pred)

        result = {}
        task_names = all_predictions[0].keys()
        for task in task_names:
            preds = torch.stack([p[task] for p in all_predictions], dim=0)
            result[task] = {
                "mean": preds.mean(dim=0),
                "std": preds.std(dim=0),
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

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        aleatoric = beta / (nu * (alpha - 1))
        epistemic = beta / (nu * alpha * (alpha - 1))

        return {
            "mean": gamma,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_std": (aleatoric + epistemic).sqrt(),
            "_gamma": gamma, "_nu": nu, "_alpha": alpha, "_beta": beta,
        }

    @staticmethod
    def evidential_loss(pred: Dict, target: torch.Tensor, coeff: float = 0.01) -> torch.Tensor:
        """NIG negative log-likelihood + evidence regularizer."""
        gamma, nu, alpha, beta = pred["_gamma"], pred["_nu"], pred["_alpha"], pred["_beta"]
        error = (target - gamma).abs()

        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=target.device) / nu)
            - alpha * torch.log(2 * beta)
            + (alpha + 0.5) * torch.log(nu * error**2 + 2 * beta)
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )
        reg = error * (2 * nu + alpha)
        return (nll + coeff * reg).mean()
