"""
Physics-Constrained Loss Functions (Enhanced)

Extends the base loss functions with:
- Voigt-Reuss-Hill bounds for elastic moduli
- GradNorm adaptive task weighting (Chen et al., 2018)
- PCGrad conflict-free gradient projection (Yu et al., 2020)
- Full Born stability criteria for all crystal systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import copy


# ─────────────────────────────────────────────────────────
#  Voigt-Reuss-Hill Bounds Loss
# ─────────────────────────────────────────────────────────

class VoigtReussBoundsLoss(nn.Module):
    """
    Enforces Voigt-Reuss bounds on predicted elastic moduli.

    For a polycrystalline material:
        K_Reuss ≤ K ≤ K_Voigt
        G_Reuss ≤ G ≤ G_Voigt

    When elastic tensor Cij is also predicted, the bounds can be
    computed from Cij and used to constrain scalar K, G predictions.

    Args:
        weight: penalty weight for bound violations
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    @staticmethod
    def voigt_average(C: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Voigt averages from elastic tensor (upper bound).

        Args:
            C: (B, 6, 6) elastic tensor in Voigt notation

        Returns:
            Dict with K_V and G_V
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

        Args:
            C: (B, 6, 6) elastic tensor

        Returns:
            Dict with K_R and G_R
        """
        # Compliance tensor S = C^{-1}
        S = torch.linalg.inv(C)

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

        Args:
            K_pred: (B, 1) predicted bulk modulus
            G_pred: (B, 1) predicted shear modulus
            C_pred: (B, 6, 6) predicted elastic tensor
        """
        voigt = self.voigt_average(C_pred)
        reuss = self.reuss_average(C_pred)

        K = K_pred.squeeze(-1)
        G = G_pred.squeeze(-1)

        # Penalty: K should be within [K_Reuss, K_Voigt]
        penalty_K = (
            F.relu(reuss["K_R"] - K)     # K < K_Reuss → penalty
            + F.relu(K - voigt["K_V"])    # K > K_Voigt → penalty
        ).mean()

        penalty_G = (
            F.relu(reuss["G_R"] - G)
            + F.relu(G - voigt["G_V"])
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

    Args:
        alpha: dict of constraint weights per type
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
        """
        Args:
            predictions: dict of predicted properties
                Keys: "bulk_modulus", "shear_modulus", "dielectric", "elastic_tensor"

        Returns:
            Total physics constraint loss
        """
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
            C_sym = 0.5 * (C + C.transpose(-1, -2))
            eigenvalues = torch.linalg.eigvalsh(C_sym)
            loss = loss + self.alpha["born_stability"] * F.relu(-eigenvalues).mean()

        # Voigt-Reuss bounds
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


# ─────────────────────────────────────────────────────────
#  GradNorm Adaptive Weighting
# ─────────────────────────────────────────────────────────

class GradNormWeighter(nn.Module):
    """
    GradNorm: Gradient Normalization (Chen et al., ICML 2018).

    Dynamically adjusts task weights so that all tasks train at
    similar rates, by normalizing gradient magnitudes.

    w_i(t+1) = w_i(t) * [G_i(t) / avg(G(t))]^α

    Args:
        task_names: list of task names
        alpha: gradient norm asymmetry parameter (1.5 recommended)
        lr: learning rate for weight update
    """

    def __init__(
        self,
        task_names: List[str],
        alpha: float = 1.5,
        lr: float = 0.025,
    ):
        super().__init__()
        self.task_names = task_names
        self.alpha = alpha
        self.lr = lr
        n = len(task_names)

        # Learnable log-weights (softmax applied to get actual weights)
        self.log_weights = nn.Parameter(torch.zeros(n))

        # Track initial losses for relative loss ratio
        self.register_buffer("initial_losses", torch.ones(n))
        self._initialized = False

    @property
    def weights(self) -> torch.Tensor:
        """Normalized task weights (sum = n_tasks)."""
        return F.softmax(self.log_weights, dim=0) * len(self.task_names)

    def update(
        self,
        losses: List[torch.Tensor],
        shared_params: nn.Module,
    ):
        """
        Update task weights based on gradient norms.

        Args:
            losses: per-task scalar losses
            shared_params: shared encoder parameters (last layer)
        """
        if not self._initialized:
            with torch.no_grad():
                for i, l in enumerate(losses):
                    self.initial_losses[i] = l.item()
            self._initialized = True

        # Compute gradient norm per task w.r.t. shared parameters
        # Use the last shared layer for efficiency
        shared_layer = None
        for name, param in shared_params.named_parameters():
            if param.requires_grad:
                shared_layer = param
                break

        if shared_layer is None:
            return

        grad_norms = []
        for i, loss in enumerate(losses):
            grad = torch.autograd.grad(
                loss, shared_layer, retain_graph=True, create_graph=True,
            )[0]
            gn = grad.norm()
            grad_norms.append(gn * self.weights[i])

        grad_norms = torch.stack(grad_norms)
        mean_norm = grad_norms.mean()

        # Relative loss ratios
        with torch.no_grad():
            current_losses = torch.tensor([l.item() for l in losses], device=self.log_weights.device)
            loss_ratios = current_losses / self.initial_losses.clamp(min=1e-6)
            relative_ratios = loss_ratios / loss_ratios.mean()
            target_norms = mean_norm * (relative_ratios ** self.alpha)

        # GradNorm loss: make gradient norms match targets
        gradnorm_loss = F.l1_loss(grad_norms, target_norms)

        # Update weights
        self.log_weights.grad = torch.autograd.grad(gradnorm_loss, self.log_weights)[0]
        with torch.no_grad():
            self.log_weights -= self.lr * self.log_weights.grad
            # Renormalize so weights sum to n_tasks
            self.log_weights -= self.log_weights.logsumexp(0) - torch.tensor(float(len(self.task_names))).log()


# ─────────────────────────────────────────────────────────
#  PCGrad: Projecting Conflicting Gradients
# ─────────────────────────────────────────────────────────

class PCGrad:
    """
    PCGrad: Projecting Conflicting Gradients (Yu et al., NeurIPS 2020).

    If two task gradients conflict (negative cosine similarity),
    project one onto the normal plane of the other to remove conflict.

    Usage:
        pcgrad = PCGrad(optimizer)
        task_losses = [loss1, loss2, loss3, loss4]
        pcgrad.backward(task_losses)
        pcgrad.step()
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def backward(self, losses: List[torch.Tensor]):
        """
        Compute PCGrad-modified gradients and accumulate.

        Args:
            losses: list of per-task scalar losses
        """
        # Compute per-task gradients
        task_grads = []
        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads = []
            for param in self._shared_params():
                if param.grad is not None:
                    grads.append(param.grad.clone().flatten())
                else:
                    grads.append(torch.zeros(param.numel(), device=param.device))
            task_grads.append(torch.cat(grads))

        # Project conflicting gradients
        n_tasks = len(task_grads)
        projected = [g.clone() for g in task_grads]

        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    continue
                dot = torch.dot(projected[i], task_grads[j])
                if dot < 0:
                    # Project out the conflicting component
                    projected[i] -= (dot / (task_grads[j].norm() ** 2 + 1e-8)) * task_grads[j]

        # Sum projected gradients and assign back
        combined = sum(projected)
        self.optimizer.zero_grad()
        idx = 0
        for param in self._shared_params():
            length = param.numel()
            param.grad = combined[idx:idx + length].reshape(param.shape)
            idx += length

    def step(self):
        self.optimizer.step()

    def _shared_params(self):
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    yield param
