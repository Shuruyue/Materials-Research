"""
Multi-Task GNN

Shared encoder with multiple task-specific prediction heads.
Supports both scalar (Eg, Ef, K, G) and tensor (Cij, εij) outputs.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Evidential Deep Learning Head (Amini et al., 2020).
    Predicts Normal-Inverse-Gamma parameters: (γ, ν, α, β).
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim * 4),
        )

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dictionary of NIG parameters."""
        raw = self.net(embedding).reshape(-1, self.output_dim, 4)

        # Enforce constraints (Softplus + epsilon)
        gamma = raw[..., 0]
        nu = F.softplus(raw[..., 1]) + 1e-6
        alpha = F.softplus(raw[..., 2]) + 1.0 + 1e-6
        beta = F.softplus(raw[..., 3]) + 1e-6

        return {
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta
        }


class ScalarHead(nn.Module):
    """Prediction head for a single scalar property."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class TensorHead(nn.Module):
    """
    Prediction head for tensor properties.

    For invariant encoder: predicts independent components of the tensor
    under the constraint of crystal symmetry.

    For equivariant encoder (Phase 2): uses irreps decomposition.
    """

    def __init__(
        self,
        embed_dim: int,
        tensor_type: str = "elastic",  # "elastic", "dielectric", "piezoelectric"
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.tensor_type = tensor_type

        # Number of independent components
        n_components = {
            "elastic": 21,       # Cij: symmetric 6×6 → 21 independent
            "dielectric": 6,     # εij: symmetric 3×3 → 6 independent
            "piezoelectric": 18,  # eijk: 3×6 → 18 independent
        }
        self.n_out = n_components.get(tensor_type, 21)

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_out),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_components) independent tensor components."""
        return self.net(embedding)

    def to_full_tensor(self, components: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct full tensor from independent components.

        Args:
            components: (B, n_components)

        Returns:
            Full tensor:
                elastic: (B, 6, 6)
                dielectric: (B, 3, 3)
                piezoelectric: (B, 3, 6)
        """
        if self.tensor_type == "elastic":
            return self._to_elastic_tensor(components)
        elif self.tensor_type == "dielectric" or self.tensor_type == "dielectric":
            return self._to_dielectric_tensor(components)
        else:
            return self._to_piezoelectric_tensor(components)

    @staticmethod
    def _to_piezoelectric_tensor(components: torch.Tensor) -> torch.Tensor:
        """Reconstruct 3x6 piezoelectric tensor from 18 components."""
        # Just reshape (B, 18) -> (B, 3, 6)
        B = components.size(0)
        return components.view(B, 3, 6)

    @staticmethod
    def _to_elastic_tensor(components: torch.Tensor) -> torch.Tensor:
        """Reconstruct symmetric 6×6 elastic tensor from 21 components."""
        B = components.size(0)
        C = torch.zeros(B, 6, 6, device=components.device)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                C[:, i, j] = components[:, idx]
                C[:, j, i] = components[:, idx]  # symmetry
                idx += 1
        return C

    @staticmethod
    def _to_dielectric_tensor(components: torch.Tensor) -> torch.Tensor:
        """Reconstruct symmetric 3×3 dielectric tensor from 6 components."""
        B = components.size(0)
        eps = torch.zeros(B, 3, 3, device=components.device)
        idx = 0
        for i in range(3):
            for j in range(i, 3):
                eps[:, i, j] = components[:, idx]
                eps[:, j, i] = components[:, idx]
                idx += 1
        return eps


class MultiTaskGNN(nn.Module):
    """
    Multi-Task Graph Neural Network.

    Shared encoder produces a graph-level embedding, which is fed to
    multiple task-specific prediction heads.

    Args:
        encoder: shared GNN encoder (CGCNN or EquivariantGNN)
        tasks: dict mapping task_name → {"type": "scalar"|"tensor", ...config}
    """

    def __init__(
        self,
        encoder: nn.Module,
        tasks: dict[str, dict] | None = None,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

        # Default tasks for PhD thesis
        if tasks is None:
            tasks = {
                "band_gap": {"type": "scalar"},
                "formation_energy": {"type": "scalar"},
                "bulk_modulus": {"type": "scalar"},
                "shear_modulus": {"type": "scalar"},
            }

        self.heads = nn.ModuleDict()
        for name, cfg in tasks.items():
            if cfg["type"] == "scalar":
                self.heads[name] = ScalarHead(embed_dim)
            elif cfg["type"] == "evidential":
                self.heads[name] = EvidentialHead(embed_dim)
            elif cfg["type"] == "tensor":
                tensor_type = cfg.get("tensor_type", "elastic")
                self.heads[name] = TensorHead(embed_dim, tensor_type=tensor_type)

        self.task_names = list(tasks.keys())

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor | None = None,
        tasks: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: predict multiple properties.

        Args:
            node_feats, edge_index, edge_feats, batch: graph inputs
            tasks: which tasks to predict (None = all)

        Returns:
            Dict[task_name → (B, output_dim) predictions]
        """
        # Shared encoding
        embedding = self.encoder.encode(node_feats, edge_index, edge_feats, batch)

        # Task-specific predictions
        predictions = {}
        for name in (tasks or self.task_names):
            if name in self.heads:
                predictions[name] = self.heads[name](embedding)

        return predictions

    def add_task(self, name: str, task_type: str = "scalar", **kwargs):
        """Dynamically add a new prediction task."""
        if task_type == "scalar":
            self.heads[name] = ScalarHead(self.embed_dim, **kwargs)
        elif task_type == "evidential":
            self.heads[name] = EvidentialHead(self.embed_dim, **kwargs)
        elif task_type == "tensor":
            self.heads[name] = TensorHead(self.embed_dim, **kwargs)
        self.task_names.append(name)
