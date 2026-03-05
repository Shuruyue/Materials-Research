"""
Multi-Task GNN

Shared encoder with multiple task-specific prediction heads.
Supports both scalar (Eg, Ef, K, G) and tensor (Cij, εij) outputs.
"""


import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

_TASK_TYPES = {"scalar", "evidential", "tensor"}
_TENSOR_TYPES = {"elastic", "dielectric", "piezoelectric"}


def _normalize_task_names(tasks: list[str] | tuple[str, ...] | str | None) -> list[str] | None:
    if tasks is None:
        return None
    if isinstance(tasks, str):
        candidates = [tasks]
    else:
        try:
            candidates = list(tasks)
        except TypeError as exc:
            raise ValueError("tasks must be a task name string or an iterable of task names") from exc

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in candidates:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("tasks contains an empty task name")
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


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
        self.tensor_type = str(tensor_type).strip().lower()
        if self.tensor_type not in _TENSOR_TYPES:
            raise ValueError(
                f"Unsupported tensor_type: {tensor_type}. Choices: {', '.join(sorted(_TENSOR_TYPES))}"
            )

        # Number of independent components
        n_components = {
            "elastic": 21,       # Cij: symmetric 6×6 → 21 independent
            "dielectric": 6,     # εij: symmetric 3×3 → 6 independent
            "piezoelectric": 18,  # eijk: 3×6 → 18 independent
        }
        self.n_out = n_components[self.tensor_type]

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
        if components.ndim != 2:
            raise ValueError(
                f"components must be rank-2 [B, n_components], got shape {tuple(components.shape)}"
            )
        if components.shape[0] <= 0:
            raise ValueError("components must contain at least one sample")
        if components.shape[1] != self.n_out:
            raise ValueError(
                f"components second dimension must match expected n_components={self.n_out}, "
                f"got {components.shape[1]}"
            )
        if not torch.isfinite(components).all():
            raise ValueError("components contains NaN or Inf values")

        if self.tensor_type == "elastic":
            return self._to_elastic_tensor(components)
        elif self.tensor_type == "dielectric":
            return self._to_dielectric_tensor(components)
        else:
            return self._to_piezoelectric_tensor(components)

    @staticmethod
    def _to_piezoelectric_tensor(components: torch.Tensor) -> torch.Tensor:
        """Reconstruct 3x6 piezoelectric tensor from 18 components."""
        # Just reshape (B, 18) -> (B, 3, 6)
        B = components.size(0)
        return components.reshape(B, 3, 6)

    @staticmethod
    def _to_elastic_tensor(components: torch.Tensor) -> torch.Tensor:
        """Reconstruct symmetric 6×6 elastic tensor from 21 components."""
        B = components.size(0)
        C = torch.zeros(B, 6, 6, device=components.device, dtype=components.dtype)
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
        eps = torch.zeros(B, 3, 3, device=components.device, dtype=components.dtype)
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
        try:
            encode_fn = encoder.encode
        except AttributeError as exc:
            raise ValueError("encoder must expose a callable encode(...) method") from exc
        if not callable(encode_fn):
            raise ValueError("encoder must expose a callable encode(...) method")
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
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError("tasks must be a non-empty dict[str, dict].")

        self.heads = nn.ModuleDict()
        for name, cfg in tasks.items():
            task_name = str(name)
            if not isinstance(cfg, dict):
                raise ValueError(f"Task config for '{task_name}' must be a dict.")
            task_type = str(cfg.get("type", "scalar")).strip().lower()
            if task_type not in _TASK_TYPES:
                raise ValueError(
                    f"Unsupported task type for '{task_name}': {task_type}. "
                    f"Choices: {', '.join(sorted(_TASK_TYPES))}"
                )
            if task_type == "scalar":
                self.heads[task_name] = ScalarHead(embed_dim)
            elif task_type == "evidential":
                self.heads[task_name] = EvidentialHead(embed_dim)
            elif task_type == "tensor":
                tensor_type = cfg.get("tensor_type", "elastic")
                self.heads[task_name] = TensorHead(embed_dim, tensor_type=tensor_type)

        self.task_names = list(dict.fromkeys(str(k) for k in tasks))

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch: torch.Tensor | None = None,
        tasks: list[str] | None = None,
        edge_vectors: torch.Tensor | None = None,
        edge_index_3body: torch.Tensor | None = None,
        encoder_kwargs: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: predict multiple properties.

        Args:
            node_feats, edge_index, edge_feats, batch: graph inputs
            tasks: which tasks to predict (None = all)

        Returns:
            Dict[task_name → (B, output_dim) predictions]
        """
        # Shared encoding with optional encoder-specific kwargs
        extra_kwargs = {}
        if edge_vectors is not None:
            extra_kwargs["edge_vectors"] = edge_vectors
        if edge_index_3body is not None:
            extra_kwargs["edge_index_3body"] = edge_index_3body
        if isinstance(encoder_kwargs, dict) and encoder_kwargs:
            extra_kwargs.update(encoder_kwargs)

        if extra_kwargs:
            try:
                signature = inspect.signature(self.encoder.encode)
                supports_var_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in signature.parameters.values()
                )
                if not supports_var_kwargs:
                    allowed = set(signature.parameters.keys())
                    extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in allowed}
            except (TypeError, ValueError):
                extra_kwargs = {}

        if extra_kwargs:
            embedding = self.encoder.encode(
                node_feats,
                edge_index,
                edge_feats,
                batch,
                **extra_kwargs,
            )
        else:
            embedding = self.encoder.encode(node_feats, edge_index, edge_feats, batch)

        # Task-specific predictions
        predictions = {}
        selected_tasks = _normalize_task_names(tasks) or self.task_names
        unknown_tasks = [name for name in selected_tasks if name not in self.heads]
        if unknown_tasks:
            raise ValueError(
                "Unknown task(s) requested: "
                f"{unknown_tasks}. Available tasks: {sorted(self.heads.keys())}"
            )
        for name in selected_tasks:
            predictions[name] = self.heads[name](embedding)

        return predictions

    def add_task(self, name: str, task_type: str = "scalar", **kwargs):
        """Dynamically add a new prediction task."""
        task_name = str(name)
        task_kind = str(task_type).strip().lower()
        if task_kind not in _TASK_TYPES:
            raise ValueError(
                f"Unsupported task type: {task_type}. Choices: {', '.join(sorted(_TASK_TYPES))}"
            )
        if task_name in self.heads:
            raise ValueError(f"Task '{task_name}' already exists.")
        if task_kind == "scalar":
            self.heads[task_name] = ScalarHead(self.embed_dim, **kwargs)
        elif task_kind == "evidential":
            self.heads[task_name] = EvidentialHead(self.embed_dim, **kwargs)
        elif task_kind == "tensor":
            self.heads[task_name] = TensorHead(self.embed_dim, **kwargs)
        self.task_names.append(task_name)
