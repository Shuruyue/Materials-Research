"""
Utilities for normalizing model prediction payloads.

Supports common ATLAS output formats:
- plain tensor predictions
- ``{"mean": ..., "std": ...}``
- evidential payloads ``{"gamma", "nu", "alpha", "beta"}``
- tuple/list ``(mean, std)``
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _to_tensor(value: Any, *, reference: torch.Tensor | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        if reference is not None and (
            value.dtype != reference.dtype or value.device != reference.device
        ):
            return value.to(dtype=reference.dtype, device=reference.device)
        return value
    if reference is not None:
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(value)


def _broadcast_to_reference(
    value: Any,
    *,
    reference: torch.Tensor,
    field_name: str,
) -> torch.Tensor:
    tensor = _to_tensor(value, reference=reference)
    if tensor.shape == reference.shape:
        return tensor
    try:
        ref_b, tensor_b = torch.broadcast_tensors(reference, tensor)
    except RuntimeError as exc:
        raise ValueError(
            f"{field_name} shape {tuple(tensor.shape)} is not broadcastable to mean shape {tuple(reference.shape)}"
        ) from exc
    if ref_b.shape != reference.shape:
        raise ValueError(
            f"{field_name} shape {tuple(tensor.shape)} would expand mean shape {tuple(reference.shape)} to {tuple(ref_b.shape)}"
        )
    return tensor_b


def _sanitize_std_like(value: Any, *, reference: torch.Tensor) -> torch.Tensor:
    std = _broadcast_to_reference(value, reference=reference, field_name="std")
    if not torch.is_floating_point(std):
        std = std.to(dtype=reference.dtype if torch.is_floating_point(reference) else torch.float32)
    std = torch.nan_to_num(std, nan=0.0, posinf=1e6, neginf=0.0)
    return std.clamp_min(0.0)


def _sanitize_mean_like(value: Any, *, reference: torch.Tensor | None = None) -> torch.Tensor:
    mean = _to_tensor(value, reference=reference)
    if not torch.is_floating_point(mean):
        mean = mean.to(dtype=reference.dtype if (reference is not None and torch.is_floating_point(reference)) else torch.float32)
    return torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)


def _from_evidential_payload(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    # MultiTask ``EvidentialHead`` format
    gamma = _sanitize_mean_like(payload["gamma"])
    nu = _broadcast_to_reference(payload["nu"], reference=gamma, field_name="nu")
    alpha = _broadcast_to_reference(payload["alpha"], reference=gamma, field_name="alpha")
    beta = _broadcast_to_reference(payload["beta"], reference=gamma, field_name="beta")

    # Clip denominators to remain numerically stable near alpha->1.
    alpha = torch.nan_to_num(alpha, nan=1.0 + 1e-6, posinf=1e6, neginf=1.0 + 1e-6)
    nu = torch.nan_to_num(nu, nan=1e-6, posinf=1e6, neginf=1e-6)
    beta = torch.nan_to_num(beta, nan=0.0, posinf=1e6, neginf=0.0)
    alpha_m1 = (alpha - 1.0).clamp_min(1e-6)
    nu_safe = nu.clamp_min(1e-6)
    alpha_safe = alpha.clamp_min(1.0 + 1e-6)
    beta_safe = beta.clamp_min(0.0)

    aleatoric = beta_safe / (nu_safe * alpha_m1)
    epistemic = beta_safe / (nu_safe * alpha_safe * alpha_m1)
    total_var = torch.nan_to_num(aleatoric + epistemic, nan=0.0, posinf=1e6, neginf=0.0)
    total_std = total_var.clamp_min(0.0).sqrt()
    return gamma, total_std


def extract_mean_and_std(prediction: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Normalize model outputs into ``(mean, std_or_none)``.

    Args:
        prediction: model output payload

    Returns:
        mean tensor and optional std tensor.
    """
    if torch.is_tensor(prediction):
        return _sanitize_mean_like(prediction), None

    if isinstance(prediction, (tuple, list)) and len(prediction) >= 2:
        mean = _sanitize_mean_like(prediction[0])
        if prediction[1] is None:
            return mean, None
        std = _sanitize_std_like(prediction[1], reference=mean)
        return mean, std

    if isinstance(prediction, dict):
        # atlas.models.uncertainty.EvidentialRegression style
        if "mean" in prediction:
            mean = _sanitize_mean_like(prediction["mean"])
            if "total_std" in prediction:
                std = _sanitize_std_like(prediction["total_std"], reference=mean)
                return mean, std
            if "std" in prediction:
                std = _sanitize_std_like(prediction["std"], reference=mean)
                return mean, std
            if "sigma" in prediction:
                std = _sanitize_std_like(prediction["sigma"], reference=mean)
                return mean, std
            return mean, None

        if "mu" in prediction:
            mean = _sanitize_mean_like(prediction["mu"])
            if "sigma" in prediction:
                std = _sanitize_std_like(prediction["sigma"], reference=mean)
                return mean, std
            if "std" in prediction:
                std = _sanitize_std_like(prediction["std"], reference=mean)
                return mean, std
            return mean, None

        if {"gamma", "nu", "alpha", "beta"}.issubset(prediction.keys()):
            return _from_evidential_payload(prediction)

        # Some heads expose only uncertainty terms with implicit mean keys.
        if "gamma" in prediction:
            mean = _sanitize_mean_like(prediction["gamma"])
            if "total_std" in prediction:
                std = _sanitize_std_like(prediction["total_std"], reference=mean)
                return mean, std
            return mean, None

    # Last resort: coerce to tensor and treat as deterministic mean.
    return _sanitize_mean_like(prediction), None


def resolve_primary_edge_features(model: Any, batch: Any) -> torch.Tensor:
    """
    Select the primary edge feature tensor for model forward.

    Preference:
    1) Equivariant encoders -> edge_vec
    2) edge_attr
    3) edge_vec fallback
    """
    encoder = getattr(model, "encoder", None)
    edge_attr = getattr(batch, "edge_attr", None)
    edge_vec = getattr(batch, "edge_vec", None)

    prefers_edge_vec = bool(hasattr(encoder, "sh_irreps"))
    if prefers_edge_vec and edge_vec is not None:
        return edge_vec
    if edge_attr is not None:
        return edge_attr
    if edge_vec is not None:
        return edge_vec
    raise AttributeError("Batch does not contain edge_attr or edge_vec.")


def forward_graph_model(
    model: Any,
    batch: Any,
    *,
    tasks: list[str] | None = None,
    encoder_kwargs: dict[str, Any] | None = None,
):
    """
    Forward helper for graph models used across ATLAS scripts.

    Handles:
    - edge feature selection (edge_attr vs edge_vec)
    - optional encoder kwargs (edge_vectors / edge_index_3body)
    - model signatures with or without optional multitask args
    """
    x_attr = getattr(batch, "x", None)
    edge_index_attr = getattr(batch, "edge_index", None)
    if x_attr is None:
        raise AttributeError("Batch must contain node features `x`.")
    if edge_index_attr is None:
        raise AttributeError("Batch must contain `edge_index`.")
    if tasks is not None and (
        isinstance(tasks, str)
        or not isinstance(tasks, Sequence)
        or any(not isinstance(task, str) or not task.strip() for task in tasks)
    ):
        raise ValueError("tasks must be a sequence of non-empty strings when provided.")
    if encoder_kwargs is not None and not isinstance(encoder_kwargs, Mapping):
        raise TypeError("encoder_kwargs must be a mapping when provided.")

    edge_feats = resolve_primary_edge_features(model, batch)
    batch_index = getattr(batch, "batch", None)
    if batch_index is None:
        n_nodes = int(getattr(x_attr, "shape", [0])[0])
        if n_nodes <= 0:
            raise ValueError("Cannot infer batch indices because batch.x is empty.")
        batch_index = torch.zeros(
            n_nodes,
            dtype=torch.long,
            device=x_attr.device if torch.is_tensor(x_attr) else None,
        )

    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        params = {}
    call_kwargs = {}

    if "tasks" in params and tasks is not None:
        call_kwargs["tasks"] = tasks
    if "edge_vectors" in params and hasattr(batch, "edge_vec") and edge_feats is not batch.edge_vec:
        # Avoid passing duplicate edge-vector information when the 3rd positional
        # argument is already edge_vec (common in EquivariantGNN/MultiTaskGNN paths).
        call_kwargs["edge_vectors"] = batch.edge_vec
    if "edge_index_3body" in params and hasattr(batch, "edge_index_3body"):
        call_kwargs["edge_index_3body"] = batch.edge_index_3body
    if "encoder_kwargs" in params and encoder_kwargs:
        call_kwargs["encoder_kwargs"] = encoder_kwargs

    return model(x_attr, edge_index_attr, edge_feats, batch_index, **call_kwargs)
