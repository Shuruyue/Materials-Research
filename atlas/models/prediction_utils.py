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
from typing import Any

import torch


def _to_tensor(value: Any, *, reference: torch.Tensor | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if reference is not None:
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(value)


def _from_evidential_payload(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    # MultiTask ``EvidentialHead`` format
    gamma = _to_tensor(payload["gamma"])
    nu = _to_tensor(payload["nu"], reference=gamma)
    alpha = _to_tensor(payload["alpha"], reference=gamma)
    beta = _to_tensor(payload["beta"], reference=gamma)

    # Clip denominators to remain numerically stable near alpha->1.
    alpha_m1 = (alpha - 1.0).clamp_min(1e-6)
    nu_safe = nu.clamp_min(1e-6)
    alpha_safe = alpha.clamp_min(1.0 + 1e-6)

    aleatoric = beta / (nu_safe * alpha_m1)
    epistemic = beta / (nu_safe * alpha_safe * alpha_m1)
    total_std = (aleatoric + epistemic).clamp_min(0.0).sqrt()
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
        return prediction, None

    if isinstance(prediction, (tuple, list)) and len(prediction) >= 2:
        mean = _to_tensor(prediction[0])
        std = _to_tensor(prediction[1], reference=mean).clamp_min(0.0)
        return mean, std

    if isinstance(prediction, dict):
        # atlas.models.uncertainty.EvidentialRegression style
        if "mean" in prediction:
            mean = _to_tensor(prediction["mean"])
            if "total_std" in prediction:
                std = _to_tensor(prediction["total_std"], reference=mean).clamp_min(0.0)
                return mean, std
            if "std" in prediction:
                std = _to_tensor(prediction["std"], reference=mean).clamp_min(0.0)
                return mean, std
            if "sigma" in prediction:
                std = _to_tensor(prediction["sigma"], reference=mean).clamp_min(0.0)
                return mean, std
            return mean, None

        if "mu" in prediction:
            mean = _to_tensor(prediction["mu"])
            if "sigma" in prediction:
                std = _to_tensor(prediction["sigma"], reference=mean).clamp_min(0.0)
                return mean, std
            if "std" in prediction:
                std = _to_tensor(prediction["std"], reference=mean).clamp_min(0.0)
                return mean, std
            return mean, None

        if {"gamma", "nu", "alpha", "beta"}.issubset(prediction.keys()):
            return _from_evidential_payload(prediction)

        # Some heads expose only uncertainty terms with implicit mean keys.
        if "gamma" in prediction:
            mean = _to_tensor(prediction["gamma"])
            if "total_std" in prediction:
                std = _to_tensor(prediction["total_std"], reference=mean).clamp_min(0.0)
                return mean, std
            return mean, None

    # Last resort: coerce to tensor and treat as deterministic mean.
    return _to_tensor(prediction), None


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
    edge_feats = resolve_primary_edge_features(model, batch)
    batch_index = getattr(batch, "batch", None)

    params = inspect.signature(model.forward).parameters
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

    return model(batch.x, batch.edge_index, edge_feats, batch_index, **call_kwargs)
