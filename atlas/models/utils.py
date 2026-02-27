"""
Model loading helpers for script-level inference utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

import torch

from atlas.models.cgcnn import CGCNN
from atlas.models.equivariant import EquivariantGNN
from atlas.models.m3gnet import M3GNet
from atlas.models.multi_task import MultiTaskGNN
from atlas.training.normalizers import MultiTargetNormalizer, TargetNormalizer

_PHASE2_PRESETS = [
    {"irreps_hidden": "16x0e + 8x1o", "max_ell": 1, "n_layers": 2, "n_radial_basis": 10, "radial_hidden": 32},
    {"irreps_hidden": "64x0e + 32x1o + 16x2e", "max_ell": 2, "n_layers": 3, "n_radial_basis": 20, "radial_hidden": 128},
    {"irreps_hidden": "128x0e + 64x1o + 32x2e", "max_ell": 2, "n_layers": 4, "n_radial_basis": 32, "radial_hidden": 256},
]


def _extract_tasks_from_state_dict(state_dict: dict) -> list[str]:
    tasks = set()
    for key in state_dict:
        if key.startswith("heads."):
            parts = key.split(".")
            if len(parts) >= 2:
                tasks.add(parts[1])
    return sorted(tasks) or ["formation_energy", "band_gap", "bulk_modulus", "shear_modulus"]


def _build_equivariant_multitask(tasks: Iterable[str], preset: dict) -> MultiTaskGNN:
    encoder = EquivariantGNN(
        irreps_hidden=preset["irreps_hidden"],
        max_ell=preset["max_ell"],
        n_layers=preset["n_layers"],
        max_radius=5.0,
        n_species=86,
        n_radial_basis=preset["n_radial_basis"],
        radial_hidden=preset["radial_hidden"],
        output_dim=1,
    )
    return MultiTaskGNN(
        encoder=encoder,
        tasks={t: {"type": "scalar"} for t in tasks},
        embed_dim=encoder.scalar_dim,
    )


def _infer_cgcnn_config_from_state_dict(state_dict: dict, *, prefix: str = "encoder.") -> dict:
    """
    Infer CGCNN hyperparameters from state_dict keys.

    Args:
        state_dict: checkpoint state dict.
        prefix: parameter prefix, e.g. "encoder." for phase2 multitask, "" for phase1.
    """
    node_embed_key = f"{prefix}node_embed.weight"
    if node_embed_key not in state_dict:
        raise KeyError(f"Missing {node_embed_key} in state_dict")

    node_embed = state_dict[node_embed_key]
    hidden_dim = int(node_embed.shape[0])
    node_dim = int(node_embed.shape[1])

    conv_indices = []
    conv_pattern = re.compile(rf"^{re.escape(prefix)}convs\.(\d+)\.")
    for key in state_dict:
        m = conv_pattern.match(key)
        if m:
            conv_indices.append(int(m.group(1)))
    n_conv = (max(conv_indices) + 1) if conv_indices else 3

    msg_weight_key = f"{prefix}convs.0.msg_mlp.0.weight"
    if msg_weight_key in state_dict:
        msg_in = int(state_dict[msg_weight_key].shape[1])
        edge_dim = max(msg_in - 2 * hidden_dim, 1)
    else:
        edge_dim = 20

    fc_linear_pattern = re.compile(rf"^{re.escape(prefix)}fc\.(\d+)\.weight$")
    fc_linear_items: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        m = fc_linear_pattern.match(key)
        if m and isinstance(value, torch.Tensor) and value.ndim == 2:
            fc_linear_items.append((int(m.group(1)), value))
    fc_linear_items.sort(key=lambda x: x[0])
    n_fc = max(len(fc_linear_items), 1)

    has_attn_gate = any(key.startswith(f"{prefix}attn_gate.") for key in state_dict)
    if has_attn_gate:
        pooling = "attn"
    elif fc_linear_items and int(fc_linear_items[0][1].shape[1]) == hidden_dim * 2:
        pooling = "mean_max"
    else:
        pooling = "mean"

    jk = "concat" if f"{prefix}jk_proj.weight" in state_dict else "last"
    use_edge_gates = any(".gate_mlp." in key for key in state_dict if key.startswith(f"{prefix}convs."))

    return {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": hidden_dim,
        "n_conv": n_conv,
        "n_fc": n_fc,
        "pooling": pooling,
        "jk": jk,
        "message_aggr": "mean",
        "use_edge_gates": use_edge_gates,
    }


def _infer_cgcnn_output_dim_from_state_dict(state_dict: dict, *, prefix: str = "") -> int:
    fc_linear_pattern = re.compile(rf"^{re.escape(prefix)}fc\.(\d+)\.weight$")
    fc_linear_items: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        m = fc_linear_pattern.match(key)
        if m and isinstance(value, torch.Tensor) and value.ndim == 2:
            fc_linear_items.append((int(m.group(1)), value))
    if not fc_linear_items:
        return 1
    fc_linear_items.sort(key=lambda x: x[0])
    return int(fc_linear_items[-1][1].shape[0])


def _build_cgcnn_multitask(tasks: Iterable[str], cfg: dict | None = None) -> MultiTaskGNN:
    if cfg is None:
        cfg = {
            "node_dim": 91,
            "edge_dim": 20,
            "hidden_dim": 128,
            "n_conv": 3,
            "n_fc": 2,
            "pooling": "mean_max",
            "jk": "concat",
            "message_aggr": "mean",
            "use_edge_gates": True,
        }
    encoder = CGCNN(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_conv=cfg["n_conv"],
        n_fc=cfg["n_fc"],
        output_dim=1,
        dropout=0.0,
        pooling=cfg["pooling"],
        jk=cfg["jk"],
        message_aggr=cfg["message_aggr"],
        use_edge_gates=cfg["use_edge_gates"],
    )
    return MultiTaskGNN(
        encoder=encoder,
        tasks={t: {"type": "scalar"} for t in tasks},
        embed_dim=encoder.graph_dim,
    )


def _infer_m3gnet_config_from_state_dict(state_dict: dict, *, prefix: str = "encoder.") -> dict:
    embedding_key = f"{prefix}embedding.weight"
    edge_embed_key = f"{prefix}edge_embedding.weight"
    if embedding_key not in state_dict or edge_embed_key not in state_dict:
        raise KeyError("State dict does not look like an M3GNet encoder payload.")

    embedding = state_dict[embedding_key]
    edge_embedding = state_dict[edge_embed_key]
    n_species = int(embedding.shape[0])
    embed_dim = int(embedding.shape[1])
    n_rbf = int(edge_embedding.shape[1])

    layer_indices = []
    layer_pattern = re.compile(rf"^{re.escape(prefix)}layers\.(\d+)\.")
    for key in state_dict:
        m = layer_pattern.match(key)
        if m:
            layer_indices.append(int(m.group(1)))
    n_layers = (max(layer_indices) + 1) if layer_indices else 3

    return {
        "n_species": n_species,
        "embed_dim": embed_dim,
        "n_layers": n_layers,
        "n_rbf": n_rbf,
        "max_radius": 5.0,
    }


def _build_m3gnet_multitask(tasks: Iterable[str], cfg: dict) -> MultiTaskGNN:
    encoder = M3GNet(
        n_species=cfg["n_species"],
        embed_dim=cfg["embed_dim"],
        n_layers=cfg["n_layers"],
        n_rbf=cfg["n_rbf"],
        max_radius=cfg.get("max_radius", 5.0),
    )
    return MultiTaskGNN(
        encoder=encoder,
        tasks={t: {"type": "scalar"} for t in tasks},
        embed_dim=encoder.embed_dim,
    )


def _build_cgcnn_single_task(cfg: dict, *, output_dim: int = 1) -> CGCNN:
    return CGCNN(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_conv=cfg["n_conv"],
        n_fc=cfg["n_fc"],
        output_dim=output_dim,
        dropout=0.0,
        pooling=cfg["pooling"],
        jk=cfg["jk"],
        message_aggr=cfg["message_aggr"],
        use_edge_gates=cfg["use_edge_gates"],
    )


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    """
    Normalize legacy checkpoint key layouts.

    Supported conversions:
    - DDP/DataParallel prefix: module.*
    - legacy phase2 CGCNN wrapper: encoder.cgcnn.* -> encoder.*
    """
    normalized = dict(state_dict)

    if any(key.startswith("module.") for key in normalized):
        normalized = {
            (key[len("module."):] if key.startswith("module.") else key): value
            for key, value in normalized.items()
        }

    if any(key.startswith("encoder.cgcnn.") for key in normalized):
        normalized = {
            (
                "encoder." + key[len("encoder.cgcnn."):]
                if key.startswith("encoder.cgcnn.")
                else key
            ): value
            for key, value in normalized.items()
        }

    if any(key.startswith("cgcnn.") for key in normalized):
        normalized = {
            (
                key[len("cgcnn."):]
                if key.startswith("cgcnn.")
                else key
            ): value
            for key, value in normalized.items()
        }

    return normalized


def _try_load_candidates(state_dict: dict, tasks: list[str]) -> MultiTaskGNN:
    # Try phase-2 equivariant presets first, then CGCNN multitask fallback.
    for preset in _PHASE2_PRESETS:
        try:
            model = _build_equivariant_multitask(tasks, preset)
            model.load_state_dict(state_dict, strict=True)
            return model
        except Exception:
            continue

    try:
        m3_cfg = _infer_m3gnet_config_from_state_dict(state_dict)
        model = _build_m3gnet_multitask(tasks, m3_cfg)
        model.load_state_dict(state_dict, strict=True)
        return model
    except Exception:
        pass

    cgcnn_cfg = _infer_cgcnn_config_from_state_dict(state_dict)
    model = _build_cgcnn_multitask(tasks, cfg=cgcnn_cfg)
    model.load_state_dict(state_dict, strict=True)
    return model


def _load_normalizer(payload: dict) -> MultiTargetNormalizer | None:
    state = payload.get("normalizer")
    if not isinstance(state, dict):
        return None
    norm = MultiTargetNormalizer()
    norm.load_state_dict(state)
    return norm


def _load_scalar_normalizer(payload: dict) -> TargetNormalizer | None:
    state = payload.get("normalizer")
    if not isinstance(state, dict):
        return None
    norm = TargetNormalizer()
    norm.load_state_dict(state)
    return norm


def load_phase2_model(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    """
    Load a Phase-2 model checkpoint and infer architecture from state_dict shape.

    Returns:
        tuple(model, normalizer_or_none)
    """
    ckpt_path = Path(checkpoint_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise KeyError(f"{ckpt_path} does not contain 'model_state_dict'")

    state_dict = payload["model_state_dict"]
    state_dict = _normalize_state_dict_keys(state_dict)
    tasks = _extract_tasks_from_state_dict(state_dict)
    model = _try_load_candidates(state_dict, tasks)
    model.to(device)
    model.eval()

    normalizer = _load_normalizer(payload)
    return model, normalizer


def load_phase1_model(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    """
    Load a Phase-1 CGCNN checkpoint with automatic architecture inference.

    Returns:
        tuple(model, normalizer_or_none)
    """
    ckpt_path = Path(checkpoint_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        normalizer = _load_scalar_normalizer(payload)
    elif isinstance(payload, dict):
        state_dict = payload
        normalizer = None
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    state_dict = _normalize_state_dict_keys(state_dict)
    cfg = _infer_cgcnn_config_from_state_dict(state_dict, prefix="")
    output_dim = _infer_cgcnn_output_dim_from_state_dict(state_dict, prefix="")
    model = _build_cgcnn_single_task(cfg, output_dim=output_dim)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, normalizer
