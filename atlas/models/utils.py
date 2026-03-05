"""
Model loading helpers for script-level inference utilities.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from pathlib import Path

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


def _coerce_checkpoint_path(checkpoint_path: str | Path) -> Path:
    if isinstance(checkpoint_path, bool) or type(checkpoint_path).__name__ == "bool_":
        raise ValueError("checkpoint_path must be a path-like string, not boolean")
    path = Path(checkpoint_path).expanduser()
    if not str(path).strip():
        raise ValueError("checkpoint_path must be non-empty")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"checkpoint_path must point to a file, got directory: {path}")
    return path


def _payload_values_compatible(lhs, rhs) -> bool:
    if torch.is_tensor(lhs) and torch.is_tensor(rhs):
        return lhs.shape == rhs.shape and lhs.dtype == rhs.dtype and torch.equal(lhs, rhs)
    return lhs == rhs


def _is_valid_scalar_normalizer(norm: TargetNormalizer) -> bool:
    try:
        mean = float(norm.mean)
        std = float(norm.std)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(mean) and math.isfinite(std) and std > 0.0)


def _is_valid_multi_normalizer(norm: MultiTargetNormalizer) -> bool:
    return all(_is_valid_scalar_normalizer(sub_norm) for sub_norm in norm.normalizers.values())


def _extract_tasks_from_state_dict(state_dict: dict) -> list[str]:
    tasks = set()
    for key in state_dict:
        if not isinstance(key, str):
            continue
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
    if not isinstance(node_embed, torch.Tensor) or node_embed.ndim != 2:
        raise ValueError(f"Invalid tensor layout for {node_embed_key}: expected 2D tensor.")
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
    if msg_weight_key in state_dict and isinstance(state_dict[msg_weight_key], torch.Tensor):
        msg_weight = state_dict[msg_weight_key]
        if msg_weight.ndim < 2:
            raise ValueError(f"Invalid tensor layout for {msg_weight_key}: expected rank>=2 tensor.")
        msg_in = int(msg_weight.shape[1])
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
    def canonical_key(key: str) -> str:
        out = key
        if out.startswith("module."):
            out = out[len("module."):]
        if out.startswith("encoder.cgcnn."):
            out = "encoder." + out[len("encoder.cgcnn."):]
        if out.startswith("cgcnn."):
            out = out[len("cgcnn."):]
        return out

    normalized: dict = {}
    for key, value in dict(state_dict).items():
        key_str = str(key)
        mapped = canonical_key(key_str)
        if mapped in normalized:
            if _payload_values_compatible(normalized[mapped], value):
                continue
            raise ValueError(
                "Conflicting state_dict keys after normalization: "
                f"{key_str!r} -> {mapped!r} collides with existing entry."
            )
        normalized[mapped] = value
    return normalized


def _try_load_candidates(state_dict: dict, tasks: list[str]) -> MultiTaskGNN:
    # Try phase-2 equivariant presets first, then CGCNN multitask fallback.
    errors: list[str] = []
    for preset in _PHASE2_PRESETS:
        try:
            model = _build_equivariant_multitask(tasks, preset)
            model.load_state_dict(state_dict, strict=True)
            return model
        except (RuntimeError, KeyError, ValueError) as exc:
            errors.append(f"equivariant[{preset['irreps_hidden']}]: {type(exc).__name__}")
            continue

    try:
        m3_cfg = _infer_m3gnet_config_from_state_dict(state_dict)
        model = _build_m3gnet_multitask(tasks, m3_cfg)
        model.load_state_dict(state_dict, strict=True)
        return model
    except (RuntimeError, KeyError, ValueError) as exc:
        errors.append(f"m3gnet: {type(exc).__name__}")

    try:
        cgcnn_cfg = _infer_cgcnn_config_from_state_dict(state_dict)
        model = _build_cgcnn_multitask(tasks, cfg=cgcnn_cfg)
        model.load_state_dict(state_dict, strict=True)
        return model
    except (RuntimeError, KeyError, ValueError) as exc:
        errors.append(f"cgcnn: {type(exc).__name__}")

    details = "; ".join(errors[-5:]) if errors else "no candidate model loaders succeeded"
    raise ValueError(f"Unable to load checkpoint with known Phase-2 model candidates: {details}")


def _load_normalizer(payload: dict) -> MultiTargetNormalizer | None:
    state = payload.get("normalizer")
    if not isinstance(state, dict):
        return None
    norm = MultiTargetNormalizer()
    try:
        norm.load_state_dict(state)
        if _is_valid_multi_normalizer(norm):
            return norm
        return None
    except Exception:
        return None


def _load_scalar_normalizer(payload: dict) -> TargetNormalizer | None:
    state = payload.get("normalizer")
    if not isinstance(state, dict):
        return None
    norm = TargetNormalizer()
    try:
        norm.load_state_dict(state)
        if _is_valid_scalar_normalizer(norm):
            return norm
        return None
    except Exception:
        return None


def load_phase2_model(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    """
    Load a Phase-2 model checkpoint and infer architecture from state_dict shape.

    Returns:
        tuple(model, normalizer_or_none)
    """
    ckpt_path = _coerce_checkpoint_path(checkpoint_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    if "model_state_dict" not in payload:
        raise KeyError(f"{ckpt_path} does not contain 'model_state_dict'")

    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"{ckpt_path} has invalid 'model_state_dict' payload.")
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
    ckpt_path = _coerce_checkpoint_path(checkpoint_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        normalizer = _load_scalar_normalizer(payload)
    elif isinstance(payload, dict):
        state_dict = payload
        normalizer = None
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{ckpt_path} has invalid 'model_state_dict' payload.")

    state_dict = _normalize_state_dict_keys(state_dict)
    cfg = _infer_cgcnn_config_from_state_dict(state_dict, prefix="")
    output_dim = _infer_cgcnn_output_dim_from_state_dict(state_dict, prefix="")
    model = _build_cgcnn_single_task(cfg, output_dim=output_dim)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, normalizer
