"""
Model loading helpers for script-level inference utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from atlas.models.cgcnn import CGCNN
from atlas.models.equivariant import EquivariantGNN
from atlas.models.multi_task import MultiTaskGNN
from atlas.training.normalizers import MultiTargetNormalizer

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


def _build_cgcnn_multitask(tasks: Iterable[str]) -> MultiTaskGNN:
    encoder = CGCNN(
        node_dim=91,
        edge_dim=20,
        hidden_dim=128,
        n_conv=3,
        n_fc=2,
        output_dim=1,
        dropout=0.0,
    )
    return MultiTaskGNN(
        encoder=encoder,
        tasks={t: {"type": "scalar"} for t in tasks},
        embed_dim=encoder.hidden_dim,
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

    return normalized


def _try_load_candidates(state_dict: dict, tasks: list[str]) -> MultiTaskGNN:
    # Try phase-2 equivariant presets first, then CGCNN multitask fallback.
    for preset in _PHASE2_PRESETS:
        model = _build_equivariant_multitask(tasks, preset)
        try:
            model.load_state_dict(state_dict, strict=True)
            return model
        except Exception:
            continue

    model = _build_cgcnn_multitask(tasks)
    model.load_state_dict(state_dict, strict=True)
    return model


def _load_normalizer(payload: dict) -> MultiTargetNormalizer | None:
    state = payload.get("normalizer")
    if not isinstance(state, dict):
        return None
    norm = MultiTargetNormalizer()
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
