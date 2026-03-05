from __future__ import annotations

import logging
import sys
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import ase
import numpy as np
import torch

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MEPIN_REPO_PATH = _PROJECT_ROOT / "references" / "recisic" / "mepin"
_SUPPORTED_MODEL_TYPES = {"cyclo_L", "t1x_L"}
_MODEL_TYPE_MAP = {"cyclo_l": "cyclo_L", "t1x_l": "t1x_L"}
_MEPIN_API: tuple[Any, Any] | None = None
_MEPIN_IMPORT_ERROR: str | None = None


def _append_repo_path() -> None:
    if _MEPIN_REPO_PATH.exists() and str(_MEPIN_REPO_PATH) not in sys.path:
        sys.path.append(str(_MEPIN_REPO_PATH))


def _load_mepin_api() -> tuple[Any, Any]:
    global _MEPIN_API, _MEPIN_IMPORT_ERROR
    if _MEPIN_API is not None:
        return _MEPIN_API
    _append_repo_path()
    try:
        from mepin.model.modules import TripleCrossPaiNNModule
        from mepin.tools.inference import create_reaction_batch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional external repo
        _MEPIN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        raise ImportError(
            f"Could not import MEPIN from {_MEPIN_REPO_PATH}. "
            f"Underlying error: {_MEPIN_IMPORT_ERROR}"
        ) from exc
    _MEPIN_API = (TripleCrossPaiNNModule, create_reaction_batch)
    _MEPIN_IMPORT_ERROR = None
    return _MEPIN_API


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _normalize_device(device: str) -> str:
    value = str(device).strip().lower()
    if value == "cpu":
        return "cpu"
    if value.startswith("cuda"):
        if torch.cuda.is_available():
            return value
        logger.warning("CUDA requested for MEPIN but unavailable; falling back to CPU.")
        return "cpu"
    logger.warning("Unknown MEPIN device '%s'; falling back to CPU.", device)
    return "cpu"


def _normalize_model_type(model_type: str) -> str:
    key = str(model_type).strip().lower()
    if not key:
        raise ValueError("model_type must be a non-empty string")
    if key not in _MODEL_TYPE_MAP:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: {sorted(_SUPPORTED_MODEL_TYPES)}"
        )
    return _MODEL_TYPE_MAP[key]


def _coerce_int_with_min(value: object, *, name: str, minimum: int) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not np.isfinite(scalar) or not scalar.is_integer():
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        number = int(scalar)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}") from exc
    if number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return number


def _resolve_checkpoint_path(checkpoint_path: str | None, model_type: str) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path).expanduser().resolve()
    return _MEPIN_REPO_PATH / "ckpt" / f"{model_type}.ckpt"


class MEPINStabilityEvaluator:
    """
    Wrapper for MEPIN (minimum-energy-path inference).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_type: str = "cyclo_L",
    ):
        TripleCrossPaiNNModule, _ = _load_mepin_api()

        self.device = _normalize_device(device)
        self.model_type = _normalize_model_type(model_type)
        ckpt_path = _resolve_checkpoint_path(checkpoint_path, self.model_type)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"MEPIN checkpoint not found at {ckpt_path}")

        self.model = TripleCrossPaiNNModule.load_from_checkpoint(str(ckpt_path), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def predict_path(
        self,
        reactant: ase.Atoms,
        product: ase.Atoms,
        num_images: int = 20,
    ) -> list[ase.Atoms]:
        _, create_reaction_batch = _load_mepin_api()
        if not isinstance(reactant, ase.Atoms) or not isinstance(product, ase.Atoms):
            raise TypeError("reactant and product must be ase.Atoms")
        image_count = _coerce_int_with_min(num_images, name="num_images", minimum=2)
        if len(reactant) != len(product):
            raise ValueError("reactant and product must have identical atom counts")
        if not np.isfinite(reactant.get_positions()).all() or not np.isfinite(product.get_positions()).all():
            raise ValueError("reactant/product positions must be finite")

        use_geodesic = "G" in self.model_type
        batch = create_reaction_batch(
            reactant,
            product,
            interp_traj=None,
            use_geodesic=use_geodesic,
            num_images=image_count,
        )
        if not hasattr(batch, "to"):
            raise RuntimeError("MEPIN create_reaction_batch returned an invalid batch object.")
        batch = batch.to(self.device)

        with torch.no_grad():
            out = self.model(batch)
            n_atoms = len(reactant)
            if not torch.is_tensor(out):
                raise RuntimeError("MEPIN model output must be a torch.Tensor.")
            if not torch.isfinite(out).all():
                raise RuntimeError("MEPIN model produced non-finite path coordinates.")
            expected = image_count * n_atoms * 3
            if out.numel() != expected:
                raise RuntimeError(
                    f"MEPIN output size mismatch: got {out.numel()}, expected {expected}"
                )
            output_positions = out.reshape(image_count, n_atoms, 3).detach().cpu().numpy()

        trajectory: list[ase.Atoms] = []
        for i in range(image_count):
            atoms = reactant.copy()
            atoms.set_positions(output_positions[i])
            trajectory.append(atoms)
        return trajectory


__all__ = ("MEPINStabilityEvaluator",)
