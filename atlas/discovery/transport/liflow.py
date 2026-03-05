from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import ase
import numpy as np
import torch

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LIFLOW_REPO_PATH = _PROJECT_ROOT / "references" / "recisic" / "liflow"
_LIFLOW_API: tuple[Any, Any, Any] | None = None
_LIFLOW_IMPORT_ERROR: str | None = None


def _append_repo_path() -> None:
    if _LIFLOW_REPO_PATH.exists() and str(_LIFLOW_REPO_PATH) not in sys.path:
        sys.path.append(str(_LIFLOW_REPO_PATH))


def _load_liflow_api() -> tuple[Any, Any, Any]:
    global _LIFLOW_API, _LIFLOW_IMPORT_ERROR
    if _LIFLOW_API is not None:
        return _LIFLOW_API
    _append_repo_path()
    try:
        from liflow.model.modules import FlowModule
        from liflow.utils.inference import FlowSimulator
        from liflow.utils.prior import get_prior
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional external repo
        _LIFLOW_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        raise ImportError(
            f"Could not import LiFlow from {_LIFLOW_REPO_PATH}. "
            f"Underlying error: {_LIFLOW_IMPORT_ERROR}"
        ) from exc
    _LIFLOW_API = (FlowModule, FlowSimulator, get_prior)
    _LIFLOW_IMPORT_ERROR = None
    return _LIFLOW_API


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _normalize_device(device: str) -> str:
    value = str(device).strip().lower()
    if value == "cpu":
        return "cpu"
    if value.startswith("cuda"):
        if torch.cuda.is_available():
            return value
        logger.warning("CUDA requested for LiFlow but unavailable; falling back to CPU.")
        return "cpu"
    logger.warning("Unknown LiFlow device '%s'; falling back to CPU.", device)
    return "cpu"


def _normalize_temperature_list(temp_list: Sequence[int] | None) -> list[int]:
    raw_values = [600, 800, 1000] if temp_list is None else list(temp_list)
    if isinstance(temp_list, (str, bytes)):
        raise ValueError("temp_list must be a sequence of positive integers, not a string")
    values: list[int] = []
    for idx, raw in enumerate(raw_values):
        if _is_boolean_like(raw):
            raise ValueError(f"temp_list[{idx}] must be a positive integer")
        if isinstance(raw, Integral):
            value = int(raw)
        elif isinstance(raw, Real):
            scalar = float(raw)
            if not np.isfinite(scalar) or not scalar.is_integer():
                raise ValueError(f"temp_list[{idx}] must be a positive integer")
            value = int(scalar)
        else:
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"temp_list[{idx}] must be a positive integer") from exc
        if value > 0:
            values.append(value)
    if not values:
        raise ValueError("temp_list must contain at least one positive integer temperature")
    return values


def _resolve_checkpoint_path(checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path).expanduser().resolve()
    return _LIFLOW_REPO_PATH / "ckpt" / "P_universal.ckpt"


def _coerce_positive_int(value: object, *, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not np.isfinite(scalar) or not scalar.is_integer():
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        number = int(scalar)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return number


class LiFlowEvaluator:
    """
    Wrapper for LiFlow (flow-matching for atomic transport).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        element_index_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temp_list: list[int] | None = None,
    ):
        FlowModule, _, get_prior = _load_liflow_api()

        self.device = _normalize_device(device)
        self.temp_list = _normalize_temperature_list(temp_list)

        ckpt_path = _resolve_checkpoint_path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"LiFlow checkpoint not found at {ckpt_path}")

        self.model = FlowModule.load_from_checkpoint(str(ckpt_path), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        self.element_idx = self._load_element_index(element_index_path)

        if hasattr(self.model, "cfg") and hasattr(self.model.cfg, "propagate_prior"):
            cfg_prior = self.model.cfg.propagate_prior
            self.prior = get_prior(cfg_prior.class_name, **cfg_prior.params, seed=42)
        else:
            logger.warning(
                "LiFlow propagate_prior config missing, using AdaptiveMaxwellBoltzmannPrior fallback."
            )
            self.prior = get_prior("AdaptiveMaxwellBoltzmannPrior", seed=42)

    def _load_element_index(self, path: str | None) -> np.ndarray:
        if path:
            candidate = Path(path).expanduser().resolve()
            if candidate.is_file():
                loaded = np.asarray(np.load(str(candidate))).reshape(-1)
                if loaded.size == 0:
                    raise ValueError("element_index array must be non-empty")
                return loaded.astype(int, copy=False)

        default_path = _LIFLOW_REPO_PATH / "data" / "universal" / "element_index.npy"
        if default_path.is_file():
            loaded = np.asarray(np.load(str(default_path))).reshape(-1)
            if loaded.size == 0:
                raise ValueError("element_index array must be non-empty")
            return loaded.astype(int, copy=False)

        logger.warning(
            "LiFlow element_index.npy missing, using fallback mapping Z->Z-1. "
            "Transport estimates may be noisy."
        )
        mapping = np.arange(119, dtype=int) - 1
        mapping[0] = 0
        return mapping

    def simulate(
        self,
        atoms: ase.Atoms,
        steps: int = 500,
        flow_steps: int = 10,
    ) -> tuple[list[ase.Atoms], float]:
        _, FlowSimulator, _ = _load_liflow_api()
        if not isinstance(atoms, ase.Atoms):
            raise TypeError("atoms must be ase.Atoms")
        if len(atoms) == 0:
            raise ValueError("atoms must be non-empty")
        n_steps = _coerce_positive_int(steps, name="steps")
        n_flow_steps = _coerce_positive_int(flow_steps, name="flow_steps")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if positions.shape != (len(atoms), 3):
            raise ValueError(
                f"atoms positions must have shape {(len(atoms), 3)}, got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("atoms positions must be finite")
        z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        if z.size != len(atoms):
            raise ValueError("atoms atomic_numbers size mismatch")
        if np.any(z < 0):
            raise ValueError("atoms atomic_numbers must be non-negative")
        if np.any(z >= int(self.element_idx.size)):
            max_z = int(np.max(z))
            raise ValueError(
                f"element_idx size {self.element_idx.size} cannot index atomic number {max_z}"
            )

        temp = self.temp_list[0]
        simulator = FlowSimulator(
            propagate_model=self.model,
            propagate_prior=self.prior,
            atomic_numbers=z,
            element_idx=self.element_idx,
            lattice=atoms.cell.array,
            temp=temp,
            correct_model=None,
            correct_prior=None,
            pbc=True,
            scale_Li_index=1,
            scale_frame_index=0,
        )

        traj_pos = simulator.run(
            positions=atoms.get_positions(),
            steps=n_steps,
            flow_steps=n_flow_steps,
            solver="euler",
            verbose=False,
            fix_com=True,
        )
        if len(traj_pos) == 0:
            raise RuntimeError("LiFlow simulator returned an empty trajectory.")

        trajectory: list[ase.Atoms] = []
        for pos in traj_pos:
            pos_array = np.asarray(pos, dtype=float)
            if pos_array.shape != (len(atoms), 3):
                raise RuntimeError(
                    f"LiFlow position frame shape mismatch: got {pos_array.shape}, expected {(len(atoms), 3)}"
                )
            if not np.isfinite(pos_array).all():
                raise RuntimeError("LiFlow produced non-finite coordinates.")
            new_atoms = atoms.copy()
            new_atoms.set_positions(pos_array)
            trajectory.append(new_atoms)

        # Rough diffusion estimate from Li MSD between first/last frame.
        diff_coeff = 0.0
        li_mask = z == 3
        if np.any(li_mask) and len(traj_pos) >= 2:
            first = np.asarray(traj_pos[0], dtype=float)
            last = np.asarray(traj_pos[-1], dtype=float)
            disp = last[li_mask] - first[li_mask]
            msd = float(np.mean(np.sum(disp**2, axis=1)))
            # Approximate dt in ps (coarse fallback estimate).
            dt_ps = max(n_steps * 1e-3, 1e-8)
            diff_coeff = float(max(msd / (6.0 * dt_ps), 0.0))

        return trajectory, diff_coeff


__all__ = ("LiFlowEvaluator",)
