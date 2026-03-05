"""
MACE inference and structure relaxation utilities.
"""

from __future__ import annotations

import logging
import math
import warnings
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from atlas.utils.registry import RELAXERS

logger = logging.getLogger(__name__)

_VALID_MODEL_SIZES = {"small", "medium", "large"}
_VALID_DTYPES = {"float32", "float64"}
_VALID_CELL_FILTERS = {"frechet", "exp", "unit", "fixed", None}


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not math.isfinite(scalar) or not scalar.is_integer():
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
        number = int(scalar)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    return number


def _normalize_device(device: str) -> str:
    if _is_boolean_like(device):
        raise ValueError(f"Invalid device string: {device!r}")
    text = str(device).strip()
    if text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    try:
        parsed = torch.device(text)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid device string: {device!r}") from exc

    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            logger.warning(
                "CUDA device index %s out of range. Falling back to cuda:0.",
                parsed.index,
            )
            return "cuda:0"
    return str(parsed)


@RELAXERS.register("mlip_arena_mace")
class MACERelaxer:
    """
    Fast structure relaxation using MACE potentials.

    Can use either:
    1. A custom-trained MACE model.
    2. The pre-trained MACE-MP foundation model.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        use_foundation: bool = True,
        model_size: str = "large",
        default_dtype: str = "float32",
    ):
        if model_size not in _VALID_MODEL_SIZES:
            raise ValueError(
                f"model_size must be one of {sorted(_VALID_MODEL_SIZES)}, got {model_size!r}"
            )
        if default_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"default_dtype must be one of {sorted(_VALID_DTYPES)}, got {default_dtype!r}"
            )

        self.model_path = Path(model_path).expanduser() if model_path is not None else None
        self._calculator = None
        self.use_foundation = bool(use_foundation)
        self.model_size = model_size
        self.dtype = default_dtype
        self.device = _normalize_device(device)

        logger.info("MACERelaxer initialized on %s", self.device)

    @property
    def calculator(self):
        """Lazy-load the MACE calculator."""
        if self._calculator is not None:
            return self._calculator

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.model_path is not None:
                if self.model_path.exists() and self.model_path.is_file():
                    logger.info("Loading custom MACE model: %s", self.model_path)
                    try:
                        from mace.calculators import MACECalculator

                        self._calculator = MACECalculator(
                            model_paths=str(self.model_path),
                            device=self.device,
                            default_dtype=self.dtype,
                        )
                        return self._calculator
                    except Exception as exc:
                        logger.error("Failed to load custom MACE model: %s", exc)
                else:
                    logger.warning(
                        "model_path does not exist or is not a file: %s", self.model_path
                    )

            if self.use_foundation:
                logger.info("Loading MACE-MP foundation model (%s).", self.model_size)
                try:
                    from mace.calculators import mace_mp

                    self._calculator = mace_mp(
                        model=self.model_size,
                        device=self.device,
                        default_dtype=self.dtype,
                    )
                    return self._calculator
                except Exception as exc:
                    logger.warning("Could not load MACE-MP foundation model: %s", exc)

        logger.warning("No MACE calculator available; using heuristic fallback.")
        self._calculator = None
        return None

    def _resolve_cell_filter(self, atoms, cell_filter: str | None):
        from ase.constraints import ExpCellFilter, UnitCellFilter

        normalized = None if cell_filter is None else str(cell_filter).lower()
        if normalized not in _VALID_CELL_FILTERS:
            raise ValueError(
                f"cell_filter must be one of {sorted(x for x in _VALID_CELL_FILTERS if x is not None)}"
                f" or None, got {cell_filter!r}"
            )

        if normalized in (None, "fixed"):
            return atoms
        if normalized == "exp":
            return ExpCellFilter(atoms)
        if normalized == "unit":
            return UnitCellFilter(atoms)

        # normalized == "frechet"
        try:
            from ase.filters import FrechetCellFilter

            return FrechetCellFilter(atoms)
        except ImportError:
            logger.warning("FrechetCellFilter unavailable; falling back to ExpCellFilter.")
            return ExpCellFilter(atoms)

    def relax_structure(
        self,
        structure,
        fmax: float = 0.05,
        steps: int = 200,
        cell_filter: str | None = "frechet",
        trajectory_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Relax a crystal structure using MACE potential.

        Returns:
            dict with: relaxed_structure, energy_per_atom, converged, n_steps
        """
        from ase.constraints import FixSymmetry
        from ase.optimize import BFGSLineSearch

        from atlas.utils.structure import ase_to_pymatgen, pymatgen_to_ase

        if not math.isfinite(float(fmax)) or float(fmax) <= 0:
            raise ValueError(f"fmax must be finite and > 0, got {fmax!r}")
        step_count = _coerce_non_negative_int(steps, name="steps")

        atoms = pymatgen_to_ase(structure)
        if len(atoms) == 0:
            return self._heuristic_result(structure, "Cannot relax empty structure")

        if self.calculator is None:
            return self._heuristic_result(structure, "No calculator loaded")

        atoms.calc = self.calculator

        trajectory = None
        if trajectory_file is not None:
            path = Path(trajectory_file).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            trajectory = str(path)

        try:
            try:
                space_group_number = structure.get_space_group_info()[1]
            except Exception:
                space_group_number = 0

            if space_group_number > 1:
                try:
                    atoms.set_constraint(FixSymmetry(atoms))
                except Exception as exc:
                    logger.debug("FixSymmetry not applied: %s", exc)

            target = self._resolve_cell_filter(atoms, cell_filter)
            optimizer = BFGSLineSearch(target, trajectory=trajectory, logfile=None)
            converged = bool(optimizer.run(fmax=float(fmax), steps=step_count))

            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            if not np.isfinite(energy):
                raise ValueError("Non-finite potential energy from calculator")
            if forces.size > 0 and not np.isfinite(forces).all():
                raise ValueError("Non-finite forces from calculator")

            relaxed_struct = ase_to_pymatgen(atoms)
            n_atoms = len(atoms)
            if n_atoms <= 0:
                raise ValueError("Relaxation returned empty atoms object")

            init_vol = float(getattr(structure, "volume", 0.0) or 0.0)
            if math.isfinite(init_vol) and init_vol > 1e-12:
                volume_change = float(relaxed_struct.volume / init_vol)
            else:
                volume_change = 1.0

            return {
                "relaxed_structure": relaxed_struct,
                "energy_per_atom": float(energy / n_atoms),
                "energy_total": energy,
                "converged": converged,
                "n_steps": int(getattr(optimizer, "nsteps", 0)),
                "forces_max": float(np.max(np.abs(forces))) if forces.size > 0 else 0.0,
                "volume_change": volume_change,
            }

        except Exception as exc:
            formula = getattr(getattr(structure, "composition", None), "reduced_formula", "unknown")
            logger.warning("Relaxation failed for %s: %s", formula, exc)
            return self._heuristic_result(structure, str(exc))

    def _heuristic_result(self, structure, error_msg: str) -> dict[str, Any]:
        """Return a standardized failure result."""
        return {
            "relaxed_structure": structure,
            "energy_per_atom": self._heuristic_energy(structure),
            "energy_total": None,
            "converged": False,
            "n_steps": 0,
            "forces_max": None,
            "volume_change": 1.0,
            "error": error_msg,
        }

    def score_stability(self, structure) -> float:
        """Quick stability score in [0, 1] from relaxed per-atom energy."""
        res = self.relax_structure(structure, fmax=0.1, steps=50)
        e_pa = res.get("energy_per_atom")
        if e_pa is None or not np.isfinite(e_pa):
            return 0.5

        e_pa_f = float(e_pa)
        if e_pa_f > 0:
            return 0.0
        return float(np.clip(-e_pa_f / 8.0, 0.0, 1.0))

    def batch_relax(
        self,
        structures: list[Any],
        fmax: float = 0.05,
        steps: int = 100,
        n_jobs: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Relax a batch of structures.
        Parallel GPU calculators are non-trivial; serial path is the default.
        """
        n_jobs_i = _coerce_non_negative_int(n_jobs, name="n_jobs")
        if n_jobs_i == 0:
            raise ValueError(f"n_jobs must be >= 1, got {n_jobs!r}")
        if n_jobs_i != 1:
            logger.warning("n_jobs=%s requested; MACERelaxer currently runs serially.", n_jobs)

        results: list[dict[str, Any]] = []
        try:
            from tqdm import tqdm

            iterator = tqdm(structures, desc="Relaxing", leave=False)
        except ImportError:
            iterator = structures

        for i, struct in enumerate(iterator):
            res = self.relax_structure(struct, fmax=fmax, steps=steps)
            res["index"] = i
            results.append(res)
        return results

    def _heuristic_energy(self, structure) -> float:
        """Rough energy estimate based on Pauling electronegativity."""
        try:
            from pymatgen.core import Element

            values: list[float] = []
            for site in structure:
                elem = Element(str(site.specie))
                eneg = elem.X or 2.0
                values.append(float(-1.0 * eneg))
            if not values:
                return -2.0
            return float(np.mean(values))
        except Exception:
            return -2.0
