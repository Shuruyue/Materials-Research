from __future__ import annotations

import logging
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from atlas.utils.registry import RELAXERS

logger = logging.getLogger(__name__)

_VALID_CELL_FILTERS = {"frechet", "exp", "unit", "fixed", None}
_VALID_OPTIMIZERS = {
    "bfgs": "BFGS",
    "fire": "FIRE",
    "lbfgs": "LBFGS",
    "bfgs_ls": "BFGSLineSearch",
    "lbfgs_ls": "LBFGSLineSearch",
}


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not np.isfinite(scalar) or not scalar.is_integer():
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


def _get_optimizer_class(name: str):
    from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, LBFGSLineSearch

    table = {
        "BFGS": BFGS,
        "FIRE": FIRE,
        "LBFGS": LBFGS,
        "BFGSLineSearch": BFGSLineSearch,
        "LBFGSLineSearch": LBFGSLineSearch,
    }
    return table[name]


@RELAXERS.register("mlip_arena_native")
class NativeMlipArenaRelaxer:
    """
    ASE-based relaxation loop aligned with MLIP-Arena defaults.

    This wrapper preserves the expected registry contract while avoiding hard runtime
    dependency on upstream task modules that may not be import-stable in local installs.
    """

    def __init__(
        self,
        fmax: float = 0.05,
        steps: int = 200,
        cell_filter: str | None = "frechet",
        optimizer: str = "bfgs_ls",
        symmetry: bool = True,
        constraints: list[Any] | None = None,
    ):
        if not math.isfinite(float(fmax)) or float(fmax) <= 0:
            raise ValueError(f"fmax must be finite and > 0, got {fmax!r}")
        step_count = _coerce_non_negative_int(steps, name="steps")

        normalized_filter = None if cell_filter is None else str(cell_filter).lower()
        if normalized_filter not in _VALID_CELL_FILTERS:
            raise ValueError(
                f"cell_filter must be one of {sorted(x for x in _VALID_CELL_FILTERS if x is not None)}"
                f" or None, got {cell_filter!r}"
            )

        normalized_optimizer = str(optimizer).lower()
        if normalized_optimizer not in _VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {sorted(_VALID_OPTIMIZERS)}, got {optimizer!r}"
            )

        self.fmax = float(fmax)
        self.steps = step_count
        self.cell_filter = normalized_filter
        self.optimizer = normalized_optimizer
        if _is_boolean_like(symmetry):
            self.symmetry = bool(symmetry)
        elif isinstance(symmetry, str):
            text = symmetry.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                self.symmetry = True
            elif text in {"0", "false", "no", "off"}:
                self.symmetry = False
            else:
                raise ValueError(f"symmetry must be boolean-like, got {symmetry!r}")
        else:
            raise ValueError(f"symmetry must be boolean-like, got {symmetry!r}")
        self.constraints = list(constraints or [])

    def _apply_cell_filter(self, atoms):
        from ase.constraints import ExpCellFilter, UnitCellFilter

        if self.cell_filter in (None, "fixed"):
            return atoms
        if self.cell_filter == "exp":
            return ExpCellFilter(atoms)
        if self.cell_filter == "unit":
            return UnitCellFilter(atoms)

        try:
            from ase.filters import FrechetCellFilter

            return FrechetCellFilter(atoms)
        except ImportError:
            logger.warning("FrechetCellFilter unavailable; falling back to ExpCellFilter.")
            return ExpCellFilter(atoms)

    def relax(self, pmg_structure, calculator) -> dict[str, Any]:
        """Relax a pymatgen Structure and return standardized result payload."""
        from ase.constraints import FixSymmetry
        from pymatgen.io.ase import AseAtomsAdaptor

        if calculator is None:
            raise ValueError("calculator must not be None")

        atoms = AseAtomsAdaptor.get_atoms(pmg_structure)
        if len(atoms) == 0:
            return {
                "structure": pmg_structure,
                "energy": None,
                "success": False,
                "converged": False,
                "n_steps": 0,
                "error": "Cannot relax empty structure",
            }

        atoms.calc = calculator

        try:
            constraints: list[Any] = []
            if self.symmetry:
                try:
                    constraints.append(FixSymmetry(atoms))
                except Exception as exc:
                    logger.debug("FixSymmetry not applied: %s", exc)
            constraints.extend(self.constraints)
            if constraints:
                atoms.set_constraint(constraints)

            target = self._apply_cell_filter(atoms)
            opt_class = _get_optimizer_class(_VALID_OPTIMIZERS[self.optimizer])
            optimizer = opt_class(target, trajectory=None, logfile=None)
            converged = bool(optimizer.run(fmax=self.fmax, steps=self.steps))

            energy = float(atoms.get_potential_energy())
            if not np.isfinite(energy):
                raise ValueError("Non-finite potential energy from calculator")

            structure = AseAtomsAdaptor.get_structure(atoms)
            return {
                "structure": structure,
                "energy": energy,
                "success": True,
                "converged": converged,
                "n_steps": int(getattr(optimizer, "nsteps", 0)),
            }

        except Exception as exc:
            logger.error("Native MLIP-Arena relaxation failed: %s", exc)
            return {
                "structure": pmg_structure,
                "energy": None,
                "success": False,
                "converged": False,
                "n_steps": 0,
                "error": str(exc),
            }
