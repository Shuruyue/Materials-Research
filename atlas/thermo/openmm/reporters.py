"""OpenMM reporters for pymatgen-compatible trajectory export."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import openmm.unit as unit
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory

_KJ_MOL_TO_EV = 1.0 / 96.48533212331002
_KJ_MOL_NM_TO_EV_ANG = 0.01036427230133138
_PS_TO_FS = 1000.0


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _coerce_positive_int(value: Any, *, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not math.isfinite(scalar) or not scalar.is_integer():
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


def _coerce_bool(value: Any, *, name: str) -> bool:
    if _is_boolean_like(value):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be boolean-like, got {value!r}")


class PymatgenTrajectoryReporter:
    """
    Reporter that accumulates OpenMM states and exports a pymatgen ``Trajectory``.

    Adapted from atomate2 patterns with extra input/finite-value validation.
    """

    def __init__(
        self,
        reportInterval: int,
        structure: Structure,
        enforcePeriodicBox: bool = True,
    ):
        interval = _coerce_positive_int(reportInterval, name="reportInterval")
        if not isinstance(structure, Structure):
            raise TypeError(f"structure must be pymatgen Structure, got {type(structure)!r}")
        if len(structure) == 0:
            raise ValueError("structure must contain at least one site")

        self._reportInterval = interval
        self._structure = structure
        self._n_sites = int(len(structure))
        self._enforcePeriodicBox = _coerce_bool(enforcePeriodicBox, name="enforcePeriodicBox")

        self.coords: list[np.ndarray] = []
        self.time_steps_ps: list[float] = []
        self.energies_ev: list[float] = []
        self.forces_ev_per_ang: list[np.ndarray] = []

    def describeNextReport(self, simulation: Any) -> tuple[int, bool, bool, bool, bool, bool]:
        current_step = max(0, int(getattr(simulation, "currentStep", 0)))
        steps = self._reportInterval - (current_step % self._reportInterval)
        return (max(1, steps), True, True, True, True, self._enforcePeriodicBox)

    def report(self, simulation: Any, state: Any) -> None:
        # Positions (A)
        positions = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
            dtype=float,
        )
        if positions.shape != (self._n_sites, 3):
            raise ValueError(
                f"Positions shape mismatch: expected {(self._n_sites, 3)}, got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("Non-finite positions encountered in OpenMM reporter")
        self.coords.append(np.array(positions, copy=True))

        # Time (ps)
        time_ps = float(state.getTime().value_in_unit(unit.picosecond))
        if not math.isfinite(time_ps):
            raise ValueError("Non-finite simulation time encountered in OpenMM reporter")
        if self.time_steps_ps and time_ps < (self.time_steps_ps[-1] - 1e-12):
            raise ValueError(
                "Non-monotonic simulation time encountered in OpenMM reporter"
            )
        self.time_steps_ps.append(time_ps)

        # Potential energy (eV)
        pe_ev = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)) * _KJ_MOL_TO_EV
        if not math.isfinite(pe_ev):
            raise ValueError("Non-finite potential energy encountered in OpenMM reporter")
        self.energies_ev.append(pe_ev)

        # Forces (eV/A)
        forces = np.asarray(
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
            dtype=float,
        ) * _KJ_MOL_NM_TO_EV_ANG
        if not np.isfinite(forces).all():
            raise ValueError("Non-finite forces encountered in OpenMM reporter")
        if forces.shape != (self._n_sites, 3):
            raise ValueError(
                f"Forces shape mismatch: expected {(self._n_sites, 3)}, got {forces.shape}"
            )
        self.forces_ev_per_ang.append(np.array(forces, copy=True))

    def _validate_collected_frames(self) -> None:
        n_coords = len(self.coords)
        n_times = len(self.time_steps_ps)
        n_energies = len(self.energies_ev)
        n_forces = len(self.forces_ev_per_ang)
        if not (n_coords == n_times == n_energies == n_forces):
            raise RuntimeError(
                "Reporter frame buffers are inconsistent: "
                f"coords={n_coords}, times={n_times}, energies={n_energies}, forces={n_forces}"
            )
        if n_coords == 0:
            raise RuntimeError("No frames were collected; cannot build trajectory")
        diffs = np.diff(np.asarray(self.time_steps_ps, dtype=float))
        if not np.isfinite(diffs).all():
            raise RuntimeError("Reporter time axis contains non-finite deltas")
        if np.any(diffs < -1e-12):
            raise RuntimeError("Reporter time axis must be non-decreasing")

    def get_trajectory(self) -> Trajectory:
        """Convert gathered data into a pymatgen ``Trajectory`` object."""
        self._validate_collected_frames()

        frame_properties = [
            {
                "energy_ev": e,
                "forces_ev_per_ang": f,
                "time_ps": t,
            }
            for e, f, t in zip(
                self.energies_ev,
                self.forces_ev_per_ang,
                self.time_steps_ps,
                strict=True,
            )
        ]

        if len(self.time_steps_ps) >= 2:
            diffs = np.diff(np.asarray(self.time_steps_ps, dtype=float))
            finite_positive = diffs[np.isfinite(diffs) & (diffs > 0)]
            dt = float(np.median(finite_positive)) if finite_positive.size else 0.0
            time_step_fs = dt * _PS_TO_FS if math.isfinite(dt) and dt > 0 else 0.0
        else:
            time_step_fs = 0.0

        return Trajectory(
            species=self._structure.species,
            coords=self.coords,
            # pymatgen trajectory time_step is defined in femtoseconds.
            time_step=time_step_fs,
            lattice=self._structure.lattice,
            frame_properties=frame_properties,
        )
