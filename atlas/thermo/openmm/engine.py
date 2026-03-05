from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import ase
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

from atlas.thermo.openmm.reporters import PymatgenTrajectoryReporter
from atlas.utils.registry import RELAXERS
from atlas.utils.structure import ase_to_pymatgen

logger = logging.getLogger(__name__)

_KJ_MOL_PER_EV = 96.48533212331002


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _is_periodic(atoms: ase.Atoms) -> bool:
    pbc = np.asarray(atoms.pbc, dtype=bool).reshape(-1)
    return bool(np.any(pbc))


def _coerce_positive_int(value: Any, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")

    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)):
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        number = int(number_f)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc

    if number <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return number


@RELAXERS.register("atomate2_openmm")
class OpenMMEngine:
    """
    OpenMM wrapper for molecular dynamics simulations.

    Supports MACE via `openmm-ml` (`forcefield_path='mace'`) and a built-in LJ fallback.
    """

    def __init__(
        self,
        temperature: float = 300.0,
        friction: float = 1.0,
        step_size: float = 1.0,
    ):
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
        if not math.isfinite(float(friction)) or float(friction) < 0:
            raise ValueError(f"friction must be finite and >= 0, got {friction!r}")
        if not math.isfinite(float(step_size)) or float(step_size) <= 0:
            raise ValueError(f"step_size must be finite and > 0, got {step_size!r}")

        self.temperature = float(temperature) * unit.kelvin
        self.friction = float(friction) / unit.picosecond
        self.step_size = float(step_size) * unit.femtoseconds

        self.atoms: ase.Atoms | None = None
        self.system: openmm.System | None = None
        self.integrator: openmm.Integrator | None = None
        self.simulation: app.Simulation | None = None

    def _set_periodic_box(self, topology: app.Topology, atoms: ase.Atoms) -> tuple[Any, Any, Any] | None:
        pbc = np.asarray(atoms.pbc, dtype=bool).reshape(-1)
        if not bool(np.any(pbc)):
            return None
        if not bool(np.all(pbc)):
            raise ValueError(
                "OpenMM engine requires either fully periodic (x/y/z) or non-periodic boundary conditions"
            )

        cell = np.asarray(atoms.get_cell(), dtype=float)
        if cell.shape != (3, 3):
            raise ValueError(f"Expected 3x3 cell matrix for periodic system, got {cell.shape}")

        vecs_nm = tuple(openmm.Vec3(*map(float, row / 10.0)) for row in cell)

        if hasattr(topology, "setPeriodicBoxVectors"):
            topology.setPeriodicBoxVectors(vecs_nm)
        else:  # pragma: no cover - legacy OpenMM compatibility
            lengths_nm = np.asarray(atoms.get_cell().lengths(), dtype=float) / 10.0
            topology.setUnitCellDimensions(unit.Quantity(lengths_nm.tolist(), unit.nanometer))

        return vecs_nm

    def _build_system(self, topology: app.Topology, atoms: ase.Atoms, forcefield_path: str | None) -> openmm.System:
        ff_name = str(forcefield_path).strip().lower() if forcefield_path is not None else ""

        if ff_name == "mace":
            try:
                from openmmml import MLPotential

                potential = MLPotential("mace-mpa-0-medium")
                system = potential.createSystem(topology)
                logger.info("Attached MACE potential via openmm-ml.")
                return system
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("Failed to initialize MACE potential. Falling back to LJ: %s", exc)

        if forcefield_path and ff_name not in {"mace", "lj"}:
            path = Path(forcefield_path).expanduser()
            if not path.exists():
                logger.warning("forcefield_path does not exist (%s). Falling back to LJ.", path)
            else:
                logger.warning(
                    "Custom forcefield files are currently unsupported (%s). Falling back to LJ.",
                    path,
                )

        system = openmm.System()
        masses = np.asarray(atoms.get_masses(), dtype=float)
        if masses.size != len(atoms):
            raise ValueError("Failed to resolve ASE atomic masses")
        for mass in masses:
            if not math.isfinite(float(mass)) or float(mass) <= 0:
                raise ValueError(f"Invalid atomic mass: {mass!r}")
            system.addParticle(float(mass) * unit.dalton)

        self._add_lj_force(system, atoms)
        return system

    def setup_system(self, atoms: ase.Atoms, forcefield_path: str | None = None) -> None:
        """Setup OpenMM System from ASE Atoms."""
        if atoms is None:
            raise ValueError("atoms must not be None")
        if len(atoms) == 0:
            raise ValueError("atoms must contain at least one site")

        self.atoms = atoms.copy()

        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("RES", chain)

        vecs_nm = self._set_periodic_box(topology, self.atoms)

        for atom in self.atoms:
            try:
                element = app.Element.getBySymbol(str(atom.symbol))
            except Exception as exc:
                raise ValueError(f"Unsupported element symbol for OpenMM topology: {atom.symbol!r}") from exc
            topology.addAtom(str(atom.symbol), element, residue)

        self.system = self._build_system(topology, self.atoms, forcefield_path)

        if vecs_nm is not None:
            self.system.setDefaultPeriodicBoxVectors(*vecs_nm)

        self.integrator = openmm.LangevinMiddleIntegrator(
            self.temperature,
            self.friction,
            self.step_size,
        )
        self.simulation = app.Simulation(topology, self.system, self.integrator)

        positions = np.asarray(self.atoms.get_positions(), dtype=float)
        if not np.isfinite(positions).all():
            raise ValueError("Non-finite atom coordinates provided")
        self.simulation.context.setPositions(positions * unit.angstrom)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

        logger.info("OpenMM system configured. Particles: %d", self.system.getNumParticles())

    def _add_lj_force(self, system: openmm.System, atoms: ase.Atoms) -> None:
        """Add simple Lennard-Jones force for fallback/testing."""
        force = openmm.NonbondedForce()
        use_periodic = _is_periodic(atoms)
        if use_periodic:
            force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
            if lengths.size != 3 or not np.isfinite(lengths).all():
                raise ValueError("Periodic system requires finite 3-vector cell lengths")
            min_box = float(np.min(lengths))
            if min_box <= 0:
                raise ValueError(f"Invalid periodic cell lengths: {lengths.tolist()!r}")

            half_box = 0.5 * min_box
            if half_box <= 0.25:
                raise ValueError(
                    "Periodic box too small for stable LJ cutoff; increase lattice vectors or use non-periodic setup"
                )
            # Keep cutoff strictly below half min box to satisfy minimum image convention.
            cutoff_angstrom = min(5.0, max(0.5, 0.95 * half_box))
            cutoff_angstrom = min(cutoff_angstrom, half_box - 1e-3)
            force.setCutoffDistance(cutoff_angstrom * unit.angstrom)
        else:
            force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

        for _ in range(len(atoms)):
            force.addParticle(0.0, 3.4 * unit.angstrom, 0.2 * unit.kilocalories_per_mole)

        system.addForce(force)

    def run(self, steps: int, trajectory_interval: int = 100) -> Any:
        """Run simulation for N steps and return a pymatgen Trajectory."""
        if self.simulation is None or self.atoms is None:
            raise RuntimeError("Simulation not initialized. Call setup_system first.")

        n_steps = _coerce_positive_int(steps, "steps")

        interval = _coerce_positive_int(trajectory_interval, "trajectory_interval")
        interval = min(interval, n_steps)

        logger.info("Running OpenMM simulation for %d steps (interval=%d)", n_steps, interval)

        reporter = PymatgenTrajectoryReporter(
            reportInterval=interval,
            structure=ase_to_pymatgen(self.atoms),
        )

        self.simulation.reporters.append(reporter)
        try:
            self.simulation.step(n_steps)
        finally:
            with np.errstate(all="ignore"):
                if self.simulation.reporters and self.simulation.reporters[-1] is reporter:
                    self.simulation.reporters.pop()
                else:
                    self.simulation.reporters = [r for r in self.simulation.reporters if r is not reporter]

        state = self.simulation.context.getState(getEnergy=True)
        pe_kj = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        if not math.isfinite(pe_kj):
            raise RuntimeError("Non-finite potential energy after simulation")
        pot_energy_ev = pe_kj / _KJ_MOL_PER_EV
        logger.info("Simulation completed. Final potential energy: %.4f eV", pot_energy_ev)

        return reporter.get_trajectory()
