"""Tests for atlas.thermo.openmm lazy exports and fallback stack."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import ase
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

import atlas.thermo.openmm as openmm_pkg


def _install_fake_openmm(monkeypatch):
    unit_mod = types.ModuleType("openmm.unit")
    unit_mod.angstrom = 1.0
    unit_mod.picosecond = 1.0
    unit_mod.nanometer = 1.0
    unit_mod.kilojoule_per_mole = 1.0
    unit_mod.kilocalories_per_mole = 1.0
    unit_mod.kelvin = 1.0
    unit_mod.femtoseconds = 1.0
    unit_mod.dalton = 1.0

    class NonbondedForce:
        CutoffPeriodic = 1
        NoCutoff = 0

        def __init__(self):
            self.method = None
            self.cutoff = None
            self.particles = []

        def setNonbondedMethod(self, method):
            self.method = method

        def setCutoffDistance(self, cutoff):
            self.cutoff = float(cutoff)

        def addParticle(self, charge, sigma, epsilon):
            self.particles.append((charge, sigma, epsilon))

    class System:
        def __init__(self):
            self.particles = []
            self.forces = []
            self.box = None

        def addParticle(self, mass):
            self.particles.append(float(mass))

        def addForce(self, force):
            self.forces.append(force)

        def setDefaultPeriodicBoxVectors(self, *vecs):
            self.box = vecs

        def getNumParticles(self):
            return len(self.particles)

    class Vec3(tuple):
        def __new__(cls, x, y, z):
            return super().__new__(cls, (float(x), float(y), float(z)))

    class LangevinMiddleIntegrator:
        def __init__(self, *args, **kwargs):  # noqa: D401,ARG002
            pass

    openmm_mod = types.ModuleType("openmm")
    openmm_mod.NonbondedForce = NonbondedForce
    openmm_mod.System = System
    openmm_mod.Vec3 = Vec3
    openmm_mod.LangevinMiddleIntegrator = LangevinMiddleIntegrator
    openmm_mod.Integrator = object
    openmm_mod.unit = unit_mod

    app_mod = types.ModuleType("openmm.app")

    class Topology:
        def __init__(self):
            self.vectors = None

        def addChain(self):
            return object()

        def addResidue(self, _name, _chain):
            return object()

        def addAtom(self, _name, _element, _residue):
            return object()

        def setPeriodicBoxVectors(self, *vecs):
            self.vectors = vecs

        def setUnitCellDimensions(self, _dims):
            self.vectors = "legacy"

    class Element:
        @staticmethod
        def getBySymbol(symbol):
            return symbol

    class Simulation:
        def __init__(self, _topology, _system, _integrator):
            self.reporters = []
            self.currentStep = 0

    app_mod.Topology = Topology
    app_mod.Element = Element
    app_mod.Simulation = Simulation

    monkeypatch.setitem(sys.modules, "openmm", openmm_mod)
    monkeypatch.setitem(sys.modules, "openmm.unit", unit_mod)
    monkeypatch.setitem(sys.modules, "openmm.app", app_mod)
    return openmm_mod, app_mod


def test_openmm_lazy_export_cache(monkeypatch):
    marker = object()
    calls = {"count": 0}

    def _fake_import(name: str):
        assert name == "atlas.thermo.openmm.engine"
        calls["count"] += 1
        return types.SimpleNamespace(OpenMMEngine=marker)

    monkeypatch.setattr(openmm_pkg.importlib, "import_module", _fake_import)
    openmm_pkg.__dict__.pop("OpenMMEngine", None)

    first = openmm_pkg.__getattr__("OpenMMEngine")
    second = openmm_pkg.__getattr__("OpenMMEngine")
    assert first is marker
    assert second is marker
    assert calls["count"] == 1


def test_openmm_lazy_export_missing_attr_raises(monkeypatch):
    def _fake_import(name: str):
        assert name == "atlas.thermo.openmm.reporters"
        return types.SimpleNamespace()

    monkeypatch.setattr(openmm_pkg.importlib, "import_module", _fake_import)
    openmm_pkg.__dict__.pop("PymatgenTrajectoryReporter", None)
    with pytest.raises(AttributeError, match="does not define expected attribute"):
        openmm_pkg.__getattr__("PymatgenTrajectoryReporter")


def test_openmm_lazy_export_runtime_error_not_suppressed(monkeypatch):
    def _fake_import(name: str):
        assert name == "atlas.thermo.openmm.engine"
        raise RuntimeError("broken module import")

    monkeypatch.setattr(openmm_pkg.importlib, "import_module", _fake_import)
    openmm_pkg.__dict__.pop("OpenMMEngine", None)
    with pytest.raises(RuntimeError, match="broken module import"):
        openmm_pkg.__getattr__("OpenMMEngine")


def test_openmm_optional_import_failure_returns_none_and_records_error(monkeypatch):
    def _fake_import(name: str):
        assert name == "atlas.thermo.openmm.engine"
        raise ModuleNotFoundError("synthetic missing openmm dependency")

    monkeypatch.setattr(openmm_pkg.importlib, "import_module", _fake_import)
    openmm_pkg.__dict__.pop("OpenMMEngine", None)
    openmm_pkg._OPTIONAL_UNAVAILABLE.discard("OpenMMEngine")
    openmm_pkg._OPTIONAL_IMPORT_ERRORS.pop("OpenMMEngine", None)
    assert openmm_pkg.__getattr__("OpenMMEngine") is None
    assert "OpenMMEngine" in openmm_pkg.get_optional_import_errors()


@dataclass
class _Quantity:
    value: object

    def value_in_unit(self, _unit):
        return self.value


class _StateShapeMismatch:
    def getPositions(self, asNumpy=True):  # noqa: ARG002
        return _Quantity(np.zeros((2, 3), dtype=float))


def test_reporter_rejects_shape_mismatch(monkeypatch):
    _install_fake_openmm(monkeypatch)
    reporters = importlib.reload(importlib.import_module("atlas.thermo.openmm.reporters"))
    reporter = reporters.PymatgenTrajectoryReporter(
        reportInterval=10,
        structure=Structure(Lattice.cubic(3.5), ["Si"], [[0.0, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="Positions shape mismatch"):
        reporter.report(simulation=None, state=_StateShapeMismatch())


def test_reporter_detects_inconsistent_frame_buffers(monkeypatch):
    _install_fake_openmm(monkeypatch)
    reporters = importlib.reload(importlib.import_module("atlas.thermo.openmm.reporters"))
    reporter = reporters.PymatgenTrajectoryReporter(
        reportInterval=10,
        structure=Structure(Lattice.cubic(3.5), ["Si"], [[0.0, 0.0, 0.0]]),
    )
    reporter.coords.append(np.zeros((1, 3), dtype=float))
    reporter.time_steps_ps.append(0.0)
    reporter.energies_ev.append(0.0)
    with pytest.raises(RuntimeError, match="inconsistent"):
        reporter.get_trajectory()


def test_engine_periodic_guards_and_cutoff(monkeypatch):
    fake_openmm, fake_app = _install_fake_openmm(monkeypatch)
    engine_module = importlib.reload(importlib.import_module("atlas.thermo.openmm.engine"))
    engine = engine_module.OpenMMEngine()

    atoms_partial = ase.Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=[True, False, True])
    with pytest.raises(ValueError, match="fully periodic"):
        engine._set_periodic_box(fake_app.Topology(), atoms_partial)

    atoms_periodic = ase.Atoms("Ar", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)
    system = fake_openmm.System()
    engine._add_lj_force(system, atoms_periodic)
    assert len(system.forces) == 1
    force = system.forces[0]
    assert force.method == fake_openmm.NonbondedForce.CutoffPeriodic
    assert force.cutoff < 2.0


def test_engine_run_rejects_non_integral_step_values(monkeypatch):
    _install_fake_openmm(monkeypatch)
    engine_module = importlib.reload(importlib.import_module("atlas.thermo.openmm.engine"))

    class _Reporter:
        def __init__(self, reportInterval, structure):  # noqa: N803,ARG002
            self._trajectory = {"ok": True}

        def get_trajectory(self):
            return self._trajectory

    class _EnergyState:
        def getPotentialEnergy(self):
            return _Quantity(0.0)

    class _Context:
        def getState(self, getEnergy=True):  # noqa: ARG002,N802
            return _EnergyState()

    class _Simulation:
        def __init__(self):
            self.reporters = []
            self.context = _Context()

        def step(self, _n_steps):
            return None

    monkeypatch.setattr(engine_module, "PymatgenTrajectoryReporter", _Reporter)

    engine = engine_module.OpenMMEngine()
    engine.atoms = ase.Atoms("Ar", positions=[[0.0, 0.0, 0.0]])
    engine.simulation = _Simulation()

    with pytest.raises(ValueError, match="positive integer"):
        engine.run(steps=10.5, trajectory_interval=1)
    with pytest.raises(ValueError, match="positive integer"):
        engine.run(steps=10, trajectory_interval=2.2)
