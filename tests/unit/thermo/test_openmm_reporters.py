"""Focused tests for atlas.thermo.openmm.reporters."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure


def _install_fake_openmm_unit(monkeypatch):
    unit_mod = types.ModuleType("openmm.unit")
    unit_mod.angstrom = 1.0
    unit_mod.picosecond = 1.0
    unit_mod.kilojoule_per_mole = 1.0
    unit_mod.nanometer = 1.0

    openmm_mod = types.ModuleType("openmm")
    openmm_mod.unit = unit_mod

    monkeypatch.setitem(sys.modules, "openmm", openmm_mod)
    monkeypatch.setitem(sys.modules, "openmm.unit", unit_mod)


@dataclass
class _Quantity:
    value: object

    def value_in_unit(self, _unit):
        return self.value


class _State:
    def __init__(self, time_ps: float):
        self._time_ps = float(time_ps)

    def getPositions(self, asNumpy=True):  # noqa: ARG002,N802
        return _Quantity(np.array([[0.0, 0.0, 0.0]], dtype=float))

    def getTime(self):  # noqa: N802
        return _Quantity(self._time_ps)

    def getPotentialEnergy(self):  # noqa: N802
        return _Quantity(0.0)

    def getForces(self, asNumpy=True):  # noqa: ARG002,N802
        return _Quantity(np.array([[0.0, 0.0, 0.0]], dtype=float))


def test_reporter_converts_time_step_from_ps_to_fs(monkeypatch):
    _install_fake_openmm_unit(monkeypatch)
    reporters = importlib.reload(importlib.import_module("atlas.thermo.openmm.reporters"))

    reporter = reporters.PymatgenTrajectoryReporter(
        reportInterval=1,
        structure=Structure(Lattice.cubic(3.5), ["Si"], [[0.0, 0.0, 0.0]]),
    )
    reporter.report(simulation=None, state=_State(time_ps=0.000))
    reporter.report(simulation=None, state=_State(time_ps=0.002))

    trajectory = reporter.get_trajectory()
    assert trajectory.time_step == pytest.approx(2.0)


def test_reporter_rejects_non_monotonic_time(monkeypatch):
    _install_fake_openmm_unit(monkeypatch)
    reporters = importlib.reload(importlib.import_module("atlas.thermo.openmm.reporters"))

    reporter = reporters.PymatgenTrajectoryReporter(
        reportInterval=1,
        structure=Structure(Lattice.cubic(3.5), ["Si"], [[0.0, 0.0, 0.0]]),
    )
    reporter.report(simulation=None, state=_State(time_ps=1.0))

    with pytest.raises(ValueError, match="Non-monotonic simulation time"):
        reporter.report(simulation=None, state=_State(time_ps=0.5))


def test_reporter_rejects_non_integral_or_boolean_interval(monkeypatch):
    _install_fake_openmm_unit(monkeypatch)
    reporters = importlib.reload(importlib.import_module("atlas.thermo.openmm.reporters"))
    structure = Structure(Lattice.cubic(3.5), ["Si"], [[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="positive integer"):
        reporters.PymatgenTrajectoryReporter(reportInterval=1.5, structure=structure)
    with pytest.raises(ValueError, match="positive integer"):
        reporters.PymatgenTrajectoryReporter(reportInterval=True, structure=structure)


def test_reporter_parses_boolean_like_enforce_periodic_box(monkeypatch):
    _install_fake_openmm_unit(monkeypatch)
    reporters = importlib.reload(importlib.import_module("atlas.thermo.openmm.reporters"))
    structure = Structure(Lattice.cubic(3.5), ["Si"], [[0.0, 0.0, 0.0]])

    reporter = reporters.PymatgenTrajectoryReporter(
        reportInterval=1,
        structure=structure,
        enforcePeriodicBox="false",
    )
    desc = reporter.describeNextReport(simulation=types.SimpleNamespace(currentStep=0))
    assert desc[-1] is False

    with pytest.raises(ValueError, match="boolean-like"):
        reporters.PymatgenTrajectoryReporter(
            reportInterval=1,
            structure=structure,
            enforcePeriodicBox="not-bool",
        )
