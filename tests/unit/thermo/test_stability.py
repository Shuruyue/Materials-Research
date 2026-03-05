"""Unit tests for atlas.thermo.stability."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import atlas.thermo.stability as stability_module
from atlas.thermo.stability import PhaseStabilityAnalyst, ReferenceDatabase


def test_get_entries_is_case_insensitive():
    db = ReferenceDatabase()
    db.entries = [
        SimpleNamespace(composition=SimpleNamespace(elements=["Sn", "Ag"])),
        SimpleNamespace(composition=SimpleNamespace(elements=["Sn", "Cu"])),
    ]
    entries = db.get_entries(["sn", "AG"])
    assert len(entries) == 1
    assert entries[0].composition.elements == ["Sn", "Ag"]


def test_get_entries_rejects_invalid_symbol_tokens():
    db = ReferenceDatabase()
    with pytest.raises(TypeError, match="element symbol must be a string"):
        db.get_entries(["Sn", True])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="invalid element symbol"):
        db.get_entries(["Sn", "Silver"])


def test_load_from_list_rejects_non_mapping_item():
    db = ReferenceDatabase()
    with pytest.raises(TypeError, match="mapping"):
        db.load_from_list([["Fe2O3", -5.0]])


def test_add_entry_rejects_boolean_and_non_string_inputs():
    db = ReferenceDatabase()
    with pytest.raises(TypeError, match="composition"):
        db.add_entry(True, -1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="energy_per_atom"):
        db.add_entry("Fe2O3", True)  # type: ignore[arg-type]


def test_analyze_stability_uses_reduced_formula_when_decomposition_empty(monkeypatch):
    class _DummyPD:
        def __init__(self, _entries):
            pass

        def get_e_above_hull(self, _entry):
            return 0.0

        def get_decomposition(self, _composition):
            return {}

        def get_form_energy_per_atom(self, _entry):
            return -0.5

    monkeypatch.setattr(stability_module, "PhaseDiagram", _DummyPD)
    analyst = PhaseStabilityAnalyst(ReferenceDatabase())
    result = analyst.analyze_stability("Fe2O3", -1.0)
    assert result["is_stable"] is True
    assert result["decomposition"] == "Fe2O3"
    assert result["formation_energy"] == pytest.approx(-0.5)


def test_analyze_stability_handles_non_finite_pd_outputs(monkeypatch):
    class _DummyPD:
        def __init__(self, _entries):
            pass

        def get_e_above_hull(self, _entry):
            return float("nan")

        def get_decomposition(self, _composition):
            return {}

        def get_form_energy_per_atom(self, _entry):
            return -0.5

    monkeypatch.setattr(stability_module, "PhaseDiagram", _DummyPD)
    analyst = PhaseStabilityAnalyst(ReferenceDatabase())
    result = analyst.analyze_stability("Fe2O3", -1.0)
    assert result["is_stable"] is False
    assert math.isnan(result["e_above_hull"])
    assert "error" in result


def test_analyze_stability_filters_non_finite_decomposition_amounts(monkeypatch):
    class _Entry:
        def __init__(self):
            self.composition = SimpleNamespace(reduced_formula="FeO")

    class _DummyPD:
        def __init__(self, _entries):
            pass

        def get_e_above_hull(self, _entry):
            return 0.01

        def get_decomposition(self, _composition):
            return {_Entry(): float("nan")}

        def get_form_energy_per_atom(self, _entry):
            return -0.2

    monkeypatch.setattr(stability_module, "PhaseDiagram", _DummyPD)
    analyst = PhaseStabilityAnalyst(ReferenceDatabase())
    result = analyst.analyze_stability("Fe2O3", -1.0)
    assert result["decomposition"] == "Fe2O3"
    assert result["is_stable"] is False


def test_analyze_stability_rejects_non_finite_target_energy():
    analyst = PhaseStabilityAnalyst(ReferenceDatabase())
    with pytest.raises(ValueError, match="target_energy_per_atom must be finite"):
        analyst.analyze_stability("Fe2O3", float("nan"))


def test_analyze_stability_rejects_invalid_target_formula_type():
    analyst = PhaseStabilityAnalyst(ReferenceDatabase())
    with pytest.raises(TypeError, match="target_formula"):
        analyst.analyze_stability(True, -1.0)  # type: ignore[arg-type]
