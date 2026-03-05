"""Tests for atlas.thermo lazy exports."""

from __future__ import annotations

import pytest

import atlas.thermo as thermo


def test_thermo_dir_lists_expected_exports():
    names = dir(thermo)
    assert "CalphadCalculator" in names
    assert "PhaseStabilityAnalyst" in names
    assert "ReferenceDatabase" in names


def test_thermo_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(thermo, "DefinitelyMissingSymbol")


def test_thermo_optional_import_failure_returns_none(monkeypatch):
    def _fake_import(name: str):
        if name == "atlas.thermo.calphad":
            raise ModuleNotFoundError("synthetic missing optional dependency")
        raise AssertionError(f"Unexpected import target: {name}")

    monkeypatch.setattr(thermo.importlib, "import_module", _fake_import)
    thermo.__dict__.pop("CalphadCalculator", None)
    thermo._OPTIONAL_UNAVAILABLE.discard("CalphadCalculator")
    thermo._OPTIONAL_IMPORT_ERRORS.pop("CalphadCalculator", None)
    assert thermo.__getattr__("CalphadCalculator") is None
    assert "CalphadCalculator" in thermo.get_optional_import_errors()


def test_thermo_getattr_caches_resolved_export(monkeypatch):
    class _Dummy:
        pass

    dummy = _Dummy()
    calls = {"count": 0}

    def _fake_import(name: str):
        assert name == "atlas.thermo.stability"
        calls["count"] += 1
        return type("M", (), {"ReferenceDatabase": dummy})

    monkeypatch.setattr(thermo.importlib, "import_module", _fake_import)
    thermo.__dict__.pop("ReferenceDatabase", None)

    first = thermo.__getattr__("ReferenceDatabase")
    second = thermo.__getattr__("ReferenceDatabase")
    assert first is dummy
    assert second is dummy
    assert calls["count"] == 1


def test_thermo_missing_expected_attr_raises(monkeypatch):
    def _fake_import(name: str):
        assert name == "atlas.thermo.stability"
        return type("M", (), {})()

    monkeypatch.setattr(thermo.importlib, "import_module", _fake_import)
    thermo.__dict__.pop("PhaseStabilityAnalyst", None)

    with pytest.raises(AttributeError, match="does not define expected attribute"):
        thermo.__getattr__("PhaseStabilityAnalyst")
