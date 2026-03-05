"""Tests for atlas.potentials lazy exports."""

from __future__ import annotations

import types

import pytest

import atlas.potentials as potentials


def test_potentials_dir_lists_expected_exports():
    names = dir(potentials)
    assert "MACERelaxer" in names
    assert "NativeMlipArenaRelaxer" in names


def test_potentials_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(potentials, "DefinitelyMissingSymbol")


def test_potentials_optional_import_failure_returns_none(monkeypatch):
    def _fake_import(name: str):
        if name == "atlas.potentials.mace_relaxer":
            raise ModuleNotFoundError("synthetic missing optional dependency")
        raise AssertionError(f"Unexpected import target: {name}")

    monkeypatch.setattr(potentials.importlib, "import_module", _fake_import)
    potentials.__dict__.pop("MACERelaxer", None)
    potentials._OPTIONAL_UNAVAILABLE.discard("MACERelaxer")
    potentials._OPTIONAL_IMPORT_ERRORS.pop("MACERelaxer", None)
    assert potentials.__getattr__("MACERelaxer") is None
    assert "MACERelaxer" in potentials.get_optional_import_errors()


def test_potentials_missing_expected_attr_raises(monkeypatch):
    def _fake_import(name: str):
        assert name == "atlas.potentials.mace_relaxer"
        return types.SimpleNamespace()

    monkeypatch.setattr(potentials.importlib, "import_module", _fake_import)
    potentials.__dict__.pop("MACERelaxer", None)
    potentials._OPTIONAL_UNAVAILABLE.discard("MACERelaxer")
    potentials._OPTIONAL_IMPORT_ERRORS.pop("MACERelaxer", None)
    with pytest.raises(AttributeError, match="does not define expected attribute"):
        potentials.__getattr__("MACERelaxer")
