"""Tests for atlas.discovery.alchemy lazy exports."""

from __future__ import annotations

import pytest

import atlas.discovery.alchemy as alchemy


def test_alchemy_dir_lists_expected_exports():
    names = dir(alchemy)
    assert "AlchemicalModel" in names
    assert "AlchemyManager" in names
    assert "AlchemicalMACECalculator" in names


def test_alchemy_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(alchemy, "DefinitelyMissingSymbol")


def test_alchemy_optional_calculator_import_failure_returns_placeholder(monkeypatch):
    def _fake_import(name: str):
        if name == "atlas.discovery.alchemy.calculator":
            raise ModuleNotFoundError("synthetic missing dependency")
        raise AssertionError(f"Unexpected import target: {name}")

    monkeypatch.setattr(alchemy, "import_module", _fake_import)
    alchemy.__dict__.pop("AlchemicalMACECalculator", None)
    alchemy._OPTIONAL_UNAVAILABLE.discard("AlchemicalMACECalculator")
    alchemy._OPTIONAL_IMPORT_ERRORS.pop("AlchemicalMACECalculator", None)

    placeholder = alchemy.__getattr__("AlchemicalMACECalculator")
    with pytest.raises(ImportError, match="optional alchemical dependencies"):
        placeholder()
    assert "AlchemicalMACECalculator" in alchemy.get_optional_import_errors()


def test_alchemy_non_import_runtime_error_is_not_suppressed(monkeypatch):
    def _fake_import(name: str):
        assert name == "atlas.discovery.alchemy.model"
        raise RuntimeError("synthetic runtime import failure")

    monkeypatch.setattr(alchemy, "import_module", _fake_import)
    alchemy.__dict__.pop("AlchemicalModel", None)
    alchemy._OPTIONAL_UNAVAILABLE.discard("AlchemicalModel")
    alchemy._OPTIONAL_IMPORT_ERRORS.pop("AlchemicalModel", None)
    with pytest.raises(RuntimeError, match="synthetic runtime import failure"):
        alchemy.__getattr__("AlchemicalModel")
