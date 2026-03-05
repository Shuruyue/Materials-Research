"""Tests for atlas.discovery.stability lazy exports."""

from __future__ import annotations

import pytest

import atlas.discovery.stability as stability_pkg


def test_stability_dir_lists_expected_exports():
    names = dir(stability_pkg)
    assert "MEPINStabilityEvaluator" in names


def test_stability_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(stability_pkg, "NotARealSymbol")


def test_stability_optional_import_failure_returns_none(monkeypatch):
    def _fake_import(name: str):
        assert name == "atlas.discovery.stability.mepin"
        raise ModuleNotFoundError("synthetic mepin missing")

    monkeypatch.setattr(stability_pkg, "import_module", _fake_import)
    stability_pkg.__dict__.pop("MEPINStabilityEvaluator", None)
    stability_pkg._OPTIONAL_UNAVAILABLE.discard("MEPINStabilityEvaluator")
    stability_pkg._OPTIONAL_IMPORT_ERRORS.pop("MEPINStabilityEvaluator", None)
    assert stability_pkg.__getattr__("MEPINStabilityEvaluator") is None
    assert "MEPINStabilityEvaluator" in stability_pkg.get_optional_import_errors()
