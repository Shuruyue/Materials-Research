"""Tests for atlas.research lazy exports."""

from __future__ import annotations

import pytest

import atlas.research as research


def test_lazy_exports_are_cached_after_first_access(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(research.__dict__, "MethodSpec", raising=False)

    first = research.MethodSpec
    second = research.MethodSpec

    assert first is second
    assert research.__dict__["MethodSpec"] is first


def test_dir_includes_lazy_export_names():
    names = dir(research)
    assert "MethodSpec" in names
    assert "WorkflowReproducibleGraph" in names


def test_getattr_raises_attribute_error_for_unknown_name():
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(research, "__unknown_export__")


def test_lazy_import_error_is_recorded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        research._LAZY_EXPORTS,  # type: ignore[attr-defined]
        "BrokenSymbol",
        ("atlas.__missing_module_for_test__", "X"),
    )
    monkeypatch.delitem(research.__dict__, "BrokenSymbol", raising=False)
    with pytest.raises(ImportError, match="Unable to import dependency"):
        getattr(research, "BrokenSymbol")
    errors = research.get_import_errors()
    assert "BrokenSymbol" in errors
