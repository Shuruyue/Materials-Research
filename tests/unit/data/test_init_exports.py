"""Tests for atlas.data lazy exports."""

from __future__ import annotations

import pytest

import atlas.data as data_pkg


def test_data_lazy_exports_are_cached_after_first_access(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(data_pkg.__dict__, "DataSourceRegistry", raising=False)
    first = data_pkg.DataSourceRegistry
    second = data_pkg.DataSourceRegistry
    assert first is second
    assert data_pkg.__dict__["DataSourceRegistry"] is first


def test_data_dir_includes_lazy_export_names():
    names = dir(data_pkg)
    assert "TopoDB" in names
    assert "validate_dataset" in names


def test_data_getattr_raises_attribute_error_for_unknown_name():
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(data_pkg, "__missing_export__")


def test_data_export_mismatch_raises_helpful_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(data_pkg.__dict__, "TopoDB", raising=False)

    class _ModuleWithoutAttr:
        pass

    monkeypatch.setattr(data_pkg, "import_module", lambda _name: _ModuleWithoutAttr())
    with pytest.raises(AttributeError, match="does not define expected attribute"):
        getattr(data_pkg, "TopoDB")


def test_data_lazy_import_error_is_recorded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        data_pkg._EXPORTS,  # type: ignore[attr-defined]
        "BrokenDataExport",
        ("atlas.__missing_data_module_for_test__", "X"),
    )
    monkeypatch.delitem(data_pkg.__dict__, "BrokenDataExport", raising=False)
    with pytest.raises(ImportError, match="Unable to import dependency"):
        getattr(data_pkg, "BrokenDataExport")
    errors = data_pkg.get_import_errors()
    assert "BrokenDataExport" in errors
