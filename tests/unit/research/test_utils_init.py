"""Tests for atlas.utils lazy exports."""

from __future__ import annotations

import pytest

import atlas.utils as utils


def test_utils_lazy_exports_are_cached_after_first_access(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(utils.__dict__, "Registry", raising=False)

    first = utils.Registry
    second = utils.Registry

    assert first is second
    assert utils.__dict__["Registry"] is first


def test_utils_dir_includes_lazy_export_names():
    names = dir(utils)
    assert "collect_runtime_metadata" in names
    assert "pymatgen_to_ase" in names


def test_utils_getattr_raises_attribute_error_for_unknown_name():
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(utils, "__missing_symbol__")
