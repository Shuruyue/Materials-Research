"""Tests for atlas.benchmark lazy exports."""

from __future__ import annotations

import types

import pytest

import atlas.benchmark as benchmark_pkg


def test_benchmark_dir_lists_expected_exports():
    names = dir(benchmark_pkg)
    for expected in (
        "MatbenchRunner",
        "FoldResult",
        "TaskReport",
        "compute_regression_metrics",
        "compute_uncertainty_metrics",
        "aggregate_fold_results",
    ):
        assert expected in names


def test_benchmark_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        benchmark_pkg.__getattr__("DefinitelyMissingSymbol")


def test_benchmark_getattr_caches_resolved_export(monkeypatch):
    marker = object()
    calls = {"count": 0}

    def _fake_import(name: str):
        assert name == "atlas.benchmark.runner"
        calls["count"] += 1
        return types.SimpleNamespace(MatbenchRunner=marker)

    monkeypatch.setattr(benchmark_pkg, "import_module", _fake_import)
    benchmark_pkg.__dict__.pop("MatbenchRunner", None)

    first = benchmark_pkg.__getattr__("MatbenchRunner")
    second = benchmark_pkg.__getattr__("MatbenchRunner")
    assert first is marker
    assert second is marker
    assert calls["count"] == 1


def test_benchmark_getattr_missing_export_raises(monkeypatch):
    def _fake_import(name: str):
        assert name == "atlas.benchmark.runner"
        return types.SimpleNamespace()

    monkeypatch.setattr(benchmark_pkg, "import_module", _fake_import)
    benchmark_pkg.__dict__.pop("FoldResult", None)

    with pytest.raises(AttributeError, match="does not define expected attribute"):
        benchmark_pkg.__getattr__("FoldResult")


def test_benchmark_lazy_import_error_is_recorded(monkeypatch):
    benchmark_pkg.__dict__.pop("BrokenBenchmarkExport", None)
    monkeypatch.setitem(
        benchmark_pkg._EXPORTS,  # type: ignore[attr-defined]
        "BrokenBenchmarkExport",
        ("atlas.__missing_benchmark_module_for_test__", "X"),
    )
    with pytest.raises(ImportError, match="Unable to import dependency"):
        benchmark_pkg.__getattr__("BrokenBenchmarkExport")
    errors = benchmark_pkg.get_import_errors()
    assert "BrokenBenchmarkExport" in errors
