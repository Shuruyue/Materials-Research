"""Tests for atlas.training package lazy exports."""

from __future__ import annotations

import types

import pytest

import atlas.training as training


def test_training_dir_lists_expected_symbols():
    names = dir(training)
    assert "Trainer" in names
    assert "MultiTaskLoss" in names
    assert "CheckpointManager" in names


def test_training_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        getattr(training, "DefinitelyMissingSymbol")


def test_training_lazy_exports_are_cached():
    training.__dict__.pop("filter_outliers", None)
    first = training.filter_outliers
    second = training.filter_outliers
    assert first is second


def test_training_all_matches_expected_surface():
    assert "Trainer" in training.__all__
    assert "CheckpointManager" in training.__all__
    assert len(training.__all__) == len(set(training.__all__))


def test_training_export_mismatch_raises_helpful_error(monkeypatch):
    training.__dict__.pop("CheckpointManager", None)
    monkeypatch.setattr(training.importlib, "import_module", lambda _name: types.SimpleNamespace())
    with pytest.raises(AttributeError, match="does not define expected export"):
        getattr(training, "CheckpointManager")


def test_training_lazy_import_error_is_recorded(monkeypatch):
    training.__dict__.pop("BrokenTrainingExport", None)
    monkeypatch.setattr(
        training,
        "_LAZY_EXPORTS",
        {
            **dict(training._LAZY_EXPORTS),  # type: ignore[attr-defined]
            "BrokenTrainingExport": ("atlas.__missing_training_module_for_test__", "X"),
        },
    )
    with pytest.raises(ImportError, match="Unable to import dependency"):
        getattr(training, "BrokenTrainingExport")
    errors = training.get_import_errors()
    assert "BrokenTrainingExport" in errors
