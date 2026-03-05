"""Tests for atlas.discovery.transport.liflow."""

from __future__ import annotations

import ase
import numpy as np
import pytest

import atlas.discovery.transport.liflow as liflow


def test_temperature_and_step_normalizers():
    assert liflow._normalize_temperature_list([600, 800]) == [600, 800]
    with pytest.raises(ValueError, match="temp_list\\[0\\]"):
        liflow._normalize_temperature_list([True])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="temp_list\\[0\\]"):
        liflow._normalize_temperature_list([600.5])

    assert liflow._coerce_positive_int(3, name="steps") == 3
    with pytest.raises(ValueError, match="steps"):
        liflow._coerce_positive_int(True, name="steps")
    with pytest.raises(ValueError, match="steps"):
        liflow._coerce_positive_int(1.2, name="steps")


def test_load_element_index_fallback_and_validation(monkeypatch, tmp_path):
    evaluator = object.__new__(liflow.LiFlowEvaluator)
    fallback = evaluator._load_element_index(path=None)
    assert fallback.ndim == 1
    assert fallback.size >= 100

    empty = tmp_path / "element_index.npy"
    np.save(str(empty), np.array([], dtype=int))
    with pytest.raises(ValueError, match="non-empty"):
        evaluator._load_element_index(path=str(empty))


def test_simulate_rejects_invalid_atomic_number_mapping(monkeypatch):
    class _FakeSimulator:
        def __init__(self, **kwargs):  # noqa: D401,ARG002
            pass

        def run(self, **kwargs):  # noqa: ARG002
            return []

    monkeypatch.setattr(
        liflow,
        "_LIFLOW_API",
        (object(), _FakeSimulator, lambda *_args, **_kwargs: object()),
    )
    evaluator = object.__new__(liflow.LiFlowEvaluator)
    evaluator.temp_list = [600]
    evaluator.model = object()
    evaluator.prior = object()
    evaluator.element_idx = np.arange(3, dtype=int)

    atoms = ase.Atoms("Si", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="cannot index atomic number"):
        evaluator.simulate(atoms, steps=1, flow_steps=1)
