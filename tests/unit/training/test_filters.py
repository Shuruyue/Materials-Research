"""Unit tests for outlier filtering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import torch

from atlas.training.filters import filter_outliers


@dataclass
class _Item:
    jid: str
    target: object | None = None


class _Dataset:
    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_filter_outliers_zscore_and_csv_export(tmp_path: Path):
    dataset = _Dataset(
        [
            _Item("a", torch.tensor(1.0)),
            _Item("b", torch.tensor(1.1)),
            _Item("c", torch.tensor(0.9)),
            _Item("d", torch.tensor(9.5)),
        ]
    )
    subset = filter_outliers(dataset, properties=["target"], save_dir=tmp_path, n_sigma=1.5)
    assert sorted(subset.indices) == [0, 1, 2]

    outlier_path = tmp_path / "outliers.csv"
    assert outlier_path.exists()
    frame = pd.read_csv(outlier_path)
    assert set(frame.columns) >= {"jid", "property", "value", "sigma", "method"}
    assert frame.loc[0, "jid"] == "d"


def test_filter_outliers_handles_non_finite_and_missing_values():
    dataset = _Dataset(
        [
            _Item("a", torch.tensor(1.0)),
            _Item("b", torch.tensor(float("nan"))),
            _Item("c", torch.tensor(float("inf"))),
            _Item("d", None),
            _Item("e", "oops"),
        ]
    )
    subset = filter_outliers(dataset, properties=["target"], n_sigma=3.0)
    assert sorted(subset.indices) == [0, 1, 2, 3, 4]


def test_filter_outliers_ignores_boolean_like_scalar_payloads():
    dataset = _Dataset(
        [
            _Item("a", True),
            _Item("b", torch.tensor(False)),
            _Item("c", torch.tensor(1.0)),
            _Item("d", torch.tensor(1.1)),
        ]
    )
    subset = filter_outliers(dataset, properties=["target"], n_sigma=2.0)
    assert sorted(subset.indices) == [0, 1, 2, 3]


def test_filter_outliers_modified_zscore_detects_extreme_values():
    dataset = _Dataset(
        [
            _Item("a", torch.tensor(0.0)),
            _Item("b", torch.tensor(0.0)),
            _Item("c", torch.tensor(1.0)),
            _Item("d", torch.tensor(10.0)),
        ]
    )
    subset = filter_outliers(dataset, properties=["target"], n_sigma=3.5, method="modified_zscore")
    assert 3 not in subset.indices
    assert sorted(subset.indices) == [0, 1, 2]


def test_filter_outliers_validates_arguments(tmp_path: Path):
    dataset = _Dataset([_Item("a", torch.tensor(1.0))])
    with pytest.raises(ValueError, match="n_sigma"):
        filter_outliers(dataset, properties=["target"], n_sigma=0.0)
    with pytest.raises(ValueError, match="Unsupported method"):
        filter_outliers(dataset, properties=["target"], method="iqr")
    with pytest.raises(TypeError, match="sequence of property names"):
        filter_outliers(dataset, properties="target")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string property names"):
        filter_outliers(dataset, properties=["target", 1])  # type: ignore[list-item]

    subset = filter_outliers(dataset, properties=[], save_dir=tmp_path)
    assert sorted(subset.indices) == [0]


def test_filter_outliers_normalizes_property_names_and_handles_empty_dataset():
    empty = _Dataset([])
    subset = filter_outliers(empty, properties=[" target ", "target"])
    assert subset.indices == []
