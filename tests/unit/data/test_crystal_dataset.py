"""Unit tests for atlas.data.crystal_dataset."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import atlas.data as data_pkg
from atlas.data import crystal_dataset as dataset_mod
from atlas.data.split_governance import compositional_split, prototype_split


def _patch_runtime(monkeypatch, tmp_path, df: pd.DataFrame):
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            processed_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
        )
    )
    monkeypatch.setattr(dataset_mod, "get_config", lambda: cfg)
    monkeypatch.delenv("ATLAS_SPLIT_MANIFEST", raising=False)

    class _FakeClient:
        def load_dft_3d(self):
            return df.copy()

    monkeypatch.setattr(data_pkg, "JARVISClient", _FakeClient)
    monkeypatch.setattr(
        dataset_mod,
        "_worker_process_row",
        lambda row_data, properties, rev_map, *args: SimpleNamespace(jid=row_data.get("jid", "unknown")),
    )


def test_min_labeled_properties_filters_rows(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {
                "jid": "J1",
                "atoms": {"dummy": 1},
                "formula": "Li2O",
                "ehull": 0.01,
                "formation_energy_peratom": -1.0,
                "optb88vdw_bandgap": "na",
            },
            {
                "jid": "J2",
                "atoms": {"dummy": 1},
                "formula": "LiFeO2",
                "ehull": 0.01,
                "formation_energy_peratom": -1.1,
                "optb88vdw_bandgap": 1.2,
            },
            {
                "jid": "J3",
                "atoms": {"dummy": 1},
                "formula": "Fe2O3",
                "ehull": 0.01,
                "formation_energy_peratom": "na",
                "optb88vdw_bandgap": 2.1,
            },
        ]
    )
    _patch_runtime(monkeypatch, tmp_path, df)

    ds = dataset_mod.CrystalPropertyDataset(
        properties=["formation_energy", "band_gap"],
        min_labeled_properties=2,
        stability_filter=None,
        split="train",
        split_ratio=(1.0, 0.0, 0.0),
    ).prepare(force_reload=True)

    assert list(ds._df["jid"]) == ["J2"]


def test_fallback_split_strategy_compositional_matches_governance(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"jid": "J1", "atoms": {"dummy": 1}, "formula": "Li2O", "spg_number": 225, "ehull": 0.02, "formation_energy_peratom": -1.0},
            {"jid": "J2", "atoms": {"dummy": 1}, "formula": "Li2O", "spg_number": 225, "ehull": 0.03, "formation_energy_peratom": -1.1},
            {"jid": "J3", "atoms": {"dummy": 1}, "formula": "Fe2O3", "spg_number": 167, "ehull": 0.01, "formation_energy_peratom": -1.2},
            {"jid": "J4", "atoms": {"dummy": 1}, "formula": "Fe2O3", "spg_number": 167, "ehull": 0.04, "formation_energy_peratom": -1.3},
            {"jid": "J5", "atoms": {"dummy": 1}, "formula": "NaCl", "spg_number": 225, "ehull": 0.05, "formation_energy_peratom": -0.9},
            {"jid": "J6", "atoms": {"dummy": 1}, "formula": "NaCl", "spg_number": 225, "ehull": 0.06, "formation_energy_peratom": -1.0},
        ]
    )
    _patch_runtime(monkeypatch, tmp_path, df)

    ratio = (0.5, 0.0, 0.5)
    ds = dataset_mod.CrystalPropertyDataset(
        properties=["formation_energy"],
        stability_filter=None,
        split="test",
        split_ratio=ratio,
        fallback_split_strategy="compositional",
        split_manifest_path=None,
    ).prepare(force_reload=True)

    expected = compositional_split(
        [str(x) for x in df["jid"].tolist()],
        [str(x) for x in df["formula"].tolist()],
        seed=42,
        ratios=ratio,
    )["test"]
    assert sorted(ds._df["jid"].tolist()) == sorted(expected)


def test_kcenter_formula_sampling_uses_coreset_indices(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"jid": "J1", "atoms": {"dummy": 1}, "formula": "A", "ehull": 0.01, "formation_energy_peratom": -1.0},
            {"jid": "J2", "atoms": {"dummy": 1}, "formula": "B", "ehull": 0.01, "formation_energy_peratom": -1.0},
            {"jid": "J3", "atoms": {"dummy": 1}, "formula": "C", "ehull": 0.01, "formation_energy_peratom": -1.0},
            {"jid": "J4", "atoms": {"dummy": 1}, "formula": "D", "ehull": 0.01, "formation_energy_peratom": -1.0},
            {"jid": "J5", "atoms": {"dummy": 1}, "formula": "E", "ehull": 0.01, "formation_energy_peratom": -1.0},
        ]
    )
    _patch_runtime(monkeypatch, tmp_path, df)

    fake_vectors = {
        "A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "C": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "D": np.array([0.5, 0.5, 0.0], dtype=np.float32),
        "E": np.array([0.0, 0.5, 0.5], dtype=np.float32),
    }
    monkeypatch.setattr(
        dataset_mod,
        "_formula_fraction_vector",
        lambda formula: fake_vectors[formula],
    )

    k = 3
    ds = dataset_mod.CrystalPropertyDataset(
        properties=["formation_energy"],
        max_samples=k,
        stability_filter=None,
        split="train",
        split_ratio=(1.0, 0.0, 0.0),
        sampling_strategy="kcenter_formula",
    ).prepare(force_reload=True)

    features = np.vstack([fake_vectors[x] for x in df["formula"].tolist()])
    expected_idx = dataset_mod._kcenter_coreset_indices(features, k, seed=42)
    expected_jids = df.iloc[expected_idx]["jid"].tolist()
    assert ds._df["jid"].tolist() == expected_jids


def test_fallback_split_strategy_prototype_matches_governance(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"jid": "J1", "atoms": {"dummy": 1}, "formula": "Li2O", "spg_number": 225, "ehull": 0.02, "formation_energy_peratom": -1.0},
            {"jid": "J2", "atoms": {"dummy": 1}, "formula": "LiFeO2", "spg_number": 225, "ehull": 0.03, "formation_energy_peratom": -1.1},
            {"jid": "J3", "atoms": {"dummy": 1}, "formula": "Fe2O3", "spg_number": 167, "ehull": 0.01, "formation_energy_peratom": -1.2},
            {"jid": "J4", "atoms": {"dummy": 1}, "formula": "Fe2O3", "spg_number": 167, "ehull": 0.04, "formation_energy_peratom": -1.3},
            {"jid": "J5", "atoms": {"dummy": 1}, "formula": "NaCl", "spg_number": 62, "ehull": 0.05, "formation_energy_peratom": -0.9},
            {"jid": "J6", "atoms": {"dummy": 1}, "formula": "NaCl", "spg_number": 62, "ehull": 0.06, "formation_energy_peratom": -1.0},
        ]
    )
    _patch_runtime(monkeypatch, tmp_path, df)

    ratio = (0.5, 0.0, 0.5)
    ds = dataset_mod.CrystalPropertyDataset(
        properties=["formation_energy"],
        stability_filter=None,
        split="test",
        split_ratio=ratio,
        fallback_split_strategy="prototype",
        split_manifest_path=None,
    ).prepare(force_reload=True)

    expected = prototype_split(
        [str(x) for x in df["jid"].tolist()],
        [str(x) for x in df["spg_number"].tolist()],
        seed=42,
        ratios=ratio,
    )["test"]
    assert sorted(ds._df["jid"].tolist()) == sorted(expected)


def test_empty_compositional_split_falls_back_to_iid(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"jid": "J1", "atoms": {"dummy": 1}, "formula": "Li2O", "spg_number": 225, "ehull": 0.01, "formation_energy_peratom": -1.0},
            {"jid": "J2", "atoms": {"dummy": 1}, "formula": "Li2O", "spg_number": 225, "ehull": 0.01, "formation_energy_peratom": -1.1},
        ]
    )
    _patch_runtime(monkeypatch, tmp_path, df)

    ds = dataset_mod.CrystalPropertyDataset(
        properties=["formation_energy"],
        stability_filter=None,
        split="val",
        split_ratio=(0.5, 0.5, 0.0),
        fallback_split_strategy="compositional",
        split_manifest_path=None,
    ).prepare(force_reload=True)

    assert len(ds._df) > 0


def test_invalid_strategies_and_unknown_property_raise():
    with pytest.raises(ValueError):
        dataset_mod.CrystalPropertyDataset(
            properties=["formation_energy"],
            sampling_strategy="bad_strategy",
        )
    with pytest.raises(ValueError):
        dataset_mod.CrystalPropertyDataset(
            properties=["formation_energy"],
            fallback_split_strategy="bad_split",
        )
    with pytest.raises(ValueError):
        dataset_mod.CrystalPropertyDataset(properties=["unknown_property"])


def test_enforce_nonempty_split_raises_on_empty_ood_split(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {"jid": "J1", "atoms": {"dummy": 1}, "formula": "Li2O", "spg_number": 225, "ehull": 0.01, "formation_energy_peratom": -1.0},
            {"jid": "J2", "atoms": {"dummy": 1}, "formula": "Li2O", "spg_number": 225, "ehull": 0.01, "formation_energy_peratom": -1.1},
        ]
    )
    _patch_runtime(monkeypatch, tmp_path, df)

    with pytest.raises(ValueError, match="produced empty split"):
        dataset_mod.CrystalPropertyDataset(
            properties=["formation_energy"],
            stability_filter=None,
            split="val",
            split_ratio=(0.5, 0.5, 0.0),
            fallback_split_strategy="compositional",
            enforce_nonempty_split=True,
            split_manifest_path=None,
        ).prepare(force_reload=True)
