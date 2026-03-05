"""Unit tests for atlas.data.jarvis_client."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from atlas.data.jarvis_client import JARVISClient


def _make_client(df: pd.DataFrame) -> JARVISClient:
    client = JARVISClient.__new__(JARVISClient)
    client._dft_3d = df.copy()
    client.load_dft_3d = lambda force_reload=False: client._dft_3d.copy()
    return client


class TestStableMaterials:
    def test_hard_stability_filter(self):
        df = pd.DataFrame(
            [
                {"jid": "A", "ehull": 0.05, "optb88vdw_bandgap": 1.0},
                {"jid": "B", "ehull": 0.15, "optb88vdw_bandgap": 1.2},
                {"jid": "C", "ehull": None, "optb88vdw_bandgap": 0.9},
            ]
        )
        client = _make_client(df)
        out = client.get_stable_materials(ehull_max=0.1, mode="hard")
        assert list(out["jid"]) == ["A"]

    def test_probabilistic_stability_filter(self):
        df = pd.DataFrame(
            [
                {"jid": "A", "ehull": 0.03, "optb88vdw_bandgap": 1.0},
                {"jid": "B", "ehull": 0.11, "optb88vdw_bandgap": 1.1},
                {"jid": "C", "ehull": 0.30, "optb88vdw_bandgap": 1.2},
            ]
        )
        client = _make_client(df)
        out = client.get_stable_materials(
            ehull_max=0.1,
            mode="probabilistic",
            min_stability_prob=0.25,
            ehull_noise=0.05,
        )
        assert "stability_probability" in out.columns
        assert set(out["jid"]) == {"A", "B"}
        vals = out["stability_probability"].to_numpy()
        assert vals[0] >= vals[1]

    def test_adaptive_noise_changes_probability(self):
        df = pd.DataFrame(
            [
                {
                    "jid": "A",
                    "ehull": 0.12,
                    "formation_energy_peratom": -1.0,
                },
                {
                    "jid": "B",
                    "ehull": 0.12,
                    "formation_energy_peratom": -1.2,
                },
                {
                    "jid": "C",
                    "ehull": 0.12,
                    "formation_energy_peratom": -10.0,
                },
            ]
        )
        client = _make_client(df)
        out = client.get_stable_materials(
            ehull_max=0.1,
            mode="probabilistic",
            min_stability_prob=0.0,
            ehull_noise=0.03,
            ehull_noise_mode="adaptive",
            ehull_noise_adaptive_slope=0.8,
        )
        probs = dict(zip(out["jid"], out["stability_probability"]))
        assert probs["A"] != probs["C"]

    def test_probabilistic_stability_sanitizes_invalid_inputs(self):
        df = pd.DataFrame(
            [
                {"jid": "A", "ehull": 0.03, "formation_energy_peratom": -1.0},
                {"jid": "B", "ehull": 0.20, "formation_energy_peratom": -2.0},
            ]
        )
        client = _make_client(df)
        out = client.get_stable_materials(
            ehull_max=float("nan"),
            mode="probabilistic",
            min_stability_prob=float("nan"),
            ehull_noise=float("-inf"),
            ehull_noise_mode="adaptive",
            ehull_noise_adaptive_slope=float("nan"),
        )
        assert "stability_probability" in out.columns
        assert out["stability_probability"].between(0.0, 1.0).all()

    def test_stability_filter_rejects_inverted_bandgap_window(self):
        df = pd.DataFrame(
            [
                {"jid": "A", "ehull": 0.03, "optb88vdw_bandgap": 1.0},
            ]
        )
        client = _make_client(df)
        with pytest.raises(ValueError, match="min_band_gap"):
            client.get_stable_materials(min_band_gap=2.0, max_band_gap=1.0)


class TestTopologicalMaterials:
    def test_probabilistic_topology_uses_multi_signal_score(self):
        df = pd.DataFrame(
            [
                {
                    "jid": "T1",
                    "spillage": 0.82,
                    "ehull": 0.02,
                    "optb88vdw_bandgap": 0.06,
                },
                {
                    "jid": "T2",
                    "spillage": 0.90,
                    "ehull": 0.45,
                    "optb88vdw_bandgap": 2.00,
                },
                {
                    "jid": "T3",
                    "spillage": 0.56,
                    "ehull": 0.08,
                    "optb88vdw_bandgap": 0.20,
                },
            ]
        )
        client = _make_client(df)
        out = client.get_topological_materials(
            mode="probabilistic",
            min_topology_prob=0.50,
        )
        assert "topological_probability" in out.columns
        assert "spillage_probability" in out.columns
        assert "stability_probability" in out.columns
        # High spillage but very unstable/large-gap entry should be filtered out.
        assert "T2" not in set(out["jid"])
        assert len(out) >= 1

    def test_topology_quantile_threshold(self):
        df = pd.DataFrame(
            [
                {"jid": f"T{i}", "spillage": 0.5 + 0.02 * i, "ehull": 0.05, "optb88vdw_bandgap": 0.1}
                for i in range(10)
            ]
        )
        client = _make_client(df)
        out = client.get_topological_materials(
            mode="probabilistic",
            top_quantile=0.5,
            min_topology_prob=0.99,
        )
        assert 4 <= len(out) <= 6

    def test_probabilistic_topology_sanitizes_invalid_numeric_inputs(self):
        df = pd.DataFrame(
            [
                {"jid": "T1", "spillage": 0.7, "ehull": 0.02, "optb88vdw_bandgap": 0.1},
                {"jid": "T2", "spillage": 0.4, "ehull": 0.15, "optb88vdw_bandgap": 1.5},
            ]
        )
        client = _make_client(df)
        out = client.get_topological_materials(
            mode="probabilistic",
            min_topology_prob=float("nan"),
            spillage_temperature=float("nan"),
            ehull_noise=float("-inf"),
            low_gap_scale=float("nan"),
            fusion_weights=(float("nan"), float("inf"), -1.0),
            score_calibration="temperature",
            calibration_temperature=float("nan"),
        )
        assert "topological_probability" in out.columns
        assert out["topological_probability"].between(0.0, 1.0).all()


class TestTrainingData:
    def test_kcenter_sampling_is_deterministic(self):
        rows = []
        for i in range(20):
            rows.append(
                {
                    "jid": f"TOP-{i}",
                    "spillage": 0.65 + 0.01 * i,
                    "ehull": 0.01 + 0.002 * (i % 5),
                    "optb88vdw_bandgap": 0.05 + 0.01 * (i % 3),
                    "formation_energy_peratom": -1.5 + 0.05 * i,
                    "atoms": {"elements": ["Bi", "Te", "Bi"]},
                }
            )
        for i in range(30):
            rows.append(
                {
                    "jid": f"TRI-{i}",
                    "spillage": 0.05 + 0.01 * (i % 4),
                    "ehull": 0.02 + 0.002 * (i % 6),
                    "optb88vdw_bandgap": 2.5 + 0.05 * (i % 10),
                    "formation_energy_peratom": -3.0 + 0.03 * i,
                    "atoms": {"elements": ["Si", "O", "O"]},
                }
            )
        df = pd.DataFrame(rows)
        client = _make_client(df)
        r1 = client.get_training_data(
            n_topo=6,
            n_trivial=7,
            sampling_strategy="kcenter",
            random_state=11,
            feature_space="hybrid",
        )
        r2 = client.get_training_data(
            n_topo=6,
            n_trivial=7,
            sampling_strategy="kcenter",
            random_state=11,
            feature_space="hybrid",
        )
        assert len(r1["topo"]) == 6
        assert len(r1["trivial"]) == 7
        assert list(r1["topo"]["jid"]) == list(r2["topo"]["jid"])
        assert list(r1["trivial"]["jid"]) == list(r2["trivial"]["jid"])

    def test_training_data_removes_overlap_between_classes(self):
        df = pd.DataFrame(
            [
                {
                    "jid": "OVERLAP",
                    "spillage": 0.9,
                    "ehull": 0.01,
                    "optb88vdw_bandgap": 3.0,
                    "formation_energy_peratom": -2.0,
                    "atoms": {"elements": ["Bi", "Te"]},
                },
                {
                    "jid": "TOP-ONLY",
                    "spillage": 0.92,
                    "ehull": 0.02,
                    "optb88vdw_bandgap": 0.1,
                    "formation_energy_peratom": -1.5,
                    "atoms": {"elements": ["Sb", "Te"]},
                },
                {
                    "jid": "TRI-ONLY",
                    "spillage": 0.02,
                    "ehull": 0.01,
                    "optb88vdw_bandgap": 3.2,
                    "formation_energy_peratom": -3.0,
                    "atoms": {"elements": ["Si", "O", "O"]},
                },
            ]
        )
        client = _make_client(df)
        out = client.get_training_data(
            n_topo=10,
            n_trivial=10,
            sampling_strategy="random",
            feature_space="property",
            topological_mode="probabilistic",
            trivial_stability_mode="probabilistic",
            min_topology_prob=0.5,
            min_stability_prob=0.3,
        )
        topo_ids = set(out["topo"]["jid"].astype(str))
        trivial_ids = set(out["trivial"]["jid"].astype(str))
        assert topo_ids.isdisjoint(trivial_ids)

    def test_unknown_sampling_strategy_raises(self):
        df = pd.DataFrame(
            [
                {
                    "jid": "X",
                    "spillage": 0.8,
                    "ehull": 0.01,
                    "optb88vdw_bandgap": 0.1,
                    "formation_energy_peratom": -1.0,
                }
            ]
        )
        client = _make_client(df)
        with pytest.raises(ValueError):
            client.get_training_data(
                n_topo=0,
                n_trivial=0,
                sampling_strategy="unknown",
            )

    def test_kcenter_sampling_handles_non_finite_features(self):
        df = pd.DataFrame(
            [
                {
                    "jid": f"T{i}",
                    "spillage": np.nan if i % 2 == 0 else 0.6 + 0.01 * i,
                    "ehull": np.inf if i % 3 == 0 else 0.02,
                    "optb88vdw_bandgap": -np.inf if i % 4 == 0 else 0.2,
                    "formation_energy_peratom": np.nan if i % 5 == 0 else -1.0,
                    "atoms": {"elements": ["Bi", "Te"]},
                }
                for i in range(12)
            ]
        )
        client = _make_client(df)
        out = client.get_training_data(
            n_topo=4,
            n_trivial=0,
            sampling_strategy="kcenter",
            random_state=7,
            feature_space="hybrid",
            topological_mode="probabilistic",
            min_topology_prob=0.0,
        )
        assert len(out["topo"]) == 4


class TestIOAndLookup:
    def test_get_structure_normalizes_jid_whitespace(self, monkeypatch):
        df = pd.DataFrame(
            [
                {"jid": "  J1  ", "atoms": {"elements": ["Si"]}},
            ]
        )
        client = _make_client(df)

        class _FakeAtoms:
            @staticmethod
            def from_dict(payload):
                return types.SimpleNamespace(pymatgen_converter=lambda: {"payload": payload})

        jarvis_mod = types.ModuleType("jarvis")
        core_mod = types.ModuleType("jarvis.core")
        atoms_mod = types.ModuleType("jarvis.core.atoms")
        atoms_mod.Atoms = _FakeAtoms
        core_mod.atoms = atoms_mod
        jarvis_mod.core = core_mod
        monkeypatch.setitem(sys.modules, "jarvis", jarvis_mod)
        monkeypatch.setitem(sys.modules, "jarvis.core", core_mod)
        monkeypatch.setitem(sys.modules, "jarvis.core.atoms", atoms_mod)

        out = client.get_structure("J1")
        assert out["payload"]["elements"] == ["Si"]

    def test_download_file_detects_incomplete_payload(self, tmp_path, monkeypatch):
        client = _make_client(pd.DataFrame([{"jid": "X"}]))

        class _FakeResponse:
            headers = {"content-length": "10"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, block_size):
                yield b"12345"

        monkeypatch.setattr("atlas.data.jarvis_client.requests.get", lambda *a, **k: _FakeResponse())

        with pytest.raises(RuntimeError, match="Failed to download data"):
            client._download_file("https://example.com/fake", tmp_path / "dft_3d.json", max_retries=1)
