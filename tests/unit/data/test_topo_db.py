"""Tests for atlas.data.topo_db module."""


import numpy as np
import pytest


@pytest.fixture
def topo_db(tmp_path, monkeypatch):
    """Create a TopoDB with a temporary directory."""
    from atlas.config import get_config
    get_config()

    # Redirect DB directory to tmp
    from atlas.data.topo_db import TopoDB
    db = TopoDB()
    db.db_dir = tmp_path
    db.db_file = tmp_path / "topological_materials.csv"
    db._df = None  # reset cache
    return db


def test_empty_db(topo_db):
    """New DB should be empty."""
    assert len(topo_db.df) == 0


def test_load_seed_data(topo_db):
    """load_seed_data should populate the database."""
    topo_db.load_seed_data()
    assert len(topo_db.df) > 0
    # Should have both topological and trivial materials
    classes = set(topo_db.df["topo_class"].unique())
    assert "TI" in classes
    assert "TSM" in classes
    assert "TRIVIAL" in classes


def test_query_by_class(topo_db):
    """Query by topo_class should filter correctly."""
    topo_db.load_seed_data()

    ti = topo_db.query(topo_class="TI")
    assert len(ti) > 0
    assert all(ti["topo_class"] == "TI")

    tsm = topo_db.query(topo_class="TSM")
    assert len(tsm) > 0
    assert all(tsm["topo_class"] == "TSM")


def test_query_by_band_gap(topo_db):
    """Query by band gap range should filter correctly."""
    topo_db.load_seed_data()

    # Narrow gap materials (0 to 0.5 eV)
    narrow = topo_db.query(band_gap_range=(0.0, 0.5))
    assert len(narrow) > 0
    assert all(narrow["band_gap"] <= 0.5)


def test_query_by_elements(topo_db):
    """Query by elements should filter correctly."""
    topo_db.load_seed_data()

    bi_mats = topo_db.query(elements=["Bi"])
    assert len(bi_mats) > 0
    assert all("Bi" in f for f in bi_mats["formula"])


def test_add_materials_dedup(topo_db):
    """Adding duplicate materials should deduplicate."""
    from atlas.data.topo_db import TopoMaterial

    m1 = TopoMaterial("test-001", "TestMat", 225, "TI", 0.1)
    m2 = TopoMaterial("test-001", "TestMat", 225, "TSM", 0.2)  # same id, different class

    topo_db.add_materials([m1])
    assert len(topo_db.df) == 1
    assert topo_db.df.iloc[0]["topo_class"] == "TI"

    topo_db.add_materials([m2])
    assert len(topo_db.df) == 1  # deduplicated
    assert topo_db.df.iloc[0]["topo_class"] == "TSM"  # last wins


def test_stats(topo_db):
    """stats() should return correct counts."""
    topo_db.load_seed_data()
    stats = topo_db.stats()
    assert stats["total"] == len(topo_db.df)
    assert "by_class" in stats
    assert "by_source" in stats
    assert "channel_reliability" in stats
    assert "channel_correlation" in stats


def test_save_and_reload(topo_db):
    """Save and reload should preserve data."""
    from atlas.data.topo_db import TopoDB

    topo_db.load_seed_data()
    n_before = len(topo_db.df)
    topo_db.save()

    # Create new instance pointing to same file
    db2 = TopoDB()
    db2.db_dir = topo_db.db_dir
    db2.db_file = topo_db.db_file
    db2._df = None

    assert len(db2.df) == n_before


def test_topo_material_is_topological():
    """TopoMaterial.is_topological should classify correctly."""
    from atlas.data.topo_db import TopoMaterial

    assert TopoMaterial("id1", "Bi2Se3", 166, "TI").is_topological() is True
    assert TopoMaterial("id2", "TaAs", 109, "TSM").is_topological() is True
    assert TopoMaterial("id3", "Si", 227, "TRIVIAL").is_topological() is False
    assert TopoMaterial("id4", "X", 1, "UNKNOWN").is_topological() is False


def test_query_elements_uses_exact_element_membership(topo_db):
    topo_db.load_seed_data()
    out = topo_db.query(elements=["Si"])
    formulas = set(out["formula"].astype(str).tolist())
    assert "Si" in formulas
    assert "Bi2Se3" not in formulas


def test_infer_topology_probabilities_from_evidence(topo_db):
    topo_db.add_material(
        "Bi2Se3",
        "TRIVIAL",
        {
            "jid": "evi-topo",
            "space_group": 166,
            "spillage": 1.2,
            "si_score": 1.5,
            "ml_probability": 0.90,
            "ml_uncertainty": 0.05,
        },
    )
    topo_db.add_material(
        "SiO2",
        "TRIVIAL",
        {
            "jid": "evi-trivial",
            "space_group": 227,
            "spillage": 0.02,
            "si_score": -1.2,
            "ml_probability": 0.10,
            "ml_uncertainty": 0.05,
        },
    )
    out = topo_db.infer_topology_probabilities(calibrate_reliability=False)
    p_topo = float(out[out["jid"] == "evi-topo"]["topological_probability"].iloc[0])
    p_tri = float(out[out["jid"] == "evi-trivial"]["topological_probability"].iloc[0])
    assert p_topo > p_tri
    row = out[out["jid"] == "evi-topo"].iloc[0]
    assert 0.0 <= float(row["topology_ci_low"]) <= float(row["topological_probability"]) <= float(row["topology_ci_high"]) <= 1.0


def test_channel_reliability_calibration_downweights_bad_channel(topo_db):
    # Build data where SI and spillage align with truth, ML is adversarial.
    for i in range(6):
        topo_db.add_material(
            f"Bi2Se{i}3",
            "TI",
            {
                "jid": f"ti-{i}",
                "space_group": 166,
                "spillage": 1.0,
                "si_score": 1.2,
                "ml_probability": 0.05,  # wrong
                "ml_uncertainty": 0.05,
            },
        )
    for i in range(6):
        topo_db.add_material(
            f"Si{i}O2",
            "TRIVIAL",
            {
                "jid": f"tr-{i}",
                "space_group": 227,
                "spillage": 0.02,
                "si_score": -1.1,
                "ml_probability": 0.95,  # wrong
                "ml_uncertainty": 0.05,
            },
        )

    topo_db.reset_channel_reliability()
    rel = topo_db.calibrate_channel_reliability(min_samples=4)
    assert rel["ml"] < rel["si"]
    assert rel["ml"] < rel["spillage"]


def test_calibration_estimates_channel_correlation(topo_db):
    # Correlated channel errors: SI and spillage always move together.
    for i in range(10):
        topo_db.add_material(
            f"Bi{i}Se3",
            "TI",
            {
                "jid": f"corr-ti-{i}",
                "space_group": 166,
                "spillage": 0.9,
                "si_score": 1.1,
                "ml_probability": 0.7,
                "ml_uncertainty": 0.1,
            },
        )
    for i in range(10):
        topo_db.add_material(
            f"Si{i}O2",
            "TRIVIAL",
            {
                "jid": f"corr-tr-{i}",
                "space_group": 227,
                "spillage": 0.1,
                "si_score": -1.1,
                "ml_probability": 0.3,
                "ml_uncertainty": 0.1,
            },
        )
    topo_db.reset_channel_reliability()
    topo_db.calibrate_channel_reliability(min_samples=6, corr_shrinkage=0.1)
    corr = topo_db.channel_correlation()
    assert corr["si"]["spillage"] > 0.1


def test_ood_scoring_flags_far_chemistry(topo_db):
    topo_db.add_material(
        "Bi2Se3",
        "TI",
        {"jid": "known-1", "space_group": 166, "spillage": 1.0, "si_score": 1.0},
    )
    topo_db.add_material(
        "Bi2Te3",
        "TI",
        {"jid": "known-2", "space_group": 166, "spillage": 1.0, "si_score": 1.0},
    )
    topo_db.add_material(
        "Xx2Yy",
        "TI",
        {"jid": "far-1", "space_group": 1, "spillage": 1.0, "si_score": 1.0},
    )
    out = topo_db.infer_topology_probabilities(
        calibrate_reliability=False,
        reference_formulas=["Bi2Se3", "Bi2Te3"],
    )
    far = float(out[out["jid"] == "far-1"]["ood_score"].iloc[0])
    near = float(out[out["jid"] == "known-1"]["ood_score"].iloc[0])
    assert far > near


def test_rank_topological_candidates_orders_by_acquisition(topo_db):
    topo_db.add_material(
        "Bi2Se3",
        "TI",
        {
            "jid": "r1",
            "space_group": 166,
            "topological_probability": 0.90,
            "topology_ci_low": 0.80,
            "topology_ci_high": 0.95,
            "ood_score": 0.10,
        },
    )
    topo_db.add_material(
        "TaAs",
        "TSM",
        {
            "jid": "r2",
            "space_group": 109,
            "topological_probability": 0.80,
            "topology_ci_low": 0.60,
            "topology_ci_high": 0.95,
            "ood_score": 0.05,
        },
    )
    ranked = topo_db.rank_topological_candidates(
        min_probability=0.5,
        uncertainty_weight=0.3,
        ood_penalty=0.4,
        top_k=2,
        recompute_if_missing=False,
    )
    assert len(ranked) == 2
    assert float(ranked["acquisition_score"].iloc[0]) >= float(ranked["acquisition_score"].iloc[1])


def test_correlation_aware_fusion_widens_interval(topo_db):
    topo_db.add_material(
        "Bi2Se3",
        "TI",
        {
            "jid": "corr-fuse",
            "space_group": 166,
            "spillage": 1.0,
            "si_score": 1.2,
            "ml_probability": 0.92,
            "ml_uncertainty": 0.05,
        },
    )
    # Force strong channel correlation to simulate redundant evidence.
    topo_db._channel_corr = np.array(
        [[1.0, 0.9, 0.85], [0.9, 1.0, 0.8], [0.85, 0.8, 1.0]],
        dtype=float,
    )
    independent = topo_db.infer_topology_probabilities(
        calibrate_reliability=False,
        correlation_mode="independent",
        ci_method="exact",
        persist=False,
    )
    correlated = topo_db.infer_topology_probabilities(
        calibrate_reliability=False,
        correlation_mode="correlated",
        ci_method="exact",
        persist=False,
    )
    w_ind = float(
        independent[independent["jid"] == "corr-fuse"]["topology_ci_high"].iloc[0]
        - independent[independent["jid"] == "corr-fuse"]["topology_ci_low"].iloc[0]
    )
    w_cor = float(
        correlated[correlated["jid"] == "corr-fuse"]["topology_ci_high"].iloc[0]
        - correlated[correlated["jid"] == "corr-fuse"]["topology_ci_low"].iloc[0]
    )
    assert w_cor >= w_ind


def test_exact_and_normal_beta_interval_modes(topo_db):
    topo_db.add_material(
        "TaAs",
        "TSM",
        {
            "jid": "ci-mode",
            "space_group": 109,
            "spillage": 0.7,
            "si_score": 0.4,
            "ml_probability": 0.65,
            "ml_uncertainty": 0.2,
        },
    )
    out_exact = topo_db.infer_topology_probabilities(
        calibrate_reliability=False,
        ci_method="exact",
        evidence_strength=6.0,
        persist=False,
    )
    out_norm = topo_db.infer_topology_probabilities(
        calibrate_reliability=False,
        ci_method="normal",
        evidence_strength=6.0,
        persist=False,
    )
    row_e = out_exact[out_exact["jid"] == "ci-mode"].iloc[0]
    row_n = out_norm[out_norm["jid"] == "ci-mode"].iloc[0]
    width_e = float(row_e["topology_ci_high"] - row_e["topology_ci_low"])
    width_n = float(row_n["topology_ci_high"] - row_n["topology_ci_low"])
    assert 0.0 <= float(row_e["topology_ci_low"]) <= float(row_e["topology_ci_high"]) <= 1.0
    assert abs(width_e - width_n) > 1e-4
