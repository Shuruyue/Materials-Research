"""Tests for atlas.data.topo_db module."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def topo_db(tmp_path, monkeypatch):
    """Create a TopoDB with a temporary directory."""
    from atlas.config import get_config
    cfg = get_config()

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
