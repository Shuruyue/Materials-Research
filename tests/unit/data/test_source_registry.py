"""Tests for atlas.data.source_registry."""


def test_data_source_registry_defaults():
    from atlas.data.source_registry import DATA_SOURCES

    keys = DATA_SOURCES.list_keys()
    assert "jarvis_dft" in keys
    assert "materials_project" in keys
    assert "matbench" in keys

    jarvis = DATA_SOURCES.get("jarvis_dft")
    assert "JARVIS" in jarvis.name
    assert "formation_energy" in jarvis.primary_targets


def test_data_source_registry_missing_key():
    import pytest

    from atlas.data.source_registry import DATA_SOURCES

    with pytest.raises(KeyError):
        DATA_SOURCES.get("does_not_exist")
