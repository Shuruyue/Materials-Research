"""Unit tests for script-level StructureEnumerator fallback implementation."""

from __future__ import annotations

import pytest
from pymatgen.core import DummySpecies, Lattice, Structure

import scripts.phase5_active_learning.structure_enumerator as structure_enumerator_module
from scripts.phase5_active_learning.structure_enumerator import StructureEnumerator


def _perovskite() -> Structure:
    lattice = Lattice.cubic(3.945)
    species = ["Sr", "Ti", "O", "O", "O"]
    coords = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
    return Structure(lattice, species, coords)


def test_generate_returns_copy_when_no_sites_match():
    base = _perovskite()
    enumerator = StructureEnumerator(base)
    structures = enumerator.generate({"Zr": ["Zr"]})
    assert len(structures) == 1
    assert structures[0] is not base
    assert structures[0].composition.reduced_formula == base.composition.reduced_formula


def test_generate_rejects_invalid_substitution_payload():
    base = _perovskite()
    enumerator = StructureEnumerator(base)
    with pytest.raises(TypeError, match="substitutions"):
        enumerator.generate(["not", "a", "dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        enumerator.generate({"Ti": []})
    with pytest.raises(ValueError, match="at least one non-empty"):
        enumerator.generate({"Ti": ["", "  "]})


def test_generate_remove_incomplete_filters_dummy_species():
    base = _perovskite()
    enumerator = StructureEnumerator(base)
    substitutions = {"O": ["O", DummySpecies("X")]}

    filtered = enumerator.generate(substitutions, remove_incomplete=True)
    assert len(filtered) > 0
    assert all(
        all(not isinstance(site.specie, DummySpecies) for site in struct)
        for struct in filtered
    )

    unfiltered = enumerator.generate(substitutions, remove_incomplete=False)
    assert len(unfiltered) >= len(filtered)
    assert any(
        any(isinstance(site.specie, DummySpecies) for site in struct)
        for struct in unfiltered
    )


def test_generate_validates_max_index():
    base = _perovskite()
    enumerator = StructureEnumerator(base)
    with pytest.raises(ValueError, match="max_index"):
        enumerator.generate({"Ti": ["Ti", "Zr"]}, max_index=0)
    with pytest.raises(ValueError, match="max_index"):
        enumerator.generate({"Ti": ["Ti", "Zr"]}, max_index=1.5)


def test_build_constraints_reports_variant_space():
    base = _perovskite()
    enumerator = StructureEnumerator(base)
    constraints = enumerator._build_constraints({"Ti": ["Ti", "Zr"], "O": ["O", "F"]})
    assert constraints["sites_to_vary"] == 4
    assert constraints["variant_space"] == 16
    assert constraints["max_variants"] >= constraints["variant_space"]


def test_normalize_substitutions_deduplicates_options():
    base = _perovskite()
    enumerator = StructureEnumerator(base)
    normalized = enumerator._normalize_substitutions({"Ti": ["Ti", "Ti", "Zr"]})
    assert normalized["Ti"] == ["Ti", "Zr"]


def test_normalize_substitutions_merges_normalized_keys():
    base = _perovskite()
    enumerator = StructureEnumerator(base)
    normalized = enumerator._normalize_substitutions(
        {
            "Ti": ["Ti"],
            " Ti ": ["Zr"],
        }
    )
    assert normalized["Ti"] == ["Ti", "Zr"]


def test_select_variant_ordinals_stratifies_large_space():
    ordinals = StructureEnumerator._select_variant_ordinals(total_variants=27, limit=4)
    assert ordinals == [0, 9, 17, 26]


def test_decode_variant_ordinal_rejects_out_of_range():
    with pytest.raises(ValueError, match="ordinal out of range"):
        StructureEnumerator._decode_variant_ordinal(8, [2, 2, 2])


def test_generate_truncated_space_samples_across_high_order_dimensions(monkeypatch):
    monkeypatch.setattr(structure_enumerator_module, "_MAX_VARIANTS", 8)
    monkeypatch.setattr(structure_enumerator_module, "_TRUNCATED_VARIANTS", 4)

    base = _perovskite()
    enumerator = StructureEnumerator(base)
    substitutions = {"Ti": ["Ti", "Zr"], "O": ["O", "F"]}
    generated = enumerator.generate(substitutions, remove_superperiodic=False)

    assert len(generated) == 4
    formulas = {structure.composition.reduced_formula for structure in generated}
    assert any("Ti" in formula for formula in formulas)
    assert any("Zr" in formula for formula in formulas)
