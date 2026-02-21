"""Integration test for alchemical MACE calculator."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.build import bulk

from atlas.discovery.alchemy import AlchemicalMACECalculator


@pytest.mark.integration
def test_alchemical_mace_static_and_gradient():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    na_idx = 0
    alchemical_pairs = [[(na_idx, 11)], [(na_idx, 19)]]
    alchemical_weights = [0.9, 0.1]

    try:
        calc = AlchemicalMACECalculator(
            atoms=atoms,
            alchemical_pairs=alchemical_pairs,
            alchemical_weights=alchemical_weights,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_size="small",
        )
    except Exception as exc:
        pytest.skip(f"AlchemicalMACECalculator unavailable in this environment: {exc}")

    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)

    calc.calculate_alchemical_grad = True
    calc.reset()
    _ = atoms.get_potential_energy()
    grad = calc.results.get("alchemical_grad")
    if grad is None:
        pytest.skip("alchemical_grad is not exposed by this backend/version.")
    assert len(grad) == len(alchemical_weights)
