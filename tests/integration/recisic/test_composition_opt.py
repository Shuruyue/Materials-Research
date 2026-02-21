import numpy as np
import pytest
import torch
import ase.build

pytestmark = pytest.mark.integration


def test_composition_optimization():
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.63)
    alchemical_pairs = [[(0, 11)], [(0, 19)]]
    initial_weights = [0.5, 0.5]

    try:
        from atlas.discovery.alchemy import AlchemicalMACECalculator
        from atlas.discovery.alchemy.optimizer import CompositionOptimizer
    except Exception as exc:
        pytest.skip(f"Alchemical modules unavailable in this environment: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        calc = AlchemicalMACECalculator(
            atoms=atoms,
            alchemical_pairs=alchemical_pairs,
            alchemical_weights=initial_weights,
            device=device,
            model_size="small",
        )
    except Exception as exc:
        pytest.skip(f"Alchemical stack unavailable in this environment: {exc}")

    atoms.calc = calc
    optimizer = CompositionOptimizer(calc, learning_rate=0.1)

    traj = optimizer.run(steps=20, verbose=True)
    final_weights = traj[-1]["weights"]
    final_energy = traj[-1]["energy"]

    w_sum = np.sum(final_weights)
    assert np.isclose(w_sum, 1.0, atol=1e-4), "Weights must satisfy simplex constraint."
    assert np.isfinite(final_energy), "Final energy should be finite."
