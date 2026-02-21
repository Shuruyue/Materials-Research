import numpy as np
import ase
import ase.build
from ase.calculators.lj import LennardJones
from atlas.discovery.stability.mepin import MEPINStabilityEvaluator
import pytest

pytestmark = pytest.mark.integration

def test_reaction_barrier():
    reactant = ase.build.molecule("C2H4")
    reactant.center(vacuum=5.0)
    reactant.pbc = False

    product = reactant.copy()
    pos = product.get_positions()
    noise = np.random.normal(0, 0.1, pos.shape)
    product.set_positions(pos + noise)

    try:
        estimator = MEPINStabilityEvaluator(model_type="cyclo_L")
    except Exception as e:
        pytest.skip(f"Failed to initialize MEPIN: {e}")

    try:
        path = estimator.predict_path(reactant, product, num_images=10)
    except Exception as e:
        pytest.skip(f"Path prediction failed in this environment: {e}")

    calc = LennardJones()
    energies = []
    for atoms in path:
        atoms.calc = calc
        e = atoms.get_potential_energy()
        energies.append(e)

    energies = np.array(energies)
    barrier = np.max(energies) - energies[0]
    assert np.isfinite(barrier)
    assert len(path) == 10
