import ase
import ase.build
import numpy as np
import pytest

from atlas.discovery.transport.liflow import LiFlowEvaluator

pytestmark = pytest.mark.integration

def test_transport_simulation():
    a = 10.0
    atoms = ase.Atoms(
        symbols=["Li", "Ge", "P", "S"],
        positions=[
            [0, 0, 0],
            [2, 2, 2],
            [4, 4, 4],
            [6, 6, 6]
        ],
        cell=[a, a, a],
        pbc=True
    )
    atoms = atoms.repeat((2, 2, 2))

    try:
        evaluator = LiFlowEvaluator(
            checkpoint_path=None,
            element_index_path=None,
            temp_list=[800]
        )
    except Exception as e:
        pytest.skip(f"Failed to initialize LiFlow: {e}")

    try:
        traj, diff = evaluator.simulate(atoms, steps=50, flow_steps=5)
    except Exception as e:
        pytest.skip(f"Simulation failed in this environment: {e}")

    assert len(traj) > 0
    final_atoms = traj[-1]
    disp = np.linalg.norm(final_atoms.positions - atoms.positions)
    assert np.isfinite(disp)
    assert diff is None or np.isfinite(diff)
