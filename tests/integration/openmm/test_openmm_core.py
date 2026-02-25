import pytest
from ase.build import bulk

pytest.importorskip("openmm")
from atlas.thermo.openmm.engine import OpenMMEngine

pytestmark = pytest.mark.integration

def test_openmm_core():
    print("Testing OpenMM Core Engine...")

    # 1. Create a simple system (Argon crystal to match default LJ)
    # Using 'Ar' (Z=18).
    # Box size needs to be > 2*cutoff (2*10 = 20A) for LJ default
    # Ar lattice ~ 5.26 A. (4,4,4) -> 21 A. Safe.
    atoms = bulk("Ar", cubic=True) * (4, 4, 4)
    print(f"Created system: {atoms.get_chemical_formula()}")

    # 2. Initialize Engine
    engine = OpenMMEngine(temperature=300, step_size=1.0)

    # 3. Setup System (Default LJ)
    engine.setup_system(atoms)

    # 4. Run Simulation
    traj = engine.run(steps=100)

    # 5. Check results
    final_pos = traj[-1].get_positions()
    assert final_pos.shape == atoms.get_positions().shape
    print("Test Passed: Simulation completed and shapes match.")
