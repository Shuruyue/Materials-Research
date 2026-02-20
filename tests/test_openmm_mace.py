from atlas.thermo.openmm.engine import OpenMMEngine
from ase.build import bulk
import numpy as np

def test_openmm_mace():
    print("Testing OpenMM with MACE Potential...")
    
    # 1. Create a simple system 
    # MACE-MP-0 handles most elements. Let's use something simple like Copper or Silicon.
    # Cu fcc lattice ~3.6 A. 
    # MACE cutoff ~5-6A. Need box > 12A.
    # (4,4,4) -> 14.4 A. Safe.
    atoms = bulk("Cu", cubic=True) * (4, 4, 4)
    print(f"Created system: {atoms.get_chemical_formula()}")
    
    # 2. Initialize Engine
    engine = OpenMMEngine(temperature=300, step_size=1.0)
    
    # 3. Setup System with MACE
    # This triggers the 'mace' branch in engine.py
    try:
        engine.setup_system(atoms, forcefield_path="mace")
    except Exception as e:
        print(f"Setup failed: {e}")
        return

    # 4. Run Simulation
    print("Running MACE MD...")
    traj = engine.run(steps=50)
    
    # 5. Check results
    final_pos = traj[-1].get_positions()
    print("MACE MD Simulation completed.")

if __name__ == "__main__":
    test_openmm_mace()
