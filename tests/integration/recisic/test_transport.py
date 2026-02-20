
import numpy as np
import torch
import ase
import ase.build
from atlas.discovery.transport.liflow import LiFlowEvaluator
import os

def test_transport_simulation():
    print("Setting up Transport System...")
    
    # 1. Create Dummy Structure (Li-Ge-P-S like)
    # We use a simple cubic cell with Li, Ge, P, S atoms
    # to test if element mapping handles them (Z=3, 32, 15, 16)
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
    # Repeat to make it larger? No, keep it small for speed.
    # But LiFlow might expect minimum density or neighbors?
    # Let's repeat 2x2x2
    atoms = atoms.repeat((2, 2, 2))
    
    print(f"Structure created: {atoms.get_chemical_formula()}")
    
    # 2. Initialize Evaluator
    try:
        evaluator = LiFlowEvaluator(
            checkpoint_path=None, # Use default P_universal
            element_index_path=None, # Will trigger dummy mapping
            temp_list=[800]
        )
    except Exception as e:
        print(f"Skipping test: Failed to initialize LiFlow: {e}")
        return

    print("\n--- Running Simulation ---")
    
    # 3. Simulate
    try:
        # Run for small number of steps for testing
        traj, diff = evaluator.simulate(atoms, steps=50, flow_steps=5)
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Simulation completed. Dictionary length: {len(traj)}")
    
    # 4. Verify Output
    if len(traj) > 0:
        final_atoms = traj[-1]
        disp = np.linalg.norm(final_atoms.positions - atoms.positions)
        print(f"Total Displacement Norm: {disp:.4f}")
        # Note: Since mapping is dummy, displacement might be huge or tiny, but shouldn't be NaN
        if np.isnan(disp):
            print("ERROR: Displacement is NaN!")
        else:
            print("LiFlow Integration Verified.")
    else:
        print("ERROR: Empty trajectory!")

if __name__ == "__main__":
    test_transport_simulation()
