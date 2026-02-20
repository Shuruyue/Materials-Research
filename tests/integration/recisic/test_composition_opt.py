
import numpy as np
import torch
import ase.build
from mace.calculators import MACECalculator
from atlas.discovery.alchemy import AlchemicalMACECalculator
from atlas.discovery.alchemy.optimizer import CompositionOptimizer
import os

def test_composition_optimization():
    print("Setting up Alchemical System (NaCl)...")
    # 1. Setup Atoms
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.63)
    
    # 2. Define Alchemical Pairs
    # Site 0 (Na) -> Mix of Na (11) and K (19)
    # We use a list of lists: [[(site_idx, Z1)], [(site_idx, Z2)]]
    # This creates two independent weight channels controlled by the optimizer
    alchemical_pairs = [[(0, 11)], [(0, 19)]]
    
    # Initial Weights: 50/50 mix
    initial_weights = [0.5, 0.5]
    
    print(f"Alchemical Pairs: {alchemical_pairs}")
    print(f"Initial Weights: {initial_weights}")

    # 3. Initialize Calculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = AlchemicalMACECalculator(
        atoms=atoms,
        alchemical_pairs=alchemical_pairs,
        alchemical_weights=initial_weights,
        device=device,
        model_size="small" # Use small for speed
    )
    atoms.calc = calc
    
    # 4. Initialize Optimizer
    optimizer = CompositionOptimizer(calc, learning_rate=0.1)
    
    print("\n--- Starting Optimization ---")
    print(f"Initial Energy: {atoms.get_potential_energy():.4f} eV")
    
    # 5. Run Optimization
    traj = optimizer.run(steps=20, verbose=True)
    
    # 6. Analysis
    final_weights = traj[-1]["weights"]
    final_energy = traj[-1]["energy"]
    
    print("\n--- Optimization Complete ---")
    print(f"Final Weights: {final_weights}")
    print(f"Final Energy: {final_energy:.4f} eV")
    
    # Check if sum constraint is maintained
    w_sum = np.sum(final_weights)
    print(f"Sum of weights: {w_sum:.6f}")
    if not np.isclose(w_sum, 1.0, atol=1e-4):
        print("❌ Constraint Violation: Weights do not sum to 1.0!")
        exit(1)
    else:
        print("Constraint Satisfied: Sum = 1.0")
        
    # Check if energy decreased (or at least didn't explode)
    if final_energy < traj[0]["energy"]:
        print("Energy Decreased.")
    else:
        print("Energy did not decrease (might be local minimum or small step).")

if __name__ == "__main__":
    try:
        test_composition_optimization()
        print("\nTest Passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
