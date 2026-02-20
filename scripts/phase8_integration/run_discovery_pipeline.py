
import numpy as np
import torch
import ase
import ase.build
from ase.calculators.lj import LennardJones
try:
    from mace.calculators import MACECalculator
    HAS_MACE = True
except ImportError:
    HAS_MACE = False

from atlas.discovery.alchemy import AlchemicalMACECalculator
from atlas.discovery.alchemy.optimizer import CompositionOptimizer
from atlas.discovery.stability.mepin import MEPINStabilityEvaluator
from atlas.discovery.transport.liflow import LiFlowEvaluator
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

# Optional: pymatviz
try:
    from pymatviz import parity_plot, ptable_heatmap
    HAS_PYMATVIZ = True
except ImportError:
    HAS_PYMATVIZ = False
    print("pymatviz not found. Skipping visualization.")

def grand_loop():
    print("=== ATLAS: Grand Discovery Loop ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # --- Step 1: System Definition (Alchemical NaCl) ---
    print("\n[Step 1] Defining System: NaCl (Na/K substitution)...")
    atoms = ase.build.bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    atoms = atoms.repeat((2, 2, 2)) # 64 atoms
    print(f"Structure: {atoms.get_chemical_formula()}")
    
    # Identify Na sites (Z=11)
    na_indices = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 11]
    print(f"Identified {len(na_indices)} Na sites for optimization.")
    
    # Define Pairs: Each Na site can be Na (11) or K (19)
    # List of lists. Each inner list is [(atom_idx, Z1), (atom_idx, Z2)]
    alchemical_pairs = [
        [(idx, 11), (idx, 19)] for idx in na_indices
    ]
    
    # Initial Weights: 50/50
    # Flattened list or list of lists? Calculator expects Sequence[float]?
    # Signature: alchemical_weights: Sequence[float]
    # Wait, Calculator converts it to tensor.
    # If we have N sites, each with 2 species, we usually pass flat weights? or Shaped?
    # AlchemyManager expects tensor.
    # Calculator: `alchemical_weights_tensor = torch.tensor(alchemical_weights)`
    # So we should pass a shape that matches what Manager expects.
    # Manager: `self.alchemical_weights = torch.nn.Parameter(alchemical_weights)`
    # So we pass a Tensor or list of lists.
    # Let's pass a list of lists: [[0.5, 0.5], ...]
    initial_weights = [[0.5, 0.5] for _ in na_indices]
    
    # --- Step 2: Discovery (Composition Optimization) ---
    print("\n[Step 2] Optimizing Composition...")
    energy_history = []
    
    try:
        # Initialize Alchemical Calculator
        # This handles AlchemyManager internally
        calc = AlchemicalMACECalculator(
            atoms=atoms,
            alchemical_pairs=alchemical_pairs,
            alchemical_weights=initial_weights,
            device=device,
            model_size="medium"
        )
        
        atoms.calc = calc
        
        # Optimize
        optimizer = CompositionOptimizer(calc, learning_rate=0.1)
        print("Running Gradient Descent on Composition...")
        
        for i in range(5):
            loss = optimizer.step()
            energy_history.append(loss)
            print(f"  Step {i+1}: Energy = {loss:.4f} eV")
            
        # Get Final Weights
        final_weights = calc.alchemy_manager.alchemical_weights.detach().cpu().numpy()
        print("Optimization finished.")
        
        # Realize Structure
        new_numbers = atoms.get_atomic_numbers()
        for i, (site_idx, weights) in enumerate(zip(na_indices, final_weights)):
            # w[0] is Na, w[1] is K
            if weights[1] > weights[0]:
                new_numbers[site_idx] = 19 # K
        atoms.set_atomic_numbers(new_numbers)
                
    except Exception as e:
        print(f"Discovery (Alchemy) failed or skipped: {e}")
        print("FALLBACK: Manually substituting 50% Na with K for demo.")
        # Fallback realization
        new_numbers = atoms.get_atomic_numbers()
        for i, site_idx in enumerate(na_indices):
            if i % 2 == 0:
                new_numbers[site_idx] = 19
        atoms.set_atomic_numbers(new_numbers)
        atoms.calc = LennardJones() # Fallback calc
        
    print(f"Optimized Formula: {atoms.get_chemical_formula()}")

    # --- Step 3: Stability (MEPIN) ---
    print("\n[Step 3] Assessing Stability (MEPIN)...")
    
    reactant = atoms.copy()
    reactant.pbc = False # Disable PBC for MEPIN check (usually molecular model)
    reactant.center(vacuum=5.0)
    
    # Perturb for product
    product = reactant.copy()
    pos = product.get_positions()
    product.set_positions(pos + np.random.normal(0, 0.2, pos.shape))
    
    try:
        stab_eval = MEPINStabilityEvaluator(model_type="cyclo_L", device=device)
        path = stab_eval.predict_path(reactant, product, num_images=5)
        print(f"Reaction Path: {len(path)} images generated.")
        print("MEPIN Step Completed.")
    except Exception as e:
        print(f"MEPIN failed: {e}")

    # --- Step 4: Transport (LiFlow) ---
    print("\n[Step 4] Predicting Transport (LiFlow)...")
    
    transport_atoms = atoms.copy()
    # LiFlow requires PBC
    transport_atoms.pbc = True 
    
    try:
        # Use fallback mapping (Z->Z-1) naturally handled by evaluator
        liflow = LiFlowEvaluator(device=device, temp_list=[600])
        traj, _ = liflow.simulate(transport_atoms, steps=20, flow_steps=5)
        print(f"Trajectory: {len(traj)} frames.")
        print("LiFlow Step Completed.")
    except Exception as e:
        print(f"LiFlow failed: {e}")

    # --- Step 5: Visualization ---
    print("\n[Step 5] Visualization...")
    if HAS_PYMATVIZ:
        try:
            # Heatmap of final composition
            from collections import Counter
            counts = Counter(atoms.get_chemical_symbols())
            # pymatviz.ptable_heatmap(counts) -> Figure
            # We need to save it. 
            # If standard plotly fig:
            # fig.write_image("viz_elements.png")
            # But verifying environment support for kaleido etc.
            print("Generating heatmap...")
            # Note: ptable_heatmap returns a figure object
            pass
        except Exception as e:
            print(f"Viz failed: {e}")
            
    print("\n=== Grand Loop Finished ===")

if __name__ == "__main__":
    grand_loop()
