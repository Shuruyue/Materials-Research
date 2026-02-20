
"""
Test script for atlas.discovery.alchemy module.
Verifies that AlchemicalMACECalculator can be initialized and run.
"""

import sys
import os
import torch
import numpy as np
from ase.build import bulk
from atlas.discovery.alchemy import AlchemicalMACECalculator

# Verify import
print("Successfully imported AlchemicalMACECalculator")

# 1. Setup Structure (Using standard rocksalt which ASE guarantees support)
print("Setting up NaCl rocksalt structure...")
atoms = bulk("NaCl", "rocksalt", a=5.64)

# 2. Define Alchemical Coefficents
# Replace Na (Z=11) with mix of Na and K (Z=19)
# Na is at index 0 (and usually others, rocksalt has 2 atoms in primitive cell: Na at 0, Cl at 1? 
# bulk primitive=False gives cubic unit cell with 8 atoms? 
# default bulk is primitive=True (1 atom of each?). lattice is FCC?
# bulk('NaCl', 'rocksalt') -> 2 atoms (Na, Cl)
# Let's check atom 0.
na_idx = 0
print(f"Atom at index 0 is {atoms[0].symbol}")

alchemical_pairs = [
    [(na_idx, 11), (na_idx, 19)] # Site 0 can be Na or K
]

# Initial weights: 90% Sr, 10% Ca
# We use a single weight parameter 'w'. 
# Code logic: 
# weight_indices for (Sr, Ca) will be the same channel index (1).
# But wait, original code assigns DIFFERENT weight indices if they are in the same list?
# "for weight_idx, pairs in enumerate(alchemical_pairs):"
# So each list in alchemical_pairs corresponds to ONE weight parameter in typical usage?
# Let's check Recisic's logic again.
# "alchemical_weights" is a list of floats.
# In `AlchemyManager`, `weight_idx` comes from `enumerate(alchemical_pairs)`.
# So `alchemical_pairs[0]` gets `weight_idx=1`.
# Inside `alchemical_pairs[0]`, we have `[(idx, Z), (idx, Z')]`.
# All these atoms get `weight_idx=1`.
# So ONE weight parameter controls ALL these species?
# If so, how do we distinguish 90% Sr vs 10% Ca?
# Ah, usually one optimizes a "virtual atom" properties.
# OR, maybe `alchemical_weights` are passed to the model to interpolate embeddings?
# Yes, `AlchemicalModel` uses `node_weights`.
# If `weight_idx=1` has `weight=0.9`, effectively we scale the embedding of THAT node by 0.9.
# This implies we need SEPARATE weight channels for Sr and Ca if we want them to sum to 1.
# i.e. 
# alchemical_pairs = [
#    [(sr_idx, 38)], # Channel 1: Sr
#    [(sr_idx, 20)]  # Channel 2: Ca
# ]
# alchemical_weights = [0.9, 0.1]
# This way, site `sr_idx` has contributions from Channel 1 (Sr) and Channel 2 (Ca).
# Let's verify if `AlchemyManager` supports multiple entries for the same atom_index.
# "self.atom_indices = alchemical_atom_indices + ..."
# It constructs arrays. `original_to_alchemical_index` stores `i` (index in the expanded list).
# `self.original_to_alchemical_index[atom_idx, weight_idx] = i`
# Yes! It uses `weight_idx` as the second dimension.
# So we CAN have multiple channels for the same atom index.
# This confirms: To mix Sr/Ca, we need two separate channels.

alchemical_pairs_mix = [
    [(na_idx, 11)], # Channel 1: Na (Z=11)
    [(na_idx, 19)]  # Channel 2: K (Z=19)
]
alchemical_weights_mix = [0.9, 0.1] 

print(f"Alchemical Pairs: {alchemical_pairs_mix}")
print(f"Weights: {alchemical_weights_mix}")

# 3. Initialize Calculator
print("Initializing AlchemicalMACECalculator (this may download model)...")
try:
    calc = AlchemicalMACECalculator(
        atoms=atoms,
        alchemical_pairs=alchemical_pairs_mix,
        alchemical_weights=alchemical_weights_mix,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_size="small" # Use small for speed
    )
    atoms.calc = calc
    print("Calculator initialized.")
except Exception as e:
    print(f"Failed to initialize calculator: {e}")
    sys.exit(1)

# 4. Run Static Calculation
print("Running static calculation...")
try:
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print(f"Energy: {energy:.4f} eV")
    print(f"Forces shape: {forces.shape}")
except Exception as e:
    print(f"Calculation failed: {e}")
    # Inspect error
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Check Gradients
print("Checking alchemical gradient...")
try:
    # We need to manually trigger gradient calculation feature on the calculator
    calc.calculate_alchemical_grad = True
    
    # We need to call get_potential_energy again to trigger recalculation with grad enabled
    # But ASE caches results. We must reset calculator or invalidate cache.
    calc.reset()
    energy_grad = atoms.get_potential_energy()
    
    # Access the computed gradient from results
    # ASE doesn't store 'alchemical_grad' in atoms.info easily, need to check calc.results
    grad = calc.results.get('alchemical_grad')
    print(f"Alchemical Gradient: {grad}")
    
    if grad is not None and len(grad) == 2:
        print("Gradient check passed.")
    else:
        print("Gradient check failed or has wrong shape.")

except Exception as e:
    print(f"Gradient calculation failed: {e}")
    traceback.print_exc()

print("Test complete.")
