#!/usr/bin/env python3
"""
Script 05: Run Relaxation with MACE Potential (Phase 3)

Demonstrates how to use a trained MACE model (or M3GNet) to relax crystal structures.
This is the "Dynamic" part of Phase 3, moving beyond static property prediction.

Refactored to use the shared `atlas.potentials.MACERelaxer` module for robustness
and consistency with the active learning loop.

Usage:
    python scripts/phase3_potentials/run_relaxation.py --structure data/raw/mp-1234.cif --model models/mace/best.model
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from atlas.config import get_config

def main():
    parser = argparse.ArgumentParser(description="Relax Crystal Structure using MACE Potential")
    parser.add_argument("--structure", type=str, required=True, help="Path to input structure (CIF, POSCAR, XYZ)")
    parser.add_argument("--model", type=str, default="large", help="Path to trained MACE model OR 'medium'/'large' for pre-trained MP models")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence threshold (eV/A)")
    parser.add_argument("--steps", type=int, default=500, help="Max relaxation steps")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu)")
    args = parser.parse_args()

    config = get_config()
    
    # Resolve Device for display
    # Use robust get_device from config (handles missing torch)
    try:
        device_obj = config.get_device(args.device)
        device = str(device_obj)
    except Exception:
        device = "cpu"
        
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     🟢 MACE RELAXATION (Phase 3)                               ║")
    print(f"║     Input: {Path(args.structure).name:<40}          ║")
    print(f"║     Model: {args.model:<40}          ║")
    print(f"║     Device: {device:<40}         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # 1. Load Structure
    print("\n[1/3] Loading Structure...")
    try:
        from pymatgen.core import Structure
        structure = Structure.from_file(args.structure)
        print(f"  Composition: {structure.composition.reduced_formula}")
        print(f"  Atoms: {len(structure)}")
    except Exception as e:
        print(f"  ❌ Error loading structure: {e}")
        return

    # 2. Initialize Relaxer
    print("\n[2/3] Initializing MACE Relaxer...")
    try:
        from atlas.potentials.mace_relaxer import MACERelaxer
        
        use_foundation = args.model in ["small", "medium", "large"]
        model_path = None if use_foundation else args.model
        
        relaxer = MACERelaxer(
            model_path=model_path,
            device=args.device,
            use_foundation=use_foundation,
            model_size=args.model if use_foundation else "large"
        )
        
        # Trigger lazy load to check availability
        if relaxer.calculator is None:
             print("  ❌ Failed to load calculator (check logs).")
             return
             
        print(f"  Relaxer ready on {relaxer.device}")
        
    except Exception as e:
        print(f"  ❌ Error initializing relaxer: {e}")
        return

    # 3. Relaxation
    print(f"\n[3/3] Starting Relaxation (fmax={args.fmax})...")
    
    # Save Trajectory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/relaxations") / f"{Path(args.structure).stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_file = out_dir / "relaxation.traj"
    
    t0 = time.time()
    
    result = relaxer.relax_structure(
        structure,
        fmax=args.fmax,
        steps=args.steps,
        cell_filter="frechet", # Use best available filter
        trajectory_file=traj_file
    )
    
    dt = time.time() - t0
    
    # 4. Results
    if result.get("error"):
        print(f"\n  ❌ Relaxation Failed: {result['error']}")
        return

    e_final = result["energy_total"]
    e_pa = result["energy_per_atom"]
    steps = result["n_steps"]
    converged = result["converged"]
    
    status = "Converged" if converged else "Not Converged"
    
    print(f"  Status:         {status}")
    print(f"  Final Energy:   {e_final:.4f} eV ({e_pa:.4f} eV/atom)")
    print(f"  Time:           {dt:.2f} s")
    print(f"  Steps:          {steps}")
    print(f"  Vol Change:     {result['volume_change']:.4f}x")
    
    # Save Final Structure
    final_path = out_dir / "relaxed.cif"
    result["relaxed_structure"].to(filename=str(final_path))
    
    print(f"\n  ✅ Relaxation Complete!")
    print(f"     Trajectory: {traj_file}")
    print(f"     Final CIF:  {final_path}")

if __name__ == "__main__":
    main()

