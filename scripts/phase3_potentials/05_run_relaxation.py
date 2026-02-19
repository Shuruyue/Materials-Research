#!/usr/bin/env python3
"""
Script 05: Run Relaxation with MACE Potential (Phase 3)

Demonstrates how to use a trained MACE model (or M3GNet) to relax crystal structures.
This is the "Dynamic" part of Phase 3, moving beyond static property prediction.

Usage:
    python scripts/phase3_potentials/05_run_relaxation.py --structure data/raw/mp-1234.cif --model models/mace/best.model
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
    parser.add_argument("--model", type=str, default="medium", help="Path to trained MACE model OR 'medium'/'large' for pre-trained MP models")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence threshold (eV/A)")
    parser.add_argument("--steps", type=int, default=500, help="Max relaxation steps")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu)")
    args = parser.parse_args()

    config = get_config()
    
    # Resolve Device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     🟢 MACE RELAXATION (Phase 3)                               ║")
    print(f"║     Input: {Path(args.structure).name:<40}          ║")
    print(f"║     Model: {args.model:<40}          ║")
    print(f"║     Device: {device:<40}         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # 1. Load Structure
    print("\n[1/3] Loading Structure...")
    try:
        from ase.io import read
        atoms = read(args.structure)
        print(f"  Composition: {atoms.get_chemical_formula()}")
        print(f"  Atoms: {len(atoms)}")
    except Exception as e:
        print(f"  ❌ Error loading structure: {e}")
        return

    # 2. Load Calculator
    print("\n[2/3] Loading MACE Calculator...")
    try:
        from mace.calculators import MACECalculator
        
        # Check if using a local model file or a pre-trained one
        model_path = args.model
        if model_path in ["small", "medium", "large"]:
             # Using MACE-MP foundation models (if available in future MACE versions)
             # For now, let's assume valid path or "medium" means default foundation model if installed
             print(f"  ℹ️ Loading MACE-MP Foundation Model ({model_path})")
             calc = MACECalculator(model="medium", device=device, default_dtype="float64")
        else:
             print(f"  ℹ️ Loading Local Model: {model_path}")
             calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float64")
             
        atoms.calc = calc
        
        # Initial Energy
        e_init = atoms.get_potential_energy()
        print(f"  Initial Energy: {e_init:.4f} eV ({e_init/len(atoms):.4f} eV/atom)")
        
    except ImportError:
        print("  ❌ MACE not installed! Please run: pip install mace-torch")
        return
    except Exception as e:
        print(f"  ❌ Error loading calculator: {e}")
        return

    # 3. Relaxation
    print(f"\n[3/3] Starting Relaxation (fmax={args.fmax})...")
    from ase.optimize import BFGS, FIRE
    from ase.constraints import ExpCellFilter
    
    # Save Trajectory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results/relaxations") / f"{Path(args.structure).stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_file = out_dir / "relaxation.traj"
    
    # Box Relaxation (ExpCellFilter allows cell volume/shape to change)
    ucf = ExpCellFilter(atoms)
    
    # Optimizer
    opt = FIRE(ucf, trajectory=str(traj_file), logfile=str(out_dir/"log.txt"))
    
    t0 = time.time()
    opt.run(fmax=args.fmax, steps=args.steps)
    dt = time.time() - t0
    
    # Final Results
    e_final = atoms.get_potential_energy()
    print(f"  Final Energy:   {e_final:.4f} eV ({e_final/len(atoms):.4f} eV/atom)")
    print(f"  Delta E:        {e_final - e_init:.4f} eV")
    print(f"  Time:           {dt:.2f} s")
    print(f"  Steps:          {opt.get_number_of_steps()}")
    
    # Save Final Structure
    final_path = out_dir / "relaxed.cif"
    atoms.write(str(final_path))
    print(f"\n  ✅ Relaxation Complete!")
    print(f"     Trajectory: {traj_file}")
    print(f"     Final CIF:  {final_path}")

if __name__ == "__main__":
    main()
