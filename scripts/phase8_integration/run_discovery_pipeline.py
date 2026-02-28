#!/usr/bin/env python3
"""
Phase 8: end-to-end integration demo for alchemy + stability + transport.

This script is intentionally resilient:
- If alchemy/MEPIN/LiFlow is unavailable, it falls back and keeps running.
- Every run writes a machine-readable summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import ase.build
import numpy as np
import torch
from ase.calculators.lj import LennardJones

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional visualization
try:
    from pymatviz import ptable_heatmap

    HAS_PYMATVIZ = True
except ImportError:
    HAS_PYMATVIZ = False


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        run_name = args.run_id if args.run_id else time.strftime("run_%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parents[2] / "artifacts" / "phase8_integration" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def grand_loop(args: argparse.Namespace) -> dict:
    print("=== ATLAS: Grand Discovery Loop ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    np.random.seed(args.seed)

    summary = {
        "timestamp": time.time(),
        "run_id": args.run_id,
        "device": device,
        "seed": args.seed,
        "steps": {
            "composition_steps": args.composition_steps,
            "mepin_images": args.mepin_images,
            "liflow_steps": args.liflow_steps,
            "liflow_flow_steps": args.liflow_flow_steps,
        },
    }

    # Step 1: System definition
    print("\n[Step 1] Defining System: NaCl (Na/K substitution)...")
    atoms = ase.build.bulk("NaCl", crystalstructure="rocksalt", a=5.64).repeat((2, 2, 2))
    print(f"Structure: {atoms.get_chemical_formula()}")

    na_indices = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 11]
    alchemical_pairs = [[(idx, 11), (idx, 19)] for idx in na_indices]
    initial_weights = [[0.5, 0.5] for _ in na_indices]
    summary["n_alchemical_sites"] = len(na_indices)

    # Step 2: Alchemical optimization
    print("\n[Step 2] Optimizing Composition...")
    energy_history: list[float] = []
    alchemy_status = "skipped" if args.skip_alchemy else "ok"
    alchemy_error = ""
    used_fallback = False

    if not args.skip_alchemy:
        try:
            from atlas.discovery.alchemy import AlchemicalMACECalculator
            from atlas.discovery.alchemy.optimizer import CompositionOptimizer

            calc = AlchemicalMACECalculator(
                atoms=atoms,
                alchemical_pairs=alchemical_pairs,
                alchemical_weights=initial_weights,
                device=device,
                model_size="medium",
            )
            atoms.calc = calc
            optimizer = CompositionOptimizer(calc, learning_rate=0.1)
            for step_idx in range(args.composition_steps):
                energy, _ = optimizer.step()
                energy_history.append(float(energy))
                print(f"  Step {step_idx + 1:2d}: Energy = {energy:.4f} eV")

            final_weights = calc.alchemy_manager.alchemical_weights.detach().cpu().numpy()
            new_numbers = atoms.get_atomic_numbers()
            for site_idx, weights in zip(na_indices, final_weights):
                if weights[1] > weights[0]:
                    new_numbers[site_idx] = 19
            atoms.set_atomic_numbers(new_numbers)
        except Exception as exc:
            alchemy_status = "fallback"
            alchemy_error = str(exc)
            used_fallback = True

    if args.skip_alchemy or used_fallback:
        if used_fallback:
            print(f"  Alchemy failed ({alchemy_error}), using fallback composition.")
        else:
            print("  Skipped alchemy optimization (--skip-alchemy).")
        new_numbers = atoms.get_atomic_numbers()
        for idx, site_idx in enumerate(na_indices):
            if idx % 2 == 0:
                new_numbers[site_idx] = 19
        atoms.set_atomic_numbers(new_numbers)
        atoms.calc = LennardJones()

    optimized_formula = atoms.get_chemical_formula()
    print(f"Optimized Formula: {optimized_formula}")
    summary["alchemy"] = {
        "status": alchemy_status,
        "error": alchemy_error,
        "energy_history": energy_history,
        "optimized_formula": optimized_formula,
    }

    # Step 3: Stability (MEPIN)
    print("\n[Step 3] Assessing Stability (MEPIN)...")
    mepin_status = "skipped" if args.skip_mepin else "ok"
    mepin_error = ""
    mepin_images = 0

    if not args.skip_mepin:
        reactant = atoms.copy()
        reactant.pbc = False
        reactant.center(vacuum=5.0)

        product = reactant.copy()
        pos = product.get_positions()
        product.set_positions(pos + np.random.normal(0, 0.2, pos.shape))

        try:
            from atlas.discovery.stability.mepin import MEPINStabilityEvaluator

            stab_eval = MEPINStabilityEvaluator(model_type="cyclo_L", device=device)
            path = stab_eval.predict_path(reactant, product, num_images=args.mepin_images)
            mepin_images = len(path)
            print(f"  MEPIN completed with {mepin_images} path images.")
        except Exception as exc:
            mepin_status = "failed"
            mepin_error = str(exc)
            print(f"  MEPIN failed: {exc}")
    else:
        print("  Skipped MEPIN (--skip-mepin).")

    summary["mepin"] = {
        "status": mepin_status,
        "error": mepin_error,
        "images": mepin_images,
    }

    # Step 4: Transport (LiFlow)
    print("\n[Step 4] Predicting Transport (LiFlow)...")
    liflow_status = "skipped" if args.skip_liflow else "ok"
    liflow_error = ""
    liflow_frames = 0

    if not args.skip_liflow:
        transport_atoms = atoms.copy()
        transport_atoms.pbc = True
        try:
            from atlas.discovery.transport.liflow import LiFlowEvaluator

            liflow = LiFlowEvaluator(device=device, temp_list=[600])
            traj, _ = liflow.simulate(
                transport_atoms,
                steps=args.liflow_steps,
                flow_steps=args.liflow_flow_steps,
            )
            liflow_frames = len(traj)
            print(f"  LiFlow completed with {liflow_frames} trajectory frames.")
        except Exception as exc:
            liflow_status = "failed"
            liflow_error = str(exc)
            print(f"  LiFlow failed: {exc}")
    else:
        print("  Skipped LiFlow (--skip-liflow).")

    summary["liflow"] = {
        "status": liflow_status,
        "error": liflow_error,
        "trajectory_frames": liflow_frames,
    }

    # Step 5: Visualization
    print("\n[Step 5] Visualization...")
    counts = dict(Counter(atoms.get_chemical_symbols()))
    summary["composition_counts"] = counts
    summary["pymatviz"] = {"available": HAS_PYMATVIZ, "generated": False, "error": ""}
    if HAS_PYMATVIZ:
        try:
            # Trigger plot creation as a sanity check; saving is optional.
            _ = ptable_heatmap(counts)
            summary["pymatviz"]["generated"] = True
            print("  pymatviz heatmap object generated.")
        except Exception as exc:
            summary["pymatviz"]["error"] = str(exc)
            print(f"  Visualization failed: {exc}")
    else:
        print("  pymatviz not found, skipped visualization.")

    print("\n=== Grand Loop Finished ===")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ATLAS Phase 8 integrated discovery pipeline")
    parser.add_argument("--composition-steps", type=int, default=5, help="Alchemy optimizer steps")
    parser.add_argument("--mepin-images", type=int, default=5, help="Number of path images for MEPIN")
    parser.add_argument("--liflow-steps", type=int, default=20, help="LiFlow trajectory steps")
    parser.add_argument("--liflow-flow-steps", type=int, default=5, help="LiFlow inner flow steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-id", type=str, default=None, help="Run id (used by default output path)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for summary JSON")
    parser.add_argument("--skip-alchemy", action="store_true", help="Skip alchemy optimization stage")
    parser.add_argument("--skip-mepin", action="store_true", help="Skip MEPIN stability stage")
    parser.add_argument("--skip-liflow", action="store_true", help="Skip LiFlow transport stage")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = _resolve_output_dir(args)
    summary_path = out_dir / "summary.json"

    print(f"Output directory: {out_dir}")
    if args.dry_run:
        payload = {
            "dry_run": True,
            "args": vars(args),
            "output_dir": str(out_dir),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] Dry-run summary saved: {summary_path}")
        return 0

    summary = grand_loop(args)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[OK] Summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
