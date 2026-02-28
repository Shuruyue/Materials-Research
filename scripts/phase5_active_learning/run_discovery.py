#!/usr/bin/env python3
"""
Script 06: Run ATLAS Discovery Engine

The flagship script — runs the full autonomous discovery loop to
find new topological quantum materials.

Pipeline:
    1. Load seed structures (known topological materials from JARVIS)
    2. Initialize MACE relaxer (foundation model, no training needed)
    3. Initialize GNN classifier (trained or heuristic)
    4. Run active learning loop:
       Generate → Relax → Classify → Score → Select → Repeat
    5. Output ranked candidate list

Usage:
    python scripts/phase5_active_learning/run_discovery.py                         # Default (5 iterations)
    python scripts/phase5_active_learning/run_discovery.py --iterations 20         # More iterations
    python scripts/phase5_active_learning/run_discovery.py --candidates 100        # More candidates/iter
    python scripts/phase5_active_learning/run_discovery.py --no-mace               # Skip MACE (fast mode)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_seed_structures(n_seeds: int = 20):
    """Load seed structures from JARVIS topological materials."""
    from jarvis.core.atoms import Atoms as JAtoms

    from atlas.data.jarvis_client import JARVISClient

    print("\n=== Loading Seed Structures ===\n")
    client = JARVISClient()

    # Get topological materials from JARVIS
    topo_df = client.get_topological_materials()

    # Select diverse seeds: sort by spillage (most topological first)
    topo_df = topo_df.sort_values("spillage", ascending=False)

    structures = []
    formulas = set()

    for _, row in topo_df.iterrows():
        if len(structures) >= n_seeds:
            break
        try:
            jatoms = JAtoms.from_dict(row["atoms"])
            struct = jatoms.pymatgen_converter()
            formula = struct.composition.reduced_formula

            # Avoid duplicate formulas
            if formula not in formulas and len(struct) <= 30:
                structures.append(struct)
                formulas.add(formula)
                spillage = row.get("spillage", 0)
                print(f"  Seed {len(structures):2d}: {formula:15s} "
                      f"(spillage={spillage:.3f}, {len(struct)} atoms)")
        except Exception:
            continue

    print(f"\n  Loaded {len(structures)} seed structures")
    return structures


def load_classifier(cfg):
    """Try to load a trained GNN classifier, or return None."""
    import torch

    from atlas.topology.classifier import CrystalGraphBuilder, TopoGNN

    model_path = cfg.paths.models_dir / "topo_classifier" / "best_model.pt"
    builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12)

    if model_path.exists():
        print(f"\n  Loading trained GNN classifier from {model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = TopoGNN(
            node_dim=len(builder.ELEMENTS) + 5,
            edge_dim=20,
            hidden_dim=128,
            n_layers=3,
        ).to(device)

        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()
        return model, builder
    else:
        print(f"\n  No trained classifier found at {model_path}")
        print("  Using heuristic topological scoring instead")
        print("  (Train one with: python scripts/phase4_topology/train_topo_classifier.py)")
        return None, builder


def main():
    from atlas.active_learning.acquisition import DISCOVERY_ACQUISITION_STRATEGIES

    parser = argparse.ArgumentParser(description="ATLAS Discovery Engine")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of active learning iterations (default: 5)")
    parser.add_argument("--candidates", type=int, default=50,
                        help="Candidates per iteration (default: 50)")
    parser.add_argument("--top", type=int, default=10,
                        help="Top N selected per iteration (default: 10)")
    parser.add_argument("--seeds", type=int, default=15,
                        help="Number of seed structures (default: 15)")
    parser.add_argument("--no-mace", action="store_true",
                        help="Skip MACE relaxation (faster, less accurate)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional run id for isolated results directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest run (or specified --run-id)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Explicit results directory (overrides --run-id/--resume)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve run directory and manifest, then exit")
    parser.add_argument(
        "--acq-strategy",
        type=str,
        default="hybrid",
        choices=sorted(DISCOVERY_ACQUISITION_STRATEGIES),
        help="Acquisition strategy: hybrid/stability/ei/pi/ucb/lcb/thompson/mean/uncertainty",
    )
    parser.add_argument(
        "--acq-kappa",
        type=float,
        default=2.0,
        help="Exploration factor for UCB/LCB family strategies",
    )
    parser.add_argument(
        "--acq-best-f",
        type=float,
        default=-0.5,
        help="Reference best objective value for EI/PI (formation-energy target)",
    )
    parser.add_argument(
        "--acq-jitter",
        type=float,
        default=0.01,
        help="Jitter used in EI/PI stability acquisition",
    )
    args = parser.parse_args()

    from atlas.config import get_config
    from atlas.training.run_utils import resolve_run_dir, write_run_manifest

    cfg = get_config()
    print(cfg.summary())

    project_root = Path(__file__).resolve().parents[2]
    managed_run = args.results_dir is not None or args.run_id is not None or args.resume

    try:
        if args.results_dir:
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            created_new = not any(results_dir.glob("iteration_*.json"))
        elif managed_run:
            base_results = cfg.paths.data_dir / "discovery_results"
            results_dir, created_new = resolve_run_dir(
                base_results,
                resume=args.resume,
                run_id=args.run_id,
            )
        else:
            results_dir = cfg.paths.data_dir / "discovery_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            created_new = False
    except Exception as exc:
        print(f"\n  [ERROR] Failed to resolve results directory: {exc}")
        return 2

    manifest_path = write_run_manifest(
        results_dir,
        args=args,
        extra={
            "phase": "phase5",
            "module": "active_learning_discovery",
            "created_new_run_dir": created_new,
        },
        project_root=project_root,
    )
    print(f"\n  Results directory: {results_dir}")
    print(f"  Run manifest: {manifest_path}")

    if args.dry_run:
        write_run_manifest(
            results_dir,
            args=args,
            extra={
                "phase": "phase5",
                "status": "dry_run",
            },
            project_root=project_root,
        )
        print("\n[OK] Dry run complete. No discovery loop executed.")
        return 0

    t_start = time.time()

    # ─── Step 1: Load Seeds ───
    seeds = load_seed_structures(n_seeds=args.seeds)

    # ─── Step 2: Initialize Structure Generator ───
    from atlas.active_learning.generator import StructureGenerator
    generator = StructureGenerator(seed_structures=seeds, rng_seed=42)
    print(f"\n  Structure generator initialized with {len(seeds)} seeds")

    # ─── Step 3: Initialize MACE Relaxer ───
    relaxer = None
    if not args.no_mace:
        try:
            from atlas.potentials.mace_relaxer import MACERelaxer
            relaxer = MACERelaxer(use_foundation=True, model_size="large")
            # Trigger lazy loading
            _ = relaxer.calculator
            print("  MACE relaxer: ready")
        except Exception as e:
            print(f"  MACE relaxer: not available ({e})")
            print("  Continuing with heuristic stability estimates")
            relaxer = None
    else:
        print("\n  Skipping MACE relaxation (--no-mace)")

    # ─── Step 4: Initialize GNN Classifier ───
    classifier, graph_builder = load_classifier(cfg)

    # ─── Step 5: Initialize Discovery Controller ───
    from atlas.active_learning.controller import DiscoveryController

    controller = DiscoveryController(
        generator=generator,
        relaxer=relaxer,
        classifier_model=classifier,
        graph_builder=graph_builder,
        w_topo=0.35,
        w_stability=0.25,
        w_heuristic=0.20,
        w_novelty=0.20,
        acquisition_strategy=args.acq_strategy,
        acquisition_kappa=args.acq_kappa,
        acquisition_best_f=args.acq_best_f,
        acquisition_jitter=args.acq_jitter,
        results_dir=results_dir,
    )
    print(
        "  Acquisition: "
        f"strategy={args.acq_strategy}, "
        f"kappa={args.acq_kappa}, "
        f"best_f={args.acq_best_f}, "
        f"jitter={args.acq_jitter}"
    )

    # ─── Step 6: RUN DISCOVERY ───
    print(f"\n{'═' * 70}")
    print("  ██  ATLAS Discovery Engine  ██")
    print("  ██  Searching for New Topological Materials  ██")
    print(f"{'═' * 70}")

    controller.run_discovery_loop(
        n_iterations=args.iterations,
        n_candidates_per_iter=args.candidates,
        n_select_top=args.top,
    )

    # ─── Step 7: Save Report ───
    report_name = f"final_report_{results_dir.name}.json" if managed_run else "final_report.json"
    controller.save_final_report(filename=report_name)
    write_run_manifest(
        results_dir,
        args=args,
        extra={
            "phase": "phase5",
            "status": "completed",
            "iterations_completed": controller.iteration,
            "report_file": report_name,
        },
        project_root=project_root,
    )

    t_total = time.time() - t_start
    print(f"\n  Total runtime: {t_total:.1f}s ({t_total/60:.1f} min)")
    print("\n[OK] Discovery complete.")
    print(f"  Results: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
