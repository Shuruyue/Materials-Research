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
import math
import re
import sys
import time
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_SAFE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_STATE_DICT_CONTAINER_KEYS = ("state_dict", "model_state_dict", "model")


def _is_safe_run_id(run_id: str | None) -> bool:
    if not run_id:
        return True
    if run_id in {".", ".."}:
        return False
    if "/" in run_id or "\\" in run_id:
        return False
    return bool(_SAFE_RUN_ID_PATTERN.fullmatch(run_id))


def _coerce_int_with_bounds(
    value: object,
    *,
    arg_name: str,
    min_value: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{arg_name} must be an integer")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise ValueError(f"{arg_name} must be an integer")
        number = int(number_f)
    else:
        try:
            number = int(value)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"{arg_name} must be an integer") from exc
    if min_value is not None and number < min_value:
        comparator = "> 0" if min_value == 1 else f">= {min_value}"
        raise ValueError(f"{arg_name} must be {comparator}")
    return number


def _coerce_finite_float(
    value: object,
    *,
    arg_name: str,
    min_value: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{arg_name} must be a finite number")
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"{arg_name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{arg_name} must be finite")
    if min_value is not None and number < float(min_value):
        comparator = f">= {min_value}"
        raise ValueError(f"{arg_name} must be {comparator}")
    return number


def _looks_like_state_dict(payload: Mapping[object, object]) -> bool:
    if not payload:
        return False
    for key in payload:
        if not isinstance(key, str) or not key:
            return False
    return any(
        "." in key
        or key in {"weight", "bias"}
        or key.endswith(("weight", "bias", "running_mean", "running_var", "num_batches_tracked"))
        for key in payload
    )


def _extract_classifier_state_dict(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Invalid classifier checkpoint payload: {type(payload)!r}")

    candidate: Mapping[object, object] | None = None
    for container_key in _STATE_DICT_CONTAINER_KEYS:
        nested = payload.get(container_key)
        if isinstance(nested, Mapping) and _looks_like_state_dict(nested):
            candidate = nested
            break

    if candidate is None:
        if not _looks_like_state_dict(payload):
            raise TypeError("Invalid classifier checkpoint payload: missing state dict")
        candidate = payload

    normalized: dict[str, object] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise TypeError("Invalid classifier checkpoint payload: non-string state_dict key")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not key:
            raise TypeError("Invalid classifier checkpoint payload: empty normalized key")
        normalized[key] = value
    if not normalized:
        raise TypeError("Invalid classifier checkpoint payload: empty state dict")
    return normalized


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
        try:
            checkpoint_payload = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            # Compatibility for older torch versions without weights_only support.
            checkpoint_payload = torch.load(model_path, map_location=device)
        state_dict = _extract_classifier_state_dict(checkpoint_payload)
        model.load_state_dict(state_dict)
        model.eval()
        return model, builder
    else:
        print(f"\n  No trained classifier found at {model_path}")
        print("  Using heuristic topological scoring instead")
        print("  (Train one with: python scripts/phase4_topology/train_topo_classifier.py)")
        return None, builder


def _validate_discovery_args(args: argparse.Namespace) -> tuple[bool, str]:
    positive_int_fields = ("iterations", "candidates", "top", "seeds", "calibration_window")
    for field in positive_int_fields:
        value = getattr(args, field, None)
        if value is None:
            continue
        try:
            normalized = _coerce_int_with_bounds(value, arg_name=f"--{field.replace('_', '-')}", min_value=1)
        except ValueError as exc:
            return False, str(exc)
        setattr(args, field, normalized)

    if int(args.top) > int(args.candidates):
        return False, "--top cannot be greater than --candidates"
    try:
        args.acq_kappa = _coerce_finite_float(args.acq_kappa, arg_name="--acq-kappa", min_value=0.0)
    except ValueError:
        return False, "--acq-kappa must be finite and >= 0"
    try:
        args.acq_jitter = _coerce_finite_float(args.acq_jitter, arg_name="--acq-jitter", min_value=0.0)
    except ValueError:
        return False, "--acq-jitter must be finite and >= 0"
    try:
        args.acq_best_f = _coerce_finite_float(args.acq_best_f, arg_name="--acq-best-f")
    except ValueError:
        return False, "--acq-best-f must be finite"
    run_id = getattr(args, "run_id", None)
    if not _is_safe_run_id(run_id):
        return False, "--run-id contains unsafe characters"
    results_dir = getattr(args, "results_dir", None)
    if results_dir is not None and not str(results_dir).strip():
        return False, "--results-dir must not be empty"
    if bool(run_id) and bool(results_dir):
        return False, "--run-id and --results-dir cannot be used together"
    if bool(getattr(args, "resume", False)) and bool(results_dir):
        return False, "--resume and --results-dir cannot be used together"
    return True, ""


def _build_parser() -> argparse.ArgumentParser:
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
        help="Acquisition strategy (see choices)",
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
    parser.add_argument(
        "--policy",
        type=str,
        choices=["legacy", "cmoeic"],
        default="legacy",
        help="Decision policy engine",
    )
    parser.add_argument(
        "--risk-mode",
        type=str,
        choices=["soft", "hard", "hybrid"],
        default="soft",
        help="Risk handling mode for policy decisions",
    )
    parser.add_argument(
        "--cost-aware",
        action="store_true",
        help="Enable explicit cost-aware utility adjustment in policy scoring",
    )
    parser.add_argument(
        "--calibration-window",
        type=int,
        default=128,
        help="History window size for calibration updates in policy engine",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    ok, message = _validate_discovery_args(args)
    if not ok:
        print(f"\n  [ERROR] {message}", file=sys.stderr)
        return 2

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
        print(f"\n  [ERROR] Failed to resolve results directory: {exc}", file=sys.stderr)
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
        policy_name=args.policy,
        risk_mode=args.risk_mode,
        cost_aware=args.cost_aware,
        calibration_window=args.calibration_window,
        results_dir=results_dir,
    )
    print(
        "  Acquisition: "
        f"strategy={args.acq_strategy}, "
        f"kappa={args.acq_kappa}, "
        f"best_f={args.acq_best_f}, "
        f"jitter={args.acq_jitter}, "
        f"policy={args.policy}, "
        f"risk_mode={args.risk_mode}, "
        f"cost_aware={args.cost_aware}, "
        f"calibration_window={args.calibration_window}"
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
