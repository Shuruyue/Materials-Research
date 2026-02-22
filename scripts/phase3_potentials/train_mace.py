#!/usr/bin/env python3
"""
Train MACE Model

Trains a MACE equivariant neural network potential on the prepared
training data. Uses energy-only training (forces optional).

This creates a universal interatomic potential that can:
- Relax crystal structures (geometry optimization)
- Run molecular dynamics simulations
- Estimate formation energies

Usage:
    python scripts/phase3_potentials/train_mace.py                     # Default settings
    python scripts/phase3_potentials/train_mace.py --epochs 100        # Quick test
    python scripts/phase3_potentials/train_mace.py --r-max 6.0         # Larger cutoff
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atlas.config import get_config
from atlas.training.run_utils import resolve_run_dir, write_run_manifest
from atlas.console_style import install_console_style

install_console_style()


def check_gpu():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_memory = getattr(props, "total_memory", None)
            if total_memory is None:
                total_memory = getattr(props, "total_mem", 0)
            gpu_mem = float(total_memory) / 1e9 if total_memory else 0.0
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return "cuda"
        else:
            print("  GPU: Not available, using CPU (will be slow)")
            return "cpu"
    except ImportError:
        print("  PyTorch not installed!")
        return "cpu"


def _mace_cli_supports(flag: str) -> bool:
    """Check whether MACE CLI supports a specific flag."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mace.cli.run_train", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    hay = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return flag in hay


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MACE model")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--r-max", type=float, default=None, help="Cutoff radius (Angstrom)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from latest run checkpoint if supported")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run id (without or with 'run_' prefix)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Compatibility flag with unified launchers (unused by MACE CLI backend)",
    )
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=3,
        help="Compatibility flag with unified launchers (unused by MACE CLI backend)",
    )
    parser.add_argument(
        "--energy-only",
        dest="energy_only",
        action="store_true",
        default=True,
        help="Train on energy only (default for JARVIS data)",
    )
    parser.add_argument(
        "--with-forces",
        dest="energy_only",
        action="store_false",
        help="Enable force/stress loss weights (requires force labels)",
    )
    args = parser.parse_args()

    cfg = get_config()
    print(cfg.summary())

    device = check_gpu()

    # Override config with CLI args
    mace_cfg = cfg.mace
    epochs = args.epochs or mace_cfg.max_epochs
    r_max = args.r_max or mace_cfg.r_max
    batch_size = args.batch_size or mace_cfg.batch_size
    lr = args.lr or mace_cfg.lr

    # Check training data exists
    data_dir = cfg.paths.processed_dir / "mace_training"
    train_file = data_dir / "train.xyz"
    val_file = data_dir / "val.xyz"
    test_file = data_dir / "test.xyz"

    if not train_file.exists():
        print(f"\n  [ERROR] Training data not found at {train_file}")
        print(f"  Run first: python scripts/phase3_potentials/prepare_mace_data.py")
        return 2

    # Count structures
    from ase.io import read as ase_read
    n_train = len(ase_read(str(train_file), index=":"))
    n_val = len(ase_read(str(val_file), index=":"))
    print(f"\n  Training structures:   {n_train}")
    print(f"  Validation structures: {n_val}")

    # Set up MACE training
    model_base_dir = cfg.paths.models_dir / "mace"
    try:
        run_dir, created_new = resolve_run_dir(
            model_base_dir,
            resume=args.resume,
            run_id=args.run_id,
        )
    except (FileNotFoundError, FileExistsError) as e:
        print(f"\n  [ERROR] {e}")
        return 2
    run_msg = "Starting new run" if created_new else "Using existing run"
    print(f"  [INFO] {run_msg}: {run_dir.name}")
    manifest_path = write_run_manifest(
        run_dir,
        args=args,
        project_root=Path(__file__).resolve().parent.parent.parent,
        extra={
            "status": "started",
            "phase": "phase3",
            "model_family": "mace",
        },
    )
    print(f"  [INFO] Run manifest: {manifest_path}")

    print(f"\n=== MACE Training Configuration ===")
    print(f"  Cutoff (r_max):   {r_max} Angstrom")
    print(f"  Max L:            {mace_cfg.max_ell}")
    print(f"  Interactions:     {mace_cfg.num_interactions}")
    print(f"  Hidden irreps:    {mace_cfg.hidden_irreps}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Learning rate:    {lr}")
    print(f"  Max epochs:       {epochs}")
    print(f"  Device:           {device}")
    print(f"  Model output:     {run_dir}")

    # Use MACE's built-in training script via subprocess
    # This is the recommended way to train MACE models
    mace_cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", run_dir.name,
        "--train_file", str(train_file),
        "--valid_file", str(val_file),
        "--test_file", str(test_file),
        "--model", "MACE",
        "--num_interactions", str(mace_cfg.num_interactions),
        "--max_ell", str(mace_cfg.max_ell),
        "--correlation", str(mace_cfg.correlation),
        "--hidden_irreps", mace_cfg.hidden_irreps,
        "--r_max", str(r_max),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--max_num_epochs", str(epochs),
        "--patience", str(mace_cfg.patience),
        "--weight_decay", str(mace_cfg.weight_decay),
        "--energy_weight", str(mace_cfg.energy_weight),
        "--forces_weight", str(0.0 if args.energy_only else mace_cfg.forces_weight),
        "--stress_weight", str(0.0 if args.energy_only else mace_cfg.stress_weight),
        "--energy_key", "REF_energy",
        "--device", device,
        "--seed", "42",
        "--work_dir", str(run_dir),
        "--save_cpu",
    ]
    if args.resume and _mace_cli_supports("--restart_latest"):
        mace_cmd.append("--restart_latest")
    elif args.resume:
        print("  [WARN] MACE CLI does not expose --restart_latest in this environment.")

    print(f"\n=== Starting MACE Training ===\n")
    print(f"  Command: {' '.join(mace_cmd[:6])} ...")

    try:
        result = subprocess.run(
            mace_cmd,
            cwd=str(run_dir),
            capture_output=False,
            text=True,
        )

        if result.returncode == 0:
            print(f"\n[OK] Training complete. Model saved to {run_dir}")
        else:
            print(f"\n[ERROR] Training failed with return code {result.returncode}")
        with open(run_dir / "training_info.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "algorithm": "mace",
                    "run_id": run_dir.name,
                    "return_code": result.returncode,
                    "config": {
                        "epochs": epochs,
                        "r_max": r_max,
                        "batch_size": batch_size,
                        "lr": lr,
                        "energy_only": args.energy_only,
                        "resume": args.resume,
                    },
                },
                f,
                indent=2,
            )
        write_run_manifest(
            run_dir,
            args=args,
            project_root=Path(__file__).resolve().parent.parent.parent,
            extra={
                "status": "completed" if result.returncode == 0 else "failed",
                "result": {
                    "return_code": int(result.returncode),
                },
            },
        )
        return int(result.returncode)

    except FileNotFoundError:
        print("\n  [ERROR] MACE training script not found.")
        print("  Install MACE: pip install mace-torch")
        print("\n  Alternative: train using MACE Python API:")
        print_python_training_alternative(cfg, train_file, val_file, device)
        write_run_manifest(
            run_dir,
            args=args,
            project_root=Path(__file__).resolve().parent.parent.parent,
            extra={
                "status": "failed",
                "result": {
                    "error": "mace_cli_not_found",
                },
            },
        )
        return 127


def print_python_training_alternative(cfg, train_file, val_file, device):
    """Print instructions for Python API training if CLI fails."""
    print(f"""
    -----------------------------------------
    You can also train MACE via the Python API:

    from mace.tools import build_default_arg_parser, run
    args = build_default_arg_parser().parse_args([
        "--train_file", "{train_file}",
        "--valid_file", "{val_file}",
        "--model", "MACE",
        "--r_max", "{cfg.mace.r_max}",
        "--max_num_epochs", "100",
        "--device", "{device}",
    ])
    run(args)
    -----------------------------------------
    """)


if __name__ == "__main__":
    raise SystemExit(main())

