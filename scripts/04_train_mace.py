#!/usr/bin/env python3
"""
Script 04: Train MACE Model

Trains a MACE equivariant neural network potential on the prepared
training data. Uses energy-only training (forces optional).

This creates a universal interatomic potential that can:
- Relax crystal structures (geometry optimization)
- Run molecular dynamics simulations
- Estimate formation energies

Usage:
    python scripts/04_train_mace.py                     # Default settings
    python scripts/04_train_mace.py --epochs 100        # Quick test
    python scripts/04_train_mace.py --r-max 6.0         # Larger cutoff
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.config import get_config


def check_gpu():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return "cuda"
        else:
            print("  GPU: Not available, using CPU (will be slow)")
            return "cpu"
    except ImportError:
        print("  PyTorch not installed!")
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train MACE model")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--r-max", type=float, default=None, help="Cutoff radius (Å)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--energy-only", action="store_true", default=True,
                        help="Train on energy only (default for JARVIS data)")
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
        print(f"\n  ✗ Training data not found at {train_file}")
        print(f"  Run first: python scripts/03_prepare_mace_data.py")
        return

    # Count structures
    from ase.io import read as ase_read
    n_train = len(ase_read(str(train_file), index=":"))
    n_val = len(ase_read(str(val_file), index=":"))
    print(f"\n  Training structures:   {n_train}")
    print(f"  Validation structures: {n_val}")

    # Set up MACE training
    model_dir = cfg.paths.models_dir / "mace"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== MACE Training Configuration ===")
    print(f"  Cutoff (r_max):   {r_max} Å")
    print(f"  Max L:            {mace_cfg.max_ell}")
    print(f"  Interactions:     {mace_cfg.num_interactions}")
    print(f"  Hidden irreps:    {mace_cfg.hidden_irreps}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Learning rate:    {lr}")
    print(f"  Max epochs:       {epochs}")
    print(f"  Device:           {device}")
    print(f"  Model output:     {model_dir}")

    # Use MACE's built-in training script via subprocess
    # This is the recommended way to train MACE models
    import subprocess

    mace_cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", "atlas_mace_v1",
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
        "--work_dir", str(model_dir),
        "--save_cpu",
    ]

    print(f"\n=== Starting MACE Training ===\n")
    print(f"  Command: {' '.join(mace_cmd[:6])} ...")

    try:
        result = subprocess.run(
            mace_cmd,
            cwd=str(model_dir),
            capture_output=False,
            text=True,
        )

        if result.returncode == 0:
            print(f"\n✓ Training complete! Model saved to {model_dir}")
        else:
            print(f"\n✗ Training failed with return code {result.returncode}")

    except FileNotFoundError:
        print("\n  ✗ MACE training script not found.")
        print("  Install MACE: pip install mace-torch")
        print("\n  Alternative: train using MACE Python API:")
        print_python_training_alternative(cfg, train_file, val_file, device)


def print_python_training_alternative(cfg, train_file, val_file, device):
    """Print instructions for Python API training if CLI fails."""
    print(f"""
    ─────────────────────────────────────────
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
    ─────────────────────────────────────────
    """)


if __name__ == "__main__":
    main()
