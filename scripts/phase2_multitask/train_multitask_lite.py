"""
Phase 2: Multi-Task E(3)-Equivariant GNN — LITE Tier (Fast Debug)

Designed for rapid validation of the pipeline.
- Uses a tiny model (preset='small')
- Trains for only 2 epochs
- Limtied to 100 samples
- Verifies that data loading, model forward pass, and backprop work without crashing.

Usage:
    python scripts/phase2_multitask/train_multitask_lite.py
    python scripts/phase2_multitask/train_multitask_lite.py --property-group core4
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import (
    CrystalPropertyDataset,
    PHASE2_PROPERTY_GROUP_CHOICES,
    resolve_phase2_property_group,
)
from atlas.models.equivariant import EquivariantGNN
from atlas.models.multi_task import MultiTaskGNN
from atlas.training.checkpoint import CheckpointManager
from atlas.training.metrics import scalar_metrics
from atlas.training.run_utils import resolve_run_dir, write_run_manifest
from atlas.console_style import install_console_style

install_console_style()

# ── Lite Config ──
PROPERTIES = resolve_phase2_property_group("core4")

# 2. Tiny Model Preset
LITE_PRESET = {
    "irreps": "16x0e + 8x1o",     # Tiny capacity
    "max_ell": 1,                 # Simple spherical harmonics
    "n_layers": 2,                # Shallow
    "n_radial": 10,
    "radial_hidden": 32,
    "head_hidden": 16,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2 multitask lite debug training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=20)
    parser.add_argument(
        "--property-group",
        choices=PHASE2_PROPERTY_GROUP_CHOICES,
        default="core4",
        help="Phase 2 property group (lite defaults to core4)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run id (without or with 'run_' prefix)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Keep top-k best checkpoints")
    parser.add_argument("--keep-last-k", type=int, default=3,
                        help="Keep latest rotating checkpoints")
    args = parser.parse_args()

    global PROPERTIES
    PROPERTIES = resolve_phase2_property_group(args.property_group)

    print("=" * 66)
    print("E3NN LITE (Debug Mode)")
    print("Fast Validation / Smoke Test")
    print(f"Property group: {args.property_group} ({len(PROPERTIES)} tasks)")
    print("=" * 66)

    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name()}")
    
    base_dir = config.paths.models_dir / "multitask_lite_e3nn"
    try:
        save_dir, created_new = resolve_run_dir(base_dir, resume=args.resume, run_id=args.run_id)
    except (FileNotFoundError, FileExistsError) as e:
        print(f"[ERROR] {e}")
        return 2
    run_msg = "Starting new run" if created_new else "Using existing run"
    print(f"[INFO] {run_msg}: {save_dir.name}")
    manager = CheckpointManager(save_dir, top_k=args.top_k, keep_last_k=args.keep_last_k)
    manifest_path = write_run_manifest(
        save_dir,
        args=args,
        project_root=Path(__file__).resolve().parent.parent.parent,
        extra={
            "status": "started",
            "phase": "phase2",
            "model_family": "multitask_lite_e3nn",
        },
    )
    print(f"[INFO] Run manifest: {manifest_path}")

    # ── 1. Data (Lite: tiny samples) ──
    print(f"\n[1/4] Loading Lite Dataset (train={args.train_samples}, eval={args.eval_samples})...")
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(
            properties=PROPERTIES,
            max_samples=args.train_samples if split == "train" else args.eval_samples,
            split=split
        )
        ds.prepare()
        datasets[split] = ds
        
    print(f"    Train: {len(datasets['train'])}, Val: {len(datasets['val'])}")

    # ── 2. Model (Lite: Tiny) ──
    print("\n[2/4] Building Lite Model...")
    encoder = EquivariantGNN(
        irreps_hidden=LITE_PRESET["irreps"],
        max_ell=LITE_PRESET["max_ell"],
        n_layers=LITE_PRESET["n_layers"],
        max_radius=5.0,
        n_species=86, # Full periodic table
        n_radial_basis=LITE_PRESET["n_radial"],
        radial_hidden=LITE_PRESET["radial_hidden"],
        output_dim=1 # Dummy, handled by multitask wrapper
    )
    
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={p: {"type": "scalar"} for p in PROPERTIES},
        embed_dim=encoder.scalar_dim
    ).to(device)
    
    print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Train Loop (Lite: 2 Epochs) ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() 

    print(f"\n[3/4] Testing Training Loop ({args.epochs} Epochs)...")
    model.train()
    
    from torch_geometric.loader import DataLoader
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)
    checkpoint_path = save_dir / "checkpoint.pt"
    history = {"train_loss": []}
    start_epoch = 1
    best_train_loss = float("inf")
    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        history = ckpt.get("history", history)
        best_train_loss = float(ckpt.get("best_train_loss", best_train_loss))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[INFO] Resume epoch: {start_epoch}")
    elif args.resume:
        print(f"[WARN] --resume requested but checkpoint not found: {checkpoint_path}")

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            preds = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
            
            # Simple loss (just checking gradients)
            loss = 0
            for prop in PROPERTIES:
                if prop in preds:
                    target = getattr(batch, prop).view(-1, 1)
                    # Handle NaNs roughly for debug
                    mask = ~torch.isnan(target)
                    if mask.sum() > 0:
                        loss += criterion(preds[prop][mask], target[mask])
            
            if loss != 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        mean_loss = float(total_loss / max(len(loader), 1))
        history["train_loss"].append(mean_loss)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_train_loss": best_train_loss,
        }
        if mean_loss < best_train_loss:
            best_train_loss = mean_loss
            state["best_train_loss"] = best_train_loss
            manager.save_best(state, best_train_loss, epoch)
        manager.save_checkpoint(state, epoch)

        print(f"    Epoch {epoch}: Loss = {mean_loss:.4f}")

    print("\n[4/4] [OK] Lite Test Passed! The pipeline is functional.")
    print("      You can now proceed to 'Std' or 'Pro' tiers.")

    results = {
        "algorithm": "e3nn_multitask_lite",
        "run_id": save_dir.name,
        "property_group": args.property_group,
        "properties": list(PROPERTIES),
        "best_train_loss": best_train_loss,
        "total_epochs": last_epoch,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "property_group": args.property_group,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "top_k": args.top_k,
            "keep_last_k": args.keep_last_k,
        },
    }
    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    write_run_manifest(
        save_dir,
        args=args,
        project_root=Path(__file__).resolve().parent.parent.parent,
        extra={
            "status": "completed",
            "result": {
                "best_train_loss": float(best_train_loss),
                "total_epochs": int(last_epoch),
            },
        },
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


