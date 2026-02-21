"""
Phase 2: Multi-Task E(3)-Equivariant GNN — PRO Tier (Production)

The "Holy Grail" script for training the Universal Material Brain.
- Uses a LARGE model (preset='large')
- Trains for 500+ epochs
- Full Resume Capability (saves checkpoint.pt every epoch)
- Robust Outlier Handling (saves outliers.csv, supports --no-filter)
- Detailed JSON logging for experiment tracking

Usage:
    python scripts/phase2_multitask/train_multitask_pro.py
    python scripts/phase2_multitask/train_multitask_pro.py --resume
    python scripts/phase2_multitask/train_multitask_pro.py --no-filter  (Discovery Mode)
"""

import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
import json
import time
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset, DEFAULT_PROPERTIES
from atlas.models.equivariant import EquivariantGNN
from atlas.models.multi_task import MultiTaskGNN
from atlas.training.metrics import scalar_metrics
from atlas.training.normalizers import TargetNormalizer, MultiTargetNormalizer
from atlas.training.filters import filter_outliers
from atlas.training.checkpoint import CheckpointManager
from atlas.training.run_utils import resolve_run_dir, write_run_manifest

# ── Pro Tier Config ──
# Full discovery mode activated by --all-properties
CORE_PROPERTIES = DEFAULT_PROPERTIES

ALL_PROPERTIES = [
    "formation_energy",
    "band_gap",
    "band_gap_mbj",
    "bulk_modulus",
    "shear_modulus",
    "dielectric",
    "piezoelectric",
    "spillage",
    "ehull",
]

PRO_PRESET = {
    "irreps": "128x0e + 64x1o + 32x2e", # Optimized High capacity (Removed 3o for speed)
    "max_ell": 2, # Reduced from 3 to 2 (Massive speedup, same as Std)
    "n_layers": 4,
    "n_radial": 32,
    "radial_hidden": 256,
    "head_hidden": 128,
}



# ── Utilities ──

def pad_missing_properties(dataset, properties):
    n_padded = {p: 0 for p in properties}
    for i in range(len(dataset)):
        data = dataset[i]
        for prop in properties:
            try:
                val = getattr(data, prop)
                if val is None: raise AttributeError
            except (AttributeError, TypeError):
                setattr(data, prop, torch.tensor([float('nan')]))
                n_padded[prop] += 1
    return dataset

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks)) # Log variance

    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total
    
    def get_weights(self):
        return {p: round(float(torch.exp(-self.log_vars[i])), 4) for i, p in enumerate(PROPERTIES)}







def train_epoch(model, loss_fn, loader, optimizer, device, normalizer=None, grad_clip=0.5, accumulation_steps=4):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    n = 0
    
    optimizer.zero_grad()
    
    # Add tqdm for progress tracking
    from tqdm import tqdm
    pbar = tqdm(loader, desc="   Training", leave=False)
    
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        
        preds = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
        
        task_losses = []
        valid_tasks = 0
        for prop in PROPERTIES:
            if prop not in preds: continue
            target = getattr(batch, prop).view(-1, 1)
            mask = ~torch.isnan(target)
            if mask.sum() == 0: continue
            
            target = target[mask].view(-1, 1)
            pred = preds[prop][mask].view(-1, 1)
            
            if normalizer:
                target_norm = normalizer.normalize(prop, target)
            else:
                target_norm = target
                
            loss = nn.functional.huber_loss(pred, target_norm, delta=1.0)
            if not torch.isnan(loss):
                task_losses.append(loss)
                valid_tasks += 1
                
        if valid_tasks == 0: continue
        
        while len(task_losses) < len(PROPERTIES):
            task_losses.append(torch.tensor(0.0, device=device))
            
        loss = loss_fn(task_losses)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        n += 1
        
        # Update progress bar
        current_loss = total_loss / max(n, 1)
        pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        
    # Handle remaining gradients
    if (i + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device, normalizer=None):
    """Evaluate model on all properties."""
    model.eval()
    all_preds = {p: [] for p in PROPERTIES}
    all_targets = {p: [] for p in PROPERTIES}
    
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
        
        for prop in PROPERTIES:
            if prop not in preds: continue
            target = getattr(batch, prop).view(-1, 1)
            mask = ~torch.isnan(target)
            if mask.sum() == 0: continue
            
            target = target[mask].view(-1, 1)
            pred = preds[prop][mask].view(-1, 1)
            
            if normalizer:
                pred = normalizer.denormalize(prop, pred)
                
            all_preds[prop].append(pred.cpu())
            all_targets[prop].append(target.cpu())
            
    metrics = {}
    for prop in PROPERTIES:
        if all_preds[prop]:
            metrics.update(scalar_metrics(torch.cat(all_preds[prop]), torch.cat(all_targets[prop]), prefix=prop))
    return metrics


# ── Main ──

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-filter", action="store_true", help="Disable outlier filter")
    parser.add_argument("--all-properties", action="store_true", help="Train on ALL 9 properties (Discovery Mode)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4) # Lowered to 4 for extreme VRAM limits
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run id (without or with 'run_' prefix)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Keep top-k best checkpoints")
    parser.add_argument("--keep-last-k", type=int, default=3,
                        help="Keep latest rotating checkpoints")
    args = parser.parse_args()

    # Dynamic Property Selection
    global PROPERTIES
    if args.all_properties:
        PROPERTIES = ALL_PROPERTIES
        print("[INFO] DISCOVERY MODE: Training on ALL 9 properties!")
    else:
        PROPERTIES = CORE_PROPERTIES
        print("[INFO] STANDARD MODE: Training on core 4 properties.")

    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 66)
    print("E3NN PRO (Production Mode)")
    print(f"Tasks: {len(PROPERTIES)} Properties")
    print("=" * 66)
    
    base_dir = config.paths.models_dir / "multitask_pro_e3nn"
    try:
        save_dir, created_new = resolve_run_dir(
            base_dir,
            resume=args.resume,
            run_id=args.run_id,
        )
    except (FileNotFoundError, FileExistsError) as e:
        print(f"  [ERROR] {e}")
        return 2
    run_msg = "Starting new run" if created_new else "Using existing run"
    print(f"  [INFO] {run_msg}: {save_dir.name}")
    manifest_path = write_run_manifest(
        save_dir,
        args=args,
        project_root=Path(__file__).resolve().parent.parent.parent,
        extra={
            "status": "started",
            "phase": "phase2",
            "model_family": "multitask_pro_e3nn",
            "all_properties": bool(args.all_properties),
        },
    )
    print(f"  [INFO] Run manifest: {manifest_path}")

    # 1. Data
    print("\n[1/5] Loading Dataset...")
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(properties=PROPERTIES, split=split).prepare()
        pad_missing_properties(ds, PROPERTIES)
        datasets[split] = ds

    # 2. Outlier Filter
    if not args.no_filter:
        print("\n[2/5] Filtering Outliers (Pro: 4.0 sigma)...")
        train_data = filter_outliers(datasets["train"], PROPERTIES, save_dir, n_sigma=4.0)
        val_data = filter_outliers(datasets["val"], PROPERTIES, save_dir, n_sigma=4.0)
        test_data = filter_outliers(datasets["test"], PROPERTIES, save_dir, n_sigma=4.0)
    else:
        print("\n[2/5] [WARN] Outlier Filter DISABLED (--no-filter). Discovery Mode ON.")
        train_data = datasets["train"]
        val_data = datasets["val"]
        test_data = datasets["test"]
    
    # 3. Normalizer
    print("\n[3/5] Normalizing Targets...")
    normalizer = MultiTargetNormalizer(train_data, PROPERTIES)
    
    # Loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             num_workers=0, pin_memory=True)
    
    print(f"\n[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name()}")
    
    # 4. Model
    print("\n[4/5] Building E3NN (Large)...")
    encoder = EquivariantGNN(
        irreps_hidden=PRO_PRESET["irreps"],
        max_ell=PRO_PRESET["max_ell"],
        n_layers=PRO_PRESET["n_layers"],
        max_radius=5.0,
        n_species=86,
        n_radial_basis=PRO_PRESET["n_radial"],
        radial_hidden=PRO_PRESET["radial_hidden"],
        output_dim=1
    )
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={p: {"type": "scalar"} for p in PROPERTIES},
        embed_dim=encoder.scalar_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = UncertaintyWeightedLoss(len(PROPERTIES)).to(device)
    # Scheduler: CosineAnnealingLR (better convergence for fixed epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    start_epoch = 1
    best_val_mae = float('inf')
    history = {"train_loss": [], "val_mae": [], "weights": []}

    # Resume
    checkpoint_path = save_dir / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        print(f"    [INFO] Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_fn.load_state_dict(ckpt["loss_fn_state_dict"]) 
        # Note: Scheduler state might be tricky if params changed, but we try
        try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception: print("    [WARN] Scheduler state mismatch, restarting scheduler")
        
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt["best_val_mae"]
        history = ckpt["history"]
    elif args.resume:
        print(f"    [WARN] --resume requested but checkpoint not found: {checkpoint_path}")

    # 5. Train
    print(f"\n[5/5] Training for {args.epochs} epochs...")
    
    manager = CheckpointManager(save_dir, top_k=args.top_k, keep_last_k=args.keep_last_k)

    last_epoch = start_epoch - 1
    best_epoch = -1
    last_val_metrics = {}
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        loss = train_epoch(model, loss_fn, train_loader, optimizer, device, normalizer)
        
        val_metrics = evaluate(model, val_loader, device, normalizer)
        last_val_metrics = val_metrics
        val_maes = [val_metrics[f"{p}_MAE"] for p in PROPERTIES if f"{p}_MAE" in val_metrics]
        avg_val_mae = sum(val_maes) / len(val_maes) if val_maes else float('inf')
        
        # Scheduler step (CosineAnnealingLR steps per epoch, not on metric)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        dt = time.time() - t0
        weights = loss_fn.get_weights()
        log = f"Ep {epoch} | Loss: {loss:.3f} | {dt:.1f}s"
        # Print ALL property MAEs with better abbreviations
        prop_log = ""
        for p in PROPERTIES:
            short = p.split('_')[0][:4]
            if "energy" in p: short = "En"
            elif "gap" in p: short = "Gap"
            elif "bulk" in p: short = "Bulk"
            elif "shear" in p: short = "Shr"
            
            val = val_metrics.get(f'{p}_MAE', 0)
            prop_log += f"{short}:{val:.3f} "
            
        print(f"{log} | {prop_log}")
        
        # Save History
        history["train_loss"].append(loss)
        history["val_mae"].append(avg_val_mae)
        history["weights"].append(weights)
        
        # Checkpoint State
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "best_val_mae": best_val_mae,
            "val_mae": avg_val_mae,
            "history": history,
            "normalizer": normalizer.state_dict(), # Critical for inference
        }

        # Save Best
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch
            print(f"    * New Best! ({best_val_mae:.4f})")
            manager.save_best(state, best_val_mae, epoch)
            
        # Save Checkpoint (Rotating last-k)
        manager.save_checkpoint(state, epoch)

    best_path = save_dir / "best.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, weights_only=False)
        if "model_state_dict" in best_ckpt:
            model.load_state_dict(best_ckpt["model_state_dict"])
            best_epoch = int(best_ckpt.get("epoch", best_epoch))
    test_metrics = evaluate(model, test_loader, device, normalizer)
    test_maes = [test_metrics[f"{p}_MAE"] for p in PROPERTIES if f"{p}_MAE" in test_metrics]
    avg_test_mae = float(sum(test_maes) / len(test_maes)) if test_maes else float("inf")

    results = {
        "algorithm": "e3nn_multitask_pro",
        "run_id": save_dir.name,
        "all_properties": bool(args.all_properties),
        "best_val_mae": float(best_val_mae),
        "best_epoch": int(best_epoch),
        "total_epochs": int(last_epoch),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "val_metrics_last_epoch": last_val_metrics,
        "test_metrics": test_metrics,
        "avg_test_mae": avg_test_mae,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "top_k": args.top_k,
            "keep_last_k": args.keep_last_k,
            "outlier_filter_enabled": bool(not args.no_filter),
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
                "best_val_mae": float(best_val_mae),
                "best_epoch": int(best_epoch),
                "total_epochs": int(last_epoch),
                "avg_test_mae": float(avg_test_mae),
            },
        },
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

