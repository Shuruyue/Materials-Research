"""
Phase 2: Multi-Task E(3)-Equivariant GNN — PRO Tier (Production)

The "Holy Grail" script for training the Universal Material Brain.
- Uses a LARGE model (preset='large')
- Trains for 500+ epochs
- Full Resume Capability (saves checkpoint.pt every epoch)
- Robust Outlier Handling (saves outliers.csv, supports --no-filter)
- Detailed JSON logging for experiment tracking

Usage:
    python scripts/phase2_multitask/22_train_multitask_pro.py
    python scripts/phase2_multitask/22_train_multitask_pro.py --resume
    python scripts/phase2_multitask/22_train_multitask_pro.py --no-filter  (Discovery Mode)
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
    "irreps": "128x0e + 64x1o + 32x2e + 16x3o", # High capacity
    "max_ell": 3,
    "n_layers": 4,
    "n_radial": 32,
    "radial_hidden": 256,
    "head_hidden": 128,
}

# ── Outlier Handler ──

def filter_outliers(dataset, properties, save_dir, n_sigma=4.0):
    indices_to_keep = set(range(len(dataset)))
    outliers_found = []

    for prop in properties:
        values = []
        valid_indices = []
        for i in range(len(dataset)):
            try:
                data = dataset[i]
                val = getattr(data, prop).item()
                values.append(val)
                valid_indices.append(i)
            except: continue

        if not values: continue
        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        if std < 1e-8: continue

        for j, v in enumerate(values):
            if abs(v - mean) > n_sigma * std:
                idx = valid_indices[j]
                indices_to_keep.discard(idx)
                outliers_found.append({
                    "jid": getattr(dataset[idx], "jid", "unknown"),
                    "property": prop,
                    "value": v,
                    "mean": mean,
                    "sigma": (v - mean) / std,
                    "threshold_sigma": n_sigma
                })

    if outliers_found:
        df_out = pd.DataFrame(outliers_found)
        df_out.to_csv(save_dir / "outliers.csv", index=False)
        print(f"    ⚠️ Found {len(outliers_found)} outliers! Saved to outliers.csv")
    
    return torch.utils.data.Subset(dataset, sorted(indices_to_keep))

# ── Utilities ──

def pad_missing_properties(dataset, properties):
    n_padded = {p: 0 for p in properties}
    for i in range(len(dataset)):
        data = dataset[i]
        for prop in properties:
            try:
                val = getattr(data, prop)
                if val is None: raise AttributeError
            except:
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

class TargetNormalizer:
    def __init__(self, dataset, property_name):
        values = []
        for i in range(len(dataset)):
            try:
                val = getattr(dataset[i], property_name).item()
                if not np.isnan(val): values.append(val)
            except: continue
        arr = np.array(values)
        self.mean = float(arr.mean())
        self.std = float(arr.std()) if arr.std() > 1e-8 else 1.0

    def normalize(self, y): return (y - self.mean) / self.std
    def denormalize(self, y): return y * self.std + self.mean
    def state_dict(self): return {"mean": self.mean, "std": self.std}
    def load_state_dict(self, state): 
        self.mean = state["mean"] 
        self.std = state["std"]

class MultiTargetNormalizer:
    def __init__(self, dataset, properties):
        self.normalizers = {p: TargetNormalizer(dataset, p) for p in properties}
    def normalize(self, prop, y): return self.normalizers[prop].normalize(y)
    def denormalize(self, prop, y): return self.normalizers[prop].denormalize(y)
    def state_dict(self): return {p: n.state_dict() for p, n in self.normalizers.items()}
    def load_state_dict(self, state):
        for p, s in state.items(): self.normalizers[p].load_state_dict(s)


# ── Checkpointing Logic ──

class CheckpointManager:
    """
    Manages top-k best models and last-k checkpoints.
    1. Keeps 'best.pt' as the absolute best.
    2. Keeps 'best_2.pt', 'best_3.pt' as runners-up.
    3. Keeps 'checkpoint.pt' as latest.
    4. Keeps 'checkpoint_prev_1.pt', 'checkpoint_prev_2.pt' as history.
    """
    def __init__(self, save_dir, top_k=3, keep_last_k=3):
        self.save_dir = Path(save_dir)
        self.top_k = top_k
        self.keep_last_k = keep_last_k
        self.best_models = []  # List of (mae, epoch, filename)

    def save_best(self, state, mae, epoch):
        """Handle saving best models."""
        filename = f"best_epoch_{epoch}_mae_{mae:.4f}.pt"
        path = self.save_dir / filename
        
        # Always save the new candidate first
        torch.save(state, path)
        self.best_models.append((mae, epoch, filename))
        self.best_models.sort(key=lambda x: x[0])  # Sort by MAE (asc)
        
        # Keep top-k
        if len(self.best_models) > self.top_k:
            worst = self.best_models.pop()
            worst_path = self.save_dir / worst[2]
            if worst_path.exists():
                worst_path.unlink()
                
        # Update 'best.pt' symlink/copy (The #1 model)
        if self.best_models[0][1] == epoch:
            import shutil
            shutil.copy(path, self.save_dir / "best.pt")

    def save_checkpoint(self, state, epoch):
        """Handle saving rotating checkpoints."""
        # Delete oldest (last_k - 1)
        oldest = self.save_dir / f"checkpoint_prev_{self.keep_last_k-1}.pt"
        if oldest.exists():
            oldest.unlink()
            
        # Shift others
        import shutil
        for i in range(self.keep_last_k - 2, 0, -1):
            src = self.save_dir / f"checkpoint_prev_{i}.pt"
            dst = self.save_dir / f"checkpoint_prev_{i+1}.pt"
            if src.exists():
                shutil.move(src, dst)
                
        # Move current 'checkpoint.pt' to 'prev_1'
        current = self.save_dir / "checkpoint.pt"
        if current.exists():
            shutil.move(current, self.save_dir / "checkpoint_prev_1.pt")
            
        # Save new
        torch.save(state, current)


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-filter", action="store_true", help="Disable outlier filter")
    parser.add_argument("--all-properties", action="store_true", help="Train on ALL 9 properties (Discovery Mode)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16) # Reduced for Pro size
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    # Dynamic Property Selection
    global PROPERTIES
    if args.all_properties:
        PROPERTIES = ALL_PROPERTIES
        print("🌍 DISCOVERY MODE: Training on ALL 9 properties!")
    else:
        PROPERTIES = CORE_PROPERTIES
        print("🔹 STANDARD MODE: Training on core 4 properties.")

    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     🔴 E3NN PRO (Production Mode)                              ║")
    print(f"║     Tasks: {len(PROPERTIES)} Properties                                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # ── Output Directory ──
    # Create timestamped run folder to preserve history of experiments
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = config.paths.models_dir / "multitask_pro_e3nn"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Specific run folder (e.g. run_20231027_153000)
    if args.resume:
        # Resume from latest run directory
        runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if runs:
            save_dir = runs[-1]
            print(f"  🟡 Resuming in existing run folder: {save_dir.name}")
        else:
            save_dir = base_dir / f"run_{timestamp}"
            save_dir.mkdir()
    else:
        save_dir = base_dir / f"run_{timestamp}"
        save_dir.mkdir()
        print(f"  🔵 Starting new experiment run: {save_dir.name}")

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
    else:
        print("\n[2/5] ⚠️ Outlier Filter DISABLED (--no-filter). Discovery Mode ON.")
        train_data = datasets["train"]
        val_data = datasets["val"]
    
    # 3. Normalizer
    print("\n[3/5] Normalizing Targets...")
    normalizer = MultiTargetNormalizer(train_data, PROPERTIES)
    
    # Loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    start_epoch = 1
    best_val_mae = float('inf')
    history = {"train_loss": [], "val_mae": [], "weights": []}

    # Resume
    checkpoint_path = save_dir / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        print(f"    🔄 Resuming from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_fn.load_state_dict(ckpt["loss_fn_state_dict"]) 
        # Note: Scheduler state might be tricky if params changed, but we try
        try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except: print("    ⚠️ Scheduler state mismatch, restarting scheduler")
        
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt["best_val_mae"]
        history = ckpt["history"]

    # 5. Train
    print(f"\n[5/5] Training for {args.epochs} epochs...")
    
    manager = CheckpointManager(save_dir, top_k=3, keep_last_k=3)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, loss_fn, train_loader, optimizer, device, normalizer)
        
        val_metrics = evaluate(model, val_loader, device, normalizer)
        val_maes = [val_metrics[f"{p}_MAE"] for p in PROPERTIES if f"{p}_MAE" in val_metrics]
        avg_val_mae = sum(val_maes) / len(val_maes) if val_maes else float('inf')
        
        scheduler.step(avg_val_mae)
        
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
            print(f"    ⭐ New Best! ({best_val_mae:.4f})")
            manager.save_best(state, best_val_mae, epoch)
            
        # Save Checkpoint (Rotating last-k)
        manager.save_checkpoint(state, epoch)

if __name__ == "__main__":
    main()
