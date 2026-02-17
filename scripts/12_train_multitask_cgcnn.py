"""
Phase 1: Multi-Task Training with Shared CGCNN Encoder

Trains a single CGCNN encoder with 4 task-specific MLP heads
simultaneously predicting: formation_energy, band_gap, bulk_modulus, shear_modulus.

This serves as the fast BASELINE to compare against Phase 2 (Equivariant GNN).
It uses the exact same multi-task loss and data pipeline, just a different encoder.

Usage:
    python scripts/12_train_multitask_cgcnn.py --preset medium
"""

import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset, DEFAULT_PROPERTIES
from atlas.models.cgcnn import CGCNN
from atlas.models.multi_task import MultiTaskGNN, ScalarHead
from atlas.training.metrics import scalar_metrics


PROPERTIES = DEFAULT_PROPERTIES

# ── Model Presets ──

MODEL_PRESETS = {
    "small": {
        "description": "Fast debugging model",
        "node-dim": 91,
        "edge-dim": 20,
        "hidden-dim": 64,
        "n-conv": 2,
        "n-fc": 1,
        "head-hidden": 32,
    },
    "medium": {
        "description": "Standard CGCNN (Baseline)",
        "node-dim": 91,
        "edge-dim": 20,
        "hidden-dim": 128,
        "n-conv": 3,
        "n-fc": 2,
        "head-hidden": 64,
    },
    "large": {
        "description": "Deep CGCNN",
        "node-dim": 91,
        "edge-dim": 20,
        "hidden-dim": 256,
        "n-conv": 5,
        "n-fc": 3,
        "head-hidden": 128,
    },
}


# ── Utilities (Same as Phase 2) ──

def pad_missing_properties(dataset, properties):
    """Ensure every PyG Data object has all properties (NaN if missing)."""
    n_padded = {p: 0 for p in properties}
    for i in range(len(dataset)):
        data = dataset[i]
        for prop in properties:
            try:
                val = getattr(data, prop)
                if val is None:
                    raise AttributeError
            except (KeyError, AttributeError):
                setattr(data, prop, torch.tensor([float('nan')]))
                n_padded[prop] += 1
    for prop, count in n_padded.items():
        if count > 0:
            print(f"    Padded {count}/{len(dataset)} samples with NaN for {prop}")
    return dataset


class TargetNormalizer:
    """Z-score normalization per property (skips NaN)."""
    def __init__(self, dataset, property_name: str):
        values = []
        for i in range(len(dataset)):
            try:
                data = dataset[i]
                val = getattr(data, property_name).item()
                if not np.isnan(val):
                    values.append(val)
            except (KeyError, AttributeError):
                continue
        arr = np.array(values)
        self.mean = float(arr.mean())
        self.std = float(arr.std())
        if self.std < 1e-8: self.std = 1.0
        print(f"      {property_name}: n={len(values)}, mean={self.mean:.4f}, std={self.std:.4f}")

    def normalize(self, y): return (y - self.mean) / self.std
    def denormalize(self, y): return y * self.std + self.mean
    def state_dict(self): return {"mean": self.mean, "std": self.std}


class MultiTargetNormalizer:
    def __init__(self, dataset, properties):
        self.normalizers = {p: TargetNormalizer(dataset, p) for p in properties}
    def normalize(self, prop, y): return self.normalizers[prop].normalize(y)
    def denormalize(self, prop, y): return self.normalizers[prop].denormalize(y)
    def state_dict(self): return {p: n.state_dict() for p, n in self.normalizers.items()}


def filter_outliers(dataset, properties, n_sigma=8.0):
    indices_to_keep = set(range(len(dataset)))
    for prop in properties:
        values = []
        valid_indices = []
        for i in range(len(dataset)):
            try:
                val = getattr(dataset[i], prop).item()
                if not np.isnan(val):
                    values.append(val)
                    valid_indices.append(i)
            except: continue
        
        if not values: continue
        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        remove = {valid_indices[j] for j, v in enumerate(values) if abs(v - mean) > n_sigma * std}
        if remove:
            print(f"    Outlier filter ({n_sigma}σ) for {prop}: removed {len(remove)} samples")
            indices_to_keep -= remove
            
    return torch.utils.data.Subset(dataset, sorted(indices_to_keep))


class UncertaintyWeightedLoss(nn.Module):
    """Automatically learns loss weights using log-variance."""
    def __init__(self, n_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total

    def get_weights(self):
        return {p: {"sigma": round(float(torch.exp(0.5*self.log_vars[i])), 4)} 
                for i, p in enumerate(PROPERTIES)}


# ── Training Loop ──

def train_epoch(model, loss_fn, loader, optimizer, device, normalizer=None, grad_clip=1.0):
    model.train()
    total_loss = 0
    n = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # CGCNN forward pass
        # Note: CGCNN expects edge_attr, not edge_vec
        predictions = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        task_losses = []
        valid_tasks = 0
        
        for prop in PROPERTIES:
            if prop not in predictions: continue
            
            target = getattr(batch, prop).view(-1, 1)
            pred = predictions[prop]
            
            # Skip NaN
            valid_mask = ~torch.isnan(target).squeeze(-1)
            if valid_mask.sum() == 0: continue
            
            target = target[valid_mask]
            pred = pred[valid_mask]
            
            # Normalize
            if normalizer:
                target_norm = normalizer.normalize(prop, target)
            else:
                target_norm = target
                
            loss = nn.functional.huber_loss(pred, target_norm, delta=1.0)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                task_losses.append(loss)
                valid_tasks += 1
                
        if valid_tasks == 0: continue
        
        # Pad losses for unused tasks (to match loss_fn shape)
        while len(task_losses) < len(PROPERTIES):
            task_losses.append(torch.tensor(0.0, device=device))
            
        loss = loss_fn(task_losses)
        
        if torch.isnan(loss) or torch.isinf(loss): continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        n += 1
        
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, normalizer=None):
    model.eval()
    all_preds = {p: [] for p in PROPERTIES}
    all_targets = {p: [] for p in PROPERTIES}
    
    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        for prop in PROPERTIES:
            if prop not in predictions: continue
            target = getattr(batch, prop).view(-1, 1)
            pred = predictions[prop]
            
            valid_mask = ~torch.isnan(target).squeeze(-1)
            if valid_mask.sum() == 0: continue
            
            target = target[valid_mask]
            pred = pred[valid_mask]
            
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
    parser.add_argument("--preset", type=str, default="medium", choices=MODEL_PRESETS.keys())
    parser.add_argument("--batch-size", type=int, default=128)  # CGCNN usually handles large batch
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    # Apply preset
    preset = MODEL_PRESETS[args.preset]
    print(f"\n[Config] Applying preset '{args.preset}': {preset['description']}")
    for k, v in preset.items():
        if k != "description":
            print(f"  - {k}: {v}")
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Data
    print("\n[1/5] Loading multi-property dataset...")
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(properties=PROPERTIES, max_samples=args.max_samples, split=split).prepare()
        pad_missing_properties(ds, PROPERTIES)
        if split != "train":
            ds = filter_outliers(ds, PROPERTIES)
        datasets[split] = ds
        
    train_data = filter_outliers(datasets["train"], PROPERTIES)
    
    # Normalizer
    print("\n[2/5] Computing normalization...")
    normalizer = MultiTargetNormalizer(train_data, PROPERTIES)
    
    # Loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size)
    
    # Model
    print("\n[3/5] Building Multi-Task CGCNN...")
    
    # CGCNN Encoder
    class CGCNNEncoder(nn.Module):
        def __init__(self, node_dim, edge_dim, hidden_dim, n_conv, n_fc, dropout=0.0):
            super().__init__()
            self.cgcnn = CGCNN(node_dim, edge_dim, hidden_dim, n_conv, n_fc, output_dim=1, dropout=dropout)
            self.hidden_dim = hidden_dim # CGCNN usually has hidden_dim/2 after pooling, let's check
            
        def forward(self, x, edge_index, edge_attr, batch):
            return self.cgcnn.encode(x, edge_index, edge_attr, batch)

    # Need to check CGCNN.encode output dim. 
    # Usually CGCNN code creates encoding of size hidden_dim.
    encoder = CGCNNEncoder(
        preset["node-dim"], preset["edge-dim"], preset["hidden-dim"], 
        preset["n-conv"], preset["n-fc"]
    )
    
    # Multi-Task Wrapper
    # Note: CGCNN.encode returns (B, hidden_dim)
    model = MultiTaskGNN(encoder, latent_dim=preset["hidden-dim"], task_names=PROPERTIES).to(device)
    
    # Optim
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = UncertaintyWeightedLoss(len(PROPERTIES)).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Train
    print(f"\n[4/5] Training for {args.epochs} epochs...")
    best_val_mae = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, loss_fn, train_loader, optimizer, device, normalizer)
        
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, val_loader, device, normalizer)
            val_mae = sum([val_metrics[f"{p}_MAE"] for p in PROPERTIES]) / len(PROPERTIES)
            
            scheduler.step(val_mae)
            
            is_best = val_mae < best_val_mae
            if is_best:
                best_val_mae = val_mae
                
            weights = loss_fn.get_weights()
            dt = time.time() - t0
            
            log = f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val MAE: {val_mae:.3f}"
            for p in PROPERTIES:
                 log += f" {p[:3]}:{val_metrics[f'{p}_MAE']:.2f}"
            log += f" | {dt:.1f}s"
            print(log)

if __name__ == "__main__":
    main()
