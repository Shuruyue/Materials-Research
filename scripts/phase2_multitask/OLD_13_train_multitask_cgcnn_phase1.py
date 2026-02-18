"""
Phase 1: Multi-Task CGCNN Training (High Precision Float32)

Trains a shared CGCNN encoder with 3 task-specific heads:
1. Formation Energy (Regression, eV/atom)
2. Band Gap (Regression, eV)
3. Bulk Modulus (Regression, GPa)
4. Shear Modulus (Regression, GPa)

Uses:
- Float32 precision (No AMP) for maximum scientific accuracy.
- Robust Multi-Task Loss with Uncertainty Weighting.
- Physics Constraints (Positivity for Band Gap).
- OneCycleLR Scheduler for super-convergence.

Benchmarks (Phase 1 Targets):
- Formation Energy MAE: < 0.070 eV/atom
- Band Gap MAE: < 0.25 eV
- Bulk Modulus MAE: < 12.0 GPa
- Shear Modulus MAE: < 10.0 GPa
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset
from atlas.models.cgcnn import CGCNN
from atlas.models.multi_task import MultiTaskGNN
from atlas.training.losses import MultiTaskLoss
from atlas.training.trainer import Trainer
from atlas.training.metrics import scalar_metrics, classification_metrics

# ── Configuration ──

TASKS = {
    "formation_energy": {"type": "scalar", "loss": "huber"},
    "band_gap":         {"type": "scalar", "loss": "huber", "constraint": "positive"},
    "bulk_modulus":     {"type": "scalar", "loss": "huber", "constraint": "positive"},
    "shear_modulus":    {"type": "scalar", "loss": "huber", "constraint": "positive"},
}

BENCHMARKS = {
    "formation_energy": {"unit": "eV/atom", "target": 0.070},
    "band_gap":         {"unit": "eV",      "target": 0.250},
    "bulk_modulus":     {"unit": "GPa",     "target": 12.00},
    "shear_modulus":    {"unit": "GPa",     "target": 10.00},
}


class MultiTargetNormalizer:
    """Z-score normalization for regression targets only."""
    
    def __init__(self, dataset, tasks):
        self.normalizers = {}
        for name, cfg in tasks.items():
            if cfg["loss"] == "bce": continue # Skip classification
            
            values = []
            for i in range(len(dataset)):
                data = dataset[i]
                if hasattr(data, name):
                    val = getattr(data, name)
                    if not torch.isnan(val):
                        values.append(val.item())
            
            if not values:
                continue
                
            arr = np.array(values)
            self.normalizers[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()) if arr.std() > 1e-8 else 1.0
            }
            print(f"    Norm [{name}]: mean={self.normalizers[name]['mean']:.3f}, std={self.normalizers[name]['std']:.3f}")

    def normalize(self, name, y):
        if name not in self.normalizers: return y
        stats = self.normalizers[name]
        return (y - stats["mean"]) / stats["std"]

    def denormalize(self, name, y):
        if name not in self.normalizers: return y
        stats = self.normalizers[name]
        return y * stats["std"] + stats["mean"]


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Multi-Task CGCNN")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-conv", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    
    args = parser.parse_args()
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*60)
    print("  Phase 1: Multi-Task CGCNN (High Precision Float32)")
    print("  Targets: Formation Energy, Band Gap, Phonon Stability")
    print("="*60)

    # 1. Dataset
    print("\n  [1/4] Loading Datasets...")
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(
            list(TASKS.keys()),
            max_samples=args.max_samples,
            split=split
        )
        ds.prepare(force_reload=False)
        datasets[split] = ds
        print(f"    {split}: {len(ds)} samples")

    # 2. Normalization
    print("\n  [2/4] Computing Normalization...")
    normalizer = MultiTargetNormalizer(datasets["train"], TASKS)

    # 3. Model
    print("\n  [3/4] Building Model...")
    # Shared Encoder: CGCNN (Baseline)
    # We use a slight modification: MultiTaskGNN expects an encoder that returns embeddings
    # CGCNN.encode() does exactly that.
    
    from atlas.models.graph_builder import CrystalGraphBuilder
    builder = CrystalGraphBuilder()
    
    encoder = CGCNN(
        node_dim=builder.node_dim, # 91
        edge_dim=20,
        hidden_dim=args.hidden_dim,
        n_conv=args.n_conv,
        output_dim=args.hidden_dim, # Output embedding size
    )
    
    # Wrapper expects inputs for heads. CGCNN.encode returns (B, hidden_dim).
    # We adapt CGCNN to fit the expected interface if needed, but MultiTaskGNN 
    # just calls encoder.encode(). We ensure CGCNN has it.
    
    model = MultiTaskGNN(
        encoder=encoder,
        tasks=TASKS,
        embed_dim=args.hidden_dim
    )
    
    print(f"    Total Params: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Training Setup
    loss_fn = MultiTaskLoss(
        task_names=list(TASKS.keys()),
        task_types={k: ("classification" if v["loss"]=="bce" else "regression") for k, v in TASKS.items()},
        strategy="uncertainty",
        constraints={k: v.get("constraint") for k, v in TASKS.items()}
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # We use OneCycleLR for faster convergence
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    # Trainer (AMP Disabled -> use_amp=False)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        save_dir=config.paths.models_dir / "phase1_cgcnn_multitask",
        use_amp=False # FORCE FLOAT32
    )

    # 5. Run
    print("\n  [4/4] Starting Training...")
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=args.epochs,
        patience=args.patience,
        checkpoint_name="phase1"
    )

    # 6. Evaluation
    print("\n  [Evaluation] Phase 1 Targets Check")
    model.eval()
    
    results = {}
    with torch.no_grad():
        all_preds = {k: [] for k in TASKS}
        all_targets = {k: [] for k in TASKS}
        
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            for name in TASKS:
                if hasattr(batch, name):
                    y_true = getattr(batch, name)
                    y_pred = preds[name].view(-1)
                    
                    # Filter NaNs
                    mask = ~torch.isnan(y_true)
                    if mask.sum() == 0: continue
                    
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]
                    
                    # Denormalize regression 
                    if TASKS[name]["loss"] != "bce":
                        y_pred = normalizer.denormalize(name, y_pred)
                    
                    all_preds[name].append(y_pred.cpu())
                    all_targets[name].append(y_true.cpu())

        print(f"  {'Task':<20} | {'Metric':<10} | {'Value':<10} | {'Target':<10} | {'Status'}")
        print("-" * 75)
        
        for name in TASKS:
            if not all_preds[name]: continue
            
            pred = torch.cat(all_preds[name])
            target = torch.cat(all_targets[name])
            
            bm = BENCHMARKS.get(name, {})
            target_val = bm.get("target", 0)
            
            if TASKS[name]["loss"] == "bce":
                # Classification
                metrics = classification_metrics(pred, target)
                val = metrics["AUC"]
                metric_name = "AUC"
                passed = val >= target_val
            else:
                # Regression
                metrics = scalar_metrics(pred, target)
                val = metrics["MAE"]
                metric_name = "MAE"
                passed = val <= target_val
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {name:<20} | {metric_name:<10} | {val:<10.4f} | {target_val:<10.4f} | {status}")

    trainer._save_history("phase1_history.json")

if __name__ == "__main__":
    main()
