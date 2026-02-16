"""
Phase 1: Maximum Accuracy CGCNN Training

Full-scale training with all accuracy optimizations:
- Full JARVIS-DFT dataset (~76k materials)
- Larger model (256 hidden, 4 conv layers)
- Cosine annealing with warm restarts
- Gradient clipping + weight decay
- Multiple properties: formation_energy, band_gap, bulk_modulus, shear_modulus
- Final comparison against literature benchmarks

Usage:
    python scripts/11_train_cgcnn_full.py
    python scripts/11_train_cgcnn_full.py --property band_gap
    python scripts/11_train_cgcnn_full.py --all-properties
"""

import argparse
import torch
import torch.nn as nn
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset
from atlas.models.cgcnn import CGCNN
from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.training.metrics import scalar_metrics

import numpy as np


# ── Literature benchmarks (JARVIS-DFT full dataset) ──
BENCHMARKS = {
    "formation_energy": {
        "unit": "eV/atom",
        "cgcnn_mae": 0.063,
        "alignn_mae": 0.033,
        "target_mae": 0.070,    # PhD target: match CGCNN
    },
    "band_gap": {
        "unit": "eV",
        "cgcnn_mae": 0.20,
        "alignn_mae": 0.14,
        "target_mae": 0.25,
    },
    "bulk_modulus": {
        "unit": "GPa",
        "cgcnn_mae": 10.5,
        "alignn_mae": 8.3,
        "target_mae": 12.0,
    },
    "shear_modulus": {
        "unit": "GPa",
        "cgcnn_mae": 8.0,
        "alignn_mae": 6.5,
        "target_mae": 10.0,
    },
}


def filter_outliers(dataset, property_name, n_sigma=10.0):
    """
    Remove extreme outliers from dataset.
    Filters samples where |value - mean| > n_sigma * std.
    This prevents a few extreme DFT values from corrupting training.
    """
    values = []
    for data in dataset:
        if hasattr(data, property_name):
            values.append(getattr(data, property_name).item())
    if not values:
        return dataset

    import numpy as np
    arr = np.array(values)
    mean, std = arr.mean(), arr.std()
    if std < 1e-8:
        return dataset

    mask = np.abs(arr - mean) <= n_sigma * std
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"    Outlier filter: removed {n_removed} samples "
              f"(|val - {mean:.2f}| > {n_sigma}σ = {n_sigma*std:.2f})")
        indices = np.where(mask)[0].tolist()
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
    return dataset


# ── Target Normalization ──

class TargetNormalizer:
    """Z-score normalization using training set statistics."""

    def __init__(self, dataset, property_name: str):
        values = []
        for i in range(len(dataset)):
            data = dataset[i]
            if hasattr(data, property_name):
                values.append(getattr(data, property_name).item())
        arr = np.array(values)
        self.mean = float(arr.mean())
        self.std = float(arr.std())
        if self.std < 1e-8:
            self.std = 1.0
        print(f"    Target normalizer: mean={self.mean:.4f}, std={self.std:.4f}")
        print(f"    Raw range: [{arr.min():.2f}, {arr.max():.2f}]")

    def normalize(self, y):
        return (y - self.mean) / self.std

    def denormalize(self, y):
        return y * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}


def train_epoch(model, loader, optimizer, property_name, device,
                normalizer=None, scaler=None):
    """Train for one epoch with optional mixed precision."""
    model.train()
    total_loss = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        if not hasattr(batch, property_name):
            continue

        target = getattr(batch, property_name).view(-1, 1)

        # Normalize target to ~N(0,1) scale
        if normalizer is not None:
            target_norm = normalizer.normalize(target)
        else:
            target_norm = target

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision (AMP)
            with torch.amp.autocast(device_type='cuda'):
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = nn.functional.huber_loss(pred, target_norm, delta=1.0)

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = nn.functional.huber_loss(pred, target_norm, delta=1.0)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        total_loss += loss.item() * target.size(0)
        n += target.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, property_name, device, normalizer=None):
    """Evaluate model and return metrics in original units."""
    model.eval()
    all_pred, all_target = [], []

    for batch in loader:
        batch = batch.to(device)
        if not hasattr(batch, property_name):
            continue

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = getattr(batch, property_name).view(-1, 1)

        # Denormalize predictions back to original scale
        if normalizer is not None:
            pred = normalizer.denormalize(pred)

        all_pred.append(pred.cpu())
        all_target.append(target.cpu())

    if not all_pred:
        return {}

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    return scalar_metrics(pred, target, prefix=property_name)


def train_single_property(args, property_name: str):
    """Train for one property with maximum accuracy settings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config()
    benchmark = BENCHMARKS[property_name]

    print("\n" + "=" * 70)
    print(f"  CGCNN Full-Scale — Property: {property_name}")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Target MAE: {benchmark['target_mae']} {benchmark['unit']}")
    print(f"  CGCNN literature: {benchmark['cgcnn_mae']} {benchmark['unit']}")
    print(f"  ALIGNN literature: {benchmark['alignn_mae']} {benchmark['unit']}")
    print("=" * 70)

    # ── Data ──
    print("\n  [1/3] Loading full dataset...")
    t0 = time.time()

    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(
            properties=[property_name],
            max_samples=args.max_samples,
            stability_filter=None,  # Use ALL data for max accuracy
            split=split,
        )
        ds.prepare()
        datasets[split] = ds

    stats = datasets["train"].property_statistics()
    for prop, s in stats.items():
        print(f"    {prop}: n={s['count']}, μ={s['mean']:.3f}, σ={s['std']:.3f}")

    data_time = time.time() - t0
    print(f"    Data loading: {data_time:.1f}s")

    # Filter extreme outliers from all splits
    train_data = filter_outliers(datasets["train"], property_name, n_sigma=10.0)
    val_data = filter_outliers(datasets["val"], property_name, n_sigma=10.0)
    test_data = filter_outliers(datasets["test"], property_name, n_sigma=10.0)

    # Target normalization (compute from training set ONLY)
    print("    Computing target normalization...")
    normalizer = TargetNormalizer(train_data, property_name)

    from torch_geometric.loader import DataLoader as PyGLoader
    train_loader = PyGLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # ── Model (maximum accuracy config) ──
    print("\n  [2/3] Building CGCNN (max accuracy)...")
    builder = CrystalGraphBuilder()
    model = CGCNN(
        node_dim=builder.node_dim,
        edge_dim=20,
        hidden_dim=args.hidden_dim,
        n_conv=args.n_conv,
        output_dim=1,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    print(f"    Hidden dim: {args.hidden_dim}, Conv layers: {args.n_conv}")

    # AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ReduceLROnPlateau — more stable than cosine restart for full dataset
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=25,
        factor=0.5,
        min_lr=1e-6,
    )

    # Mixed precision: disabled by default (can cause NaN with GNN edge features)
    use_amp = device == "cuda" and args.amp
    scaler = torch.amp.GradScaler() if use_amp else None

    # ── Training ──
    print(f"\n  [3/3] Training for up to {args.epochs} epochs...")
    print(f"    Patience: {args.patience}")
    print(f"    Mixed precision: {use_amp}")
    print(f"    Loss: Huber (delta=1.0) — robust to outliers")
    print("-" * 70)

    save_dir = config.paths.models_dir / f"cgcnn_full_{property_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": [], "lr": []}
    t_train = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, property_name, device,
            normalizer=normalizer, scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, property_name, device, normalizer=normalizer)
        val_mae = val_metrics.get(f"{property_name}_MAE", float("inf"))

        scheduler.step(val_mae)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(lr)

        # NaN epoch detection
        if val_mae != val_mae or train_loss != train_loss:  # NaN check
            patience_counter += 1
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:4d} | NaN detected, skipping...")
            continue

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "property": property_name,
                "normalizer": normalizer.state_dict(),
            }, save_dir / "best.pt")
        else:
            patience_counter += 1

        dt_ep = time.time() - t_ep

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:4d} | "
                f"loss: {train_loss:.4f} | "
                f"val_MAE: {val_mae:.4f} | "
                f"best: {best_val_mae:.4f} | "
                f"lr: {lr:.2e} | "
                f"patience: {patience_counter}/{args.patience} | "
                f"{dt_ep:.1f}s"
            )

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t_train
    print(f"\n  Training time: {train_time/60:.1f} minutes ({train_time/3600:.1f} hours)")

    # ── Test Evaluation ──
    print("\n" + "=" * 70)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 70)

    checkpoint = torch.load(save_dir / "best.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint["epoch"]

    test_metrics = evaluate(model, test_loader, property_name, device, normalizer=normalizer)

    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    test_mae = test_metrics.get(f"{property_name}_MAE", float("inf"))
    unit = benchmark["unit"]
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Property: {property_name:<30s}│")
    print(f"  │  Test MAE: {test_mae:.4f} {unit:<26s}│")
    print(f"  │  CGCNN lit: {benchmark['cgcnn_mae']:.4f} {unit:<25s}│")
    print(f"  │  ALIGNN lit: {benchmark['alignn_mae']:.4f} {unit:<24s}│")
    print(f"  │  Target: {benchmark['target_mae']:.4f} {unit:<27s}│")
    passed = test_mae <= benchmark["target_mae"]
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  │  Result: {status:<32s}│")
    print(f"  │  Best epoch: {best_epoch:<27d}│")
    print(f"  └─────────────────────────────────────────┘")

    # Save results
    results = {
        "property": property_name,
        "test_metrics": test_metrics,
        "target_mae": benchmark["target_mae"],
        "cgcnn_literature_mae": benchmark["cgcnn_mae"],
        "alignn_literature_mae": benchmark["alignn_mae"],
        "passed": passed,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "training_time_minutes": train_time / 60,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "n_conv": args.n_conv,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "loss": "huber_delta1.0",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
        },
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save history
    hist_ser = {k: [float(v) for v in vs] for k, vs in history.items()}
    with open(save_dir / "history.json", "w") as f:
        json.dump(hist_ser, f, indent=2)

    print(f"  Results saved to {save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CGCNN Maximum Accuracy Training (Phase 1)"
    )
    parser.add_argument(
        "--property", type=str, default="formation_energy",
        choices=list(BENCHMARKS.keys()),
    )
    parser.add_argument("--all-properties", action="store_true",
                        help="Train all 4 properties sequentially")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on samples (None = use all ~76k)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-conv", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (may cause NaN)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        CGCNN Baseline — Maximum Accuracy Configuration         ║")
    print("║        PhD Thesis Phase 1: Validation Against Literature       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    if args.all_properties:
        all_results = {}
        for prop in BENCHMARKS:
            result = train_single_property(args, prop)
            all_results[prop] = result

        # Summary table
        print("\n" + "=" * 70)
        print("  SUMMARY — ALL PROPERTIES")
        print("=" * 70)
        print(f"  {'Property':<20s} {'Test MAE':>10s} {'CGCNN Lit':>10s} "
              f"{'Target':>10s} {'Status':>8s}")
        print("  " + "-" * 62)
        for prop, res in all_results.items():
            unit = BENCHMARKS[prop]["unit"]
            mae = res["test_metrics"].get(f"{prop}_MAE", float("inf"))
            lit = BENCHMARKS[prop]["cgcnn_mae"]
            tgt = BENCHMARKS[prop]["target_mae"]
            st = "✅" if res["passed"] else "❌"
            print(f"  {prop:<20s} {mae:>8.4f}  {lit:>8.4f}  {tgt:>8.4f}  {st:>6s}")

        # Save combined results
        config = get_config()
        combined_path = config.paths.models_dir / "cgcnn_full_summary.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Combined results: {combined_path}")
    else:
        train_single_property(args, args.property)


if __name__ == "__main__":
    main()
