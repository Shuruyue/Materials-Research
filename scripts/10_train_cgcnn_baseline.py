"""
Phase 1: CGCNN Baseline Validation

Train single-task CGCNN on JARVIS-DFT data to predict:
1. Formation energy (eV/atom) — MAE target: < 0.063
2. Band gap (eV) — MAE target: < 0.20

This validates that the data pipeline and model architecture work correctly
before proceeding to multi-task learning (Phase 2).

Usage:
    python scripts/10_train_cgcnn_baseline.py --property formation_energy
    python scripts/10_train_cgcnn_baseline.py --property band_gap
    python scripts/10_train_cgcnn_baseline.py --property formation_energy --max-samples 5000
"""

import argparse
import torch
import torch.nn as nn
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset
from atlas.models.cgcnn import CGCNN
from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.training.metrics import scalar_metrics


def train_epoch(model, loader, optimizer, property_name, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        if not hasattr(batch, property_name):
            continue

        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = getattr(batch, property_name).view(-1, 1)

        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        n += target.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, property_name, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_pred, all_target = [], []

    for batch in loader:
        batch = batch.to(device)
        if not hasattr(batch, property_name):
            continue

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = getattr(batch, property_name).view(-1, 1)

        all_pred.append(pred.cpu())
        all_target.append(target.cpu())

    if not all_pred:
        return {}

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    return scalar_metrics(pred, target, prefix=property_name)


def main():
    parser = argparse.ArgumentParser(description="CGCNN Baseline Training (Phase 1)")
    parser.add_argument(
        "--property", type=str, default="formation_energy",
        choices=["formation_energy", "band_gap", "bulk_modulus", "shear_modulus"],
        help="Property to predict",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples (for quick testing)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-conv", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config()

    print("=" * 60)
    print(f"  CGCNN Baseline — Predicting: {args.property}")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── Data ──
    print("\n  [1/3] Loading data...")

    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(
            properties=[args.property],
            max_samples=args.max_samples,
            split=split,
        )
        ds.prepare()
        datasets[split] = ds

    # Stats
    stats = datasets["train"].property_statistics()
    print(f"\n  Property statistics (train):")
    for prop, s in stats.items():
        print(f"    {prop}: count={s['count']}, "
              f"mean={s['mean']:.3f}, std={s['std']:.3f}, "
              f"range=[{s['min']:.3f}, {s['max']:.3f}]")

    train_loader = datasets["train"].to_pyg_loader(batch_size=args.batch_size, shuffle=True)
    val_loader = datasets["val"].to_pyg_loader(batch_size=args.batch_size, shuffle=False)
    test_loader = datasets["test"].to_pyg_loader(batch_size=args.batch_size, shuffle=False)

    # ── Model ──
    print("\n  [2/3] Building CGCNN...")
    builder = CrystalGraphBuilder()
    model = CGCNN(
        node_dim=builder.node_dim,
        edge_dim=20,
        hidden_dim=args.hidden_dim,
        n_conv=args.n_conv,
        output_dim=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    # ── Training ──
    print(f"\n  [3/3] Training for up to {args.epochs} epochs...")
    print(f"  Early stopping patience: {args.patience}")
    print("-" * 60)

    save_dir = config.paths.models_dir / f"cgcnn_{args.property}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.property, device)
        val_metrics = evaluate(model, val_loader, args.property, device)

        val_mae = val_metrics.get(f"{args.property}_MAE", float("inf"))
        scheduler.step(val_mae)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best.pt")
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:4d} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_MAE: {val_mae:.4f} | "
                f"best: {best_val_mae:.4f} | "
                f"lr: {lr:.2e} | "
                f"patience: {patience_counter}/{args.patience}"
            )

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Test ──
    print("\n" + "=" * 60)
    print("  Test Set Evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load(save_dir / "best.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, args.property, device)

    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Targets ──
    targets = {
        "formation_energy": {"MAE_target": 0.063, "unit": "eV/atom"},
        "band_gap": {"MAE_target": 0.20, "unit": "eV"},
        "bulk_modulus": {"MAE_target": 10.0, "unit": "GPa"},
        "shear_modulus": {"MAE_target": 8.0, "unit": "GPa"},
    }

    t = targets.get(args.property, {})
    test_mae = test_metrics.get(f"{args.property}_MAE", float("inf"))
    target_mae = t.get("MAE_target", float("inf"))
    unit = t.get("unit", "")
    passed = "✅ PASS" if test_mae <= target_mae else "❌ FAIL"
    print(f"\n  Target MAE: {target_mae} {unit}")
    print(f"  Actual MAE: {test_mae:.4f} {unit}")
    print(f"  Result: {passed}")

    # Save results
    results = {
        "property": args.property,
        "test_metrics": test_metrics,
        "target_mae": target_mae,
        "passed": test_mae <= target_mae,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "n_params": n_params,
        "best_epoch": epoch - patience_counter,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {save_dir}")


if __name__ == "__main__":
    main()
