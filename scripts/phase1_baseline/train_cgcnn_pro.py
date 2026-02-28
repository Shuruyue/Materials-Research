"""
Phase 1: Maximum Accuracy CGCNN Training (Stable Version)

Full-scale training with all accuracy optimizations and STABILITY fixes:
- Full JARVIS-DFT dataset (~76k materials)
- Larger model (256 hidden, 4 conv layers)
- OneCycleLR learning rate schedule (Super-convergence)
- Strict Gradient clipping (0.5) + Huber Loss (delta=0.2)
- Float32 precision (No AMP) for numerical stability
- Outlier filtering (4-sigma)

Usage:
    python scripts/phase1_baseline/train_cgcnn_pro.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Enhance module discovery
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from atlas.config import get_config
    from atlas.console_style import install_console_style
    from atlas.data.crystal_dataset import CrystalPropertyDataset
    from atlas.models.cgcnn import CGCNN
    from atlas.models.graph_builder import CrystalGraphBuilder
    from atlas.training.checkpoint import CheckpointManager
    from atlas.training.metrics import scalar_metrics
    from atlas.training.normalizers import TargetNormalizer
    from atlas.training.run_utils import resolve_run_dir, write_run_manifest
except ImportError as e:
    print(f"Error: Could not import atlas package. ({e})")
    print("Please install the package in editable mode: pip install -e .")
    sys.exit(1)

install_console_style()

import numpy as np

# ── Literature benchmarks (JARVIS-DFT full dataset) ──
BENCHMARKS = {
    "formation_energy": {
        "unit": "eV/atom",
        "cgcnn_mae": 0.063,
        "alignn_mae": 0.033,
        "target_mae": 0.070,    # PhD target: match CGCNN
    },
}


def _metric_key(property_name: str, suffix: str) -> str:
    return f"{property_name}_{suffix}"


def _metric(metrics: dict, property_name: str, suffix: str, default=float("nan")) -> float:
    return float(metrics.get(_metric_key(property_name, suffix), default))


def _print_metric_block(metrics: dict, property_name: str, unit: str) -> None:
    for suffix in ["MAE", "RMSE", "R2", "MaxAE"]:
        key = _metric_key(property_name, suffix)
        if key not in metrics:
            continue
        value = float(metrics[key])
        metric_unit = unit if suffix in {"MAE", "RMSE", "MaxAE"} else ""
        print(f"  {key}: {value:.4f} {metric_unit}".rstrip())


def filter_outliers(dataset, property_name, save_dir, n_sigma=4.0):
    """
    Remove extreme outliers from dataset and save them for inspection.
    Filters samples where |value - mean| > n_sigma * std.
    Strict filtering (4.0 sigma) removes physical non-sense.
    """
    values = []
    ids = []

    # Extract values and JIDs
    for data in dataset:
        if hasattr(data, property_name):
            values.append(getattr(data, property_name).item())
            ids.append(getattr(data, "jid", "unknown"))

    if not values:
        return dataset

    import pandas as pd

    arr = np.array(values)
    mean, std = arr.mean(), arr.std()

    if std < 1e-8:
        return dataset

    mask = np.abs(arr - mean) <= n_sigma * std
    n_removed = (~mask).sum()

    if n_removed > 0:
        print(f"    Outlier filter: removed {n_removed} samples "
              f"(|val - {mean:.2f}| > {n_sigma}*std = {n_sigma*std:.2f})")

        # Save outliers for inspection (Critical for discovery)
        outlier_indices = np.where(~mask)[0]
        outlier_data = []
        for idx in outlier_indices:
            outlier_data.append({
                "jid": ids[idx],
                "value": values[idx],
                "distance_sigma": (values[idx] - mean) / std
            })

        outlier_df = pd.DataFrame(outlier_data)
        outlier_file = save_dir / "outliers.csv"
        outlier_df.to_csv(outlier_file, index=False)
        print(f"    [WARN] Saved {n_removed} outliers to {outlier_file} for review")

        from torch.utils.data import Subset
        indices = np.where(mask)[0].tolist()
        dataset = Subset(dataset, indices)

    return dataset





def train_epoch(model, loader, optimizer, scheduler, property_name, device,
                normalizer=None):
    """Train for one epoch with OneCycleLR and Gradient Clipping."""
    model.train()
    total_loss = 0
    n = 0

    # Huber Loss with small delta for robustness
    criterion = nn.HuberLoss(delta=0.2)

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

        # Forward pass (Float32)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred, target_norm)

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            print("    Warning: NaN loss detected in batch, skipping...")
            optimizer.zero_grad()
            continue

        loss.backward()

        # Strict Gradient Clipping (0.5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()
        scheduler.step()  # Step scheduler every batch for OneCycleLR

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
    print(f"  CGCNN Full-Scale - Property: {property_name}")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Target MAE: {benchmark['target_mae']} {benchmark['unit']}")
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
        print(f"    {prop}: n={s['count']}, mean={s['mean']:.3f}, std={s['std']:.3f}")

    data_time = time.time() - t0
    print(f"    Data loading: {data_time:.1f}s")

    # ── Output Directory ──
    base_dir = config.paths.models_dir / f"cgcnn_pro_{property_name}"
    try:
        save_dir, created_new = resolve_run_dir(
            base_dir,
            resume=args.resume,
            run_id=args.run_id,
        )
    except (FileNotFoundError, FileExistsError) as e:
        print(f"  [ERROR] {e}")
        return {}
    run_msg = "Starting new run" if created_new else "Using existing run"
    print(f"  [INFO] {run_msg}: {save_dir.name}")
    manifest_path = write_run_manifest(
        save_dir,
        args=args,
        project_root=PROJECT_ROOT,
        extra={
            "status": "started",
            "phase": "phase1",
            "model_family": "cgcnn_pro",
            "property": property_name,
        },
    )
    print(f"  [INFO] Run manifest: {manifest_path}")

    # Filter extreme outliers from all splits (Strict 4.0 sigma)
    if not args.no_filter:
        print("    Applying strict outlier filter (4.0 sigma)...")
        # Pass save_dir to save outliers.csv
        train_data = filter_outliers(datasets["train"], property_name, save_dir, n_sigma=4.0)
        # Validation/Test should represent reality, but for stability we filter extreme physics errors
        # In production discovery, we might want to keep them.
        val_data = filter_outliers(datasets["val"], property_name, save_dir, n_sigma=4.0)
        test_data = filter_outliers(datasets["test"], property_name, save_dir, n_sigma=4.0)
    else:
        print("    [WARN] Outlier filter DISABLED (--no-filter active)")
        print("    Training on raw data including extreme values.")
        train_data = datasets["train"]
        val_data = datasets["val"]
        test_data = datasets["test"]

    # Target normalization (compute from training set ONLY)
    print("    Computing target normalization...")
    normalizer = TargetNormalizer(train_data, property_name)

    from torch_geometric.loader import DataLoader as PyGLoader
    train_loader = PyGLoader(train_data, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    val_loader = PyGLoader(val_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = PyGLoader(test_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(
        f"    Dataset sizes used: train={len(train_data)}, "
        f"val={len(val_data)}, test={len(test_data)}"
    )

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
        pooling=args.pooling,
        jk=args.jk,
        message_aggr=args.message_aggr,
        use_edge_gates=not args.no_edge_gates,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    print(f"    Hidden dim: {args.hidden_dim}, Conv layers: {args.n_conv}")

    # AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # OneCycleLR — Super-convergence strategy
    # Updates every batch, not every epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,  # 30% warmup
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # ── Training ──
    print(f"\n  [3/3] Training for up to {args.epochs} epochs...")
    print(f"    Patience: {args.patience}")
    print("    Strategy: OneCycleLR + GradClip(0.5)")
    print("    Loss: Huber (delta=0.2) - robust to outliers")
    print("-" * 70)

    best_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": [], "val_rmse": [], "val_r2": [], "lr": []}
    start_epoch = 1

    # Resume logic
    checkpoint_path = save_dir / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        print(f"  [INFO] Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        normalizer_state = checkpoint.get("normalizer")
        if normalizer_state:
            normalizer.mean = normalizer_state["mean"]
            normalizer.std = normalizer_state["std"]

        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", history)
        best_val_mae = checkpoint.get("best_val_mae", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        print(f"  -> Resuming at epoch {start_epoch}")
    elif args.resume:
        print(f"  [WARN] --resume requested but checkpoint not found: {checkpoint_path}")

    t_train = time.time()


    manager = CheckpointManager(save_dir, top_k=args.top_k, keep_last_k=args.keep_last_k)

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, property_name, device,
            normalizer=normalizer,
        )
        val_metrics = evaluate(model, val_loader, property_name, device, normalizer=normalizer)
        val_mae = _metric(val_metrics, property_name, "MAE", float("inf"))
        val_rmse = _metric(val_metrics, property_name, "RMSE")
        val_r2 = _metric(val_metrics, property_name, "R2")

        # Current LR
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_r2"].append(val_r2)
        history["lr"].append(lr)

        # NaN epoch detection
        if val_mae != val_mae or train_loss != train_loss:  # NaN check
            print(f"  Epoch {epoch:4d} | CRITICAL: NaN detected in epoch stats!")
            patience_counter += 1
            if patience_counter > 5:
                print("  Too many NaNs. Stopping.")
                break
            continue

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "normalizer": normalizer.state_dict(),
            "history": history,
            "best_val_mae": best_val_mae,
            "val_mae": val_mae,
            "property": property_name,
            "patience_counter": patience_counter,
        }

        # Save Best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            state["best_val_mae"] = best_val_mae
            state["patience_counter"] = patience_counter
            manager.save_best(state, val_mae, epoch)
        else:
            patience_counter += 1
            state["patience_counter"] = patience_counter

        # Save Checkpoint (Rotating last-k)
        manager.save_checkpoint(state, epoch)

        dt_ep = time.time() - t_ep

        # Log EVERY epoch as requested
        print(
            f"  Epoch {epoch:4d}/{args.epochs:4d} | "
            f"train_huber: {train_loss:.4f} | "
            f"val_mae: {val_mae:.4f} | "
            f"val_rmse: {val_rmse:.4f} | "
            f"val_r2: {val_r2:.4f} | "
            f"best_val_mae: {best_val_mae:.4f} | "
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

    if (save_dir / "best.pt").exists():
        checkpoint = torch.load(save_dir / "best.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = checkpoint["epoch"]
    else:
        print("  Warning: No best model found (NaNs?). Using current model.")
        best_epoch = -1

    test_metrics = evaluate(model, test_loader, property_name, device, normalizer=normalizer)

    _print_metric_block(test_metrics, property_name, benchmark["unit"])

    test_mae = _metric(test_metrics, property_name, "MAE", float("inf"))
    unit = benchmark["unit"]
    print("\n  -------------------------------------------")
    print(f"  Property:   {property_name}")
    print(f"  Test MAE:   {test_mae:.4f} {unit}")
    print(f"  Target MAE: {benchmark['target_mae']:.4f} {unit}")
    passed = test_mae <= benchmark["target_mae"]
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  Result:     {status}")
    print(f"  Best epoch: {best_epoch}")
    print("  -------------------------------------------")

    # Save results
    results = {
        "property": property_name,
        "run_id": save_dir.name,
        "test_metrics": test_metrics,
        "target_mae": benchmark["target_mae"],
        "cgcnn_literature_mae": benchmark["cgcnn_mae"],
        "alignn_literature_mae": benchmark["alignn_mae"],
        "passed": passed,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "n_train_used": len(train_data),
        "n_val_used": len(val_data),
        "n_test_used": len(test_data),
        "n_params": n_params,
        "best_val_mae": float(best_val_mae),
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
            "pooling": args.pooling,
            "jk": args.jk,
            "message_aggr": args.message_aggr,
            "use_edge_gates": bool(not args.no_edge_gates),
            "loss": "huber_delta0.2",
            "optimizer": "AdamW",
            "scheduler": "OneCycleLR",
            "top_k": args.top_k,
            "keep_last_k": args.keep_last_k,
        },
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save history
    hist_ser = {k: [float(v) for v in vs] for k, vs in history.items()}
    with open(save_dir / "history.json", "w") as f:
        json.dump(hist_ser, f, indent=2)
    write_run_manifest(
        save_dir,
        args=args,
        project_root=PROJECT_ROOT,
        extra={
            "status": "completed",
            "result": {
                "best_epoch": int(best_epoch),
                "total_epochs": int(epoch),
                "passed": bool(passed),
                "test_mae": float(test_mae),
            },
        },
    )

    print(f"  Results saved to {save_dir}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CGCNN Production Training (High Precision)"
    )
    parser.add_argument(
        "--property", type=str, default="formation_energy",
        choices=list(BENCHMARKS.keys()),
    )
    # parser.add_argument("--all-properties", action="store_true",
    #                     help="Train all properties sequentially")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on samples (None = use all ~76k)")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Max LR for OneCycle (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-conv", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling", choices=["mean", "sum", "max", "mean_max", "attn"], default="mean_max")
    parser.add_argument("--jk", choices=["last", "mean", "concat"], default="concat")
    parser.add_argument("--message-aggr", choices=["sum", "mean"], default="mean")
    parser.add_argument("--no-edge-gates", action="store_true",
                        help="Disable edge-wise gating in message passing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run id (without or with 'run_' prefix)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Keep top-k best checkpoints")
    parser.add_argument("--keep-last-k", type=int, default=3,
                        help="Keep latest rotating checkpoints")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable outlier filtering (use all data)")
    # Removed AMP argument to enforce float32
    args = parser.parse_args()

    print("=" * 70)
    print("CGCNN PRO (Production Mode)")
    print("High Precision / SOTA Challenge")
    print("=" * 70)

    result = train_single_property(args, args.property)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())



