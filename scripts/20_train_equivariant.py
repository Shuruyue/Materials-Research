"""
Phase 2: E(3)-Equivariant GNN Training — Maximum Precision

Full-scale training with NequIP-inspired equivariant architecture:
- Spherical harmonics edge features (respects SO(3) symmetry)
- Bessel radial basis with smooth cutoff
- Tensor product message passing with gated activation
- EMA (Exponential Moving Average) for smoother final model
- 8σ outlier filtering for clean training data
- Conservative learning rate with gradual decay

Usage:
    python scripts/20_train_equivariant.py --property shear_modulus
    python scripts/20_train_equivariant.py --all-properties
    python scripts/20_train_equivariant.py --max-samples 5000  # quick test
"""

import argparse
import copy
import torch
import torch.nn as nn
import json
import time
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset
from atlas.models.equivariant import EquivariantGNN
from atlas.training.metrics import scalar_metrics


# ── Literature benchmarks ──
BENCHMARKS = {
    "formation_energy": {
        "unit": "eV/atom",
        "cgcnn_mae": 0.063,
        "alignn_mae": 0.033,
        "target_mae": 0.050,    # Phase 2 target: beat CGCNN
    },
    "band_gap": {
        "unit": "eV",
        "cgcnn_mae": 0.20,
        "alignn_mae": 0.14,
        "target_mae": 0.18,
    },
    "bulk_modulus": {
        "unit": "GPa",
        "cgcnn_mae": 10.5,
        "alignn_mae": 8.3,
        "target_mae": 9.5,
    },
    "shear_modulus": {
        "unit": "GPa",
        "cgcnn_mae": 8.0,
        "alignn_mae": 6.5,
        "target_mae": 7.5,
    },
}


# ─────────────────────────────────────────────────────────
#  EMA — Exponential Moving Average
# ─────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights:
        shadow_w = decay * shadow_w + (1 - decay) * model_w

    Using the EMA weights for evaluation typically gives 2-5% better MAE
    because it smooths out noise from individual gradient updates.

    Reference: NequIP, Polyak averaging, SWA.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update shadow weights after each optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Replace model weights with EMA shadow weights (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model weights (after evaluation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ─────────────────────────────────────────────────────────
#  Outlier Filter
# ─────────────────────────────────────────────────────────

def filter_outliers(dataset, property_name, n_sigma=8.0):
    """
    Remove extreme outliers from dataset.
    Filters samples where |value - mean| > n_sigma * std.

    Args:
        dataset: PyG dataset (list of Data objects)
        property_name: target property attribute name
        n_sigma: number of standard deviations for threshold

    Returns:
        Filtered dataset (list or Subset)
    """
    values = []
    for data in dataset:
        if hasattr(data, property_name):
            values.append(getattr(data, property_name).item())
    if not values:
        return dataset

    arr = np.array(values)
    mean, std = arr.mean(), arr.std()
    if std < 1e-8:
        return dataset

    mask = np.abs(arr - mean) <= n_sigma * std
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"    Outlier filter ({n_sigma}σ): removed {n_removed} samples "
              f"(|val - {mean:.2f}| > {n_sigma * std:.2f})")
        indices = np.where(mask)[0].tolist()
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
    else:
        print(f"    Outlier filter ({n_sigma}σ): no outliers found")
    return dataset


# ─────────────────────────────────────────────────────────
#  Target Normalization (CRITICAL for non-eV properties)
# ─────────────────────────────────────────────────────────

class TargetNormalizer:
    """
    Z-score normalization for target property values.

    Computes mean/std from training set, then:
      - normalize(y) = (y - mean) / std   (for training)
      - denormalize(y) = y * std + mean    (for evaluation/inference)

    This is CRITICAL for properties like shear_modulus (σ=37 GPa)
    where raw values are far from the ~N(0,1) scale that BatchNorm
    keeps internal features at. Without this, the model's output
    layer must learn a large bias (~40) and scale (~37), which is
    extremely difficult.

    Reference: ALIGNN uses sklearn.StandardScaler for the same purpose.
    """

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
            self.std = 1.0  # Avoid division by zero

        print(f"    Target normalizer: mean={self.mean:.4f}, std={self.std:.4f}")
        print(f"    Raw range: [{arr.min():.2f}, {arr.max():.2f}]")
        print(f"    Normalized range: [{(arr.min()-self.mean)/self.std:.2f}, "
              f"{(arr.max()-self.mean)/self.std:.2f}]")

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        """Transform raw values to standardized scale."""
        return (y - self.mean) / self.std

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """Transform standardized predictions back to original scale."""
        return y * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict):
        obj = cls.__new__(cls)
        obj.mean = state["mean"]
        obj.std = state["std"]
        return obj


# ─────────────────────────────────────────────────────────
#  Training & Evaluation
# ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, property_name, device,
                normalizer=None, ema=None, grad_clip=0.5):
    """Train for one epoch with conservative gradient clipping and EMA."""
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

        # EquivariantGNN uses edge_vec (3D vectors), NOT edge_attr (Gaussian)
        pred = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
        loss = nn.functional.huber_loss(pred, target_norm, delta=1.0)

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # Update EMA after each optimizer step
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * target.size(0)
        n += target.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, property_name, device, normalizer=None):
    """Evaluate model and return metrics in ORIGINAL units."""
    model.eval()
    all_pred, all_target = [], []

    for batch in loader:
        batch = batch.to(device)
        if not hasattr(batch, property_name):
            continue

        pred = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
        target = getattr(batch, property_name).view(-1, 1)

        # Denormalize predictions back to original scale for metrics
        if normalizer is not None:
            pred = normalizer.denormalize(pred)

        all_pred.append(pred.cpu())
        all_target.append(target.cpu())  # target is always in original scale

    if not all_pred:
        return {}

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    return scalar_metrics(pred, target, prefix=property_name)


def format_time(seconds):
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def train_single_property(args, property_name: str):
    """Train equivariant GNN for one property with maximum precision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config()
    benchmark = BENCHMARKS[property_name]

    print("\n" + "=" * 70)
    print(f"  EquivariantGNN — Property: {property_name}")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Target MAE: {benchmark['target_mae']} {benchmark['unit']}")
    print(f"  CGCNN benchmark: {benchmark['cgcnn_mae']} {benchmark['unit']}")
    print(f"  ALIGNN benchmark: {benchmark['alignn_mae']} {benchmark['unit']}")
    print(f"  Mode: MAXIMUM PRECISION")
    print("=" * 70)

    # ── Data ──
    print("\n  [1/4] Loading dataset...")
    t0 = time.time()

    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(
            properties=[property_name],
            max_samples=args.max_samples,
            stability_filter=None,
            split=split,
        )
        ds.prepare()
        datasets[split] = ds

    stats = datasets["train"].property_statistics()
    for prop, s in stats.items():
        print(f"    {prop}: n={s['count']}, μ={s['mean']:.3f}, σ={s['std']:.3f}")

    # ── Outlier filtering ──
    print(f"\n  [2/4] Filtering outliers ({args.outlier_sigma}σ)...")
    for split in ["train", "val", "test"]:
        ds = datasets[split]
        data_list = [ds[i] for i in range(len(ds))]
        filtered = filter_outliers(data_list, property_name, n_sigma=args.outlier_sigma)
        if isinstance(filtered, list):
            datasets[split]._data_list = filtered
        else:
            # Subset — extract actual data
            datasets[split]._data_list = [filtered[i] for i in range(len(filtered))]

    data_time = time.time() - t0
    print(f"    Data loading + filtering: {data_time:.1f}s")
    print(f"    Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, "
          f"Test: {len(datasets['test'])}")

    # ── Target Normalization (compute from training set ONLY) ──
    print(f"\n  [3/5] Computing target normalization...")
    normalizer = TargetNormalizer(datasets["train"], property_name)

    train_loader = datasets["train"].to_pyg_loader(
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = datasets["val"].to_pyg_loader(
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = datasets["test"].to_pyg_loader(
        batch_size=args.batch_size, shuffle=False,
    )

    n_batches = len(train_loader)

    # ── Model ──
    print(f"\n  [4/5] Building EquivariantGNN...")
    model = EquivariantGNN(
        irreps_hidden=args.irreps,
        max_ell=args.max_ell,
        n_layers=args.n_layers,
        max_radius=5.0,
        n_species=86,
        n_radial_basis=args.n_radial,
        radial_hidden=args.radial_hidden,
        output_dim=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    print(f"    Irreps: {args.irreps}")
    print(f"    Layers: {args.n_layers}, max_ell: {args.max_ell}")
    print(f"    Radial basis: {args.n_radial}, radial hidden: {args.radial_hidden}")

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        print(f"    EMA decay: {args.ema_decay}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=args.sched_patience,
        factor=args.sched_factor,
        min_lr=args.min_lr,
    )

    # ── Training ──
    print(f"\n  [5/5] Training for up to {args.epochs} epochs...")
    print(f"    LR: {args.lr} → min {args.min_lr}")
    print(f"    Patience: {args.patience} (early stop), "
          f"{args.sched_patience} (scheduler)")
    print(f"    Scheduler factor: {args.sched_factor}")
    print(f"    Grad clip: {args.grad_clip}")
    print(f"    Loss: Huber (delta=1.0)")
    print(f"    Batches/epoch: {n_batches}")
    print("-" * 70)

    save_dir = config.paths.models_dir / f"equivariant_{property_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_ema_val_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_mae": [], "ema_val_mae": [], "lr": []}
    t_train = time.time()
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, property_name, device,
            normalizer=normalizer, ema=ema, grad_clip=args.grad_clip,
        )

        # Evaluate with regular weights (metrics in original units)
        val_metrics = evaluate(model, val_loader, property_name, device, normalizer=normalizer)
        val_mae = val_metrics.get(f"{property_name}_MAE", float("inf"))

        # Evaluate with EMA weights (if enabled)
        ema_val_mae = float("inf")
        if ema is not None:
            ema.apply_shadow(model)
            ema_metrics = evaluate(model, val_loader, property_name, device, normalizer=normalizer)
            ema_val_mae = ema_metrics.get(f"{property_name}_MAE", float("inf"))
            ema.restore(model)

        # Use the better of regular vs EMA for scheduling
        effective_mae = min(val_mae, ema_val_mae) if ema else val_mae
        scheduler.step(effective_mae)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["ema_val_mae"].append(ema_val_mae)
        history["lr"].append(lr)

        # NaN epoch detection
        if val_mae != val_mae or train_loss != train_loss:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:4d} | NaN detected, skipping...")
            continue

        # Track best (save both regular and EMA checkpoints)
        improved = False
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            improved = True
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "property": property_name,
                "checkpoint_type": "best_regular",
                "normalizer": normalizer.state_dict(),
            }, save_dir / "best.pt")

        if ema is not None and ema_val_mae < best_ema_val_mae:
            best_ema_val_mae = ema_val_mae
            improved = True
            # Save EMA weights
            ema.apply_shadow(model)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": ema_val_mae,
                "property": property_name,
                "checkpoint_type": "best_ema",
                "normalizer": normalizer.state_dict(),
            }, save_dir / "best_ema.pt")
            ema.restore(model)

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        dt_ep = time.time() - t_ep
        epoch_times.append(dt_ep)
        elapsed = time.time() - t_train

        # ── Progress display ──
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            # ETA estimation
            avg_ep_time = sum(epoch_times[-20:]) / len(epoch_times[-20:])
            remaining_patience = args.patience - patience_counter
            est_remaining = avg_ep_time * min(remaining_patience, args.epochs - epoch)

            best_display = best_val_mae
            ema_str = ""
            if ema is not None:
                ema_str = f" | ema: {best_ema_val_mae:.4f}"

            progress_pct = epoch / args.epochs * 100
            print(
                f"  Epoch {epoch:4d}/{args.epochs} ({progress_pct:5.1f}%) | "
                f"loss: {train_loss:.4f} | "
                f"val: {val_mae:.4f} | "
                f"best: {best_display:.4f}{ema_str} | "
                f"lr: {lr:.2e} | "
                f"p: {patience_counter}/{args.patience} | "
                f"{dt_ep:.1f}s | "
                f"ETA: {format_time(est_remaining)}"
            )

        if patience_counter >= args.patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    train_time = time.time() - t_train
    print(f"\n  Training time: {train_time/60:.1f} minutes ({train_time/3600:.1f} hours)")

    # ── Test Evaluation ──
    print("\n" + "=" * 70)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 70)

    # Load best checkpoint (prefer EMA if it's better)
    use_ema = False
    if ema is not None and (save_dir / "best_ema.pt").exists():
        if best_ema_val_mae <= best_val_mae:
            use_ema = True

    if use_ema:
        print("  Using EMA checkpoint (better validation)")
        checkpoint = torch.load(save_dir / "best_ema.pt", weights_only=False)
    else:
        print("  Using regular checkpoint")
        checkpoint = torch.load(save_dir / "best.pt", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint["epoch"]

    test_metrics = evaluate(model, test_loader, property_name, device, normalizer=normalizer)

    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    test_mae = test_metrics.get(f"{property_name}_MAE", float("inf"))
    unit = benchmark["unit"]
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Property: {property_name:<34s}│")
    print(f"  │  Test MAE: {test_mae:.4f} {unit:<30s}│")
    print(f"  │  CGCNN lit: {benchmark['cgcnn_mae']:.4f} {unit:<29s}│")
    print(f"  │  ALIGNN lit: {benchmark['alignn_mae']:.4f} {unit:<28s}│")
    print(f"  │  Target: {benchmark['target_mae']:.4f} {unit:<31s}│")
    passed = test_mae <= benchmark["target_mae"]
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  │  Result: {status:<36s}│")
    print(f"  │  Best epoch: {best_epoch:<32d}│")
    print(f"  │  Checkpoint: {'EMA' if use_ema else 'regular':<30s}│")
    print(f"  │  Training: {train_time/3600:.1f}h ({epoch} epochs){' '*(22-len(f'{train_time/3600:.1f}h ({epoch} epochs)'))}│")
    print(f"  └─────────────────────────────────────────────┘")

    # Save results
    results = {
        "property": property_name,
        "model": "EquivariantGNN",
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
        "training_time_hours": train_time / 3600,
        "used_ema_checkpoint": use_ema,
        "best_val_mae": best_val_mae,
        "best_ema_val_mae": best_ema_val_mae if ema else None,
        "hyperparameters": {
            "irreps": args.irreps,
            "max_ell": args.max_ell,
            "n_layers": args.n_layers,
            "n_radial": args.n_radial,
            "radial_hidden": args.radial_hidden,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "ema_decay": args.ema_decay,
            "outlier_sigma": args.outlier_sigma,
            "sched_patience": args.sched_patience,
            "sched_factor": args.sched_factor,
            "loss": "huber_delta1.0",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
        },
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    hist_ser = {k: [float(v) for v in vs] for k, vs in history.items()}
    with open(save_dir / "history.json", "w") as f:
        json.dump(hist_ser, f, indent=2)

    print(f"  Results saved to {save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="E(3)-Equivariant GNN Training — Maximum Precision (Phase 2)"
    )
    parser.add_argument(
        "--property", type=str, default="formation_energy",
        choices=list(BENCHMARKS.keys()),
    )
    parser.add_argument("--all-properties", action="store_true",
                        help="Train all 4 properties sequentially")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on samples (None = use all ~76k)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (32 for 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Initial learning rate (ultra-precise)")
    parser.add_argument("--min-lr", type=float, default=1e-8,
                        help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6,
                        help="L2 regularization (minimal)")
    parser.add_argument("--patience", type=int, default=250,
                        help="Early stopping patience")
    parser.add_argument("--grad-clip", type=float, default=0.5,
                        help="Gradient clipping max norm")

    # Scheduler
    parser.add_argument("--sched-patience", type=int, default=50,
                        help="ReduceLROnPlateau patience")
    parser.add_argument("--sched-factor", type=float, default=0.7,
                        help="LR reduction factor")

    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay (0 to disable)")

    # Outlier filter
    parser.add_argument("--outlier-sigma", type=float, default=8.0,
                        help="Outlier filter threshold in σ")

    # Model architecture
    parser.add_argument("--irreps", type=str, default="32x0e + 16x1o + 8x2e",
                        help="Hidden irreps")
    parser.add_argument("--max-ell", type=int, default=2,
                        help="Max spherical harmonic order")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of interaction blocks")
    parser.add_argument("--n-radial", type=int, default=20,
                        help="Number of Bessel radial basis functions")
    parser.add_argument("--radial-hidden", type=int, default=128,
                        help="Radial MLP hidden dimension")

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     E(3)-Equivariant GNN — Phase 2 Maximum Precision          ║")
    print("║     PhD Thesis: Improvement Over CGCNN Baseline               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    if args.all_properties:
        all_results = {}
        for prop in BENCHMARKS:
            result = train_single_property(args, prop)
            all_results[prop] = result

        # Summary table
        print("\n" + "=" * 70)
        print("  SUMMARY — ALL PROPERTIES (EquivariantGNN)")
        print("=" * 70)
        print(f"  {'Property':<20s} {'Test MAE':>10s} {'CGCNN Lit':>10s} "
              f"{'Target':>10s} {'Status':>8s}")
        print("  " + "-" * 62)
        for prop, res in all_results.items():
            mae = res["test_metrics"].get(f"{prop}_MAE", float("inf"))
            lit = BENCHMARKS[prop]["cgcnn_mae"]
            tgt = BENCHMARKS[prop]["target_mae"]
            st = "✅" if res["passed"] else "❌"
            print(f"  {prop:<20s} {mae:>8.4f}  {lit:>8.4f}  {tgt:>8.4f}  {st:>6s}")

        config = get_config()
        combined_path = config.paths.models_dir / "equivariant_summary.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Combined results: {combined_path}")
    else:
        train_single_property(args, args.property)


if __name__ == "__main__":
    main()
