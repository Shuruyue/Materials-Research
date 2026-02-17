"""
Phase 2: Multi-Task Training with Shared Equivariant Encoder

Trains a single EquivariantGNN encoder with 4 task-specific MLP heads
simultaneously predicting: formation_energy, band_gap, bulk_modulus, shear_modulus.

Multi-task loss uses uncertainty-weighted approach (Kendall et al., CVPR 2018):
    L = Σ_t (1/(2σ_t²)) L_t + log(σ_t)
where σ_t is a learnable task uncertainty that auto-balances loss magnitudes.

Key improvement over single-task: target normalization per property.

Usage:
    python scripts/21_train_multitask.py
    python scripts/21_train_multitask.py --max-samples 5000  # quick test
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
from atlas.models.equivariant import EquivariantGNN
from atlas.models.multi_task import MultiTaskGNN, ScalarHead
from atlas.training.metrics import scalar_metrics


PROPERTIES = DEFAULT_PROPERTIES  # ["formation_energy", "band_gap", "bulk_modulus", "shear_modulus"]

# Literature benchmarks
BENCHMARKS = {
    "formation_energy": {"unit": "eV/atom", "cgcnn_mae": 0.063, "alignn_mae": 0.033, "target_mae": 0.070},
    "band_gap":         {"unit": "eV",      "cgcnn_mae": 0.20,  "alignn_mae": 0.14,  "target_mae": 0.25},
    "bulk_modulus":     {"unit": "GPa",     "cgcnn_mae": 10.5,  "alignn_mae": 8.3,   "target_mae": 12.0},
    "shear_modulus":    {"unit": "GPa",     "cgcnn_mae": 8.0,   "alignn_mae": 6.5,   "target_mae": 10.0},
}


# ── Model Presets ──

MODEL_PRESETS = {
    "small": {
        "description": "Fast debugging model (Speed: ~10x Medium)",
        "irreps": "32x0e + 16x1o",
        "max-ell": 1,
        "n-layers": 2,
        "n-radial": 16,
        "radial-hidden": 64,
        "head-hidden": 32,
    },
    "medium": {
        "description": "Balanced model (Default)",
        "irreps": "64x0e + 32x1o + 16x2e",
        "max-ell": 2,
        "n-layers": 3,
        "n-radial": 20,
        "radial-hidden": 128,
        "head-hidden": 64,
    },
    "large": {
        "description": "High performance model",
        "irreps": "128x0e + 64x1o + 32x2e + 16x3o",
        "max-ell": 3,
        "n-layers": 4,
        "n-radial": 32,
        "radial-hidden": 256,
        "head-hidden": 128,
    },
    "ultra": {
        "description": "Max precision (No time limit)",
        "irreps": "256x0e + 128x1o + 64x2e + 32x3o",
        "max-ell": 3,
        "n-layers": 6,
        "n-radial": 64,
        "radial-hidden": 512,
        "head-hidden": 256,
    },
}


# ── Target Normalization ──

# ── Ensure all Data objects have all properties (NaN for missing) ──

def pad_missing_properties(dataset, properties):
    """Ensure every PyG Data object has all properties.
    Missing properties are set to NaN so DataLoader can collate."""
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
    """Z-score normalization per property, using training set statistics (skips NaN)."""

    def __init__(self, dataset, property_name: str):
        values = []
        for i in range(len(dataset)):
            data = dataset[i]
            try:
                val = getattr(data, property_name).item()
            except (KeyError, AttributeError):
                continue
            if not np.isnan(val):
                values.append(val)
        arr = np.array(values)
        self.mean = float(arr.mean())
        self.std = float(arr.std())
        if self.std < 1e-8:
            self.std = 1.0
        print(f"      {property_name}: n={len(values)}, mean={self.mean:.4f}, std={self.std:.4f}, "
              f"range=[{arr.min():.2f}, {arr.max():.2f}]")

    def normalize(self, y):
        return (y - self.mean) / self.std

    def denormalize(self, y):
        return y * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}


class MultiTargetNormalizer:
    """Container for per-property normalizers."""

    def __init__(self, dataset, properties):
        self.normalizers = {}
        for prop in properties:
            self.normalizers[prop] = TargetNormalizer(dataset, prop)

    def normalize(self, prop, y):
        return self.normalizers[prop].normalize(y)

    def denormalize(self, prop, y):
        return self.normalizers[prop].denormalize(y)

    def state_dict(self):
        return {prop: n.state_dict() for prop, n in self.normalizers.items()}


# ── Outlier Filtering ──

def filter_outliers(dataset, properties, n_sigma=8.0):
    """Filter extreme outliers across all properties."""
    indices_to_keep = set(range(len(dataset)))
    total_removed = 0

    for prop in properties:
        values = []
        valid_indices = []
        for i in range(len(dataset)):
            try:
                data = dataset[i]
                val = getattr(data, prop).item()
                values.append(val)
                valid_indices.append(i)
            except (KeyError, AttributeError):
                continue

        if not values:
            continue

        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        if std < 1e-8:
            continue

        remove = set()
        for j, v in enumerate(values):
            if abs(v - mean) > n_sigma * std:
                remove.add(valid_indices[j])

        if remove:
            print(f"    Outlier filter ({n_sigma}σ) for {prop}: removed {len(remove)} samples")
            indices_to_keep -= remove
            total_removed += len(remove)

    kept = torch.utils.data.Subset(dataset, sorted(indices_to_keep))
    return kept


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Learns a log-variance parameter per task that automatically
    balances the contribution of each task's loss.

    L = Σ_t [1/(2σ²_t) * L_t + log(σ_t)]
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list) -> torch.Tensor:
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total

    def get_weights(self) -> dict:
        """Return current task weights for logging."""
        weights = {}
        with torch.no_grad():
            for i, name in enumerate(PROPERTIES):
                sigma = torch.exp(0.5 * self.log_vars[i]).item()
                weight = torch.exp(-self.log_vars[i]).item()
                weights[name] = {"sigma": round(sigma, 4), "weight": round(weight, 4)}
        return weights


def train_epoch(model, loss_fn, loader, optimizer, device,
                normalizer=None, ema=None, grad_clip=0.5):
    """Train one epoch with multi-task loss and per-property normalization."""
    model.train()
    total_loss = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        predictions = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)

        task_losses = []
        valid_tasks = 0
        for prop in PROPERTIES:
            if prop not in predictions:
                continue
            target = getattr(batch, prop).view(-1, 1)
            pred = predictions[prop]

            # Skip NaN targets (missing properties)
            valid_mask = ~torch.isnan(target).squeeze(-1)
            if valid_mask.sum() == 0:
                continue
            target = target[valid_mask]
            pred = pred[valid_mask]

            # Normalize target to ~N(0,1)
            if normalizer is not None:
                target_norm = normalizer.normalize(prop, target)
            else:
                target_norm = target

            task_loss = nn.functional.huber_loss(pred, target_norm, delta=1.0)
            if not (torch.isnan(task_loss) or torch.isinf(task_loss)):
                task_losses.append(task_loss)
                valid_tasks += 1

        if valid_tasks == 0:
            continue

        # Pad with zeros for missing tasks
        while len(task_losses) < len(PROPERTIES):
            task_losses.append(torch.tensor(0.0, device=device))

        loss = loss_fn(task_losses)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # EMA update
        if ema is not None:
            with torch.no_grad():
                for p_ema, p_model in zip(ema.parameters(), model.parameters()):
                    p_ema.data.mul_(0.999).add_(p_model.data, alpha=0.001)

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, normalizer=None):
    """Evaluate model: denormalize predictions, return metrics in original units."""
    model.eval()
    all_preds = {p: [] for p in PROPERTIES}
    all_targets = {p: [] for p in PROPERTIES}

    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)

        for prop in PROPERTIES:
            if prop not in predictions:
                continue
            target = getattr(batch, prop).view(-1, 1)
            pred = predictions[prop]

            # Skip NaN targets (missing properties)
            valid_mask = ~torch.isnan(target).squeeze(-1)
            if valid_mask.sum() == 0:
                continue
            target = target[valid_mask]
            pred = pred[valid_mask]

            # Denormalize predictions back to original scale
            if normalizer is not None:
                pred = normalizer.denormalize(prop, pred)

            all_preds[prop].append(pred.cpu())
            all_targets[prop].append(target.cpu())

    metrics = {}
    for prop in PROPERTIES:
        if all_preds[prop]:
            pred = torch.cat(all_preds[prop])
            target = torch.cat(all_targets[prop])
            prop_metrics = scalar_metrics(pred, target, prefix=prop)
            metrics.update(prop_metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Task Equivariant GNN Training (Phase 2)"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--min-lr", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--sched-patience", type=int, default=50)
    parser.add_argument("--sched-factor", type=float, default=0.7)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--outlier-sigma", type=float, default=8.0)

    # Model architecture
    parser.add_argument("--preset", type=str, choices=MODEL_PRESETS.keys(),
                        help="Model size preset (overrides other model args)")
    parser.add_argument("--irreps", type=str, default="64x0e + 32x1o + 16x2e")
    parser.add_argument("--max-ell", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-radial", type=int, default=20)
    parser.add_argument("--radial-hidden", type=int, default=128)
    parser.add_argument("--head-hidden", type=int, default=64,
                        help="Hidden dim of per-task prediction heads")

    args = parser.parse_args()

    # Apply preset if specified
    if args.preset:
        preset = MODEL_PRESETS[args.preset]
        print(f"\n  [Config] Applying preset '{args.preset}': {preset['description']}")
        for k, v in preset.items():
            if k != "description":
                arg_name = k.replace("-", "_")
                setattr(args, arg_name, v)
                print(f"    - {arg_name}: {v}")

    print("=" * 70)
    print("  Multi-Task E(3)-Equivariant GNN — Phase 2")
    print("  4 Properties: Ef, Eg, K, G (Shared Encoder)")
    print("  Target Normalization + Uncertainty-Weighted Loss")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config()
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # ── Data (all 4 properties) ──
    print("\n  [1/5] Loading multi-property dataset...")
    t0 = time.time()

    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(
            properties=PROPERTIES,
            max_samples=args.max_samples,
            stability_filter=None,
            split=split,
        )
        ds.prepare()
        datasets[split] = ds

    stats = datasets["train"].property_statistics()
    for prop, s in stats.items():
        print(f"    {prop}: n={s['count']}, mean={s['mean']:.3f}, std={s['std']:.3f}")

    data_time = time.time() - t0
    print(f"    Data loading: {data_time:.1f}s")

    # ── Pad missing properties with NaN for collation ──
    print(f"\n  [1.5/5] Padding missing properties with NaN...")
    for split in ["train", "val", "test"]:
        pad_missing_properties(datasets[split], PROPERTIES)

    # ── Outlier filtering ──
    print(f"\n  [2/5] Filtering outliers ({args.outlier_sigma}sigma)...")
    train_data = filter_outliers(datasets["train"], PROPERTIES, n_sigma=args.outlier_sigma)
    val_data = filter_outliers(datasets["val"], PROPERTIES, n_sigma=args.outlier_sigma)
    test_data = filter_outliers(datasets["test"], PROPERTIES, n_sigma=args.outlier_sigma)
    print(f"    Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # ── Target normalization (from training set) ──
    print(f"\n  [3/5] Computing target normalization...")
    normalizer = MultiTargetNormalizer(train_data, PROPERTIES)

    from torch_geometric.loader import DataLoader as PyGLoader
    train_loader = PyGLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # ── Model ──
    print(f"\n  [4/5] Building Multi-Task EquivariantGNN...")

    encoder = EquivariantGNN(
        irreps_hidden=args.irreps,
        max_ell=args.max_ell,
        n_layers=args.n_layers,
        max_radius=5.0,
        n_species=86,
        n_radial_basis=args.n_radial,
        radial_hidden=args.radial_hidden,
        output_dim=1,
    )

    tasks = {prop: {"type": "scalar"} for prop in PROPERTIES}
    model = MultiTaskGNN(
        encoder=encoder,
        tasks=tasks,
        embed_dim=encoder.scalar_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in encoder.parameters())
    head_params = n_params - enc_params
    print(f"    Total parameters: {n_params:,}")
    print(f"    Encoder params:   {enc_params:,}")
    print(f"    Head params:      {head_params:,} ({len(PROPERTIES)} heads)")
    print(f"    Irreps: {args.irreps}")
    print(f"    Layers: {args.n_layers}, Radial: {args.n_radial}, Hidden: {args.radial_hidden}")

    # EMA model
    ema = None
    if args.ema_decay > 0:
        ema = copy.deepcopy(model)
        ema.eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        print(f"    EMA decay: {args.ema_decay}")

    # Loss and optimizer
    loss_fn = UncertaintyWeightedLoss(len(PROPERTIES)).to(device)
    all_params = list(model.parameters()) + list(loss_fn.parameters())

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.sched_patience,
        factor=args.sched_factor, min_lr=args.min_lr,
    )

    # ── Training ──
    print(f"\n  [5/5] Training for up to {args.epochs} epochs...")
    print(f"    LR: {args.lr} -> min {args.min_lr}")
    print(f"    Patience: {args.patience} (early stop), {args.sched_patience} (scheduler)")
    print(f"    Loss: Uncertainty-weighted Huber (Kendall et al.)")
    print("-" * 70)

    save_dir = config.paths.models_dir / "multitask_equivariant"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_avg_mae = float("inf")
    best_ema_val_avg_mae = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_avg_mae": [], "lr": []}
    t_train = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss = train_epoch(
            model, loss_fn, train_loader, optimizer, device,
            normalizer=normalizer, ema=ema, grad_clip=args.grad_clip,
        )
        val_metrics = evaluate(model, val_loader, device, normalizer=normalizer)

        # Average MAE across all properties (in original units)
        val_maes = [val_metrics.get(f"{p}_MAE", float("inf")) for p in PROPERTIES]
        valid_maes = [m for m in val_maes if m != float("inf") and m == m]
        val_avg_mae = sum(valid_maes) / max(len(valid_maes), 1)

        scheduler.step(val_avg_mae)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_avg_mae"].append(val_avg_mae)
        history["lr"].append(lr)

        if val_avg_mae != val_avg_mae or train_loss != train_loss:
            patience_counter += 1
            continue

        improved = False
        if val_avg_mae < best_val_avg_mae:
            best_val_avg_mae = val_avg_mae
            improved = True
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss_fn_state_dict": loss_fn.state_dict(),
                "normalizer": normalizer.state_dict(),
                "val_avg_mae": val_avg_mae,
                "val_metrics": val_metrics,
            }, save_dir / "best.pt")

        # EMA evaluation every 10 epochs
        if ema is not None and epoch % 10 == 0:
            ema_metrics = evaluate(ema, val_loader, device, normalizer=normalizer)
            ema_maes = [ema_metrics.get(f"{p}_MAE", float("inf")) for p in PROPERTIES]
            ema_valid = [m for m in ema_maes if m != float("inf") and m == m]
            ema_avg = sum(ema_valid) / max(len(ema_valid), 1)
            if ema_avg < best_ema_val_avg_mae:
                best_ema_val_avg_mae = ema_avg
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": ema.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "val_avg_mae": ema_avg,
                    "val_metrics": ema_metrics,
                }, save_dir / "best_ema.pt")

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        dt_ep = time.time() - t_ep

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            weights = loss_fn.get_weights()
            w_str = " | ".join(f"{p[:3]}={w['weight']:.2f}" for p, w in weights.items())
            per_prop = " ".join(
                f"{p[:3]}:{val_metrics.get(f'{p}_MAE', 0):.2f}"
                for p in PROPERTIES
            )
            eta_min = (args.epochs - epoch) * dt_ep / 60
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | "
                f"loss: {train_loss:.4f} | "
                f"avg: {val_avg_mae:.3f} | "
                f"best: {best_val_avg_mae:.3f} | "
                f"[{per_prop}] | "
                f"lr: {lr:.2e} | p: {patience_counter}/{args.patience} | "
                f"{dt_ep:.1f}s | ETA: {eta_min:.0f}m"
            )

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t_train
    print(f"\n  Training time: {train_time/60:.1f} minutes ({train_time/3600:.1f} hours)")

    # ── Test Evaluation ──
    print("\n" + "=" * 70)
    print("  FINAL TEST SET EVALUATION — MULTI-TASK")
    print("=" * 70)

    # Try EMA checkpoint first, fall back to regular
    ema_path = save_dir / "best_ema.pt"
    reg_path = save_dir / "best.pt"

    use_ema = False
    if ema_path.exists():
        ema_ckpt = torch.load(ema_path, weights_only=False)
        if ema_ckpt["val_avg_mae"] < torch.load(reg_path, weights_only=False)["val_avg_mae"]:
            model.load_state_dict(ema_ckpt["model_state_dict"])
            best_epoch = ema_ckpt["epoch"]
            use_ema = True
            print("  Using EMA checkpoint (better val MAE)")

    if not use_ema:
        checkpoint = torch.load(reg_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = checkpoint["epoch"]
        print("  Using regular checkpoint")

    test_metrics = evaluate(model, test_loader, device, normalizer=normalizer)

    print(f"\n  {'Property':<20s} {'Test MAE':>10s} {'Unit':>8s} {'Target':>8s} {'Status':>8s} {'R2':>6s}")
    print("  " + "-" * 64)
    all_passed = True
    for prop in PROPERTIES:
        mae = test_metrics.get(f"{prop}_MAE", float("inf"))
        r2 = test_metrics.get(f"{prop}_R2", 0.0)
        bm = BENCHMARKS[prop]
        passed = mae <= bm["target_mae"]
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  {prop:<20s} {mae:>10.4f} {bm['unit']:>8s} {bm['target_mae']:>8.4f} {status:>8s} {r2:>6.3f}")

    # Save results
    results = {
        "model": "MultiTaskEquivariantGNN",
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "all_passed": all_passed,
        "n_params": n_params,
        "n_encoder_params": enc_params,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "training_time_minutes": train_time / 60,
        "training_time_hours": train_time / 3600,
        "used_ema_checkpoint": use_ema,
        "best_val_avg_mae": best_val_avg_mae,
        "best_ema_val_avg_mae": best_ema_val_avg_mae if ema else None,
        "task_weights": loss_fn.get_weights(),
        "normalizer": normalizer.state_dict(),
        "benchmarks": BENCHMARKS,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "hyperparameters": {
            "irreps": args.irreps,
            "max_ell": args.max_ell,
            "n_layers": args.n_layers,
            "n_radial": args.n_radial,
            "radial_hidden": args.radial_hidden,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "ema_decay": args.ema_decay,
            "sched_patience": args.sched_patience,
            "sched_factor": args.sched_factor,
            "outlier_sigma": args.outlier_sigma,
            "loss": "uncertainty_weighted_huber_delta1.0",
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

    print(f"\n  Results saved to {save_dir}")


if __name__ == "__main__":
    main()
