"""
Phase 3: Single-Task E(3)-Equivariant GNN — PRO Tier (Specialist)

The "Single-Task Laser" for high-precision, specific property prediction.
- Ideal for benchmarking a single property (e.g., band_gap) with maximum accuracy.
- Supports Transfer Learning from Phase 2 Multi-Task models.

Full-scale training with NequIP-inspired equivariant architecture:
- Spherical harmonics edge features (respects SO(3) symmetry)
- Bessel radial basis with smooth cutoff
- Tensor product message passing with gated activation
- EMA (Exponential Moving Average) & SWA (Stochastic Weight Averaging)
- 8σ outlier filtering for clean training data
- Gradient Accumulation for large batch simulation
"""

import argparse
import copy
import torch
import torch.nn as nn
import json
import time
import numpy as np
from pathlib import Path
from torch.optim.swa_utils import AveragedModel, SWALR

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset
from atlas.models.equivariant import EquivariantGNN, LARGE_PRESET
from atlas.training.metrics import scalar_metrics


# ── Literature benchmarks ──
BENCHMARKS = {
    "formation_energy": {
        "unit": "eV/atom",
        "cgcnn_mae": 0.063,
        "alignn_mae": 0.033,
        "target_mae": 0.030,    # Phase 3 target: Approach DFT accuracy
    },
    "band_gap": {
        "unit": "eV",
        "cgcnn_mae": 0.20,
        "alignn_mae": 0.14,
        "target_mae": 0.15,
    },
    "bulk_modulus": {
        "unit": "GPa",
        "cgcnn_mae": 10.5,
        "alignn_mae": 8.3,
        "target_mae": 9.0,
    },
    "shear_modulus": {
        "unit": "GPa",
        "cgcnn_mae": 8.0,
        "alignn_mae": 6.5,
        "target_mae": 7.0,
    },
}


# ─────────────────────────────────────────────────────────
#  EMA — Exponential Moving Average
# ─────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters.
    Maintains a shadow copy of the model weights.
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
#  Outlier Filter & Normalizer
# ─────────────────────────────────────────────────────────

def filter_outliers(dataset, property_name, n_sigma=8.0):
    """Remove extreme outliers from dataset."""
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
        print(f"    Outlier filter ({n_sigma}σ): removed {n_removed} samples")
        indices = np.where(mask)[0].tolist()
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
    else:
        print(f"    Outlier filter ({n_sigma}σ): no outliers found")
    return dataset


class TargetNormalizer:
    """Z-score normalization for target property values."""
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

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
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
                normalizer=None, ema=None, grad_clip=0.5, acc_steps=1):
    """Train for one epoch with gradient accumulation and EMA."""
    model.train()
    total_loss = 0
    n = 0
    
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        if not hasattr(batch, property_name):
            continue

        target = getattr(batch, property_name).view(-1, 1)

        # Normalize target
        if normalizer is not None:
            target_norm = normalizer.normalize(target)
        else:
            target_norm = target

        # Forward
        pred = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
        
        # Loss (Huber is robust)
        loss = nn.functional.huber_loss(pred, target_norm, delta=0.5)
        
        # Scale loss for accumulation
        loss = loss / acc_steps
        loss.backward()

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        if (i + 1) % acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            
            # Update EMA only after optimizer step
            if ema is not None:
                ema.update(model)
                
            optimizer.zero_grad()

        total_loss += loss.item() * acc_steps * target.size(0)
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

        if normalizer is not None:
            pred = normalizer.denormalize(pred)

        all_pred.append(pred.cpu())
        all_target.append(target.cpu())

    if not all_pred:
        return {}

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    return scalar_metrics(pred, target, prefix=property_name)


def format_time(seconds):
    if seconds < 60: return f"{seconds:.0f}s"
    elif seconds < 3600: return f"{seconds / 60:.1f}m"
    else: return f"{seconds / 3600:.1f}h"


def train_single_property(args, property_name: str):
    """Train equivariant GNN for one property."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_config()
    benchmark = BENCHMARKS[property_name]

    print("\n" + "=" * 70)
    print(f"  EquivariantGNN PRO — Property: {property_name}")
    print(f"  Mode: {'Fine-Tuning' if args.finetune_from else 'Scratch'}")
    print(f"  SWA: {'Enabled' if args.use_swa else 'Disabled'}")
    print(f"  Acc. Steps: {args.acc_steps}")
    print("=" * 70)

    # ── Data ──
    print("\n  [1/4] Loading dataset...")
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(
            properties=[property_name],
            max_samples=args.max_samples,
            split=split,
        )
        ds.prepare()
        datasets[split] = ds

    # ── Outlier filtering ──
    print(f"\n  [2/4] Filtering outliers ({args.outlier_sigma}σ)...")
    for split in ["train", "val", "test"]:
        datasets[split]._data_list = filter_outliers(
            [datasets[split][i] for i in range(len(datasets[split]))],
            property_name, n_sigma=args.outlier_sigma
        )

    # ── Target Normalization ──
    print(f"\n  [3/5] Computing target normalization...")
    normalizer = TargetNormalizer(datasets["train"], property_name)

    # Loaders
    train_loader = datasets["train"].to_pyg_loader(batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = datasets["val"].to_pyg_loader(batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = datasets["test"].to_pyg_loader(batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ──
    print(f"\n  [4/5] Building EquivariantGNN (Pro Tier)...")
    model = EquivariantGNN(
        irreps_hidden=LARGE_PRESET["irreps"],
        max_ell=LARGE_PRESET["max_ell"],
        n_layers=LARGE_PRESET["n_layers"],
        max_radius=5.0,
        n_species=86,
        n_radial_basis=LARGE_PRESET["n_radial"],
        radial_hidden=LARGE_PRESET["radial_hidden"],
        output_dim=1,
    ).to(device)

    # Load Transfer Learning Weights
    if args.finetune_from:
        print(f"  ⬇️ Loading pre-trained weights from {args.finetune_from}")
        checkpoint = torch.load(args.finetune_from, map_location=device)
        pretrained_dict = checkpoint['model_state_dict']
        encoder_dict = model.state_dict()
        filtered_dict = {k.replace('encoder.', ''): v for k, v in pretrained_dict.items() if k.startswith('encoder.')}
        encoder_dict.update(filtered_dict)
        model.load_state_dict(encoder_dict)
        print("  ✅ Encoder weights loaded successfully")
        
        if args.freeze_encoder:
            print("  🔒 Freezing encoder weights")
            for param in model.species_embed.parameters(): param.requires_grad = False
            for param in model.interactions.parameters(): param.requires_grad = False
            for param in model.input_proj.parameters(): param.requires_grad = False

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    
    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.sched_patience, factor=args.sched_factor, min_lr=args.min_lr, verbose=True)

    # SWA
    swa_model = AveragedModel(model) if args.use_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1) if args.use_swa else None
    swa_start = int(args.epochs * 0.75) 

    # ── Training ──
    print(f"\n  [5/5] Training for up to {args.epochs} epochs...")
    save_dir = config.paths.models_dir / f"specialist_{property_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_ema_val_mae = float("inf")
    patience_counter = 0
    t_train = time.time()
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        
        # SWA Phase Logic
        if args.use_swa and epoch >= swa_start:
            is_swa_phase = True
            scheduler = swa_scheduler
        else:
            is_swa_phase = False

        train_loss = train_epoch(
            model, train_loader, optimizer, property_name, device,
            normalizer=normalizer, ema=ema, grad_clip=args.grad_clip, acc_steps=args.acc_steps
        )
        
        if args.use_swa and is_swa_phase:
            swa_model.update_parameters(model)
            scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, property_name, device, normalizer=normalizer)
        val_mae = val_metrics.get(f"{property_name}_MAE", float("inf"))

        # Evaluate EMA
        ema_val_mae = float("inf")
        if ema is not None:
            ema.apply_shadow(model)
            ema_metrics = evaluate(model, val_loader, property_name, device, normalizer=normalizer)
            ema_val_mae = ema_metrics.get(f"{property_name}_MAE", float("inf"))
            ema.restore(model)

        # Step Scheduler (if not SWA)
        if not is_swa_phase:
            effective_mae = min(val_mae, ema_val_mae) if ema else val_mae
            scheduler.step(effective_mae)

        # Checkpointing
        improved = False
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            improved = True
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae,
                "normalizer": normalizer.state_dict(),
            }, save_dir / "best.pt")

        if ema is not None and ema_val_mae < best_ema_val_mae:
            best_ema_val_mae = ema_val_mae
            improved = True
            ema.apply_shadow(model)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": ema_val_mae,
                "normalizer": normalizer.state_dict(),
            }, save_dir / "best_ema.pt")
            ema.restore(model)

        if improved: patience_counter = 0
        else: patience_counter += 1

        dt_ep = time.time() - t_ep
        epoch_times.append(dt_ep)

        if epoch % 5 == 0 or epoch == 1:
            sw_tag = " [SWA]" if is_swa_phase else ""
            print(f"  Epoch {epoch:4d}{sw_tag} | loss: {train_loss:.4f} | "
                  f"val: {val_mae:.4f} | best: {best_val_mae:.4f} | "
                  f"ema: {ema_val_mae:.4f} | {dt_ep:.1f}s")

        if patience_counter >= args.patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch}")
            break

    # Save SWA
    if args.use_swa:
        torch.save(swa_model.state_dict(), save_dir / "swa_final.pt")

    print(f"\n✅ Training Complete.")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="E(3)-Equivariant GNN Specialist Training (Phase 3)")
    parser.add_argument("--property", type=str, default="formation_energy", choices=list(BENCHMARKS.keys()))
    parser.add_argument("--all-properties", action="store_true", help="Train all properties")
    parser.add_argument("--finetune-from", type=str, default=None, help="Path to Phase 2 model")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder layers")
    parser.add_argument("--use-swa", action="store_true", help="Enable SWA")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--acc-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--min-lr", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--sched-patience", type=int, default=50)
    parser.add_argument("--sched-factor", type=float, default=0.7)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--outlier-sigma", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     PHASE 3: SPECIALIST PRO TRAINING (Fine-Tuning/SWA)         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    if args.all_properties:
        for prop in BENCHMARKS:
            train_single_property(args, prop)
    else:
        train_single_property(args, args.property)


if __name__ == "__main__":
    main()
