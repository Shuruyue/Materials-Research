"""
Phase 1: Multi-Task Training with Shared CGCNN Encoder

Trains a single CGCNN encoder with 4 task-specific MLP heads
simultaneously predicting: formation_energy, band_gap, bulk_modulus, shear_modulus.

This serves as the fast BASELINE to compare against Phase 2 (Equivariant GNN).
It uses the exact same multi-task loss and data pipeline, just a different encoder.

Usage:
    python scripts/phase2_multitask/train_multitask_cgcnn.py --preset medium
    python scripts/phase2_multitask/train_multitask_cgcnn.py --preset medium --property-group priority7
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import (
    CrystalPropertyDataset,
    PHASE2_PROPERTY_GROUP_CHOICES,
    resolve_phase2_property_group,
)
from atlas.models.cgcnn import CGCNN
from atlas.models.m3gnet import M3GNet
from atlas.models.multi_task import MultiTaskGNN, ScalarHead
from atlas.models.prediction_utils import forward_graph_model
from atlas.training.checkpoint import CheckpointManager
from atlas.training.metrics import scalar_metrics
from atlas.training.normalizers import TargetNormalizer, MultiTargetNormalizer
from atlas.training.run_utils import resolve_run_dir, write_run_manifest
from atlas.console_style import install_console_style

install_console_style()


PROPERTIES = resolve_phase2_property_group("priority7")

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
        "pooling": "mean",
        "jk": "last",
        "message-aggr": "mean",
        "edge-gates": True,
    },
    "medium": {
        "description": "Standard CGCNN (Baseline)",
        "node-dim": 91,
        "edge-dim": 20,
        "hidden-dim": 128,
        "n-conv": 3,
        "n-fc": 2,
        "head-hidden": 64,
        "pooling": "mean_max",
        "jk": "concat",
        "message-aggr": "mean",
        "edge-gates": True,
    },
    "large": {
        "description": "Deep CGCNN",
        "node-dim": 91,
        "edge-dim": 20,
        "hidden-dim": 256,
        "n-conv": 5,
        "n-fc": 3,
        "head-hidden": 128,
        "pooling": "mean_max",
        "jk": "concat",
        "message-aggr": "mean",
        "edge-gates": True,
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
            except Exception: continue
        
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
        predictions = forward_graph_model(model, batch)
        
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
        predictions = forward_graph_model(model, batch)
        
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

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["cgcnn", "m3gnet"], default="cgcnn")
    parser.add_argument("--preset", type=str, default="medium", choices=MODEL_PRESETS.keys())
    parser.add_argument("--batch-size", type=int, default=128)  # CGCNN usually handles large batch
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pooling", choices=["mean", "sum", "max", "mean_max", "attn"], default=None)
    parser.add_argument("--jk", choices=["last", "mean", "concat"], default=None)
    parser.add_argument("--message-aggr", choices=["sum", "mean"], default=None)
    parser.add_argument("--no-edge-gates", action="store_true")
    parser.add_argument(
        "--property-group",
        choices=PHASE2_PROPERTY_GROUP_CHOICES,
        default="priority7",
        help="Phase 2 property group: core4/priority7/secondary2/all9",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run id (without or with 'run_' prefix)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Keep top-k best checkpoints")
    parser.add_argument("--keep-last-k", type=int, default=3,
                        help="Keep latest rotating checkpoints")
    args = parser.parse_args()

    global PROPERTIES
    PROPERTIES = resolve_phase2_property_group(args.property_group)

    # Apply preset
    print(f"\n[Config] Property group '{args.property_group}' ({len(PROPERTIES)} tasks)")
    preset = MODEL_PRESETS[args.preset]
    print(f"\n[Config] Applying preset '{args.preset}': {preset['description']}")
    for k, v in preset.items():
        if k != "description":
            print(f"  - {k}: {v}")
    print(f"  - encoder: {args.encoder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    config = get_config()
    model_family = "multitask_cgcnn" if args.encoder == "cgcnn" else "multitask_m3gnet"
    base_dir = config.paths.models_dir / model_family
    try:
        save_dir, created_new = resolve_run_dir(base_dir, resume=args.resume, run_id=args.run_id)
    except (FileNotFoundError, FileExistsError) as e:
        print(f"[ERROR] {e}")
        return 2
    run_msg = "Starting new run" if created_new else "Using existing run"
    print(f"[INFO] {run_msg}: {save_dir.name}")
    manager = CheckpointManager(save_dir, top_k=args.top_k, keep_last_k=args.keep_last_k)
    manifest_path = write_run_manifest(
        save_dir,
        args=args,
        project_root=Path(__file__).resolve().parent.parent.parent,
        extra={
            "status": "started",
            "phase": "phase2",
            "model_family": model_family,
        },
    )
    print(f"[INFO] Run manifest: {manifest_path}")

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
    # Loaders
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size,
                            num_workers=0, pin_memory=True)
    
    # Model
    print(f"\n[3/5] Building Multi-Task {args.encoder.upper()}...")
    
    # CGCNN Encoder
    class CGCNNEncoder(nn.Module):
        def __init__(
            self,
            node_dim,
            edge_dim,
            hidden_dim,
            n_conv,
            n_fc,
            *,
            dropout=0.0,
            pooling="mean_max",
            jk="concat",
            message_aggr="mean",
            use_edge_gates=True,
        ):
            super().__init__()
            self.cgcnn = CGCNN(
                node_dim,
                edge_dim,
                hidden_dim,
                n_conv,
                n_fc,
                output_dim=1,
                dropout=dropout,
                pooling=pooling,
                jk=jk,
                message_aggr=message_aggr,
                use_edge_gates=use_edge_gates,
            )
            self.hidden_dim = self.cgcnn.graph_dim
            
        def encode(self, x, edge_index, edge_attr, batch):
            return self.cgcnn.encode(x, edge_index, edge_attr, batch)

        def forward(self, x, edge_index, edge_attr, batch):
            return self.encode(x, edge_index, edge_attr, batch)

    pooling_mode = None
    jk_mode = None
    message_aggr = None
    use_edge_gates = None

    if args.encoder == "cgcnn":
        pooling_mode = args.pooling or preset["pooling"]
        jk_mode = args.jk or preset["jk"]
        message_aggr = args.message_aggr or preset["message-aggr"]
        use_edge_gates = preset["edge-gates"] and (not args.no_edge_gates)
        print(
            f"  - pooling: {pooling_mode}, jk: {jk_mode}, "
            f"message_aggr: {message_aggr}, edge_gates: {use_edge_gates}"
        )
        encoder = CGCNNEncoder(
            preset["node-dim"],
            preset["edge-dim"],
            preset["hidden-dim"],
            preset["n-conv"],
            preset["n-fc"],
            pooling=pooling_mode,
            jk=jk_mode,
            message_aggr=message_aggr,
            use_edge_gates=use_edge_gates,
        )
        embed_dim = encoder.hidden_dim
    else:
        if (
            args.pooling is not None
            or args.jk is not None
            or args.message_aggr is not None
            or args.no_edge_gates
        ):
            print("[WARN] CGCNN-specific args ignored for encoder=m3gnet.")
        encoder = M3GNet(
            n_species=86,
            embed_dim=preset["hidden-dim"],
            n_layers=preset["n-conv"],
            n_rbf=preset["edge-dim"],
            max_radius=5.0,
        )
        embed_dim = int(getattr(encoder, "embed_dim", preset["hidden-dim"]))
        print(
            f"  - m3gnet: embed_dim={embed_dim}, "
            f"layers={preset['n-conv']}, n_rbf={preset['edge-dim']}"
        )
    
    # Multi-Task Wrapper
    # Note: CGCNN.encode returns (B, hidden_dim)
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={p: {"type": "scalar"} for p in PROPERTIES},
        embed_dim=embed_dim,
    ).to(device)
    
    # Optim
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = UncertaintyWeightedLoss(len(PROPERTIES)).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    start_epoch = 1
    best_val_mae = float('inf')
    history = {"train_loss": [], "val_mae": [], "lr": []}
    checkpoint_path = save_dir / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "loss_fn_state_dict" in ckpt:
            loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
        if "normalizer" in ckpt:
            normalizer.load_state_dict(ckpt["normalizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))
        history = ckpt.get("history", history)
        print(f"[INFO] Resume epoch: {start_epoch}")
    elif args.resume:
        print(f"[WARN] --resume requested but checkpoint not found: {checkpoint_path}")

    # Train
    print(f"\n[4/5] Training for {args.epochs} epochs...")
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        loss = train_epoch(model, loss_fn, train_loader, optimizer, device, normalizer)

        val_metrics = evaluate(model, val_loader, device, normalizer)
        val_maes = [val_metrics[f"{p}_MAE"] for p in PROPERTIES if f"{p}_MAE" in val_metrics]
        val_mae = float(sum(val_maes) / len(val_maes)) if val_maes else float("inf")

        scheduler.step(val_mae)
        dt = time.time() - t0
        lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(float(loss))
        history["val_mae"].append(val_mae)
        history["lr"].append(lr)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_fn_state_dict": loss_fn.state_dict(),
            "normalizer": normalizer.state_dict(),
            "history": history,
            "best_val_mae": best_val_mae,
            "val_mae": val_mae,
            "preset": args.preset,
        }

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            state["best_val_mae"] = best_val_mae
            manager.save_best(state, val_mae, epoch)

        manager.save_checkpoint(state, epoch)

        log = f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val MAE: {val_mae:.3f}"
        for p in PROPERTIES:
            key = f"{p}_MAE"
            if key in val_metrics:
                log += f" {p[:3]}:{val_metrics[key]:.2f}"
        log += f" | lr:{lr:.2e} | {dt:.1f}s"
        print(log)

    # Final test evaluation with best checkpoint
    best_path = save_dir / "best.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, weights_only=False)
        if "model_state_dict" in best_ckpt:
            model.load_state_dict(best_ckpt["model_state_dict"])
            best_epoch = int(best_ckpt.get("epoch", -1))
        else:
            best_epoch = -1
    else:
        best_epoch = -1

    test_metrics = evaluate(model, test_loader, device, normalizer)
    test_maes = [test_metrics[f"{p}_MAE"] for p in PROPERTIES if f"{p}_MAE" in test_metrics]
    avg_test_mae = float(sum(test_maes) / len(test_maes)) if test_maes else float("inf")

    results = {
        "algorithm": f"{args.encoder}_multitask",
        "run_id": save_dir.name,
        "encoder": args.encoder,
        "preset": args.preset,
        "property_group": args.property_group,
        "properties": list(PROPERTIES),
        "test_metrics": test_metrics,
        "avg_test_mae": avg_test_mae,
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "total_epochs": last_epoch,
        "n_train": len(train_data),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "hyperparameters": {
            "preset": args.preset,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "encoder": args.encoder,
            "property_group": args.property_group,
            "max_samples": args.max_samples,
            "pooling": pooling_mode,
            "jk": jk_mode,
            "message_aggr": message_aggr,
            "use_edge_gates": bool(use_edge_gates) if use_edge_gates is not None else None,
            "top_k": args.top_k,
            "keep_last_k": args.keep_last_k,
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
                "best_epoch": int(best_epoch),
                "total_epochs": int(last_epoch),
                "avg_test_mae": float(avg_test_mae),
            },
        },
    )
    print(f"\n[5/5] [OK] Results saved to {save_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

