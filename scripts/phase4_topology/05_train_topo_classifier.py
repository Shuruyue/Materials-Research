#!/usr/bin/env python3
"""
Script 05: Train Topological GNN Classifier

Trains a Graph Neural Network to classify crystal structures as
topological (spillage > 0.5) or trivial using JARVIS-DFT data.

This is a surrogate model — it learns to predict topological character
from structure alone, without running expensive DFT+SOC calculations.

Usage:
    python scripts/05_train_topo_classifier.py                 # Full training
    python scripts/05_train_topo_classifier.py --epochs 20     # Quick test
    python scripts/05_train_topo_classifier.py --max 500       # Small dataset
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atlas.config import get_config
from atlas.data.jarvis_client import JARVISClient
from atlas.topology.classifier import CrystalGraphBuilder, TopoGNN


class TopoDataset(Dataset):
    """Dataset of crystal graphs with topological labels."""

    def __init__(self, structures, labels, builder: CrystalGraphBuilder):
        self.structures = structures
        self.labels = labels
        self.builder = builder
        self._cache = {}

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        struct = self.structures[idx]
        label = self.labels[idx]

        try:
            graph = self.builder.structure_to_graph(struct)
            graph["label"] = torch.FloatTensor([label])
            self._cache[idx] = graph
            return graph
        except Exception as e:
            # Return a dummy graph on failure
            return {
                "node_features": torch.zeros(1, 69),
                "edge_index": torch.zeros(2, 1, dtype=torch.long),
                "edge_features": torch.zeros(1, 20),
                "num_nodes": 1,
                "label": torch.FloatTensor([label]),
            }


def collate_graphs(batch):
    """Custom collate: merge individual graphs into a batched graph."""
    node_feats = []
    edge_indices = []
    edge_feats = []
    labels = []
    batch_idx = []

    offset = 0
    for i, g in enumerate(batch):
        n = g["num_nodes"]
        node_feats.append(g["node_features"])
        edge_indices.append(g["edge_index"] + offset)
        edge_feats.append(g["edge_features"])
        labels.append(g["label"])
        batch_idx.extend([i] * n)
        offset += n

    return {
        "node_features": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_features": torch.cat(edge_feats, dim=0),
        "batch": torch.LongTensor(batch_idx),
        "labels": torch.cat(labels, dim=0),
    }


def prepare_data(client: JARVISClient, max_samples: int = 5000):
    """Prepare balanced dataset of topological/trivial materials."""
    from jarvis.core.atoms import Atoms as JAtoms

    print("\n=== Preparing Training Data ===\n")

    df = client.load_dft_3d()

    # Filter materials with spillage data
    has_spillage = df["spillage"].notna()
    df_labeled = df[has_spillage].copy()
    print(f"  Materials with spillage data: {len(df_labeled)}")

    # Label: spillage > 0.5 → topological (1), else trivial (0)
    df_labeled["topo_label"] = (df_labeled["spillage"] > 0.5).astype(int)

    n_topo = df_labeled["topo_label"].sum()
    n_trivial = len(df_labeled) - n_topo
    print(f"  Topological: {n_topo}, Trivial: {n_trivial}")

    # Balance the dataset
    n_each = min(n_topo, n_trivial, max_samples // 2)
    topo_df = df_labeled[df_labeled["topo_label"] == 1].sample(n=n_each, random_state=42)
    trivial_df = df_labeled[df_labeled["topo_label"] == 0].sample(n=n_each, random_state=42)
    balanced = pd.concat([topo_df, trivial_df]).sample(frac=1, random_state=42)
    print(f"  Balanced dataset: {len(balanced)} ({n_each} each)")

    # Convert to pymatgen structures
    print(f"  Converting to structures...")
    structures = []
    labels = []
    skipped = 0

    for _, row in balanced.iterrows():
        try:
            jatoms = JAtoms.from_dict(row["atoms"])
            struct = jatoms.pymatgen_converter()
            structures.append(struct)
            labels.append(int(row["topo_label"]))
        except Exception:
            skipped += 1

    print(f"  Valid structures: {len(structures)} (skipped {skipped})")

    # Split: 80/10/10
    n = len(structures)
    idx = np.random.RandomState(42).permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    splits = {
        "train": (
            [structures[i] for i in idx[:n_train]],
            [labels[i] for i in idx[:n_train]],
        ),
        "val": (
            [structures[i] for i in idx[n_train:n_train + n_val]],
            [labels[i] for i in idx[n_train:n_train + n_val]],
        ),
        "test": (
            [structures[i] for i in idx[n_train + n_val:]],
            [labels[i] for i in idx[n_train + n_val:]],
        ),
    }

    print(f"  Train: {len(splits['train'][0])}, "
          f"Val: {len(splits['val'][0])}, "
          f"Test: {len(splits['test'][0])}")

    return splits


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        node_feats = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_feats = batch["edge_features"].to(device)
        batch_idx = batch["batch"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(node_feats, edge_index, edge_feats, batch_idx)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        node_feats = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_feats = batch["edge_features"].to(device)
        batch_idx = batch["batch"].to(device)
        labels = batch["labels"].to(device)

        logits = model(node_feats, edge_index, edge_feats, batch_idx)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description="Train topological classifier")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--max", type=int, default=5000, help="Max samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dim")
    args = parser.parse_args()

    cfg = get_config()
    print(cfg.summary())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Prepare data
    client = JARVISClient()
    splits = prepare_data(client, max_samples=args.max)

    # Build datasets
    builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12)

    train_ds = TopoDataset(*splits["train"], builder)
    val_ds = TopoDataset(*splits["val"], builder)
    test_ds = TopoDataset(*splits["test"], builder)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_graphs, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        collate_fn=collate_graphs, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        collate_fn=collate_graphs, num_workers=0,
    )

    # Build model
    model = TopoGNN(
        node_dim=len(builder.ELEMENTS) + 5,
        edge_dim=20,
        hidden_dim=args.hidden,
        n_layers=3,
        dropout=0.2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params:,}")

    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    criterion = nn.BCEWithLogitsLoss()

    model_dir = cfg.paths.models_dir / "topo_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} | "
                  f"{val_loss:8.4f} | {val_acc:6.1%} | {lr:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_dir / "best_model.pt")

        # Early stopping
        if epoch - best_epoch > 30:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best: epoch {best_epoch}, acc {best_val_acc:.1%})")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(model_dir / "best_model.pt", weights_only=True))
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best val acc:   {best_val_acc:.1%}")
    print(f"  Test accuracy:  {test_acc:.1%}")

    # Confusion matrix
    preds = np.array(preds)
    labels = np.array(labels)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Confusion Matrix:")
    print(f"                   Predicted")
    print(f"                  Triv  Topo")
    print(f"  Actual Triv  | {tn:4d}  {fp:4d}")
    print(f"         Topo  | {fn:4d}  {tp:4d}")
    print(f"\n  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    # Save model info
    info = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_params": n_params,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }

    import json
    with open(model_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Model saved to {model_dir}")
    print(f"✓ Training complete!")


if __name__ == "__main__":
    main()
