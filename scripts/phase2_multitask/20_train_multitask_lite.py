"""
Phase 2: Multi-Task E(3)-Equivariant GNN — LITE Tier (Fast Debug)

Designed for rapid validation of the pipeline.
- Uses a tiny model (preset='small')
- Trains for only 2 epochs
- Limtied to 100 samples
- Verifies that data loading, model forward pass, and backprop work without crashing.

Usage:
    python scripts/phase2_multitask/20_train_multitask_lite.py
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset
from atlas.models.equivariant import EquivariantGNN
from atlas.models.multi_task import MultiTaskGNN
from atlas.training.metrics import scalar_metrics

# ── Lite Config ──
# 1. Targeting only the 4 core properties for speed, but code supports 9
CORE_PROPERTIES = ["formation_energy", "band_gap", "bulk_modulus", "shear_modulus"]

# 2. Tiny Model Preset
LITE_PRESET = {
    "irreps": "16x0e + 8x1o",     # Tiny capacity
    "max_ell": 1,                 # Simple spherical harmonics
    "n_layers": 2,                # Shallow
    "n_radial": 10,
    "radial_hidden": 32,
    "head_hidden": 16,
}


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     🟢 E3NN LITE (Debug Mode)                                  ║")
    print("║     Fast Validation / Smoke Test                               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name()}")
    
    # ── 1. Data (Lite: 100 samples) ──
    print("\n[1/4] Loading Lite Dataset (100 samples)...")
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = CrystalPropertyDataset(
            properties=CORE_PROPERTIES,
            max_samples=100 if split == "train" else 20, # Tiny subsets
            split=split
        )
        ds.prepare()
        datasets[split] = ds
        
    print(f"    Train: {len(datasets['train'])}, Val: {len(datasets['val'])}")

    # ── 2. Model (Lite: Tiny) ──
    print("\n[2/4] Building Lite Model...")
    encoder = EquivariantGNN(
        irreps_hidden=LITE_PRESET["irreps"],
        max_ell=LITE_PRESET["max_ell"],
        n_layers=LITE_PRESET["n_layers"],
        max_radius=5.0,
        n_species=86, # Full periodic table
        n_radial_basis=LITE_PRESET["n_radial"],
        radial_hidden=LITE_PRESET["radial_hidden"],
        output_dim=1 # Dummy, handled by multitask wrapper
    )
    
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={p: {"type": "scalar"} for p in CORE_PROPERTIES},
        embed_dim=encoder.scalar_dim
    ).to(device)
    
    print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Train Loop (Lite: 2 Epochs) ──
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss() 
    
    print("\n[3/4] Testing Training Loop (2 Epochs)...")
    model.train()
    
    from torch_geometric.loader import DataLoader
    loader = DataLoader(datasets["train"], batch_size=4, shuffle=True,
                        num_workers=0, pin_memory=True)
    
    for epoch in range(1, 3):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            preds = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
            
            # Simple loss (just checking gradients)
            loss = 0
            for prop in CORE_PROPERTIES:
                if prop in preds:
                    target = getattr(batch, prop).view(-1, 1)
                    # Handle NaNs roughly for debug
                    mask = ~torch.isnan(target)
                    if mask.sum() > 0:
                        loss += criterion(preds[prop][mask], target[mask])
            
            if loss != 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        print(f"    Epoch {epoch}: Loss = {total_loss:.4f}")

    print("\n[4/4] ✅ Lite Test Passed! The pipeline is functional.")
    print("      You can now proceed to 'Std' or 'Pro' tiers.")

if __name__ == "__main__":
    main()
