import torch
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT)) # Insert at 0 to prioritize local modules

# Import necessary modules
try:
    from atlas.data.crystal_dataset import CrystalPropertyDataset, DEFAULT_PROPERTIES
    from atlas.models.equivariant import EquivariantGNN
    from atlas.models.multi_task import MultiTaskGNN
    from atlas.training.normalizers import TargetNormalizer, MultiTargetNormalizer
except ImportError as e:
    print(f"[ERROR] ImportError: {e}")
    print(f"   sys.path: {sys.path}")
    sys.exit(1)

import numpy as np



# Define STD_PRESET locally to avoid script import issues
STD_PRESET = {
    "irreps": "64x0e + 32x1o + 16x2e",
    "max_ell": 2,
    "n_layers": 3,
    "n_radial": 20,
    "radial_hidden": 128,
    "head_hidden": 64,
}
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error

def evaluate_run(run_dir):
    print(f"\n[INFO] Inspecting run: {run_dir}")
    run_path = Path(run_dir)
    checkpoint_path = run_path / "best.pt"
    
    if not checkpoint_path.exists():
        print(f"[ERROR] No best.pt found in {run_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Load Checkpoint
    print("   Loading Checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct Model (Assuming Std architecture based on folder name 'std')
    # If Pro, we need Large Preset. But usually Std is what users run first.
    # Let's try to infer from checkpoint structure or just use Std preset.
    print("   Building Model...")
    encoder = EquivariantGNN(
        irreps_hidden=STD_PRESET["irreps"],
        max_ell=STD_PRESET["max_ell"],
        n_layers=STD_PRESET["n_layers"],
        max_radius=5.0,
        n_species=86,
        n_radial_basis=STD_PRESET["n_radial"],
        radial_hidden=STD_PRESET["radial_hidden"],
        output_dim=1
    ).to(device)
    
    # Get properties from dataset or default
    properties = DEFAULT_PROPERTIES 
    
    model = MultiTaskGNN(
        encoder=encoder,
        tasks={p: {"type": "scalar"} for p in properties},
        embed_dim=encoder.scalar_dim
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # Load Normalizer
    print("   Loading Normalizer...")
    # We need a dummy dataset to init normalizer structure, then load state
    dummy_ds = CrystalPropertyDataset(split="val", max_samples=10).prepare()
    normalizer = MultiTargetNormalizer(dummy_ds, properties)
    normalizer.load_state_dict(ckpt["normalizer"])

    # Load Validation Data
    print("   Loading Validation Data...")
    val_data = CrystalPropertyDataset(split="val").prepare()
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # Evaluate
    print("   Evaluating...")
    all_preds = {p: [] for p in properties}
    all_targets = {p: [] for p in properties}
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
            
            for prop in properties:
                if prop not in preds: continue
                target = getattr(batch, prop).view(-1, 1)
                mask = ~torch.isnan(target)
                if mask.sum() == 0: continue
                
                target = target[mask].view(-1, 1)
                pred = preds[prop][mask].view(-1, 1)
                
                # Denormalize
                pred_denorm = normalizer.denormalize(prop, pred)
                
                all_preds[prop].append(pred_denorm.cpu())
                all_targets[prop].append(target.cpu())

    # Calculate Metrics
    print("\n   Validation Metrics (MAE):")
    print(f"   {'Property':<20} | {'MAE':<10} | {'Target (approx)'}")
    print("-" * 50)
    
    avg_mae = 0
    count = 0
    
    for prop in properties:
        if all_preds[prop]:
            y_pred = torch.cat(all_preds[prop]).numpy()
            y_true = torch.cat(all_targets[prop]).numpy()
            mae = mean_absolute_error(y_true, y_pred)
            
            target_mae = "0.05" if "energy" in prop else "10.0" if "modulus" in prop else "0.3"
            print(f"   {prop:<20} | {mae:.4f}     | < {target_mae}")
            
            avg_mae += mae
            count += 1
            
    print("-" * 50)
    import glob
    import os
    # Auto-find latest run in std folder
    base_dir = PROJECT_ROOT / "models" / "multitask_std_e3nn"
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if runs:
        latest = runs[-1]
        evaluate_run(latest)
    else:
        print("No run folder found.")
