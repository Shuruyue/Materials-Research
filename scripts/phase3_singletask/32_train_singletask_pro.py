import os
import sys
import torch
import torch.nn as nn
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from atlas.data.crystal_dataset import CrystalPropertyDataset, DEFAULT_PROPERTIES
from atlas.models.equivariant import EquivariantGNN, STD_PRESET, LARGE_PRESET
from atlas.utils.checkpoint import CheckpointManager
from atlas.config import get_config
from torch_geometric.loader import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", type=str, required=True, help="Target property to fine-tune (e.g., formation_energy)")
    parser.add_argument("--finetune-from", type=str, default=None, help="Path to Phase 2 checkpoint (e.g., models/multitask_std/best.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005) # Lower LR for fine-tuning
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights, only train output head")
    args = parser.parse_args()

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     🔵 E3NN SPECIALIST (Phase 3)                               ║")
    print(f"║     Target: {args.property:<40}           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── Data Loading ──
    print(f"\n[1/4] Loading Data for {args.property}...")
    # Only load the single property to save memory/speed
    train_data = CrystalPropertyDataset(properties=[args.property], split="train").prepare()
    val_data = CrystalPropertyDataset(properties=[args.property], split="val").prepare()
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # ── Model Construction ──
    print("\n[2/4] Building Specialist Model...")
    
    # Use Large Preset for Pro Tier
    encoder = EquivariantGNN(
        irreps_hidden=LARGE_PRESET["irreps"],
        max_ell=LARGE_PRESET["max_ell"],
        n_layers=LARGE_PRESET["n_layers"],
        max_radius=5.0,
        n_species=86,
        n_radial_basis=LARGE_PRESET["n_radial"],
        radial_hidden=LARGE_PRESET["radial_hidden"],
        output_dim=1 # Single output
    ).to(device)
    
    # Load Pre-trained Weights if provided
    if args.finetune_from:
        print(f"  ⬇️ Loading pre-trained weights from {args.finetune_from}")
        checkpoint = torch.load(args.finetune_from, map_location=device)
        
        # Load Encoder State Dictionary
        # Note: The Phase 2 checkpoint has a 'model_state_dict' which contains 'encoder.xxx' and 'heads.xxx'
        # We need to filter and load only the 'encoder.' keys
        
        pretrained_dict = checkpoint['model_state_dict']
        encoder_dict = encoder.state_dict()
        
        # Filter: preserve 'encoder.' prefix in checkpoint? 
        # Usually Phase 2 model is: self.encoder = EquivariantGNN(...)
        # So keys are 'encoder.layers.0...', which matches directly if we load into a wrapped model.
        # But here 'encoder' IS the model. So we strip 'encoder.' prefix.
        
        filtered_dict = {k.replace('encoder.', ''): v for k, v in pretrained_dict.items() if k.startswith('encoder.')}
        
        # Update current model
        encoder_dict.update(filtered_dict)
        encoder.load_state_dict(encoder_dict)
        print("  ✅ Encoder weights loaded successfully")
        
        if args.freeze_encoder:
            print("  🔒 Freezing encoder weights")
            for param in encoder.parameters():
                param.requires_grad = False
    
    model = encoder # In single task, the encoder output IS the prediction (after pooling inside encoder)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # ── Training Loop ──
    print(f"\n[3/4] Starting Fine-Tuning ({args.epochs} epochs)...")
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_dir = config.paths.models_dir / f"specialist_{args.property}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_manager = CheckpointManager(save_dir, metric_name="val_mae_best", mode="min")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        n_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            target = getattr(batch, args.property).view(-1, 1)
            mask = ~torch.isnan(target)
            if mask.sum() == 0: continue
            
            target = target[mask]
            
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
            pred = pred[mask] # EquivariantGNN output is (N_graphs, 1)
            
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * mask.sum().item()
            n_train += mask.sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / n_train if n_train > 0 else 0.0
        
        # Validation
        model.eval()
        val_mae = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                target = getattr(batch, args.property).view(-1, 1)
                mask = ~torch.isnan(target)
                if mask.sum() == 0: continue
                target = target[mask]
                
                pred = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
                pred = pred[mask]
                
                val_mae += torch.abs(pred - target).sum().item()
                n_val += mask.sum().item()
        
        avg_val_mae = val_mae / n_val if n_val > 0 else float('inf')
        
        scheduler.step(avg_val_mae)
        
        # Checkpoint
        is_best = ckpt_manager.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': avg_val_mae
            },
            metric_value=avg_val_mae,
            epoch=epoch
        )
        
        if epoch % 5 == 0 or is_best:
            print(f"Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.4f} {'🌟' if is_best else ''}")

    print(f"\n✅ Training Complete. Best MAE: {ckpt_manager.best_value:.4f}")
    if ckpt_manager.best_path:
        print(f"   Saved to: {ckpt_manager.best_path}")

if __name__ == "__main__":
    main()
