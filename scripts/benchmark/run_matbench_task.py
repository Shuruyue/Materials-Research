#!/usr/bin/env python3
"""
Script: Run Matbench Benchmark Task

Evaluates a trained ATLAS model on standard Matbench tasks.
This script performs ZERO-SHOT inference (no fine-tuning) to test generalization,
or acts as a template for fine-tuning benchmarks.

Usage:
    python scripts/benchmark/run_matbench_task.py --task matbench_mp_e_form --model models/multitask_pro_e3nn/best.pt
"""

import argparse
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.benchmark.runner import MatbenchRunner
from atlas.models.utils import load_phase2_model, load_phase1_model

from atlas.models.utils import load_phase2_model, load_phase1_model

def fine_tune_fold(model, task, fold, property_name, device, epochs=50, lr=1e-4):
    """Fine-tune the model on the fold's training set."""
    print(f"\n⚡ Fine-tuning on Fold {fold} (Train: {len(task.get_train_and_val_data(fold)[0])} samples)")
    
    # 1. Prepare Data
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    
    runner = MatbenchRunner(model, property_name, device=device)
    
    from joblib import Parallel, delayed
    print("  Converting training structures...")
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(runner.structure_to_data)(s) for s in tqdm(train_inputs, leave=False)
    )
    
    data_list = []
    failed = 0
    for i, data in enumerate(results):
        if data is not None:
            # Attach Target manually since structure_to_data dummy target logic is limited
            target_val = train_outputs.iloc[i]
            # Handle log scale tasks if needed (Matbench targets are largely linear, log tasks are pre-logged?)
            # Matbench usually provides RAW values. User needs to check property.
            # For simplicity, we train on provided values.
            data.y = torch.tensor([target_val], dtype=torch.float)
            data_list.append(data)
        else:
            failed += 1
            
    print(f"  Valid training samples: {len(data_list)} (Failed: {failed})")
    
    from torch_geometric.loader import DataLoader
    loader = DataLoader(data_list, batch_size=16, shuffle=True) # Smaller batch for stability
    
    # 2. Optimizer (Only fine-tune heads? Or all?)
    # Usually fine-tuning all with small LR is best for deep transfer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.L1Loss() # MAE optimized
    
    model.train()
    
    for epoch in range(1, epochs+1):
        total_loss = 0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward
            out = model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
            
            if isinstance(out, dict):
                pred = out.get(property_name).flatten()
            else:
                pred = out.flatten()
                
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
            
        avg_loss = total_loss / n
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Ep {epoch}: Loss {avg_loss:.4f}")
            
    return model

def main():
    parser = argparse.ArgumentParser(description="Run Matbench Benchmark")
    parser.add_argument("--task", type=str, required=True, 
                        help="Task name (e.g., matbench_mp_e_form, matbench_mp_gap)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate (0-4)")
    
    # New optimization args
    parser.add_argument("--finetune", action="store_true", help="Fine-tune on training fold before evaluation")
    parser.add_argument("--epochs", type=int, default=30, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
    print(f"Running {args.task} on {device}")
    
    # 1. Load Model
    model = None
    normalizer = None
    try:
        model, normalizer = load_phase2_model(args.model, device)
    except Exception as e:
        print(f"Standard load failed: {e}")
        try:
            model, normalizer = load_phase1_model(args.model, device)
        except Exception as e2:
             print(f"Legacy load failed: {e2}")
             return

    # 2. Map Task to ATLAS Property
    # MatbenchRunner.TASKS maps matbench_name -> atlas_property_name
    runner = MatbenchRunner(model, None, device=device) # Init first to access TASKS
    
    if args.task not in runner.TASKS:
        print(f"Error: Unknown task {args.task}")
        print(f"Available: {list(runner.TASKS.keys())}")
        return
        
    atlas_property = runner.TASKS[args.task]
    runner.property_name = atlas_property
    print(f"Targeting property: {atlas_property}")
    
    # Check if model has this task, if not, add it (for fine-tuning)
    if hasattr(model, "task_names") and atlas_property not in model.task_names:
        print(f"⚠️ Task '{atlas_property}' not found in model. Adding new initialized Head.")
        if hasattr(model, "add_task"):
            # Most Matbench tasks are scalars. 
            # If tensor, we'd need to know type, but for now default scalar.
            model.add_task(atlas_property, task_type="scalar")
            model.to(device) # Ensure new parameters are on device
            print(f"  Head added. Model now supports: {model.task_names}")
    
    # 3. Load Matbench Data
    try:
        from matbench.bench import MatbenchBenchmark
        mb = MatbenchBenchmark(autoload=False)
        task = mb.tasks_map[args.task]
        task.load()
    except ImportError:
        print("Please install matbench: pip install matbench")
        return
        
    # 3.5 Fine-tuning (Optimization)
    if args.finetune:
        model = fine_tune_fold(
            model, task, args.fold, atlas_property, device, 
            epochs=args.epochs, lr=args.lr
        )
    else:
        print("\n🚀 running Zero-shot Inference (Use --finetune for better results)")
        
    # 4. Run Inference on Fold
    print(f"Evaluating Fold {args.fold}...")
    targets, preds = runner.run_fold(task, args.fold)
    
    # Denormalize if needed
    if normalizer is not None and not args.finetune:
        # Only denormalize if we didn't fine-tune. 
        # If we fine-tuned, the model adapted to raw values (assuming our fine-tune loop used raw values).
        # Actually our fine-tune loop aboves uses raw values from train_outputs.
        # So the model drifts from normalized space to raw space.
        preds_denorm = preds
    elif normalizer is not None:
         # Zero shot: Model predicts normalized.
        preds_tensor = torch.tensor(preds).view(-1, 1).to(device)
        preds_denorm = normalizer.denormalize(atlas_property, preds_tensor).cpu().numpy().flatten()
    else:
        preds_denorm = preds

    # 5. Metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    
    mae = mean_absolute_error(targets, preds_denorm)
    r2 = r2_score(targets, preds_denorm)
    
    print("\n" + "="*40)
    print(f"BENCHMARK RESULTS: {args.task}")
    print("="*40)
    print(f"Samples: {len(targets)}")
    print(f"MAE:     {mae:.4f}")
    print(f"R²:      {r2:.4f}")
    if args.finetune:
        print("Mode:    Fine-tuned")
    else:
        print("Mode:    Zero-shot")
    print("="*40)
    
    # Save results
    out_dir = Path("results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame({"target": targets, "prediction": preds_denorm})
    res_df.to_csv(out_dir / f"{args.task}_fold{args.fold}.csv", index=False)
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
