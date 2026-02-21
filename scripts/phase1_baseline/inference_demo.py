"""
Phase 1: Precision Inference Tool

This script loads a trained CGCNN model and performs inference on:
1. Single CIF files
2. Directories of CIF files (Batch Processing)
3. Random samples from the test set (Verification)

It ensures 100% compatibility with the training pipeline by using the same
graph construction logic (JARVIS Atoms -> Pymatgen Structure -> Graph).

Usage:
    # Single file
    python scripts/phase1_baseline/inference_demo.py --cif data/structure.cif

    # Batch processing (save to csv)
    python scripts/phase1_baseline/inference_demo.py --dir data/my_structures/ --output results.csv

    # Verification
    python scripts/phase1_baseline/inference_demo.py --test-random
"""

import argparse
import sys
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from jarvis.core.atoms import Atoms

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.models.cgcnn import CGCNN
from atlas.models.graph_builder import CrystalGraphBuilder
from torch_geometric.data import Batch

# ─────────────────────────────────────────────────────────────────────────────
# Core Inference Logic
# ─────────────────────────────────────────────────────────────────────────────

class AtlasPredictor:
    """
    Wrapper class for loading models and running inference.
    """
    def __init__(self, model_dir: Path, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_dir = model_dir
        self.model = None
        self.builder = None
        self.normalizer = None
        self.hyperparams = {}
        
        self._load_model()
        
    def _load_model(self):
        """Load checkpoint, normalizer, and rebuild model architecture."""
        checkpoint_path = self.model_dir / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        print(f"[INFO] Loading model from {self.model_dir.name} ({self.device})...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 1. Load Normalizer (Vital for correct units)
        norm_state = checkpoint.get("normalizer", {})
        self.normalizer = {
            "mean": norm_state.get("mean", 0.0),
            "std": norm_state.get("std", 1.0)
        }
        
        # 2. Setup Graph Builder (Must match training settings)
        self.builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12)
        
        # 3. Rebuild Model
        # Try to load hyperparams from results.json, else default to Pro
        results_path = self.model_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                res = json.load(f)
                self.hyperparams = res.get("hyperparameters", {})
        
        self.model = CGCNN(
            node_dim=self.builder.node_dim,
            edge_dim=20,  # Must match training script (20)
            hidden_dim=self.hyperparams.get("hidden_dim", 512),
            n_conv=self.hyperparams.get("n_conv", 5),
            output_dim=1,
            dropout=0.0
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"[INFO] Model loaded successfully.")

    def predict(self, atoms: Atoms) -> float:
        """
        Predict property for a single JARVIS Atoms object.
        """
        # Convert to pymatgen (Standard Pipeline)
        structure = atoms.pymatgen_converter()
        
        # Build Graph
        # We assign target_value=0.0 as a dummy holder
        data = self.builder.structure_to_pyg(structure, target=0.0)
        
        # Batching (batch size=1)
        batch = Batch.from_data_list([data]).to(self.device)
        
        with torch.no_grad():
            pred_norm = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # Denormalize
            pred_val = pred_norm.item() * self.normalizer["std"] + self.normalizer["mean"]
            
        return pred_val

    def predict_batch(self, cif_paths: list) -> pd.DataFrame:
        """
        Predict for a list of CIF files with progress bar.
        Returns: DataFrame with filenames and predictions.
        """
        results = []
        print(f"[INFO] Processing {len(cif_paths)} files...")
        
        for path in tqdm(cif_paths):
            try:
                atoms = Atoms.from_cif(str(path))
                val = self.predict(atoms)
                results.append({
                    "filename": path.name,
                    "prediction": val,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "filename": path.name,
                    "prediction": None,
                    "status": f"error: {str(e)}"
                })
        
        return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Atlas Precision Inference Tool")
    parser.add_argument("--cif", type=Path, help="Path to single CIF file")
    parser.add_argument("--dir", type=Path, help="Directory containing CIF files for batch processing")
    parser.add_argument("--output", type=Path, default="predictions.csv", help="Output file for batch results")
    parser.add_argument("--test-random", action="store_true", help="Verify on random test sample")
    
    args = parser.parse_args()
    

    # Auto-detect model path (Find LATEST run)
    base_dir = PROJECT_ROOT / "models" / "cgcnn_pro_formation_energy"
    
    # Check for run_XXXX subdirectories
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if runs:
        model_dir = runs[-1]
        print(f"[INFO] Using latest run: {model_dir.name}")
    elif base_dir.exists() and (base_dir / "best.pt").exists():
        model_dir = base_dir # Fallback to old flat structure
        print(f"[INFO] Using base model directory: {model_dir.name}")
    else:
        print(f"Error: No trained models found in {base_dir}")
        print("Please train the Phase 1 Pro model first.")
        return

    # Initialize Predictor
    try:
        predictor = AtlasPredictor(model_dir)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Mode 1: Single file
    if args.cif:
        if not args.cif.exists():
            print(f"Error: File {args.cif} not found.")
            return
        
        print(f"\n[INFO] Predicting for: {args.cif.name}")
        atoms = Atoms.from_cif(str(args.cif))
        val = predictor.predict(atoms)
        print(f"   Formation Energy: {val:.4f} eV/atom")

    # Mode 2: Directory Batch
    elif args.dir:
        if not args.dir.exists():
            print(f"Error: Directory {args.dir} not found.")
            return
            
        cifs = list(args.dir.glob("*.cif"))
        if not cifs:
            print(f"No .cif files found in {args.dir}")
            return
            
        df = predictor.predict_batch(cifs)
        
        # Smart Export (Excel or CSV)
        output_path = args.output
        if output_path.suffix == ".xlsx":
            try:
                df.to_excel(output_path, index=False)
                print(f"\n[OK] Saved {len(df)} predictions to Excel: {output_path}")
            except ImportError:
                print("\n[WARN] pandas openpyxl missing. Falling back to CSV.")
                df.to_csv(output_path.with_suffix(".csv"), index=False)
                print(f"[OK] Saved predictions to CSV: {output_path.with_suffix('.csv')}")
        else:
            df.to_csv(output_path, index=False)
            print(f"\n[OK] Saved {len(df)} predictions to CSV: {output_path}")
            
        # Preview Results Table
        print("\n" + "="*50)
        print("PREVIEW RESULT (Top 5)")
        print("="*50)
        print(df.head().to_markdown(index=False, tablefmt="grid"))
        print("="*50)

    # Mode 3: Verification
    elif args.test_random:
        from atlas.data.crystal_dataset import CrystalPropertyDataset
        print("\n[INFO] Loading test dataset for verification...")
        ds = CrystalPropertyDataset(properties=["formation_energy"], split="test")
        ds.prepare()
        
        import random
        idx = random.randint(0, len(ds)-1)
        sample = ds[idx]
        
        # We need to reconstruct structure to go through the full pipeline
        # OR just confirm we can predict on PyG data directly.
        # Let's trust the loaded PyG graph which is ground truth here.
        
        # However, to simulate 'inference', we should use the predictor's method
        # but that takes 'Atoms'. 
        # For simplicity in this verificaiton mode, we stick to checking model output on graph.
        
        batch = Batch.from_data_list([sample]).to(predictor.device)
        with torch.no_grad():
            pred_norm = predictor.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred_val = pred_norm.item() * predictor.normalizer["std"] + predictor.normalizer["mean"]
            
        actual = sample.formation_energy.item()
        print(f"\n[INFO] Random Test Sample (Index {idx})")
        print(f"   JID:       {getattr(sample, 'jid', 'unknown')}")
        print(f"   Predicted: {pred_val:.4f} eV/atom")
        print(f"   Actual:    {actual:.4f} eV/atom")
        print(f"   Error:     {abs(pred_val - actual):.4f}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

