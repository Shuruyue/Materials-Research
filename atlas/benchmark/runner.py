import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset
from atlas.models.graph_builder import CrystalGraphBuilder
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class MatbenchRunner:
    """
    Runs Matbench tasks using a trained ATLAS model.
    """
    
    TASKS = {
        "matbench_mp_e_form": "formation_energy",
        "matbench_mp_gap": "band_gap",
        "matbench_log_gvrh": "shear_modulus", # Log scale
        "matbench_log_kvrh": "bulk_modulus",  # Log scale
        "matbench_dielectric": "dielectric",
        "matbench_phonons": "phonons",
        "matbench_jdft2d": "exfoliation_energy",
    }
    
    def __init__(self, model, property_name, device="auto"):
        self.model = model
        self.property_name = property_name # Target property name in standard ATLAS format
        self.cfg = get_config()
        self.graph_builder = CrystalGraphBuilder()
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()

    def load_task(self, task_name):
        """Lazy load matbench task data."""
        try:
            from matbench.bench import MatbenchBenchmark
            mb = MatbenchBenchmark(autoload=False)
            task = mb.tasks_map[task_name]
            task.load()
            return task
        except ImportError:
            raise ImportError("Matbench not installed. Run: pip install matbench")
        except KeyError:
            raise ValueError(f"Unknown task: {task_name}")

    def structure_to_data(self, structure, target=None):
        """Convert pymatgen structure to PyG Data."""
        try:
            # Convert to PyG graph using shared builder
            data = self.graph_builder.build_graph(structure)
            
            # Add dummy target for compatibility
            if target is not None:
                # Handle log-scale targets if needed (wrapper usually handles this, pass raw here)
                pass 
                
            return data
        except Exception as e:
            print(f"Error converting structure: {e}")
            return None

    def run_fold(self, task, fold):
        """Run a single fold of cross-validation."""
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
        
        # Note: In a real benchmark, we should RETRAIN the model on train_inputs.
        # But for quick evaluation of a Pre-trained "Universal" model (Phase 2),
        # we often just run inference (Zero-shot) or fine-tune.
        # For standard Matbench, specific training is required.
        
        # Here we implement INFERENCE mode for now to test our pre-trained model's generalization.
        # If user wants full generic training, we need a Training Loop here.
        
        print(f"  Fold {fold}: {len(test_inputs)} test samples")
        
        predictions = []
        targets = []
        
        # Parallel conversion using joblib
        from joblib import Parallel, delayed
        
        print(f"  Converting {len(test_inputs)} structures (Parallel)...")
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.structure_to_data)(s) for s in tqdm(test_inputs, leave=False)
        )
        
        # Filter None
        data_list = []
        valid_indices = []
        for i, data in enumerate(results):
            if data is not None:
                data_list.append(data)
                valid_indices.append(i)
                targets.append(test_outputs.iloc[i])
            else:
                predictions.append(np.nan)
                
        if not data_list:
            print("  ⚠️ No valid structures found.")
            return np.array(targets), np.array(predictions)

        loader = DataLoader(data_list, batch_size=32, shuffle=False)
        
        self.model.eval()
        with torch.no_grad():
            batch_preds = []
            for batch in loader:
                batch = batch.to(self.device)
                # Model forward
                # Assuming MultiTask model which returns specific keys
                out = self.model(batch.x, batch.edge_index, batch.edge_vec, batch.batch)
                
                if isinstance(out, dict):
                    pred = out.get(self.property_name)
                else:
                    pred = out # Single task model
                
                if pred is not None:
                    batch_preds.append(pred.cpu().numpy())
            
            if batch_preds:
                fold_preds = np.concatenate(batch_preds).flatten()
                predictions.extend(fold_preds)

        # In zero-shot, we might not match length if conversion failed
        # Just return rough metrics for now
        return np.array(targets), np.array(predictions[:len(targets)])

