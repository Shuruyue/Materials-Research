
import os
import sys
import torch
import numpy as np
import ase
from typing import List, Optional

# Ensure MEPIN is in path
# Assuming typical project structure: atlas/discovery/stability/mepin.py -> ... -> root -> references
# We need to add 'references/recisic/mepin' to sys.path
# Absolute path approach is safer given the environment
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MEPIN_REPO_PATH = os.path.join(ROOT_DIR, "references", "recisic", "mepin")

if MEPIN_REPO_PATH not in sys.path:
    sys.path.append(MEPIN_REPO_PATH)

try:
    from mepin.model.modules import TripleCrossPaiNNModule
    from mepin.tools.inference import create_reaction_batch
except ImportError:
    raise ImportError(f"Could not import MEPIN from {MEPIN_REPO_PATH}. Please ensure the repo is cloned there.")

class MEPINStabilityEvaluator:
    """
    Wrapper for MEPIN (Minimum Energy Path Inference Network).
    Predicts reaction pathways and energy barriers between two states without iterative NEB.

    References:
        Nam et al., "Neural Network Potentials for MEP Prediction", arXiv (2024).
    """
    
    def __init__(self, 
                 checkpoint_path: Optional[str] = None, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 model_type: str = "cyclo_L"):
        """
        Initialize the MEPIN evaluator.

        Args:
            checkpoint_path: Explicit path to the model checkpoint.
            model_type: Type of pre-trained model to load if checkpoint_path is None.
                        Options: 'cyclo_L' (Default), 't1x_L'.
            device: Computation device ('cuda' or 'cpu').
        
        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
            ValueError: If an unknown model_type is specified.
        """
        self.device = device
        self.model_type = model_type
        
        if checkpoint_path is None:
            # Construct default path based on model_type
            if model_type not in ["cyclo_L", "t1x_L"]:
                raise ValueError(f"Unknown model_type: {model_type}. Supported: cyclo_L, t1x_L")
                
            checkpoint_path = os.path.join(MEPIN_REPO_PATH, "ckpt", f"{model_type}.ckpt")
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"MEPIN checkpoint not found at {checkpoint_path}")
            
        print(f"Loading MEPIN model from {checkpoint_path}...")
        try:
            # Load model
            # TripleCrossPaiNNModule.load_from_checkpoint might fail if dependencies mismatch
            # We map_location to device
            self.model = TripleCrossPaiNNModule.load_from_checkpoint(checkpoint_path, map_location=device)
            self.model.eval()
            self.model.to(device)
            print("MEPIN model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load MEPIN model: {e}")

    def predict_path(self, 
                    reactant: ase.Atoms, 
                    product: ase.Atoms, 
                    num_images: int = 20) -> List[ase.Atoms]:
        """
        Predict reaction path from reactant to product.
        
        Args:
            reactant: Initial state.
            product: Final state.
            num_images: Number of frames in the path (including endpoints).
            
        Returns:
            List of ase.Atoms along the path.
        """
        use_geodesic = "G" in self.model_type
        
        # Prepare batch
        # Note: create_reaction_batch expects consistent atoms
        # It handles Kabsch alignment internally for product
        
        try:
            batch = create_reaction_batch(
                reactant, 
                product, 
                interp_traj=None, # Not needed for Linear mode (L)
                use_geodesic=use_geodesic, 
                num_images=num_images
            )
            batch = batch.to(self.device)
            
            with torch.no_grad():
                # Model returns (N_nodes * N_images, 3) or reshaped?
                # module.forward returns (N_nodes * N_images, 3) flattened usually?
                # Check inference_example: 
                # model(batch).reshape(batch.num_graphs, -1, 3) -> valid for batching multiple reactions
                # Here we have 1 reaction in batch. 
                # But wait, num_images is embedded in the batch as "graph_time".
                # Actually, data_list in create_reaction_batch creates `num_images` graphs!
                # So batch.num_graphs = num_images.
                
                out = self.model(batch) # [Total_nodes, 3]
                
                # Reshape: [Num_images, Num_nodes_per_image, 3]
                # Assuming all images have same num_nodes (they must)
                n_atoms = len(reactant)
                out = out.reshape(num_images, n_atoms, 3)
                output_positions = out.cpu().numpy()
                
            # Construct Trajectory
            trajectory = []
            for i in range(num_images):
                atoms = reactant.copy() # inherited cell, pbc, atomic_numbers
                atoms.set_positions(output_positions[i])
                trajectory.append(atoms)
                
            return trajectory
            
        except Exception as e:
            print(f"Error during MEPIN prediction: {e}")
            raise e
