
from typing import Sequence, Tuple, Optional, Dict, List

import ase
import numpy as np
import torch
import torch.nn.functional as F
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE
from ase.stress import full_3x3_to_voigt_6_stress

try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from mace import data
from mace.tools import torch_geometric

# Import from our local ported module
from .model import (
    AlchemicalPair,
    AlchemyManager,
    AlchemicalModel,
    load_alchemical_model
)

class AlchemicalMACECalculator(Calculator):
    """
    ASE Calculator for Alchemical MACE.
    Supports continuous interpolation of atomic species for optimization.
    """
    
    implemented_properties = ["energy", "free_energy", "forces", "stress", "alchemical_grad"]

    def __init__(
        self,
        atoms: ase.Atoms,
        alchemical_pairs: Sequence[Sequence[Tuple[int, int]]],
        alchemical_weights: Sequence[float],
        device: str = "cpu",
        model_size: str = "medium",
        model_path: Optional[str] = None, # Allow loading custom checkpoint
    ):
        """
        Initialize the Alchemical MACE calculator.

        Args:
            atoms: ASE Atoms object (provides initial structure and species).
            alchemical_pairs: List of lists of (index, atomic_number) tuples.
            alchemical_weights: Initial weights for the alchemical species [0, 1].
            device: 'cpu' or 'cuda'.
            model_size: 'small', 'medium', or 'large' (default MACE-MP models).
            model_path: Optional path to a specific model checkpoint.
        """
        Calculator.__init__(self)
        self.results = {}
        self.device = device
        
        # Load Model
        if model_path:
             # TODO: Implement loading from specific path if needed
             # For now, we default to the factory function for MACE-MP
            raise NotImplementedError("Custom model path not yet supported in this port.")
        else:
            self.model = load_alchemical_model(
                model_size=model_size, device=device
            )

        # Freeze model parameters (we only optimize alchemical weights or geometry)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Initialize Alchemy Manager
        # Extract Z-Table from the loaded model
        # AtomicNumberTable is usually in mace.tools
        from mace.tools import AtomicNumberTable
        z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        r_max = self.model.r_max.item()
        
        alchemical_weights_tensor = torch.tensor(alchemical_weights, dtype=torch.float32)
        
        self.alchemy_manager = AlchemyManager(
            atoms=atoms,
            alchemical_pairs=alchemical_pairs,
            alchemical_weights=alchemical_weights_tensor,
            z_table=z_table,
            r_max=r_max,
        ).to(self.device)

        # Optimization control
        self.calculate_alchemical_grad = False
        self.alchemy_manager.alchemical_weights.requires_grad = False
        
        self.num_atoms = len(atoms)

    def set_alchemical_weights(self, alchemical_weights: Sequence[float]):
        """Update the alchemical mixing weights."""
        tensor_weights = torch.tensor(
            alchemical_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self.alchemy_manager.alchemical_weights.data = tensor_weights
        self.results = {} # Force recalculation without clearing self.atoms (which reset() does)

    def get_alchemical_atomic_masses(self) -> np.ndarray:
        """
        Calculate effective atomic masses based on current alchemical weights.
        Useful for MD or vibrational analysis.
        """
        # 1. Get masses of all possible species in the system
        # self.alchemy_manager.atomic_numbers contains all involved Zs
        node_masses = ase.data.atomic_masses[self.alchemy_manager.atomic_numbers]
        
        # 2. Get current weights for these nodes
        weights = self.alchemy_manager.alchemical_weights.data
        # Pad with 1.0 for fixed atoms (index 0)
        weights = F.pad(weights, (1, 0), "constant", 1.0).cpu().numpy()
        node_weights = weights[self.alchemy_manager.weight_indices]

        # 3. Sum up weighted masses for each original atom site
        atom_masses = np.zeros(self.num_atoms, dtype=np.float32)
        
        # self.alchemy_manager.atom_indices maps "expanded node" -> "original atom index"
        # We accumulate mass contributions
        np.add.at(
            atom_masses, 
            self.alchemy_manager.atom_indices, 
            node_masses * node_weights
        )
        return atom_masses

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Perform the electronic structure calculation (Energy, Forces, Stress).
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Prepare Inputs
        tensor_kwargs = {"dtype": torch.float32, "device": self.device}
        positions = torch.tensor(self.atoms.get_positions(), **tensor_kwargs)
        cell = torch.tensor(self.atoms.get_cell().array, **tensor_kwargs)

        # Toggle Gradient Calculation for Alchemical Weights
        if self.calculate_alchemical_grad:
            self.alchemy_manager.alchemical_weights.requires_grad = True
        
        # Build Batch
        batch = self.alchemy_manager(positions, cell).to(self.device)

        # Forward Pass
        if self.calculate_alchemical_grad:
            out = self.model(
                batch, 
                compute_stress=True, 
                compute_alchemical_grad=True,
                retain_graph=True
            )
            
            # Compute Gradient w.r.t Weights manually if needed or extract from model
            # MACE's get_outputs_alchemical returns node_grad/edge_grad w.r.t node_weights/edge_weights inputs
            # We need grad w.r.t original self.alchemy_manager.alchemical_weights
            
            # Since node_weights/edge_weights are differentiable functions of alchemical_weights (via padding/indexing),
            # calling backward on energy should populate alchemical_weights.grad
            
            # Wait, MACE returns gradients w.r.t inputs, but we want to stick to AutoGrad graph?
            # Creating the graph:
            # batch = manager(pos, cell) -> creates node/edge weights from manager.weights
            # out = model(batch) -> computes energy
            # out['energy'].backward() -> propagates to manager.weights
            
            # The original implementation did:
            # (grad,) = torch.autograd.grad(..., inputs=[weights], ...)
            # This is cleaner.
            
            (grad_weights,) = torch.autograd.grad(
                outputs=[out["energy"]], # Scalar energy
                inputs=[self.alchemy_manager.alchemical_weights],
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
            
            # Check if None (unused)
            if grad_weights is None:
                grad_weights = torch.zeros_like(self.alchemy_manager.alchemical_weights)
            
            grad = grad_weights.cpu().numpy()
            
            # Reset
            self.alchemy_manager.alchemical_weights.requires_grad = False
            self.alchemy_manager.alchemical_weights.grad = None
            
        else:
            # Standard eval
            out = self.model(batch, retain_graph=False, compute_stress=True)
            grad = np.zeros(
                self.alchemy_manager.alchemical_weights.shape[0], dtype=np.float32
            )

        # Collect Results
        self.results = {
            "energy": out["energy"].item(),
            "free_energy": out["energy"].item(),
            "forces": out["forces"].detach().cpu().numpy(),
            "stress": full_3x3_to_voigt_6_stress(
                out["stress"][0].detach().cpu().numpy()
            ),
            "alchemical_grad": grad,
        }
