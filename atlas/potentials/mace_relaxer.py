"""
MACE Inference & Structure Relaxation Module

Wraps a trained MACE model (or uses the pre-trained MACE-MP-0 foundation model)
for fast structure relaxation and stability assessment.

Optimization:
- Robust Relaxation: Handles cell filter selection (Frechet/Exp) and errors.
- Trajectory Saving: Optional saving of optimization path for debugging.
- Logging: Use Python logging instead of print.
- Device Management: improved GPU/CPU selection.
"""

import numpy as np
import logging
import torch
import warnings
from typing import Optional, Dict, Any, Union
from pathlib import Path

from atlas.config import get_config
from atlas.utils.registry import RELAXERS

# Configure logging
logger = logging.getLogger(__name__)

@RELAXERS.register("mlip_arena_mace")
class MACERelaxer:
    """
    Fast structure relaxation using MACE potentials.
    
    Can use either:
    1. A custom-trained MACE model (from Phase 1A)
    2. The pre-trained MACE-MP-0 foundation model (zero-shot, covers all elements)
    """

    ):
        """
        Args:
            model_path: path to trained MACE model (.pt file)
            device: "cuda", "cpu", or "auto"
            use_foundation: if True and no model_path, use MACE-MP-0 foundation model
            model_size: "small", "medium", or "large" (default: "large")
            default_dtype: precision for calculation
        """
        self.cfg = get_config()
        self.model_path = model_path
        self._calculator = None
        self.use_foundation = use_foundation
        self.model_size = model_size
        self.dtype = default_dtype
        
        # Device resolution
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"MACERelaxer initialized on {self.device}")

    @property
    def calculator(self):
        """Lazy-load the MACE calculator."""
        if self._calculator is not None:
            return self._calculator

        # Suppress warnings from MACE/torch on load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.model_path and Path(self.model_path).exists():
                # Use custom-trained model
                logger.info(f"Loading custom MACE model: {self.model_path}")
                try:
                    from mace.calculators import MACECalculator
                    self._calculator = MACECalculator(
                        model_paths=self.model_path,
                        device=self.device,
                        default_dtype=self.dtype,
                    )
                except Exception as e:
                    logger.error(f"Failed to load custom MACE model: {e}")
                    self._calculator = None

            elif self.use_foundation:
                # Use pre-trained foundation model (MACE-MP-0)
                logger.info("Loading MACE-MP-0 foundation model (universal)...")
                try:
                    from mace.calculators import mace_mp
                    # "large" is most accurate (MACE-MP-0)
                    self._calculator = mace_mp(
                        model=self.model_size, 
                        device=self.device,
                        default_dtype=self.dtype,
                    )
                except Exception as e:
                    logger.warning(f"Could not load MACE-MP-0: {e}")
                    logger.warning("Falling back to heuristic energy estimates")
                    self._calculator = None
            else:
                logger.info("No MACE model available, using heuristics.")
                self._calculator = None

        return self._calculator

    def relax_structure(
        self,
        structure,
        fmax: float = 0.05,
        steps: int = 200,
        cell_filter: str = "frechet",  # 'frechet', 'exp', or None
        trajectory_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Relax a crystal structure using MACE potential.

        Args:
            structure: pymatgen Structure
            fmax: force convergence criterion (eV/Å)
            steps: max optimization steps
            cell_filter: filter type for variable cell relaxation
            trajectory_file: path to save relaxation trajectory (.traj)

        Returns:
            dict with: relaxed_structure, energy_per_atom, converged, n_steps
        """
        from atlas.utils.structure import pymatgen_to_ase, ase_to_pymatgen
        from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, LBFGSLineSearch
        from ase.constraints import ExpCellFilter, UnitCellFilter, FixSymmetry
        from ase.spacegroup import get_spacegroup
        
        # Helper for Frechet filter (more stable)
        try:
            from ase.filters import FrechetCellFilter
            HAS_FRECHET = True
        except ImportError:
            HAS_FRECHET = False

        atoms = pymatgen_to_ase(structure)

        if self.calculator is None:
            return self._heuristic_result(structure, "No calculator loaded")

        atoms.calc = self.calculator
        
        # Setup relaxation
        try:
            # 1. Symmetry Constraint (inspired by MLIP-Arena)
            if structure.get_space_group_info()[1] > 1:
                try:
                    atoms.set_constraint(FixSymmetry(atoms))
                    logger.debug("Applied FixSymmetry constraint.")
                except Exception as sym_e:
                    logger.debug(f"Could not apply FixSymmetry: {sym_e}")

            # 2. Choose filter
            if cell_filter == "frechet" and HAS_FRECHET:
                ecf = FrechetCellFilter(atoms)
            elif cell_filter == "exp" or (cell_filter == "frechet" and not HAS_FRECHET):
                ecf = ExpCellFilter(atoms)
            elif cell_filter == "unit":
                ecf = UnitCellFilter(atoms)
            else:
                 # Fixed cell
                ecf = atoms

            # 3. Choose optimizer (MLIP-Arena robust choice mapping)
            optimizers = {
                "bfgs": BFGS,
                "fire": FIRE,
                "lbfgs": LBFGS,
                "bfgs_ls": BFGSLineSearch,
                "lbfgs_ls": LBFGSLineSearch
            }
            # Defaulting to BFGSLineSearch as recommended by MLIP-Arena for general robust optimization
            opt_class = optimizers.get("bfgs_ls", BFGSLineSearch)
            
            opt = opt_class(ecf, trajectory=trajectory_file, logfile=None)
            
            # Run
            converged = opt.run(fmax=fmax, steps=steps)
            
            # Extract results
            energy = atoms.get_potential_energy()
            n_atoms = len(atoms)
            forces = atoms.get_forces()
            
            relaxed_struct = ase_to_pymatgen(atoms)
            
            return {
                "relaxed_structure": relaxed_struct,
                "energy_per_atom": energy / n_atoms,
                "energy_total": energy,
                "converged": converged,
                "n_steps": opt.nsteps,
                "forces_max": np.max(np.abs(forces)) if len(forces) > 0 else 0.0,
                "volume_change": relaxed_struct.volume / structure.volume,
            }

        except Exception as e:
            # If relaxation crashes (e.g. SCF non-convergence or segfault), return input
            logger.warning(f"Relaxation failed for {structure.composition.reduced_formula}: {e}")
            return self._heuristic_result(structure, str(e))

    def _heuristic_result(self, structure, error_msg: str) -> Dict[str, Any]:
        """Return a standardized failure result."""
        return {
            "relaxed_structure": structure,
            "energy_per_atom": self._heuristic_energy(structure),
            "energy_total": None,
            "converged": False,
            "n_steps": 0,
            "forces_max": None,
            "volume_change": 1.0,
            "error": error_msg,
        }

    def score_stability(self, structure) -> float:
        """
        Quick stability score (0–1).
        1.0 = highly stable (< -0.5 eV/atom below hull proxy)
        """
        # Quick relax with loose criteria
        res = self.relax_structure(structure, fmax=0.1, steps=50)
        e_pa = res["energy_per_atom"]

        if e_pa is None: return 0.5 

        # Approx convex hull energy for stable materials is typically -5 to -9 eV/atom
        # but pure MACE energy depends on reference states.
        # MACE-MP-0 is trained on MP formation energies? No, usually total energies.
        # So we need reference energies to get formation energy.
        # WITHOUT reference energies, raw e_pa is hard to interpret absolutely.
        #
        # Heuristic: 
        # Deeply negative usually means stable bonding.
        # We rely on relative ranking for Active Learning.
        
        # Map raw energy to 0-1 score (sigmoid-like)
        # Assume "good" energy is < -3.0 eV/atom (very rough)
        
        if e_pa > 0: return 0.0 # Unbound
        
        # Simple linear map for now, relative to a "deep" minimum
        # This needs calibration with real data
        score = min(1.0, max(0.0, -e_pa / 8.0)) 
        return score

    def batch_relax(
        self,
        structures: list,
        fmax: float = 0.05,
        steps: int = 100,
        n_jobs: int = 1
    ) -> list[dict]:
        """
        Relax a batch of structures.
        Parallelism is tricky with GPU calculators. Serial is safest.
        """
        results = []
        # TQDM optimization
        try:
            from tqdm import tqdm
            iterator = tqdm(structures, desc="Relaxing", leave=False)
        except ImportError:
            iterator = structures

        for i, struct in enumerate(iterator):
            # Pass trajectory file if debugging needed? 
            # trajectory_file=f"traj_{i}.traj"
            res = self.relax_structure(struct, fmax=fmax, steps=steps)
            res["index"] = i
            results.append(res)
            
        return results

    def _heuristic_energy(self, structure) -> float:
        """Rough energy estimate based on pauling electronegativity."""
        try:
            from pymatgen.core import Element
            energies = []
            for site in structure:
                elem = Element(str(site.specie))
                eneg = elem.X or 2.0
                e_approx = -1.0 * eneg 
                energies.append(e_approx)
            return float(np.mean(energies))
        except Exception:
            return -2.0
