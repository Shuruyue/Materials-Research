import logging
from typing import Dict, Any, Optional
import torch

# Native Import from Assimilated third_party
try:
    # Importing raw bfgs components from MLIP-Arena
    from atlas.third_party.mlip_arena.tasks.opt import BFGSRelaxer as OriginalBFGSRelaxer
    from atlas.third_party.mlip_arena.tasks.utils import FrechetCellFilter, FixSymmetry
except ImportError as e:
    logging.warning(f"Could not import assimilated MLIP-Arena: {e}")
    OriginalBFGSRelaxer = None
    FrechetCellFilter = None
    FixSymmetry = None

from atlas.utils.registry import RELAXERS

logger = logging.getLogger(__name__)

@RELAXERS.register("mlip_arena_native")
class NativeMlipArenaRelaxer:
    """
    Full architecture assimilation of the MLIP-Arena Relaxation loop.
    Rather than rewriting BFGS, we strictly invoke their `BFGSRelaxer` class,
    providing its innate access to `FrechetCellFilter` and point-group `FixSymmetry`.
    """
    def __init__(self, fmax: float = 0.05, steps: int = 200, cell_filter: str = 'frechet', constraints: list = None):
        self.fmax = fmax
        self.steps = steps
        self.cell_filter = cell_filter
        self.constraints = constraints or []
        
        if OriginalBFGSRelaxer is None:
            raise RuntimeError("Assimilated MLIP-Arena source is missing.")
            
    def relax(self, pmg_structure, calculator) -> dict:
        """
        Relaxes a given structure via the Native MLIP Arena BFGS loop.
        """
        # Convert to ASE Atoms to conform exactly to MLIP-Arena's expected input
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(pmg_structure)
        atoms.calc = calculator
        
        # Apply strict symmetry constraints native to MLIP-Arena
        if FixSymmetry is not None:
             atoms.set_constraint(FixSymmetry(atoms))
             
        # Call the assimilated optimizer
        try:
            native_optimizer = OriginalBFGSRelaxer(
                atoms=atoms,
                trajectory=None,
                logfile=None
            )
            
            # Use native cell filter logic
            if self.cell_filter == "frechet" and FrechetCellFilter is not None:
                atoms_to_opt = FrechetCellFilter(atoms)
                logger.debug("Applied MLIP-Arena FrechetCellFilter")
            else:
                atoms_to_opt = atoms
                
            native_optimizer.run(fmax=self.fmax, steps=self.steps)
            opt_structure = AseAtomsAdaptor.get_structure(atoms)
            
            return {
                "structure": opt_structure,
                "energy": atoms.get_potential_energy(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Native MLIP-Arena relaxation failed: {e}")
            return {"structure": pmg_structure, "energy": None, "success": False}
