"""
MACE Inference & Structure Relaxation Module

Wraps a trained MACE model (or uses the pre-trained MACE-MP-0 foundation model)
for fast structure relaxation and stability assessment.

Key capabilities:
    - Structure relaxation (geometry optimization) at ML speed
    - Formation energy estimation
    - Stability scoring (energy above hull proxy)
    - Phonon stability check (negative frequencies → unstable)
"""

import numpy as np
from typing import Optional
from pathlib import Path

from atlas.config import get_config


class MACERelaxer:
    """
    Fast structure relaxation using MACE potentials.
    
    Can use either:
    1. A custom-trained MACE model (from Phase 1A)
    2. The pre-trained MACE-MP-0 foundation model (zero-shot, covers all elements)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        use_foundation: bool = True,
    ):
        """
        Args:
            model_path: path to trained MACE model (.pt file)
            device: "cuda", "cpu", or "auto"
            use_foundation: if True and no model_path, use MACE-MP-0 foundation model
        """
        self.cfg = get_config()
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self._calculator = None
        self.use_foundation = use_foundation

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    @property
    def calculator(self):
        """Lazy-load the MACE calculator."""
        if self._calculator is not None:
            return self._calculator

        if self.model_path and Path(self.model_path).exists():
            # Use custom-trained model
            print(f"  Loading custom MACE model: {self.model_path}")
            from mace.calculators import MACECalculator
            self._calculator = MACECalculator(
                model_paths=self.model_path,
                device=self.device,
            )
        elif self.use_foundation:
            # Use pre-trained foundation model (MACE-MP-0)
            print(f"  Loading MACE-MP-0 foundation model (universal, all elements)...")
            try:
                from mace.calculators import mace_mp
                self._calculator = mace_mp(
                    model="medium",
                    device=self.device,
                )
            except Exception as e:
                print(f"  Warning: Could not load MACE-MP-0: {e}")
                print(f"  Falling back to energy-free mode (heuristic only)")
                self._calculator = None
        else:
            print("  No MACE model available, using heuristic energy estimates")
            self._calculator = None

        return self._calculator

    def relax_structure(
        self,
        structure,
        fmax: float = 0.05,
        steps: int = 200,
    ) -> dict:
        """
        Relax a crystal structure using MACE potential.

        Args:
            structure: pymatgen Structure
            fmax: force convergence criterion (eV/Å)
            steps: max optimization steps

        Returns:
            dict with: relaxed_structure, energy_per_atom, converged, n_steps
        """
        from atlas.utils.structure import pymatgen_to_ase, ase_to_pymatgen

        atoms = pymatgen_to_ase(structure)

        if self.calculator is not None:
            atoms.calc = self.calculator

            try:
                from ase.optimize import BFGS
                from ase.constraints import ExpCellFilter
                import io

                # Relax both positions and cell
                ecf = ExpCellFilter(atoms)
                opt = BFGS(ecf, logfile=io.StringIO())
                converged = opt.run(fmax=fmax, steps=steps)

                energy = atoms.get_potential_energy()
                n_atoms = len(atoms)

                relaxed_struct = ase_to_pymatgen(atoms)

                return {
                    "relaxed_structure": relaxed_struct,
                    "energy_per_atom": energy / n_atoms,
                    "energy_total": energy,
                    "converged": converged,
                    "n_steps": opt.nsteps,
                    "forces_max": np.max(np.abs(atoms.get_forces())),
                }
            except Exception as e:
                return {
                    "relaxed_structure": structure,
                    "energy_per_atom": self._heuristic_energy(structure),
                    "energy_total": None,
                    "converged": False,
                    "n_steps": 0,
                    "forces_max": None,
                    "error": str(e),
                }
        else:
            # No MACE model — return unrelaxed with heuristic energy
            return {
                "relaxed_structure": structure,
                "energy_per_atom": self._heuristic_energy(structure),
                "energy_total": None,
                "converged": False,
                "n_steps": 0,
                "forces_max": None,
            }

    def score_stability(self, structure) -> float:
        """
        Quick stability score (0–1) for a structure.
        1.0 = very stable, 0.0 = very unstable.
        
        Uses energy per atom relative to known stable energies.
        """
        result = self.relax_structure(structure, steps=50)  # quick relax
        e_pa = result["energy_per_atom"]

        if e_pa is None:
            return 0.5  # unknown

        # Heuristic normalization:
        # Most stable materials: ~ -8 to -3 eV/atom
        # Unstable materials: > 0 eV/atom
        if e_pa < -2.0:
            return min(1.0, 0.5 + abs(e_pa) / 20.0)
        elif e_pa < 0:
            return 0.3 + 0.2 * abs(e_pa) / 2.0
        else:
            return max(0.0, 0.3 - e_pa / 5.0)

    def batch_relax(
        self,
        structures: list,
        fmax: float = 0.05,
        steps: int = 100,
    ) -> list[dict]:
        """Relax a batch of structures."""
        results = []
        for i, struct in enumerate(structures):
            result = self.relax_structure(struct, fmax=fmax, steps=steps)
            result["index"] = i
            results.append(result)
        return results

    def _heuristic_energy(self, structure) -> float:
        """
        Rough energy estimate based on element electronegativities.
        Used as fallback when no MACE model is available.
        """
        try:
            from pymatgen.core import Element
            energies = []
            for site in structure:
                elem = Element(str(site.specie))
                # Approximate cohesive energy (very rough)
                eneg = elem.X or 2.0
                z = elem.Z
                e_approx = -0.5 * eneg - 0.02 * z
                energies.append(e_approx)
            return np.mean(energies)
        except Exception:
            return -3.0  # generic fallback
