"""
Phase Stability Analysis

Tools for constructing Convex Hulls and calculating thermodynamic stability.
Uses pymatgen's PhaseDiagram analysis.
"""

import logging

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

# Configure logging
logger = logging.getLogger(__name__)


class ReferenceDatabase:
    """
    Manages reference energies for phase diagram construction.
    Can load from MP/JARVIS or local cache.
    """
    def __init__(self):
        self.entries: list[PDEntry] = []

    def add_entry(self, composition: str, energy_per_atom: float):
        """Add a manual entry."""
        comp = Composition(composition)
        # PDEntry takes total energy, so multiply by num_atoms
        entry = PDEntry(comp, energy_per_atom * comp.num_atoms)
        self.entries.append(entry)

    def load_from_list(self, data: list[dict]):
        """
        Load entries from list of dicts.
        Format: {"formula": "Fe2O3", "energy_per_atom": -6.5}
        """
        for item in data:
            self.add_entry(item["formula"], item["energy_per_atom"])

    def get_entries(self, chemical_system: list[str]) -> list[PDEntry]:
        """Get all entries within a specific chemical system."""
        system_set = set(chemical_system)
        relevant = []
        for e in self.entries:
            entry_elems = set(str(el) for el in e.composition.elements)
            # Check if entry is subset of system
            if entry_elems.issubset(system_set):
                relevant.append(e)
        return relevant


class PhaseStabilityAnalyst:
    """
    Analyzes thermodynamic stability using Convex Hull.
    """

    def __init__(self, reference_db: ReferenceDatabase | None = None):
        self.db = reference_db or ReferenceDatabase()

    def analyze_stability(self, target_formula: str, target_energy_per_atom: float) -> dict:
        """
        Calculate energy above hull for a target material.

        Args:
            target_formula: chemical formula (e.g. "Fe2O3")
            target_energy_per_atom: DFT/MLIP energy in eV/atom

        Returns:
            Dict with stability info (e_above_hull, is_stable, decomposition)
        """
        comp = Composition(target_formula)
        target_entry = PDEntry(comp, target_energy_per_atom * comp.num_atoms)

        # 1. Get relevant system entries
        elems = [str(e) for e in comp.elements]
        entries = self.db.get_entries(elems)

        # 2. Add target to entries to construct hull INCLUDING it (to see if it hits hull)
        # Actually, standard practice: construct hull from *competitors*, then check target.
        # But PhaseDiagram usually takes all entries.
        # If we want e_above_hull, we put it in the list.
        # Note: if target is the new ground state, e_above_hull will be 0.

        all_entries = entries + [target_entry]

        try:
            pd = PhaseDiagram(all_entries)
            e_above_hull = pd.get_e_above_hull(target_entry)
            decomp = pd.get_decomposition(target_entry.composition)

            # Format decomposition
            decomp_str = " + ".join([f"{amt:.3f} {e.composition.reduced_formula}" for e, amt in decomp.items()])

            return {
                "formula": target_formula,
                "e_above_hull": e_above_hull,
                "is_stable": e_above_hull <= 1e-6,
                "decomposition": decomp_str,
                "formation_energy": pd.get_form_energy_per_atom(target_entry)
            }
        except Exception as e:
            logger.error(f"Stability analysis failed for {target_formula}: {e}")
            return {
                "formula": target_formula,
                "e_above_hull": float("nan"),
                "is_stable": False,
                "error": str(e)
            }

    def plot_phase_diagram(self, chemical_system: list[str], save_path: str = None):
        """Plot phase diagram for 2-element or 3-element system."""
        from pymatgen.analysis.phase_diagram import PDPlotter

        entries = self.db.get_entries(chemical_system)
        if not entries:
            logger.warning("No entries for phase diagram.")
            return

        pd = PhaseDiagram(entries)
        plotter = PDPlotter(pd, show_unstable=0.2) # Show unstable up to 0.2 eV/atom above hull

        if save_path:
            plotter.get_plot().savefig(save_path)
        else:
            return plotter.get_plot()
