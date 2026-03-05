"""Phase stability analysis based on pymatgen phase diagram utilities."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry, PDPlotter, PhaseDiagram
from pymatgen.core import Composition

logger = logging.getLogger(__name__)
_STABLE_EHULL_EPS = 1e-6
_ELEMENT_SYMBOL_PATTERN = re.compile(r"^[A-Za-z]{1,3}$")


def _is_boolean_like(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    if _is_boolean_like(value):
        raise ValueError(f"{field_name} must be finite numeric, not boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return number


def _normalize_element_symbol(value: Any) -> str:
    if _is_boolean_like(value) or not isinstance(value, str):
        raise TypeError("element symbol must be a string")
    symbol = value.strip()
    if not symbol:
        raise ValueError("element symbol must be a non-empty string")
    if not _ELEMENT_SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"invalid element symbol: {value!r}")
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


@dataclass(frozen=True)
class StabilityResult:
    formula: str
    e_above_hull: float
    is_stable: bool
    decomposition: str
    formation_energy: float | None
    error: str | None = None


class ReferenceDatabase:
    """
    In-memory reference-energy database for phase diagram construction.
    """

    def __init__(self):
        self.entries: list[PDEntry] = []

    def add_entry(self, composition: str, energy_per_atom: float) -> None:
        """Add a reference entry in eV/atom."""
        if _is_boolean_like(composition) or not isinstance(composition, str):
            raise TypeError("composition must be a non-empty formula string")
        formula = composition.strip()
        if not formula:
            raise ValueError("composition must be a non-empty formula string")
        comp = Composition(formula)
        energy = _coerce_finite_float(energy_per_atom, field_name="energy_per_atom")
        self.entries.append(PDEntry(comp, energy * comp.num_atoms))

    def load_from_list(self, data: list[dict[str, Any]]) -> None:
        """Load entries from list like `{'formula': 'Fe2O3', 'energy_per_atom': -6.5}`."""
        for item in data:
            if not isinstance(item, Mapping):
                raise TypeError("Each entry must be a mapping")
            if "formula" not in item or "energy_per_atom" not in item:
                raise KeyError("Each entry must contain 'formula' and 'energy_per_atom'")
            self.add_entry(str(item["formula"]), float(item["energy_per_atom"]))

    def get_entries(self, chemical_system: list[str]) -> list[PDEntry]:
        """Get all entries that are subsets of the specified chemical system."""
        system_set = {_normalize_element_symbol(el) for el in chemical_system}
        if not system_set:
            raise ValueError("chemical_system must contain at least one element symbol")

        relevant: list[PDEntry] = []
        for entry in self.entries:
            entry_elems = {_normalize_element_symbol(el) for el in entry.composition.elements}
            if entry_elems.issubset(system_set):
                relevant.append(entry)
        return relevant


class PhaseStabilityAnalyst:
    """Analyze thermodynamic stability using convex hull metrics."""

    def __init__(self, reference_db: ReferenceDatabase | None = None):
        self.db = reference_db or ReferenceDatabase()

    def analyze_stability(self, target_formula: str, target_energy_per_atom: float) -> dict[str, Any]:
        """
        Calculate energy above hull for a target material.

        Returns a dict for backward compatibility with existing callers.
        """
        if _is_boolean_like(target_formula) or not isinstance(target_formula, str):
            raise TypeError("target_formula must be a formula string")
        formula = target_formula.strip()
        if not formula:
            raise ValueError("target_formula must be a non-empty formula string")

        comp = Composition(formula)
        energy_pa = _coerce_finite_float(
            target_energy_per_atom,
            field_name="target_energy_per_atom",
        )

        target_entry = PDEntry(comp, energy_pa * comp.num_atoms)
        elems = [str(e) for e in comp.elements]
        competitor_entries = self.db.get_entries(elems)

        if not competitor_entries:
            logger.warning("No competitor entries found for system %s", elems)

        all_entries = [*competitor_entries, target_entry]

        try:
            pd = PhaseDiagram(all_entries)
            e_above_hull = float(pd.get_e_above_hull(target_entry))
            decomposition_map = pd.get_decomposition(target_entry.composition)
            if decomposition_map:
                clean_decomposition: list[tuple[PDEntry, float]] = []
                for entry, amt_raw in decomposition_map.items():
                    amt = float(amt_raw)
                    if not math.isfinite(amt) or amt <= 0:
                        continue
                    clean_decomposition.append((entry, amt))
                decomposition_str = " + ".join(
                    f"{amt:.3f} {entry.composition.reduced_formula}"
                    for entry, amt in sorted(
                        clean_decomposition,
                        key=lambda item: item[0].composition.reduced_formula,
                    )
                )
                if not decomposition_str:
                    decomposition_str = comp.reduced_formula
            else:
                decomposition_str = comp.reduced_formula
            formation_energy = float(pd.get_form_energy_per_atom(target_entry))
            if not math.isfinite(e_above_hull):
                raise ValueError("non-finite e_above_hull from phase diagram result")
            if not math.isfinite(formation_energy):
                raise ValueError("non-finite formation_energy from phase diagram result")

            result = StabilityResult(
                formula=formula,
                e_above_hull=e_above_hull,
                is_stable=e_above_hull <= _STABLE_EHULL_EPS,
                decomposition=decomposition_str,
                formation_energy=formation_energy,
            )

            return {
                "formula": result.formula,
                "e_above_hull": result.e_above_hull,
                "is_stable": result.is_stable,
                "decomposition": result.decomposition,
                "formation_energy": result.formation_energy,
            }
        except Exception as exc:
            logger.exception("Stability analysis failed for %s", formula)
            return {
                "formula": formula,
                "e_above_hull": float("nan"),
                "is_stable": False,
                "decomposition": "",
                "formation_energy": None,
                "error": str(exc),
            }

    def plot_phase_diagram(self, chemical_system: list[str], save_path: str | None = None):
        """Plot phase diagram for a chemical system."""
        entries = self.db.get_entries(chemical_system)
        if not entries:
            logger.warning("No entries for phase diagram in system %s", chemical_system)
            return None

        pd = PhaseDiagram(entries)
        plotter = PDPlotter(pd, show_unstable=0.2)
        fig = plotter.get_plot()
        if save_path:
            fig.savefig(save_path)
            return None
        return fig
