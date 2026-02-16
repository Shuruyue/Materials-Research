"""
atlas.thermo.calphad — CALPHAD phase diagram calculations.

Uses pycalphad to compute equilibrium phase diagrams, solidification
paths, and thermodynamic properties for multi-component alloy systems.

Example:
    >>> from atlas.thermo.calphad import CalphadCalculator
    >>> calc = CalphadCalculator.sn_ag_cu()
    >>> result = calc.equilibrium_at("SAC305", T=490)
    >>> calc.plot_binary("SN", "AG")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


TDB_DIR = Path(__file__).parent


@dataclass
class EquilibriumResult:
    """Result of an equilibrium calculation at a single point."""
    temperature_K: float
    composition: dict[str, float]
    stable_phases: list[str]
    phase_fractions: dict[str, float]
    phase_compositions: dict[str, dict[str, float]]


@dataclass
class SolidificationResult:
    """Result of a solidification path calculation."""
    temperatures: np.ndarray
    liquid_fraction: np.ndarray
    solid_phases: dict[str, np.ndarray]  # phase_name -> fraction vs T
    solidus_K: float
    liquidus_K: float
    alloy_name: str = ""


class CalphadCalculator:
    """CALPHAD phase diagram calculator using pycalphad.

    Wraps pycalphad's equilibrium solver for convenient use with
    solder and electronic packaging alloy systems.
    """

    # Common alloy compositions (mole fractions)
    ALLOY_COMPOSITIONS = {
        "SAC305": {"SN": 0.955, "AG": 0.034, "CU": 0.011},
        "SAC405": {"SN": 0.942, "AG": 0.046, "CU": 0.012},
        "SAC105": {"SN": 0.977, "AG": 0.012, "CU": 0.011},
        "SN100C": {"SN": 0.988, "CU": 0.012},  # Sn-0.7Cu
        "SNAG36": {"SN": 0.960, "AG": 0.040},  # Sn-3.5Ag binary eutectic
    }

    def __init__(self, tdb_path: str | Path, components: list[str]):
        """Initialize with a TDB file.

        Args:
            tdb_path: Path to TDB thermodynamic database file.
            components: List of element symbols (e.g., ["SN", "AG", "CU"]).
        """
        try:
            from pycalphad import Database
        except ImportError:
            raise ImportError("pycalphad required: pip install pycalphad")

        self.tdb_path = Path(tdb_path)
        self.components = [c.upper() for c in components]
        self.db = Database(str(self.tdb_path))

        # Get available phases from database
        self.all_phases = list(self.db.phases.keys())

    @classmethod
    def sn_ag_cu(cls) -> "CalphadCalculator":
        """Create calculator for the Sn-Ag-Cu system."""
        tdb = TDB_DIR / "Sn-Ag-Cu.tdb"
        if not tdb.exists():
            raise FileNotFoundError(f"TDB not found: {tdb}")
        return cls(tdb, ["SN", "AG", "CU"])

    def get_composition(self, alloy_name: str) -> dict[str, float]:
        """Get mole-fraction composition for a named alloy."""
        key = alloy_name.upper().replace("-", "").replace(" ", "").replace("_", "")
        if key in self.ALLOY_COMPOSITIONS:
            return self.ALLOY_COMPOSITIONS[key]
        raise ValueError(
            f"Unknown alloy '{alloy_name}'. "
            f"Available: {', '.join(self.ALLOY_COMPOSITIONS.keys())}"
        )

    def equilibrium_at(self, alloy: str | dict, T: float) -> EquilibriumResult:
        """Compute equilibrium at a single temperature.

        Args:
            alloy: Alloy name (e.g., "SAC305") or composition dict.
            T: Temperature in Kelvin.

        Returns:
            EquilibriumResult with stable phases and their fractions.
        """
        from pycalphad import equilibrium, variables as v

        if isinstance(alloy, str):
            comp = self.get_composition(alloy)
        else:
            comp = alloy

        # Build conditions
        conditions = {v.T: T, v.P: 101325, v.N: 1}
        for elem in self.components[:-1]:  # skip last (dependent)
            if elem in comp:
                conditions[v.X(elem)] = comp[elem]

        # Compute equilibrium
        phases = self.all_phases
        result = equilibrium(self.db, self.components + ["VA"], phases, conditions)

        # Extract results
        stable_phases = []
        phase_fractions = {}
        phase_compositions = {}

        try:
            # pycalphad result structure
            phase_names = result.Phase.values.squeeze()
            np_fracs = result.NP.values.squeeze()

            if phase_names.ndim == 0:
                phase_names = [str(phase_names)]
                np_fracs = [float(np_fracs)]
            else:
                phase_names = [str(p) for p in phase_names]
                np_fracs = [float(f) for f in np_fracs]

            for pname, frac in zip(phase_names, np_fracs):
                if pname.strip() and frac > 1e-6 and pname != '':
                    pname_clean = pname.strip()
                    stable_phases.append(pname_clean)
                    phase_fractions[pname_clean] = frac

        except Exception:
            pass

        return EquilibriumResult(
            temperature_K=T,
            composition=comp,
            stable_phases=stable_phases,
            phase_fractions=phase_fractions,
            phase_compositions=phase_compositions,
        )

    def solidification_path(
        self,
        alloy: str | dict,
        T_start: float = 600,
        T_end: float = 370,
        n_steps: int = 50,
    ) -> SolidificationResult:
        """Compute equilibrium solidification path (lever rule).

        Args:
            alloy: Alloy name or composition dict.
            T_start: Start temperature (K), above liquidus.
            T_end: End temperature (K), below solidus.
            n_steps: Number of temperature steps.

        Returns:
            SolidificationResult with phase fractions vs temperature.
        """
        if isinstance(alloy, str):
            comp = self.get_composition(alloy)
            alloy_name = alloy
        else:
            comp = alloy
            alloy_name = "Custom"

        temperatures = np.linspace(T_start, T_end, n_steps)
        liquid_frac = np.zeros(n_steps)
        all_phase_data = {}

        for i, T in enumerate(temperatures):
            try:
                eq = self.equilibrium_at(comp, T)
                for phase, frac in eq.phase_fractions.items():
                    if phase not in all_phase_data:
                        all_phase_data[phase] = np.zeros(n_steps)
                    all_phase_data[phase][i] = frac

                liquid_frac[i] = eq.phase_fractions.get("LIQUID", 0.0)
            except Exception:
                liquid_frac[i] = float("nan")

        # Find solidus and liquidus
        liquidus_K = T_start
        solidus_K = T_end
        for i, T in enumerate(temperatures):
            if liquid_frac[i] > 0.99:
                liquidus_K = T
            if liquid_frac[i] < 0.01 and T < liquidus_K:
                solidus_K = T
                break

        # More precise: interpolate
        for i in range(len(temperatures) - 1):
            if liquid_frac[i] > 0.99 and liquid_frac[i + 1] < 0.99:
                liquidus_K = temperatures[i]
            if liquid_frac[i] > 0.01 and liquid_frac[i + 1] < 0.01:
                solidus_K = temperatures[i + 1]

        return SolidificationResult(
            temperatures=temperatures,
            liquid_fraction=liquid_frac,
            solid_phases=all_phase_data,
            solidus_K=solidus_K,
            liquidus_K=liquidus_K,
            alloy_name=alloy_name,
        )

    def plot_solidification(
        self,
        result: SolidificationResult,
        save_path: str | Path | None = None,
    ):
        """Plot solidification curve (phase fraction vs temperature).

        Args:
            result: SolidificationResult from solidification_path().
            save_path: Optional path to save figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        T_C = result.temperatures - 273.15  # Convert to Celsius

        # Plot each phase
        colors = {
            "LIQUID": "#E63946",
            "BCT_A5": "#457B9D",
            "FCC_A1": "#2A9D8F",
            "AG3SN": "#E9C46A",
            "CU6SN5": "#F4A261",
            "CU3SN": "#264653",
            "HCP_A3": "#6A4C93",
        }

        for phase, fracs in result.solid_phases.items():
            color = colors.get(phase, None)
            label = phase
            if phase == "BCT_A5":
                label = "β-Sn (BCT_A5)"
            elif phase == "FCC_A1":
                label = "Ag/Cu solid soln (FCC)"
            elif phase == "AG3SN":
                label = "Ag₃Sn"
            elif phase == "CU6SN5":
                label = "Cu₆Sn₅"
            elif phase == "CU3SN":
                label = "Cu₃Sn"

            ax.plot(T_C, fracs, label=label, color=color, linewidth=2)

        # Mark solidus / liquidus
        liq_C = result.liquidus_K - 273.15
        sol_C = result.solidus_K - 273.15
        ax.axvline(liq_C, color="#E63946", linestyle="--", alpha=0.5)
        ax.axvline(sol_C, color="#457B9D", linestyle="--", alpha=0.5)
        ax.text(liq_C + 1, 0.95, f"Liquidus\n{liq_C:.0f}°C", fontsize=9, color="#E63946")
        ax.text(sol_C + 1, 0.85, f"Solidus\n{sol_C:.0f}°C", fontsize=9, color="#457B9D")

        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Phase Fraction", fontsize=12)
        ax.set_title(f"Equilibrium Solidification: {result.alloy_name}", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="center right", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")

        plt.show()

    def plot_binary(
        self,
        elem_x: str,
        elem_y: str,
        T_range: tuple[float, float] = (373, 773),
        n_x: int = 50,
        n_T: int = 50,
        save_path: str | Path | None = None,
    ):
        """Plot a binary phase diagram section.

        Args:
            elem_x: Element for x-axis composition.
            elem_y: Other binary element.
            T_range: Temperature range in Kelvin.
            n_x: Number of composition points.
            n_T: Number of temperature points.
            save_path: Optional path to save figure.
        """
        from pycalphad import binplot, variables as v

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        comps = [elem_x.upper(), elem_y.upper(), "VA"]
        phases = self.all_phases

        try:
            binplot(
                self.db, comps, phases,
                {v.X(elem_y.upper()): (0, 1, 0.02),
                 v.T: T_range,
                 v.P: 101325, v.N: 1},
                ax=ax,
            )
            ax.set_xlabel(f"Mole fraction {elem_y}", fontsize=12)
            ax.set_ylabel("Temperature (K)", fontsize=12)
            ax.set_title(f"{elem_x}-{elem_y} Binary Phase Diagram", fontsize=14, fontweight="bold")

            plt.tight_layout()
            if save_path:
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
                print(f"  Saved: {save_path}")
            plt.show()

        except Exception as e:
            print(f"  Binary plot failed: {e}")
            plt.close(fig)

    def print_equilibrium_table(
        self,
        alloy: str | dict,
        T_range: tuple[float, float] = (470, 530),
        step: float = 5,
    ):
        """Print a table of equilibrium phases vs temperature.

        Args:
            alloy: Alloy name or composition dict.
            T_range: Temperature range in Kelvin.
            step: Temperature step in Kelvin.
        """
        if isinstance(alloy, str):
            comp = self.get_composition(alloy)
            name = alloy
        else:
            comp = alloy
            name = "Custom"

        print(f"\n{'═' * 72}")
        print(f"  CALPHAD Equilibrium: {name}")
        print(f"  Composition: {comp}")
        print(f"{'═' * 72}")
        print(f"\n  {'T(K)':>6s}  {'T(°C)':>6s}  {'Phases':40s} {'f_liq':>6s}")
        print(f"  {'─' * 6}  {'─' * 6}  {'─' * 40} {'─' * 6}")

        temperatures = np.arange(T_range[0], T_range[1] + step, step)
        for T in temperatures:
            try:
                eq = self.equilibrium_at(comp, float(T))
                phase_str = " + ".join(
                    f"{p}({f:.2f})" for p, f in eq.phase_fractions.items()
                )
                f_liq = eq.phase_fractions.get("LIQUID", 0.0)
                T_C = T - 273.15
                print(f"  {T:6.0f}  {T_C:6.1f}  {phase_str:40s} {f_liq:6.3f}")
            except Exception as e:
                print(f"  {T:6.0f}  {T - 273.15:6.1f}  Error: {e}")

        print(f"\n{'═' * 72}")
