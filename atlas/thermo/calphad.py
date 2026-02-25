"""
atlas.thermo.calphad — CALPHAD phase diagram calculations.

Uses pycalphad to compute equilibrium phase diagrams, solidification
paths (Equilibrium & Scheil), and thermodynamic properties.

Key Features:
- Equilibrium Solidification (Lever Rule): Infinite diffusion in solid/liquid.
- Scheil-Gulliver Solidification: No diffusion in solid, infinite in liquid (realistic for fast cooling).
- Robust handling of pycalphad errors.
- Publication-quality plotting.

Example:
    >>> from atlas.thermo.calphad import CalphadCalculator
    >>> calc = CalphadCalculator.sn_ag_cu()
    >>> res_eq = calc.solidification_path("SAC305", scheil=False)
    >>> res_sch = calc.solidification_path("SAC305", scheil=True)
    >>> calc.plot_solidification(res_sch)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

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
    solid_phases: dict[str, np.ndarray]  # phase_name -> cumulative fraction vs T
    solidus_K: float
    liquidus_K: float
    alloy_name: str = ""
    method: str = "Equilibrium" # "Equilibrium" or "Scheil"


class CalphadCalculator:
    """CALPHAD phase diagram calculator using pycalphad."""

    # Common alloy compositions (mole fractions)
    ALLOY_COMPOSITIONS = {
        "SAC305": {"SN": 0.955, "AG": 0.034, "CU": 0.011},
        "SAC405": {"SN": 0.942, "AG": 0.046, "CU": 0.012},
        "SAC105": {"SN": 0.977, "AG": 0.012, "CU": 0.011},
        "SN100C": {"SN": 0.988, "CU": 0.012},  # Sn-0.7Cu
        "SNAG36": {"SN": 0.960, "AG": 0.040},  # Sn-3.5Ag binary eutectic
    }

    def __init__(self, tdb_path: str | Path, components: list[str]):
        """Initialize with a TDB file."""
        try:
            from pycalphad import Database
        except ImportError as e:
            raise ImportError("pycalphad required: pip install pycalphad") from e

        self.tdb_path = Path(tdb_path)
        self.components = [c.upper() for c in components]

        try:
            self.db = Database(str(self.tdb_path))
            self.all_phases = list(self.db.phases.keys())
            logger.info(f"Loaded TDB: {self.tdb_path.name} with phases {len(self.all_phases)}")
        except Exception as e:
            logger.error(f"Failed to load TDB {tdb_path}: {e}")
            raise

    @classmethod
    def sn_ag_cu(cls) -> CalphadCalculator:
        """Create calculator for the Sn-Ag-Cu system."""
        tdb = TDB_DIR / "Sn-Ag-Cu.tdb"
        if not tdb.exists():
            # Try to find it in atlas/data or similar if needed, or raise
            raise FileNotFoundError(f"TDB not found: {tdb}")
        return cls(tdb, ["SN", "AG", "CU"])

    def get_composition(self, alloy_name: str) -> dict[str, float]:
        """Get mole-fraction composition for a named alloy."""
        key = alloy_name.upper().replace("-", "").replace(" ", "").replace("_", "")
        if key in self.ALLOY_COMPOSITIONS:
            return self.ALLOY_COMPOSITIONS[key]
        raise ValueError(f"Unknown alloy '{alloy_name}'. Available: {list(self.ALLOY_COMPOSITIONS.keys())}")

    def equilibrium_at(self, alloy: str | dict[str, float], T: float) -> EquilibriumResult:
        """Compute equilibrium at a single temperature."""
        from pycalphad import equilibrium
        from pycalphad import variables as v

        if isinstance(alloy, str):
            comp = self.get_composition(alloy)
        else:
            comp = alloy

        # Build conditions: T, P, N=1, X_i
        conditions = {v.T: T, v.P: 101325, v.N: 1}
        for elem in self.components[:-1]:  # skip last (dependent)
            if elem in comp:
                conditions[v.X(elem)] = comp[elem]

        try:
            # We filter out phases that might cause issues if needed, strictly use all_phases for now
            result = equilibrium(self.db, self.components + ["VA"], self.all_phases, conditions)

            # Extract results safely
            Eq = result.NP.squeeze()
            Ph = result.Phase.squeeze()

            # Handle single point vs array (though here it's 1 point)
            if Eq.ndim == 0: Eq = [float(Eq)]
            else: Eq = Eq.values.tolist()
            if Ph.ndim == 0: Ph = [str(Ph)]
            else: Ph = Ph.values.tolist()

            stable_phases = []
            phase_fractions = {}
            phase_compositions = {} # TODO: Extract X for each phase if needed

            for pname, frac in zip(Ph, Eq):
                if pname and frac > 1e-4:
                    pname = str(pname)
                    stable_phases.append(pname)
                    phase_fractions[pname] = float(frac)

        except Exception as e:
            logger.warning(f"Equilibrium failed at T={T}: {e}")
            return EquilibriumResult(T, comp, [], {}, {})

        return EquilibriumResult(T, comp, stable_phases, phase_fractions, phase_compositions)

    def solidification_path(
        self,
        alloy: str | dict[str, float],
        T_start: float = 600,
        T_end: float = 370,
        n_steps: int = 50,
        scheil: bool = False
    ) -> SolidificationResult:
        """
        Compute solidification path using either Equilibrium (Lever Rule) or Scheil-Gulliver model.

        Args:
            scheil: If True, simulate Scheil solidification (no diffusion in solid).
        """

        if isinstance(alloy, str):
            comp = self.get_composition(alloy)
            alloy_name = alloy
        else:
            comp = alloy
            alloy_name = "Custom"

        if scheil:
             return self._scheil_simulation(comp, T_start, n_steps, alloy_name)
        else:
             return self._equilibrium_simulation(comp, T_start, T_end, n_steps, alloy_name)

    def _equilibrium_simulation(self, comp, T_start, T_end, n_steps, alloy_name):
        """Standard equilibrium stepping."""
        temperatures = np.linspace(T_start, T_end, n_steps)
        liquid_frac = np.zeros(n_steps)
        all_phase_data = {}

        for i, T in enumerate(temperatures):
            eq = self.equilibrium_at(comp, T)
            for phase, frac in eq.phase_fractions.items():
                if phase not in all_phase_data:
                    all_phase_data[phase] = np.zeros(n_steps)
                all_phase_data[phase][i] = frac

            liquid_frac[i] = eq.phase_fractions.get("LIQUID", 0.0)

        # Post-process liquidus/solidus
        liq_K, sol_K = self._find_transus(temperatures, liquid_frac)

        return SolidificationResult(
            temperatures=temperatures,
            liquid_fraction=liquid_frac,
            solid_phases=all_phase_data,
            solidus_K=int(sol_K) if not np.isnan(sol_K) else T_end,
            liquidus_K=int(liq_K) if not np.isnan(liq_K) else T_start,
            alloy_name=alloy_name,
            method="Equilibrium"
        )

    def _scheil_simulation(self, comp, T_start, n_steps, alloy_name):
        """
        Scheil-Gulliver simulation using pycalphad's specific Scheil utility is complex,
        so implementing a simplified manual stepper.
        Assumption: Liquid is well-mixed, Solid does not diffuse.
        """
        # Try importing official Scheil if available (newer pycalphad versions)
        try:
            from pycalphad.tools.scheil import scheil_solidification
            # PyCalphad's scheil tool is excellent
            # It returns a result object we need to parse

            # Convert comp to mole fractions {X(EL): val}
            # Note: scheil_solidification takes a flexible comp dict
            res = scheil_solidification(
                self.db, self.components, self.all_phases, comp,
                start_temperature=T_start, step_temperature=1.0, verbose=False
            )

            # Scheil result extraction
            temperatures = np.array(res.temperatures)
            liquid_frac = np.array(res.fraction_liquid)

            # Phase fractions in Scheil are tricky; usually we track CUMULATIVE solid
            # or instantaneous solid formers.
            # Pycalphad's result structure tracks cumulative phase amounts in 'phase_amounts'

            all_phase_data = {}
            # Map phase indices to names
            # This part depends on pycalphad version internals, assuming standard xarray

            # Simplified fallback: just track T and f_liq for now as it's the most robust metric
            # for Solidus. Scheil solidus is where f_liq -> 0 or epsilon.

            # To get specific solid phases, we need to iterate the result
            # Or just return T vs f_liq which is the main curve

            # Let's try to extract phase fractions if possible
            if hasattr(res, 'cum_phase_amounts'):
                for phase in self.all_phases:
                    if phase == "LIQUID": continue
                    if phase in res.cum_phase_amounts:
                        # Extract data
                        all_phase_data[phase] = np.array(res.cum_phase_amounts[phase])

            # Find approximate solidus (f_L < 0.01)
            sol_idx = np.where(liquid_frac < 0.01)[0]
            solidus_K = temperatures[sol_idx[0]] if len(sol_idx) > 0 else temperatures[-1]
            liquidus_K = temperatures[0] # Roughly start

            return SolidificationResult(
                temperatures=temperatures,
                liquid_fraction=liquid_frac,
                solid_phases=all_phase_data,
                solidus_K=solidus_K,
                liquidus_K=liquidus_K,
                alloy_name=alloy_name,
                method="Scheil"
            )

        except ImportError:
            logger.warning("pycalphad.tools.scheil not found. Falling back to simple Equilibrium.")
            return self._equilibrium_simulation(comp, T_start, T_start-200, n_steps, alloy_name)
        except Exception as e:
            logger.error(f"Scheil simulation failed: {e}")
            return self._equilibrium_simulation(comp, T_start, T_start-200, n_steps, alloy_name)

    def _find_transus(self, T, f_liq):
        """Helper to find L and S temperatures."""
        liq_K = float("nan")
        sol_K = float("nan")
        for i in range(len(T) - 1):
            # Liquidus: f_L crosses 0.99 downwards (as T decreases)
            if f_liq[i] >= 0.99 and f_liq[i+1] < 0.99:
                liq_K = T[i]
            # Solidus: f_L crosses 0.01 downwards
            if f_liq[i] >= 0.01 and f_liq[i+1] < 0.01:
                sol_K = T[i+1]
        return liq_K, sol_K

    def plot_solidification(
        self,
        result: SolidificationResult,
        save_path: str | Path | None = None,
    ):
        """Plot solidification curve (Scheil or Equilibrium)."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

        T_C = result.temperatures - 273.15

        # Plot Liquid Fraction
        ax.plot(T_C, result.liquid_fraction, label="Liquid Fraction", color="black", linewidth=2.5)

        # Plot Solid phases (cumulative or instantaneous)
        # For Scheil, these are cumulative fractions
        plt.cm.tab10(np.linspace(0, 1, len(result.solid_phases)))
        for _i, (phase, fracs) in enumerate(result.solid_phases.items()):
            # Only plot if significant
            if np.max(fracs) > 0.02:
                # Clean phase names
                label = phase.replace("_", " ").title()
                # Try to fill area for cumulative view?
                # For now just lines
                ax.plot(T_C, fracs, label=label, linewidth=1.5, alpha=0.8)

        # Labels
        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Phase Fraction (Mole)", fontsize=12)
        ax.set_title(f"{result.method} Solidification: {result.alloy_name}", fontsize=14, fontweight="bold")

        # Ranges
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(min(T_C), max(T_C))

        # Grid
        ax.grid(True, linestyle=":", alpha=0.6)

        # Legend
        ax.legend(fontsize=10)

        # Annotations for Liquidus/Solidus
        liq_C = result.liquidus_K - 273.15
        sol_C = result.solidus_K - 273.15

        ax.annotate(f"T_liq: {liq_C:.0f}°C", xy=(liq_C, 1.0), xytext=(liq_C+10, 1.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

        if result.solidus_K > 0 and result.solidus_K < 2000:
             ax.annotate(f"T_sol: {sol_C:.0f}°C", xy=(sol_C, 0.0), xytext=(sol_C-30, 0.05),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5))

        plt.tight_layout()

        if save_path:
            fig.savefig(str(save_path), bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        # Don't block execution with show() in automated env
        # plt.show()
        plt.close(fig)

    def print_equilibrium_table(
        self,
        alloy: str | dict[str, float],
        T_range: tuple[float, float] = (400, 550),
        step: float = 5.0,
    ):
        """
        Print a compact temperature-phase equilibrium table to stdout.
        """
        t_lo, t_hi = float(T_range[0]), float(T_range[1])
        if t_hi < t_lo:
            t_lo, t_hi = t_hi, t_lo
        temperatures = np.arange(t_lo, t_hi + 0.5 * step, step)

        print("\n  T(K)   T(C)  Stable phases (fraction)")
        print("  -----  ----- ---------------------------------------------")
        for t in temperatures:
            eq = self.equilibrium_at(alloy, float(t))
            if not eq.phase_fractions:
                phase_text = "-"
            else:
                phase_text = ", ".join(
                    f"{p}:{f:.3f}" for p, f in sorted(eq.phase_fractions.items(), key=lambda x: -x[1])
                )
            print(f"  {t:5.0f}  {t - 273.15:5.1f} {phase_text}")

    def plot_binary(self, elem_x: str, elem_y: str, T_range=(373, 1073)):
        """Plot binary diagram (wrapper for pycalphad binplot)."""
        from pycalphad import binplot
        from pycalphad import variables as v
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        comps = [elem_x.upper(), elem_y.upper(), "VA"]
        conditions = {
            v.X(elem_y.upper()): (0, 1, 0.02),
            v.T: T_range,
            v.P: 101325, v.N: 1
        }
        binplot(self.db, comps, self.all_phases, conditions, plot_kwargs={'ax': ax})
        ax.set_title(f"{elem_x}-{elem_y} Phase Diagram")
        plt.show()
