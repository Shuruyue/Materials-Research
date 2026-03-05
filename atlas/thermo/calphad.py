"""
atlas.thermo.calphad — CALPHAD phase diagram calculations.

Uses pycalphad to compute equilibrium phase diagrams, solidification
paths (Equilibrium & Scheil), and thermodynamic properties.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

TDB_DIR = Path(__file__).resolve().parent
_EPS = 1e-12


def _normalize_alloy_key(name: str) -> str:
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError("alloy name must be a string")
    normalized = name.upper().replace("-", "").replace("_", "").replace(" ", "")
    if not normalized:
        raise ValueError("alloy name must be non-empty")
    return normalized


def _canonical_phase_name(value: Any) -> str:
    phase = " ".join(str(value).split()).strip().upper()
    if not phase or phase == "NAN":
        return ""
    return phase


def _coerce_int_with_min(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)):
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        number = int(number_f)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}") from exc
    if number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return number


def _normalize_phase_fractions(raw: Mapping[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for phase_name_raw, frac_raw in raw.items():
        phase_name = _canonical_phase_name(phase_name_raw)
        if not phase_name:
            continue
        if isinstance(frac_raw, bool) or type(frac_raw).__name__ in {"bool", "bool_"}:
            raise ValueError(
                f"phase fraction for {phase_name!r} must be finite numeric, not boolean"
            )
        frac = float(frac_raw)
        if not math.isfinite(frac) or frac <= 1e-6:
            continue
        cleaned[phase_name] = cleaned.get(phase_name, 0.0) + max(0.0, frac)

    if not cleaned:
        return {}

    total = float(sum(cleaned.values()))
    if not math.isfinite(total) or total <= _EPS:
        return {}

    normalized = {phase: float(np.clip(value / total, 0.0, 1.0)) for phase, value in cleaned.items()}
    return dict(sorted(normalized.items(), key=lambda kv: -kv[1]))


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
    method: str = "Equilibrium"  # "Equilibrium" or "Scheil"


class CalphadCalculator:
    """CALPHAD phase diagram calculator using pycalphad."""

    # Common alloy compositions (mole fractions)
    ALLOY_COMPOSITIONS = {
        "SAC305": {"SN": 0.955, "AG": 0.034, "CU": 0.011},
        "SAC405": {"SN": 0.942, "AG": 0.046, "CU": 0.012},
        "SAC105": {"SN": 0.977, "AG": 0.012, "CU": 0.011},
        "SN100C": {"SN": 0.988, "CU": 0.012},
        "SNAG36": {"SN": 0.960, "AG": 0.040},
    }

    def __init__(self, tdb_path: str | Path, components: list[str]):
        """Initialize with a TDB file and system components."""
        try:
            from pycalphad import Database
        except ImportError as exc:
            raise ImportError("pycalphad required: pip install pycalphad") from exc

        path = Path(tdb_path).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"TDB not found or not a file: {path}")

        if len(components) < 2:
            raise ValueError("components must contain at least two elements")

        deduped: list[str] = []
        for comp in components:
            key = str(comp).strip().upper()
            if not key:
                raise ValueError("components entries must be non-empty")
            if key not in deduped:
                deduped.append(key)
        if len(deduped) < 2:
            raise ValueError("components must contain at least two unique elements")

        self.tdb_path = path
        self.components = deduped
        self.dependent_component = deduped[-1]

        try:
            self.db = Database(str(self.tdb_path))
            self.all_phases = sorted(str(p) for p in self.db.phases)
            logger.info(
                "Loaded TDB %s with %d phases",
                self.tdb_path.name,
                len(self.all_phases),
            )
        except Exception as exc:
            logger.error("Failed to load TDB %s: %s", tdb_path, exc)
            raise

    @classmethod
    def sn_ag_cu(cls) -> CalphadCalculator:
        """Create calculator for the Sn-Ag-Cu system."""
        tdb = TDB_DIR / "Sn-Ag-Cu.tdb"
        if not tdb.exists():
            raise FileNotFoundError(f"TDB not found: {tdb}")
        return cls(tdb, ["SN", "AG", "CU"])

    @classmethod
    def available_alloys(cls) -> list[str]:
        return sorted(cls.ALLOY_COMPOSITIONS)

    def _normalize_composition(self, composition: Mapping[str, float]) -> dict[str, float]:
        if not composition:
            raise ValueError("composition must not be empty")

        comp: dict[str, float] = {}
        for raw_key, raw_value in composition.items():
            key = str(raw_key).strip().upper()
            if key not in self.components:
                raise ValueError(
                    f"Unknown component {raw_key!r}. Expected subset of {self.components}"
                )
            if isinstance(raw_value, bool) or type(raw_value).__name__ in {"bool", "bool_"}:
                raise ValueError(f"Invalid mole fraction for {key}: {raw_value!r}")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid mole fraction for {key}: {raw_value!r}")
            comp[key] = value

        if not comp:
            raise ValueError("composition resolved to an empty component map")

        non_dependent_total = sum(comp.get(k, 0.0) for k in self.components[:-1])
        if non_dependent_total > 1.0 + 1e-6:
            raise ValueError(
                "Sum of independent component fractions cannot exceed 1. "
                f"Got {non_dependent_total:.6f}."
            )

        if self.dependent_component not in comp:
            comp[self.dependent_component] = max(0.0, 1.0 - non_dependent_total)

        total = sum(comp.values())
        if total <= _EPS or not math.isfinite(total):
            raise ValueError(f"Invalid composition sum: {total!r}")

        normalized = {k: (v / total) for k, v in comp.items()}
        return normalized

    def get_composition(self, alloy_name: str) -> dict[str, float]:
        """Get mole-fraction composition for a named alloy."""
        key = _normalize_alloy_key(alloy_name)
        if key in self.ALLOY_COMPOSITIONS:
            return self._normalize_composition(dict(self.ALLOY_COMPOSITIONS[key]))
        raise ValueError(
            f"Unknown alloy {alloy_name!r}. Available: {sorted(self.ALLOY_COMPOSITIONS)}"
        )

    def equilibrium_at(self, alloy: str | Mapping[str, float], T: float) -> EquilibriumResult:
        """Compute equilibrium at a single temperature."""
        from pycalphad import equilibrium
        from pycalphad import variables as v

        temp = float(T)
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError(f"T must be finite and > 0 K, got {T!r}")

        comp = self.get_composition(alloy) if isinstance(alloy, str) else self._normalize_composition(alloy)

        conditions: dict[Any, Any] = {v.T: temp, v.P: 101325, v.N: 1}
        for elem in self.components[:-1]:
            conditions[v.X(elem)] = comp.get(elem, 0.0)

        try:
            result = equilibrium(self.db, self.components + ["VA"], self.all_phases, conditions)

            phase_raw = np.asarray(getattr(result.Phase, "values", result.Phase), dtype=object).reshape(-1)
            frac_raw = np.asarray(getattr(result.NP, "values", result.NP), dtype=float).reshape(-1)

            stable: dict[str, float] = {}
            for phase_name_raw, frac in zip(phase_raw, frac_raw, strict=False):
                if not np.isfinite(frac) or frac <= 1e-4:
                    continue
                phase_name = str(phase_name_raw).strip()
                if not phase_name or phase_name.lower() == "nan":
                    continue
                stable[phase_name] = stable.get(phase_name, 0.0) + float(frac)

            phase_fractions = _normalize_phase_fractions(stable)
            stable_phases = list(phase_fractions)

        except Exception as exc:
            logger.warning("Equilibrium failed at T=%.3f K: %s", temp, exc)
            return EquilibriumResult(temp, comp, [], {}, {})

        return EquilibriumResult(temp, comp, stable_phases, phase_fractions, {})

    def solidification_path(
        self,
        alloy: str | Mapping[str, float],
        T_start: float = 600,
        T_end: float = 370,
        n_steps: int = 50,
        scheil: bool = False,
    ) -> SolidificationResult:
        """
        Compute solidification path using either equilibrium or Scheil approximation.
        """
        comp = self.get_composition(alloy) if isinstance(alloy, str) else self._normalize_composition(alloy)
        alloy_name = alloy if isinstance(alloy, str) else "Custom"

        t_start = float(T_start)
        t_end = float(T_end)
        steps = _coerce_int_with_min(n_steps, "n_steps", 2)
        if not math.isfinite(t_start) or not math.isfinite(t_end):
            raise ValueError("T_start and T_end must be finite")
        if t_start <= 0 or t_end <= 0:
            raise ValueError("Temperatures must be > 0 K")
        if t_start < t_end:
            logger.warning("T_start < T_end detected; swapping to enforce cooling trajectory.")
            t_start, t_end = t_end, t_start

        if scheil:
            return self._scheil_simulation(comp, t_start, t_end, steps, str(alloy_name))
        return self._equilibrium_simulation(comp, t_start, t_end, steps, str(alloy_name))

    def _equilibrium_simulation(
        self,
        comp: Mapping[str, float],
        T_start: float,
        T_end: float,
        n_steps: int,
        alloy_name: str,
    ) -> SolidificationResult:
        temperatures = np.linspace(float(T_start), float(T_end), int(n_steps), dtype=float)
        liquid_frac = np.zeros_like(temperatures)
        solid_phase_data: dict[str, np.ndarray] = {}

        for i, temp in enumerate(temperatures):
            eq = self.equilibrium_at(comp, float(temp))
            liquid_frac[i] = np.clip(eq.phase_fractions.get("LIQUID", 0.0), 0.0, 1.0)

            for phase, frac in eq.phase_fractions.items():
                if phase == "LIQUID":
                    continue
                if phase not in solid_phase_data:
                    solid_phase_data[phase] = np.zeros_like(temperatures)
                solid_phase_data[phase][i] = np.clip(float(frac), 0.0, 1.0)

        liquidus_K, solidus_K = self._find_transus(temperatures, liquid_frac)
        if np.isnan(liquidus_K):
            liquidus_K = float(temperatures[0])
        if np.isnan(solidus_K):
            solidus_K = float(temperatures[-1])

        return SolidificationResult(
            temperatures=temperatures,
            liquid_fraction=liquid_frac,
            solid_phases=solid_phase_data,
            solidus_K=float(solidus_K),
            liquidus_K=float(liquidus_K),
            alloy_name=alloy_name,
            method="Equilibrium",
        )

    def _scheil_simulation(
        self,
        comp: Mapping[str, float],
        T_start: float,
        T_end: float,
        n_steps: int,
        alloy_name: str,
    ) -> SolidificationResult:
        try:
            from pycalphad.tools.scheil import scheil_solidification
        except ImportError:
            logger.warning("pycalphad.tools.scheil unavailable. Falling back to equilibrium path.")
            return self._equilibrium_simulation(comp, T_start, T_end, n_steps, alloy_name)

        step_temperature = abs((T_start - T_end) / max(n_steps - 1, 1))

        try:
            res = scheil_solidification(
                self.db,
                self.components,
                self.all_phases,
                dict(comp),
                start_temperature=float(T_start),
                step_temperature=float(step_temperature),
                verbose=False,
            )

            temperatures = np.asarray(getattr(res, "temperatures", []), dtype=float).reshape(-1)
            liquid_frac = np.asarray(getattr(res, "fraction_liquid", []), dtype=float).reshape(-1)
            if temperatures.size == 0 or liquid_frac.size == 0 or temperatures.size != liquid_frac.size:
                raise ValueError("Invalid Scheil output shapes")

            liquid_frac = np.nan_to_num(liquid_frac, nan=0.0, posinf=1.0, neginf=0.0)
            liquid_frac = np.clip(liquid_frac, 0.0, 1.0)

            solid_phase_data: dict[str, np.ndarray] = {}
            cum_phase_amounts = getattr(res, "cum_phase_amounts", None)
            if cum_phase_amounts is not None:
                for phase_name, values in getattr(cum_phase_amounts, "items", lambda: [])():
                    phase = str(phase_name).strip()
                    if not phase or phase.upper() == "LIQUID":
                        continue
                    arr = np.asarray(values, dtype=float).reshape(-1)
                    if arr.size != temperatures.size:
                        continue
                    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                    solid_phase_data[phase] = np.clip(arr, 0.0, 1.0)

            liquidus_K, solidus_K = self._find_transus(temperatures, liquid_frac)
            if np.isnan(liquidus_K):
                liquidus_K = float(temperatures[0])
            if np.isnan(solidus_K):
                solidus_K = float(temperatures[-1])

            return SolidificationResult(
                temperatures=temperatures,
                liquid_fraction=liquid_frac,
                solid_phases=solid_phase_data,
                solidus_K=float(solidus_K),
                liquidus_K=float(liquidus_K),
                alloy_name=alloy_name,
                method="Scheil",
            )

        except Exception as exc:
            logger.error("Scheil simulation failed: %s", exc)
            return self._equilibrium_simulation(comp, T_start, T_end, n_steps, alloy_name)

    def _find_transus(self, temperatures: np.ndarray, liquid_fraction: np.ndarray) -> tuple[float, float]:
        """Estimate liquidus and solidus temperatures from liquid fraction trajectory."""
        temps = np.asarray(temperatures, dtype=float).reshape(-1)
        f_liq = np.asarray(liquid_fraction, dtype=float).reshape(-1)
        if temps.size == 0 or f_liq.size == 0 or temps.size != f_liq.size:
            return float("nan"), float("nan")
        finite_mask = np.isfinite(temps) & np.isfinite(f_liq)
        if not np.any(finite_mask):
            return float("nan"), float("nan")

        temps = temps[finite_mask]
        f_liq = np.clip(f_liq[finite_mask], 0.0, 1.0)
        if temps.size < 2:
            return float("nan"), float("nan")

        order = np.argsort(temps)[::-1]
        temps = temps[order]
        f_liq = f_liq[order]

        def _crossing_temperature(threshold: float) -> float:
            for i in range(len(temps) - 1):
                f1 = float(f_liq[i])
                f2 = float(f_liq[i + 1])
                if f1 >= threshold > f2:
                    t1 = float(temps[i])
                    t2 = float(temps[i + 1])
                    if abs(f2 - f1) <= _EPS:
                        return t1
                    alpha = float(np.clip((threshold - f1) / (f2 - f1), 0.0, 1.0))
                    return t1 + alpha * (t2 - t1)
            return float("nan")

        liquidus = _crossing_temperature(0.99)
        solidus = _crossing_temperature(0.01)
        return float(liquidus), float(solidus)

    def plot_solidification(
        self,
        result: SolidificationResult,
        save_path: str | Path | None = None,
    ) -> None:
        """Plot solidification curve (Scheil or Equilibrium)."""
        temperatures = np.asarray(result.temperatures, dtype=float).reshape(-1)
        liquid_frac = np.asarray(result.liquid_fraction, dtype=float).reshape(-1)
        if temperatures.size == 0 or temperatures.size != liquid_frac.size:
            raise ValueError("temperatures and liquid_fraction must have the same non-empty length")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
        t_c = temperatures - 273.15

        ax.plot(t_c, liquid_frac, label="Liquid Fraction", color="black", linewidth=2.5)

        for phase, fracs in sorted(result.solid_phases.items()):
            arr = np.asarray(fracs, dtype=float).reshape(-1)
            if arr.size != temperatures.size:
                continue
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            if np.max(arr) > 0.02:
                label = phase.replace("_", " ").title()
                ax.plot(t_c, np.clip(arr, 0.0, 1.0), label=label, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Temperature (°C)", fontsize=12)
        ax.set_ylabel("Phase Fraction (Mole)", fontsize=12)
        ax.set_title(f"{result.method} Solidification: {result.alloy_name}", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(float(np.min(t_c)), float(np.max(t_c)))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=10)

        liq_c = float(result.liquidus_K) - 273.15
        sol_c = float(result.solidus_K) - 273.15

        if math.isfinite(liq_c):
            ax.annotate(
                f"T_liq: {liq_c:.0f}°C",
                xy=(liq_c, 1.0),
                xytext=(liq_c + 10, 1.05),
                arrowprops={"facecolor": "black", "shrink": 0.05, "width": 1, "headwidth": 5},
            )

        if math.isfinite(sol_c) and -273.15 < sol_c < 2500.0:
            ax.annotate(
                f"T_sol: {sol_c:.0f}°C",
                xy=(sol_c, 0.0),
                xytext=(sol_c - 30, 0.05),
                arrowprops={"facecolor": "red", "shrink": 0.05, "width": 1, "headwidth": 5},
            )

        plt.tight_layout()
        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_path), bbox_inches="tight")
            logger.info("Saved plot to %s", out_path)
        plt.close(fig)

    def print_equilibrium_table(
        self,
        alloy: str | Mapping[str, float],
        T_range: tuple[float, float] = (400, 550),
        step: float = 5.0,
    ) -> None:
        """Print a compact temperature-phase equilibrium table to stdout."""
        step_f = float(step)
        if not math.isfinite(step_f) or step_f <= 0:
            raise ValueError(f"step must be finite and > 0, got {step!r}")

        t_lo, t_hi = float(T_range[0]), float(T_range[1])
        if t_hi < t_lo:
            t_lo, t_hi = t_hi, t_lo
        temperatures = np.arange(t_lo, t_hi + 0.5 * step_f, step_f)

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

    def plot_binary(self, elem_x: str, elem_y: str, T_range: tuple[float, float] = (373, 1073)):
        """Plot binary diagram (wrapper for pycalphad binplot)."""
        from pycalphad import binplot
        from pycalphad import variables as v

        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        comps = [elem_x.upper(), elem_y.upper(), "VA"]
        conditions = {
            v.X(elem_y.upper()): (0, 1, 0.02),
            v.T: T_range,
            v.P: 101325,
            v.N: 1,
        }
        binplot(self.db, comps, self.all_phases, conditions, plot_kwargs={"ax": ax})
        ax.set_title(f"{elem_x}-{elem_y} Phase Diagram")
        plt.show()
