"""
Comprehensive Material Property Estimator

Extracts ALL available properties from JARVIS-DFT and computes
derived/estimated properties using physics-based models.

Direct properties (from JARVIS DFT):
    - Band gap, formation energy, energy above hull
    - Bulk modulus, shear modulus, Poisson ratio
    - Dielectric constant, piezoelectric coefficient
    - Seebeck coefficient, electrical conductivity, thermal conductivity
    - Effective masses, magnetic moment
    - SLME (solar cell efficiency), exfoliation energy
    - Spillage (topological indicator)
    - Superconducting Tc estimate

Derived properties (physics-based estimation):
    - Melting point (from cohesive energy + bulk modulus)
    - Debye temperature (from elastic constants + density)
    - Thermal conductivity at 300K (Slack model)
    - Electromigration resistance (from activation energy proxy)
    - Hardness (Chen-Niu model)
    - Ductility index (Pugh ratio)
    - Thermal expansion coefficient (Grüneisen relation)
"""

import numpy as np
import pandas as pd
from typing import Optional

from atlas.config import get_config


# ═══════════════════════════════════════════════════════════════
#  Physical Constants
# ═══════════════════════════════════════════════════════════════
kB = 8.617333e-5      # Boltzmann constant (eV/K)
HBAR = 6.582119e-16   # reduced Planck (eV·s)
AMU = 1.66054e-27     # atomic mass unit (kg)
ANGSTROM = 1e-10      # Å to m
EV_TO_J = 1.602176634e-19  # eV to Joules


class PropertyEstimator:
    """
    Comprehensive material property calculator.

    Combines direct JARVIS data with physics-based derived estimates
    to produce a complete property profile for any material.
    """

    # All numeric columns in JARVIS that need cleaning
    NUMERIC_COLS = [
        "optb88vdw_bandgap", "mbj_bandgap", "hse_gap",
        "formation_energy_peratom", "optb88vdw_total_energy", "ehull",
        "bulk_modulus_kv", "shear_modulus_gv", "poisson",
        "epsx", "epsy", "epsz", "mepsx", "mepsy", "mepsz",
        "dfpt_piezo_max_dij", "dfpt_piezo_max_eij",
        "dfpt_piezo_max_dielectric", "dfpt_piezo_max_dielectric_electronic",
        "dfpt_piezo_max_dielectric_ionic",
        "n-Seebeck", "p-Seebeck", "n-powerfact", "p-powerfact",
        "ncond", "pcond", "nkappa", "pkappa",
        "spillage", "slme", "exfoliation_energy",
        "magmom_oszicar", "magmom_outcar",
        "max_efg", "max_ir_mode", "min_ir_mode",
        "Tc_supercon", "density", "avg_elec_mass", "avg_hole_mass",
    ]

    def __init__(self):
        self.cfg = get_config()

    def extract_all_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and clean all properties from a JARVIS DataFrame.
        Adds derived property columns.

        Args:
            df: raw JARVIS DFT DataFrame

        Returns:
            DataFrame with cleaned + derived properties
        """
        result = df.copy()

        # ─── Clean numeric columns ───
        for col in self.NUMERIC_COLS:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        # ─── Derived: Best band gap estimate ───
        result["bandgap_best"] = result.apply(self._best_bandgap, axis=1)

        # ─── Derived: Conductivity class ───
        result["conductivity_class"] = result["bandgap_best"].apply(
            self._classify_conductivity
        )

        # ─── Derived: Average dielectric constant ───
        for prefix in ["eps", "meps"]:
            cols = [f"{prefix}{d}" for d in "xyz"]
            if all(c in result.columns for c in cols):
                result[f"{prefix}_avg"] = result[cols].mean(axis=1)

        # ─── Derived: Debye temperature ───
        result["debye_temperature"] = result.apply(
            self._estimate_debye_temperature, axis=1
        )

        # ─── Derived: Melting point ───
        result["melting_point_est"] = result.apply(
            self._estimate_melting_point, axis=1
        )

        # ─── Derived: Thermal conductivity (Slack model) ───
        result["kappa_slack"] = result.apply(
            self._estimate_thermal_conductivity_slack, axis=1
        )

        # ─── Derived: Hardness (Chen-Niu model) ───
        result["hardness_chen"] = result.apply(
            self._estimate_hardness, axis=1
        )

        # ─── Derived: Ductility (Pugh ratio) ───
        result["pugh_ratio"] = result.apply(self._pugh_ratio, axis=1)
        result["is_ductile"] = result["pugh_ratio"].apply(
            lambda x: True if x is not None and x > 1.75 else
                      (False if x is not None else None)
        )

        # ─── Derived: Electromigration resistance ───
        result["electromigration_resistance"] = result.apply(
            self._estimate_electromigration_resistance, axis=1
        )

        # ─── Derived: Thermal expansion estimate ───
        result["thermal_expansion_est"] = result.apply(
            self._estimate_thermal_expansion, axis=1
        )

        # ─── Derived: Young's modulus ───
        result["youngs_modulus"] = result.apply(
            self._estimate_youngs_modulus, axis=1
        )

        # ─── Derived: Best Seebeck ───
        if "n-Seebeck" in result.columns and "p-Seebeck" in result.columns:
            n_seebeck = pd.to_numeric(result["n-Seebeck"], errors="coerce").abs()
            p_seebeck = pd.to_numeric(result["p-Seebeck"], errors="coerce").abs()
            result["seebeck_best"] = pd.concat(
                [n_seebeck, p_seebeck], axis=1
            ).max(axis=1)

        # ─── Derived: Best electrical conductivity ───
        if "ncond" in result.columns and "pcond" in result.columns:
            ncond = pd.to_numeric(result["ncond"], errors="coerce")
            pcond = pd.to_numeric(result["pcond"], errors="coerce")
            result["elec_cond_best"] = pd.concat(
                [ncond, pcond], axis=1
            ).max(axis=1)

        # ─── Derived: Best thermal conductivity ───
        if "nkappa" in result.columns and "pkappa" in result.columns:
            nk = pd.to_numeric(result["nkappa"], errors="coerce")
            pk = pd.to_numeric(result["pkappa"], errors="coerce")
            result["thermal_cond_best"] = pd.concat([nk, pk], axis=1).max(axis=1)

        # ─── Derived: Power factor ───
        if "n-powerfact" in result.columns and "p-powerfact" in result.columns:
            npf = pd.to_numeric(result["n-powerfact"], errors="coerce")
            ppf = pd.to_numeric(result["p-powerfact"], errors="coerce")
            result["powerfactor_best"] = pd.concat([npf, ppf], axis=1).max(axis=1)

        return result

    # ═══════════════════════════════════════════════════════════
    #  Direct Property Helpers
    # ═══════════════════════════════════════════════════════════

    def _best_bandgap(self, row) -> Optional[float]:
        """Pick the most accurate band gap estimate available."""
        for col in ["hse_gap", "mbj_bandgap", "optb88vdw_bandgap"]:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                try:
                    v = float(val)
                    if v >= 0:
                        return v
                except (ValueError, TypeError):
                    continue
        return None

    def _classify_conductivity(self, bandgap) -> Optional[str]:
        """Classify material by conductivity type."""
        if bandgap is None or (isinstance(bandgap, float) and np.isnan(bandgap)):
            return None
        if bandgap == 0 or bandgap < 0.01:
            return "metal"
        elif bandgap < 0.5:
            return "semimetal"
        elif bandgap < 3.0:
            return "semiconductor"
        else:
            return "insulator"

    # ═══════════════════════════════════════════════════════════
    #  Derived Property Estimators (Physics-Based)
    # ═══════════════════════════════════════════════════════════

    def _estimate_debye_temperature(self, row) -> Optional[float]:
        """
        Debye temperature from elastic constants.

        θ_D = (ℏ/k_B) · (6π²n)^(1/3) · v_m

        where v_m is the average sound velocity from bulk (K) and shear (G) modulus:
            v_m = [(1/3)(2/v_t³ + 1/v_l³)]^(-1/3)
            v_t = √(G/ρ),  v_l = √((K + 4G/3)/ρ)
        """
        K = row.get("bulk_modulus_kv")
        G = row.get("shear_modulus_gv")
        rho = row.get("density")

        if not self._valid(K, G, rho) or K <= 0 or G <= 0 or rho <= 0:
            return None

        try:
            # Convert GPa → Pa
            K_pa = float(K) * 1e9
            G_pa = float(G) * 1e9
            rho_kg = float(rho) * 1000  # g/cm³ → kg/m³

            v_t = np.sqrt(G_pa / rho_kg)       # transverse sound velocity
            v_l = np.sqrt((K_pa + 4*G_pa/3) / rho_kg)  # longitudinal

            # Average sound velocity
            v_m = ((1/3) * (2/v_t**3 + 1/v_l**3)) ** (-1/3)

            # Estimate number density from density and average atomic mass
            atoms_dict = row.get("atoms")
            if atoms_dict:
                from jarvis.core.atoms import Atoms as JAtoms
                jatoms = JAtoms.from_dict(atoms_dict)
                n_atoms = len(jatoms.elements)
                vol_m3 = jatoms.volume * ANGSTROM**3
                n_density = n_atoms / vol_m3
            else:
                # Fallback: estimate from density
                avg_mass = 30 * AMU  # rough average atomic mass
                n_density = rho_kg / avg_mass

            theta_D = (HBAR * EV_TO_J / kB / EV_TO_J) * \
                      (6 * np.pi**2 * n_density)**(1/3) * v_m

            # Convert from eV to K
            # Actually Hbar is in eV·s, kB in eV/K, so θ_D in K
            hbar_si = 1.0545718e-34  # J·s
            kb_si = 1.380649e-23     # J/K
            theta_D = (hbar_si / kb_si) * \
                      (6 * np.pi**2 * n_density)**(1/3) * v_m

            if 50 < theta_D < 2000:
                return round(theta_D, 1)
        except Exception:
            pass
        return None

    def _estimate_melting_point(self, row) -> Optional[float]:
        """
        Melting point estimation from Debye temperature.

        Empirical relation (Lindemann criterion):
            T_m ≈ C · θ_D² · M · V^(2/3) / (ℏ²)

        Simplified empirical correlation:
            T_m ≈ 607 + 9.3 · θ_D   (for metals, Grimvall)

        For covalent materials we use:
            T_m ≈ 0.032 · K · V_atom   (from bulk modulus)

        where K = bulk modulus (GPa), V_atom = volume per atom (ų)
        """
        # Method 1: From Debye temperature (best for metals)
        theta_D = self._estimate_debye_temperature(row)
        K = row.get("bulk_modulus_kv")
        bandgap = self._best_bandgap(row)

        if theta_D is not None:
            if bandgap is not None and bandgap < 0.5:
                # Metal: Grimvall empirical
                T_m = 607 + 9.3 * theta_D
            else:
                # Non-metal: weaker correlation
                T_m = 400 + 7.0 * theta_D
            if 200 < T_m < 5000:
                return round(T_m, 0)

        # Method 2: From bulk modulus + volume (fallback)
        if self._valid(K) and K > 0:
            try:
                atoms_dict = row.get("atoms")
                if atoms_dict:
                    from jarvis.core.atoms import Atoms as JAtoms
                    jatoms = JAtoms.from_dict(atoms_dict)
                    V_atom = jatoms.volume / len(jatoms.elements)  # ų per atom
                    T_m = 0.032 * float(K) * V_atom + 300
                    if 200 < T_m < 5000:
                        return round(T_m, 0)
            except Exception:
                pass

        return None

    def _estimate_thermal_conductivity_slack(self, row) -> Optional[float]:
        """
        Lattice thermal conductivity using the Slack model:

            κ_L = A · (M_avg · θ_D³ · V_atom^(1/3)) / (γ² · n^(2/3) · T)

        where:
            A ≈ 3.1e-6 (empirical constant)
            M_avg = average atomic mass (amu)
            θ_D = Debye temperature (K)
            V_atom = volume per atom (ų)
            γ = Grüneisen parameter ≈ 1.5 (typical)
            n = atoms per primitive cell
            T = 300 K
        """
        theta_D = self._estimate_debye_temperature(row)
        if theta_D is None:
            return None

        try:
            atoms_dict = row.get("atoms")
            if not atoms_dict:
                return None

            from jarvis.core.atoms import Atoms as JAtoms
            from pymatgen.core import Element

            jatoms = JAtoms.from_dict(atoms_dict)
            n_atoms = len(jatoms.elements)
            vol = jatoms.volume  # ų

            masses = [Element(e).atomic_mass for e in jatoms.elements]
            M_avg = float(np.mean(masses))  # amu

            V_atom = vol / n_atoms  # ų per atom
            gamma = 1.5   # Grüneisen parameter (typical)
            T = 300.0     # temperature (K)

            A = 3.1e-6
            kappa = A * (M_avg * theta_D**3 * V_atom**(1/3)) / \
                    (gamma**2 * n_atoms**(2/3) * T)

            if 0.01 < kappa < 5000:
                return round(kappa, 2)
        except Exception:
            pass
        return None

    def _estimate_hardness(self, row) -> Optional[float]:
        """
        Vickers hardness from Chen-Niu model:

            H_v = 2 · (k² · G)^0.585 - 3

        where k = G/K (Pugh ratio), G = shear modulus, K = bulk modulus.
        All in GPa. Result in GPa.
        """
        K = row.get("bulk_modulus_kv")
        G = row.get("shear_modulus_gv")

        if not self._valid(K, G) or K <= 0 or G <= 0:
            return None

        try:
            K, G = float(K), float(G)
            k = G / K
            H_v = 2.0 * (k**2 * G)**0.585 - 3.0
            if 0 < H_v < 100:
                return round(H_v, 2)
        except Exception:
            pass
        return None

    def _pugh_ratio(self, row) -> Optional[float]:
        """
        Pugh ratio K/G — indicator of ductility.
        K/G > 1.75 → ductile
        K/G < 1.75 → brittle
        """
        K = row.get("bulk_modulus_kv")
        G = row.get("shear_modulus_gv")

        if not self._valid(K, G) or G <= 0:
            return None

        try:
            return round(float(K) / float(G), 3)
        except Exception:
            return None

    def _estimate_electromigration_resistance(self, row) -> Optional[float]:
        """
        Electromigration resistance proxy.

        EM resistance correlates with:
        1. Melting point (higher T_m → stronger atomic bonds → harder to migrate)
        2. Bulk modulus (stiffer material → higher activation energy)
        3. Cohesive energy (stronger bonds → higher EM resistance)

        Empirical activation energy for electromigration:
            E_a ≈ 0.0012 · T_m  (eV, rough for metals)

        We normalize to a 0–1 score:
            0 = very susceptible to EM (like Al: E_a ≈ 0.7 eV)
            1 = very resistant (like W: E_a ≈ 3.5 eV)
        """
        T_m = self._estimate_melting_point(row)
        K = row.get("bulk_modulus_kv")

        if T_m is not None:
            E_a = 0.0012 * T_m  # approximate activation energy (eV)

            # Boost from bulk modulus
            if self._valid(K) and K > 0:
                K_factor = min(float(K) / 300.0, 1.5)  # normalize by W's K
                E_a *= K_factor

            # Normalize to 0-1 scale (0.5eV → 0, 3.5eV → 1)
            score = np.clip((E_a - 0.5) / 3.0, 0, 1)
            return round(score, 3)

        return None

    def _estimate_thermal_expansion(self, row) -> Optional[float]:
        """
        Volumetric thermal expansion coefficient (×10⁻⁶ K⁻¹)

        Grüneisen relation:
            α = γ · C_v / (K · V)

        Simplified estimate using Debye model at T >> θ_D:
            α ≈ 3 · γ · k_B · n / (K · V)

        With γ ≈ 1.5 (typical Grüneisen parameter)
        """
        K = row.get("bulk_modulus_kv")
        theta_D = self._estimate_debye_temperature(row)

        if not self._valid(K) or K <= 0:
            return None

        try:
            atoms_dict = row.get("atoms")
            if not atoms_dict:
                return None

            from jarvis.core.atoms import Atoms as JAtoms
            jatoms = JAtoms.from_dict(atoms_dict)
            n_atoms = len(jatoms.elements)
            vol_m3 = jatoms.volume * ANGSTROM**3
            K_pa = float(K) * 1e9
            gamma = 1.5
            kb_si = 1.380649e-23

            # At 300K, assume C_v ≈ 3·n·k_B (Dulong-Petit)
            C_v = 3 * n_atoms * kb_si
            alpha = gamma * C_v / (K_pa * vol_m3)

            # Convert to ×10⁻⁶ K⁻¹
            alpha_ppm = alpha * 1e6
            if 0.1 < alpha_ppm < 200:
                return round(alpha_ppm, 2)
        except Exception:
            pass
        return None

    def _estimate_youngs_modulus(self, row) -> Optional[float]:
        """
        Young's modulus from bulk and shear modulus.
        E = 9KG / (3K + G)  (in GPa)
        """
        K = row.get("bulk_modulus_kv")
        G = row.get("shear_modulus_gv")

        if not self._valid(K, G) or K <= 0 or G <= 0:
            return None

        try:
            K, G = float(K), float(G)
            E = 9 * K * G / (3 * K + G)
            if E > 0:
                return round(E, 2)
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════════════════
    #  Multi-Property Search Engine
    # ═══════════════════════════════════════════════════════════

    def search(
        self,
        df: pd.DataFrame,
        criteria: dict[str, tuple],
        sort_by: str = None,
        ascending: bool = True,
        max_results: int = 50,
    ) -> pd.DataFrame:
        """
        Search for materials matching multiple property criteria.

        Args:
            df: DataFrame with properties (from extract_all_properties)
            criteria: dict of {column: (min_val, max_val)} or {column: value}
                Examples:
                    {"bandgap_best": (0.5, 2.0)}        # semiconductor
                    {"melting_point_est": (None, 600)}   # low melting point
                    {"conductivity_class": "metal"}       # metals only
                    {"is_ductile": True}                  # ductile materials
            sort_by: column to sort results by
            ascending: sort direction
            max_results: maximum number of results

        Returns:
            Filtered and sorted DataFrame
        """
        mask = pd.Series(True, index=df.index)

        for col, criterion in criteria.items():
            if col not in df.columns:
                print(f"  Warning: column '{col}' not found, skipping")
                continue

            if isinstance(criterion, tuple):
                lo, hi = criterion
                if lo is not None:
                    mask &= df[col] >= lo
                if hi is not None:
                    mask &= df[col] <= hi
            elif isinstance(criterion, bool):
                mask &= df[col] == criterion
            elif isinstance(criterion, str):
                mask &= df[col] == criterion
            else:
                mask &= df[col] == criterion

        result = df[mask].copy()

        if sort_by and sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=ascending)

        return result.head(max_results)

    def property_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics for all properties."""
        summary = {}

        key_props = {
            "bandgap_best": "Band Gap (eV)",
            "formation_energy_peratom": "Formation Energy (eV/atom)",
            "ehull": "Energy Above Hull (eV/atom)",
            "bulk_modulus_kv": "Bulk Modulus (GPa)",
            "shear_modulus_gv": "Shear Modulus (GPa)",
            "youngs_modulus": "Young's Modulus (GPa)",
            "hardness_chen": "Hardness (GPa)",
            "melting_point_est": "Melting Point (K)",
            "debye_temperature": "Debye Temperature (K)",
            "kappa_slack": "Thermal Conductivity - Slack (W/mK)",
            "thermal_expansion_est": "Thermal Expansion (×10⁻⁶/K)",
            "electromigration_resistance": "EM Resistance (0-1)",
            "seebeck_best": "Seebeck Coefficient (μV/K)",
            "elec_cond_best": "Electrical Conductivity (S/m)",
            "thermal_cond_best": "Thermal Conductivity (W/mK)",
            "spillage": "Topological Spillage",
            "slme": "SLME Solar Efficiency (%)",
            "density": "Density (g/cm³)",
        }

        for col, label in key_props.items():
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                valid = series.dropna()
                if len(valid) > 0:
                    summary[label] = {
                        "count": len(valid),
                        "mean": round(valid.mean(), 4),
                        "min": round(valid.min(), 4),
                        "max": round(valid.max(), 4),
                        "median": round(valid.median(), 4),
                    }

        # Conductivity class distribution
        if "conductivity_class" in df.columns:
            summary["Conductivity Distribution"] = (
                df["conductivity_class"].value_counts().to_dict()
            )

        return summary

    # ═══════════════════════════════════════════════════════════
    #  Utility
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _valid(*values) -> bool:
        """Check that all values are not None and not NaN."""
        for v in values:
            if v is None:
                return False
            try:
                if np.isnan(float(v)):
                    return False
            except (ValueError, TypeError):
                return False
        return True
