"""
Alloy property estimator.

Provides a compact, test-friendly API:
- ``AlloyEstimator.from_preset(...)``
- ``AlloyEstimator.custom(...)``
- ``AlloyEstimator.estimate_properties()``
- ``AlloyEstimator.print_report(...)``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_COMMON_ELEMENT_MOLAR_MASS = {
    "SN": 118.710,
    "AG": 107.8682,
    "CU": 63.546,
    "PB": 207.2,
    "BI": 208.9804,
    "IN": 114.818,
    "SB": 121.760,
    "ZN": 65.38,
    "NI": 58.6934,
    "AU": 196.96657,
    "AL": 26.9815385,
    "FE": 55.845,
    "CO": 58.933194,
    "CR": 51.9961,
    "MN": 54.938044,
    "MO": 95.95,
    "W": 183.84,
    "TI": 47.867,
    "SI": 28.085,
    "MG": 24.305,
    "LI": 6.94,
    "NA": 22.98976928,
    "K": 39.0983,
    "CA": 40.078,
}

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


@dataclass
class AlloyPhase:
    name: str
    formula: str
    weight_fraction: float
    properties: dict[str, float]
    volume_fraction: float = field(default=0.0)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.properties.get(key, default))


class AlloyEstimator:
    def __init__(self, name: str, phases: list[AlloyPhase]):
        self.name = name
        self.phases = phases
        self._normalize_weight_fractions()
        self.convert_wt_to_vol(self.phases)

    @staticmethod
    def _normalize_preset_key(preset: str) -> str:
        return preset.strip().lower().replace("-", "").replace("_", "").replace(" ", "")

    @classmethod
    def available_presets(cls) -> tuple[str, ...]:
        return ("SAC305", "SAC405", "SnPb63", "pure_Sn", "pure_Cu")

    @staticmethod
    def convert_wt_to_vol(phases: list[AlloyPhase]):
        weights = np.array([max(p.weight_fraction, 0.0) for p in phases], dtype=float)
        rho = np.array([max(p.get("density_g_cm3", 1e-6), 1e-6) for p in phases], dtype=float)
        vol = weights / rho
        denom = float(np.sum(vol))
        if denom <= 0:
            frac = np.ones(len(phases), dtype=float) / max(len(phases), 1)
        else:
            frac = vol / denom
        for p, vf in zip(phases, frac):
            p.volume_fraction = float(vf)

    def _normalize_weight_fractions(self):
        total = float(sum(max(p.weight_fraction, 0.0) for p in self.phases))
        if total <= 0:
            raise ValueError("Total weight fraction must be > 0")
        for p in self.phases:
            p.weight_fraction = float(max(p.weight_fraction, 0.0) / total)

    @classmethod
    def from_preset(cls, preset: str) -> AlloyEstimator:
        key = cls._normalize_preset_key(preset)
        presets = {
            "sac305": cls._preset_sac305,
            "sac405": cls._preset_sac405,
            "snpb63": cls._preset_snpb63,
            "puresn": cls._preset_pure_sn,
            "purecu": cls._preset_pure_cu,
        }
        if key not in presets:
            names = ", ".join(cls.available_presets())
            raise ValueError(f"Unknown preset: {preset}. Available presets: {names}")
        return presets[key]()

    @classmethod
    def custom(cls, name: str, phases: list[dict]) -> AlloyEstimator:
        parsed = []
        for ph in phases:
            parsed.append(
                AlloyPhase(
                    name=str(ph["name"]),
                    formula=str(ph.get("formula", ph["name"])),
                    weight_fraction=float(ph["weight_fraction"]),
                    properties={k: float(v) for k, v in ph.get("properties", {}).items()},
                )
            )
        return cls(name=name, phases=parsed)

    @classmethod
    def _preset_sac305(cls) -> AlloyEstimator:
        return cls(
            name="SAC305",
            phases=[
                AlloyPhase(
                    name="Sn",
                    formula="Sn",
                    weight_fraction=0.965,
                    properties={
                        "density_g_cm3": 7.29,
                        "bulk_modulus_GPa": 56.3,
                        "shear_modulus_GPa": 18.4,
                        "thermal_conductivity_W_mK": 66.0,
                        "thermal_expansion_1e6_K": 22.0,
                        "melting_point_K": 505.0,
                    },
                ),
                AlloyPhase(
                    name="Ag",
                    formula="Ag",
                    weight_fraction=0.030,
                    properties={
                        "density_g_cm3": 10.49,
                        "bulk_modulus_GPa": 100.0,
                        "shear_modulus_GPa": 30.0,
                        "thermal_conductivity_W_mK": 429.0,
                        "thermal_expansion_1e6_K": 18.9,
                        "melting_point_K": 1234.0,
                    },
                ),
                AlloyPhase(
                    name="Cu",
                    formula="Cu",
                    weight_fraction=0.005,
                    properties={
                        "density_g_cm3": 8.96,
                        "bulk_modulus_GPa": 137.0,
                        "shear_modulus_GPa": 48.3,
                        "thermal_conductivity_W_mK": 401.0,
                        "thermal_expansion_1e6_K": 16.5,
                        "melting_point_K": 1358.0,
                    },
                ),
            ],
        )

    @classmethod
    def _preset_sac405(cls) -> AlloyEstimator:
        return cls(
            name="SAC405",
            phases=[
                AlloyPhase(
                    name="Sn",
                    formula="Sn",
                    weight_fraction=0.955,
                    properties={
                        "density_g_cm3": 7.29,
                        "bulk_modulus_GPa": 56.3,
                        "shear_modulus_GPa": 18.4,
                        "thermal_conductivity_W_mK": 66.0,
                        "thermal_expansion_1e6_K": 22.0,
                        "melting_point_K": 505.0,
                    },
                ),
                AlloyPhase(
                    name="Ag",
                    formula="Ag",
                    weight_fraction=0.040,
                    properties={
                        "density_g_cm3": 10.49,
                        "bulk_modulus_GPa": 100.0,
                        "shear_modulus_GPa": 30.0,
                        "thermal_conductivity_W_mK": 429.0,
                        "thermal_expansion_1e6_K": 18.9,
                        "melting_point_K": 1234.0,
                    },
                ),
                AlloyPhase(
                    name="Cu",
                    formula="Cu",
                    weight_fraction=0.005,
                    properties={
                        "density_g_cm3": 8.96,
                        "bulk_modulus_GPa": 137.0,
                        "shear_modulus_GPa": 48.3,
                        "thermal_conductivity_W_mK": 401.0,
                        "thermal_expansion_1e6_K": 16.5,
                        "melting_point_K": 1358.0,
                    },
                ),
            ],
        )

    @classmethod
    def _preset_snpb63(cls) -> AlloyEstimator:
        return cls(
            name="SnPb63",
            phases=[
                AlloyPhase(
                    name="Sn",
                    formula="Sn",
                    weight_fraction=0.63,
                    properties={
                        "density_g_cm3": 7.29,
                        "bulk_modulus_GPa": 56.3,
                        "shear_modulus_GPa": 18.4,
                        "thermal_conductivity_W_mK": 66.0,
                        "thermal_expansion_1e6_K": 22.0,
                        "melting_point_K": 505.0,
                    },
                ),
                AlloyPhase(
                    name="Pb",
                    formula="Pb",
                    weight_fraction=0.37,
                    properties={
                        "density_g_cm3": 11.34,
                        "bulk_modulus_GPa": 46.0,
                        "shear_modulus_GPa": 5.6,
                        "thermal_conductivity_W_mK": 35.0,
                        "thermal_expansion_1e6_K": 28.9,
                        "melting_point_K": 601.0,
                    },
                ),
            ],
        )

    @classmethod
    def _preset_pure_sn(cls) -> AlloyEstimator:
        return cls(
            name="pure_Sn",
            phases=[
                AlloyPhase(
                    name="Sn",
                    formula="Sn",
                    weight_fraction=1.0,
                    properties={
                        "density_g_cm3": 7.29,
                        "bulk_modulus_GPa": 56.3,
                        "shear_modulus_GPa": 18.4,
                        "thermal_conductivity_W_mK": 66.0,
                        "thermal_expansion_1e6_K": 22.0,
                        "melting_point_K": 505.0,
                    },
                ),
            ],
        )

    @classmethod
    def _preset_pure_cu(cls) -> AlloyEstimator:
        return cls(
            name="pure_Cu",
            phases=[
                AlloyPhase(
                    name="Cu",
                    formula="Cu",
                    weight_fraction=1.0,
                    properties={
                        "density_g_cm3": 8.96,
                        "bulk_modulus_GPa": 137.0,
                        "shear_modulus_GPa": 48.3,
                        "thermal_conductivity_W_mK": 401.0,
                        "thermal_expansion_1e6_K": 16.5,
                        "melting_point_K": 1358.0,
                    },
                ),
            ],
        )

    @staticmethod
    def _safe_reuss(vf: np.ndarray, x: np.ndarray) -> float:
        x = np.maximum(x, 1e-8)
        return float(1.0 / np.sum(vf / x))

    @staticmethod
    def _wiener_bounds(frac: np.ndarray, values: np.ndarray) -> tuple[float, float]:
        """
        Wiener bounds for effective scalar properties.

        Reference:
        - Wiener, O. (1912), relation between microstructure and effective transport.
          (Arithmetic/parallel upper bound, harmonic/series lower bound)
        """
        values = np.maximum(values.astype(float), 1e-8)
        upper = float(np.sum(frac * values))
        lower = float(1.0 / np.sum(frac / values))
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    @staticmethod
    def _safe_shannon_entropy(frac: np.ndarray) -> float:
        frac = np.asarray(frac, dtype=float)
        frac = frac[frac > 1e-12]
        if frac.size == 0:
            return 0.0
        return float(-np.sum(frac * np.log(frac)))

    @classmethod
    def _element_atomic_mass(cls, symbol: str) -> float | None:
        symbol = str(symbol).strip()
        if not symbol:
            return None
        try:
            from pymatgen.core import Element

            m = float(Element(symbol).atomic_mass)
            if np.isfinite(m) and m > 0:
                return m
        except Exception:
            pass
        fallback = _COMMON_ELEMENT_MOLAR_MASS.get(symbol.upper())
        if fallback is None:
            return None
        return float(fallback)

    @classmethod
    def _parse_formula_stoichiometry(cls, formula: str) -> dict[str, float] | None:
        formula = str(formula).strip()
        if not formula:
            return None
        try:
            from pymatgen.core import Composition

            comp = Composition(formula)
            data = {str(el): float(v) for el, v in comp.get_el_amt_dict().items() if float(v) > 0}
            return data or None
        except Exception:
            pass

        tokens = _FORMULA_TOKEN_RE.findall(formula)
        if not tokens:
            return None
        out: dict[str, float] = {}
        for sym, num_str in tokens:
            count = float(num_str) if num_str else 1.0
            if count <= 0:
                continue
            out[sym] = out.get(sym, 0.0) + count
        return out or None

    @classmethod
    def _formula_molar_mass(cls, formula: str) -> float | None:
        stoich = cls._parse_formula_stoichiometry(formula)
        if not stoich:
            return None
        mass = 0.0
        for sym, count in stoich.items():
            m = cls._element_atomic_mass(sym)
            if m is None:
                return None
            mass += count * m
        return mass if mass > 0 else None

    @classmethod
    def _estimate_mole_fractions(cls, phases: list[AlloyPhase]) -> np.ndarray:
        wf = np.array([max(p.weight_fraction, 0.0) for p in phases], dtype=float)
        wf_sum = float(np.sum(wf))
        if wf_sum <= 0:
            return np.ones(len(phases), dtype=float) / max(len(phases), 1)
        wf = wf / wf_sum

        molar_masses = []
        for p in phases:
            mass = cls._formula_molar_mass(p.formula)
            molar_masses.append(np.nan if mass is None else float(mass))
        mm = np.array(molar_masses, dtype=float)
        if not np.all(np.isfinite(mm)) or np.any(mm <= 0):
            # If formula mass is unavailable, fallback to normalized weight fractions.
            return wf

        moles = wf / mm
        denom = float(np.sum(moles))
        if denom <= 0:
            return wf
        return moles / denom

    @staticmethod
    def _estimate_element_mole_fractions(phases: list[AlloyPhase]) -> np.ndarray | None:
        """
        Estimate elemental mole fractions from phase formulas and phase weights.

        This is the physically consistent basis for ideal configurational entropy:
            S_conf = -R * sum_i x_i ln x_i
        """
        if not phases:
            return None
        wf = np.array([max(p.weight_fraction, 0.0) for p in phases], dtype=float)
        wf_sum = float(np.sum(wf))
        if wf_sum <= 0:
            return None
        wf = wf / wf_sum

        from collections import defaultdict

        elem_moles: dict[str, float] = defaultdict(float)
        for p, w in zip(phases, wf):
            stoich = AlloyEstimator._parse_formula_stoichiometry(p.formula)
            mm = AlloyEstimator._formula_molar_mass(p.formula)
            if not stoich or mm is None or mm <= 0:
                return None
            phase_moles = w / mm
            for sym, count in stoich.items():
                elem_moles[sym] += phase_moles * float(count)

        if not elem_moles:
            return None
        vec = np.array([v for v in elem_moles.values() if v > 0], dtype=float)
        total = float(np.sum(vec))
        if total <= 0:
            return None
        return vec / total

    @staticmethod
    def _maxwell_eucken(vf: np.ndarray, conductivity: np.ndarray) -> float:
        """
        Symmetric Bruggeman/Maxwell-Eucken effective medium estimate.

        Reference:
        - Maxwell (1873), Treatise on Electricity and Magnetism.
        - Eucken (1940), extension for composite media.
        - Bruggeman (1935), self-consistent effective medium equation.
        """
        vf = np.asarray(vf, dtype=float)
        k = np.maximum(np.asarray(conductivity, dtype=float), 1e-8)
        if len(vf) == 0:
            return 0.0
        vf_sum = float(np.sum(vf))
        if vf_sum <= 0:
            return float(np.exp(np.mean(np.log(k))))
        vf = vf / vf_sum

        # f(x) = sum_i vf_i * (k_i - x)/(k_i + 2x) = 0
        def f(x: float) -> float:
            den = np.maximum(k + 2.0 * x, 1e-12)
            return float(np.sum(vf * ((k - x) / den)))

        if np.allclose(k, k[0]):
            return float(k[0])

        lo = max(1e-12, float(np.min(k)) * 1e-6)
        hi = float(np.max(k))
        flo = f(lo)
        fhi = f(hi)

        # Expand upper bracket until sign change.
        step = 0
        while fhi > 0 and step < 60:
            hi *= 2.0
            fhi = f(hi)
            step += 1
        if flo < 0:
            # Numerically unlikely, but keep robust fallback.
            lo = 1e-12
            flo = f(lo)

        # If bracketing failed, fallback to geometric mean (order-invariant).
        if flo * fhi > 0:
            return float(np.exp(np.sum(vf * np.log(k))))

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < 1e-12 or abs(hi - lo) < 1e-10 * max(1.0, mid):
                return float(mid)
            if flo * fmid > 0:
                lo = mid
                flo = fmid
            else:
                hi = mid
                fhi = fmid
        return float(0.5 * (lo + hi))

    def estimate_properties(
        self,
        *,
        thermal_model: str = "geometric",
    ) -> dict[str, float]:
        vf = np.array([p.volume_fraction for p in self.phases], dtype=float)
        wf = np.array([p.weight_fraction for p in self.phases], dtype=float)
        xf_phase = self._estimate_mole_fractions(self.phases)
        xf_element = self._estimate_element_mole_fractions(self.phases)

        density = float(np.sum(vf * np.array([p.get("density_g_cm3") for p in self.phases])))
        K = np.array([p.get("bulk_modulus_GPa") for p in self.phases], dtype=float)
        G = np.array([p.get("shear_modulus_GPa") for p in self.phases], dtype=float)

        K_voigt = float(np.sum(vf * K))
        G_voigt = float(np.sum(vf * G))
        K_reuss = self._safe_reuss(vf, K)
        G_reuss = self._safe_reuss(vf, G)
        K_vrh = 0.5 * (K_voigt + K_reuss)
        G_vrh = 0.5 * (G_voigt + G_reuss)

        denom = 3.0 * K_vrh + G_vrh
        young = float(9.0 * K_vrh * G_vrh / denom) if denom > 1e-8 else 0.0
        poisson = float((3.0 * K_vrh - 2.0 * G_vrh) / (2.0 * denom)) if denom > 1e-8 else 0.0

        kappa = np.array([max(p.get("thermal_conductivity_W_mK"), 1e-8) for p in self.phases], dtype=float)
        kappa_wiener_lo, kappa_wiener_hi = self._wiener_bounds(vf, kappa)
        thermal_model_key = str(thermal_model).strip().lower()
        if thermal_model_key == "geometric":
            thermal_cond = float(np.exp(np.sum(vf * np.log(kappa))))
        elif thermal_model_key == "maxwell":
            thermal_cond = self._maxwell_eucken(vf, kappa)
        elif thermal_model_key == "wiener_mid":
            thermal_cond = 0.5 * (kappa_wiener_lo + kappa_wiener_hi)
        else:
            raise ValueError(
                "Unknown thermal_model. Choices: geometric, maxwell, wiener_mid"
            )

        cte = float(np.sum(vf * np.array([p.get("thermal_expansion_1e6_K") for p in self.phases], dtype=float)))
        # Use harmonic averaging for alloy melting point proxy.
        # This is empirically closer to solder liquidus behavior than arithmetic mean.
        tm = np.array([max(p.get("melting_point_K"), 1e-8) for p in self.phases], dtype=float)
        melting = float(1.0 / np.sum(wf / tm))
        melting_arith = float(np.sum(wf * tm))

        pugh = float(K_vrh / max(G_vrh, 1e-8))
        hardness = float(max(0.0, 2.0 * ((G_vrh / max(K_vrh, 1e-8)) ** 2 * G_vrh) ** 0.585 - 3.0))

        # Entropy-based descriptors (ideal-mixture surrogates).
        # Reference:
        # - Yeh et al. (2004), high-entropy alloy concept and configurational entropy use.
        R = 8.314462618  # J/mol/K
        if xf_element is not None:
            cfg_entropy_nat = self._safe_shannon_entropy(xf_element)
            entropy_basis = "element"
            n_config_components = int(np.sum(xf_element > 1e-12))
        else:
            # Fallback if element parsing is unavailable.
            cfg_entropy_nat = self._safe_shannon_entropy(xf_phase)
            entropy_basis = "phase_fallback"
            n_config_components = int(np.sum(xf_phase > 1e-12))
        cfg_entropy_j_molK = R * cfg_entropy_nat

        phase_entropy_vol = self._safe_shannon_entropy(vf)
        phase_entropy_wt = self._safe_shannon_entropy(wf)

        K_spread = float((K_voigt - K_reuss) / max(K_vrh, 1e-8))
        G_spread = float((G_voigt - G_reuss) / max(G_vrh, 1e-8))
        kappa_spread = float((kappa_wiener_hi - kappa_wiener_lo) / max(thermal_cond, 1e-8))

        return {
            "density_g_cm3": density,
            "bulk_modulus_GPa": K_vrh,
            "shear_modulus_GPa": G_vrh,
            "youngs_modulus_GPa": young,
            "poisson_ratio": poisson,
            "thermal_conductivity_W_mK": thermal_cond,
            "thermal_expansion_1e6_K": cte,
            "melting_point_K": melting,
            "hardness_GPa": hardness,
            "pugh_ratio": pugh,
            "bulk_modulus_voigt_GPa": K_voigt,
            "bulk_modulus_reuss_GPa": K_reuss,
            "bulk_modulus_spread": K_spread,
            "shear_modulus_spread": G_spread,
            "thermal_conductivity_wiener_lower_W_mK": kappa_wiener_lo,
            "thermal_conductivity_wiener_upper_W_mK": kappa_wiener_hi,
            "thermal_conductivity_spread": kappa_spread,
            "thermal_model_used": thermal_model_key,
            "melting_point_arithmetic_K": melting_arith,
            "mixing_entropy_J_molK": cfg_entropy_j_molK,
            "mixing_entropy_over_R": cfg_entropy_nat,
            "mixing_entropy_basis": entropy_basis,
            "n_config_components": n_config_components,
            "phase_entropy_volume_nat": phase_entropy_vol,
            "phase_entropy_weight_nat": phase_entropy_wt,
            "ductile": bool(pugh > 1.75),
        }

    def print_report(self, experimental: dict[str, float] | None = None):
        props = self.estimate_properties()
        print(f"Alloy Report: {self.name}")
        print(f"Density (g/cm3): {props['density_g_cm3']:.3f}")
        print(f"Bulk Modulus (GPa): {props['bulk_modulus_GPa']:.3f}")
        print(f"Shear Modulus (GPa): {props['shear_modulus_GPa']:.3f}")
        print(f"Young's Modulus (GPa): {props['youngs_modulus_GPa']:.3f}")
        print(f"Thermal Conductivity (W/mK): {props['thermal_conductivity_W_mK']:.3f}")
        print(f"Melting Point (K): {props['melting_point_K']:.3f}")

        if experimental:
            print("Experimental Comparison:")
            for k, v in experimental.items():
                if k in props and v != 0:
                    err = 100.0 * (props[k] - float(v)) / float(v)
                    print(f"{k}: pred={props[k]:.3f}, exp={float(v):.3f}, err={err:.2f}%")
