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
from functools import lru_cache

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
        value = self.properties.get(key, default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(parsed):
            return float(default)
        return parsed


class AlloyEstimator:
    def __init__(self, name: str, phases: list[AlloyPhase]):
        if not phases:
            raise ValueError("At least one phase is required")
        self.name = name
        self.phases = phases
        self._normalize_weight_fractions()
        self.convert_wt_to_vol(self.phases)

    @staticmethod
    def _normalize_preset_key(preset: str) -> str:
        return preset.strip().lower().replace("-", "").replace("_", "").replace(" ", "")

    @staticmethod
    def _coerce_finite_real(
        value: float,
        *,
        field_name: str,
        allow_non_finite: bool = False,
    ) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be a finite real number, got bool")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a finite real number") from exc
        if not allow_non_finite and not np.isfinite(parsed):
            raise ValueError(f"{field_name} must be finite")
        return float(parsed)

    @staticmethod
    def _coerce_nonnegative_finite(value: float, *, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(parsed):
            return float(default)
        return max(parsed, 0.0)

    @staticmethod
    def _coerce_positive_finite(
        value: float,
        *,
        default: float = 1e-8,
        floor: float = 1e-8,
    ) -> float:
        nonnegative = AlloyEstimator._coerce_nonnegative_finite(value, default=default)
        return max(nonnegative, float(floor))

    @staticmethod
    def _normalize_nonnegative_fractions(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
        total = float(np.sum(arr))
        if arr.size == 0:
            return arr
        if total <= 0:
            return np.ones(arr.size, dtype=float) / float(arr.size)
        return arr / total

    @classmethod
    def available_presets(cls) -> tuple[str, ...]:
        return ("SAC305", "SAC405", "SnPb63", "pure_Sn", "pure_Cu")

    @classmethod
    def convert_wt_to_vol(cls, phases: list[AlloyPhase]):
        if not phases:
            return
        weights = np.array(
            [cls._coerce_nonnegative_finite(p.weight_fraction) for p in phases],
            dtype=float,
        )
        rho = np.array(
            [cls._coerce_positive_finite(p.get("density_g_cm3", 1e-6), default=1e-6, floor=1e-6) for p in phases],
            dtype=float,
        )
        vol = weights / rho
        frac = cls._normalize_nonnegative_fractions(vol)
        for p, vf in zip(phases, frac):
            p.volume_fraction = float(vf)

    def _normalize_weight_fractions(self):
        weights = np.array(
            [self._coerce_nonnegative_finite(p.weight_fraction) for p in self.phases],
            dtype=float,
        )
        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("Total weight fraction must be > 0")
        normalized = weights / total
        for p, wf in zip(self.phases, normalized):
            p.weight_fraction = float(wf)

    @classmethod
    def from_preset(cls, preset: str) -> AlloyEstimator:
        if not isinstance(preset, str):
            raise ValueError("preset must be a non-empty string")
        key = cls._normalize_preset_key(preset)
        if not key:
            raise ValueError("preset must be a non-empty string")
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
        if not isinstance(phases, list) or not phases:
            raise ValueError("phases must be a non-empty list")
        parsed = []
        for i, ph in enumerate(phases):
            if not isinstance(ph, dict):
                raise ValueError(f"phases[{i}] must be a mapping")
            phase_name = str(ph.get("name", "")).strip()
            if not phase_name:
                raise ValueError(f"phases[{i}].name must be non-empty")
            formula = str(ph.get("formula", phase_name)).strip()
            if not formula:
                raise ValueError(f"phases[{i}].formula must be non-empty")
            weight_fraction = cls._coerce_finite_real(
                ph.get("weight_fraction"),
                field_name=f"phases[{i}].weight_fraction",
                allow_non_finite=True,
            )
            raw_properties = ph.get("properties", {})
            if not isinstance(raw_properties, dict):
                raise ValueError(f"phases[{i}].properties must be a mapping")
            properties: dict[str, float] = {}
            for key, value in raw_properties.items():
                properties[str(key)] = cls._coerce_finite_real(
                    value,
                    field_name=f"phases[{i}].properties[{key!r}]",
                )
            parsed.append(
                AlloyPhase(
                    name=phase_name,
                    formula=formula,
                    weight_fraction=weight_fraction,
                    properties=properties,
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
        frac = np.asarray(vf, dtype=float)
        vals = np.asarray(x, dtype=float)
        valid = np.isfinite(frac) & np.isfinite(vals) & (frac > 0)
        if not np.any(valid):
            return 0.0
        frac = AlloyEstimator._normalize_nonnegative_fractions(frac[valid])
        vals = np.maximum(vals[valid], 1e-8)
        denom = float(np.sum(frac / vals))
        if not np.isfinite(denom) or denom <= 0:
            return float(np.mean(vals))
        return float(1.0 / denom)

    @staticmethod
    def _wiener_bounds(frac: np.ndarray, values: np.ndarray) -> tuple[float, float]:
        """
        Wiener bounds for effective scalar properties.

        Reference:
        - Wiener, O. (1912), relation between microstructure and effective transport.
          (Arithmetic/parallel upper bound, harmonic/series lower bound)
        """
        weights = np.asarray(frac, dtype=float)
        vals = np.asarray(values, dtype=float)
        valid = np.isfinite(weights) & np.isfinite(vals) & (weights > 0)
        if not np.any(valid):
            return 0.0, 0.0
        weights = AlloyEstimator._normalize_nonnegative_fractions(weights[valid])
        vals = np.maximum(vals[valid], 1e-8)
        upper = float(np.sum(weights * vals))
        denom = float(np.sum(weights / vals))
        if not np.isfinite(denom) or denom <= 0:
            lower = float(np.min(vals))
        else:
            lower = float(1.0 / denom)
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    @staticmethod
    def _safe_shannon_entropy(frac: np.ndarray) -> float:
        values = np.asarray(frac, dtype=float)
        values = values[np.isfinite(values) & (values > 1e-12)]
        if values.size == 0:
            return 0.0
        values = values / max(float(np.sum(values)), 1e-12)
        return float(-np.sum(values * np.log(values)))

    @classmethod
    @lru_cache(maxsize=256)
    def _element_atomic_mass(cls, symbol: str) -> float | None:
        raw_symbol = str(symbol).strip()
        if not raw_symbol:
            return None
        symbol_norm = raw_symbol[0].upper() + raw_symbol[1:].lower()
        try:
            from pymatgen.core import Element

            m = float(Element(symbol_norm).atomic_mass)
            if np.isfinite(m) and m > 0:
                return m
        except Exception:
            pass
        fallback = _COMMON_ELEMENT_MOLAR_MASS.get(symbol_norm.upper())
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
    @lru_cache(maxsize=512)
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
        if not phases:
            return np.array([], dtype=float)
        wf = np.array(
            [cls._coerce_nonnegative_finite(p.weight_fraction) for p in phases],
            dtype=float,
        )
        wf = cls._normalize_nonnegative_fractions(wf)

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
        wf = np.array(
            [AlloyEstimator._coerce_nonnegative_finite(p.weight_fraction) for p in phases],
            dtype=float,
        )
        wf = AlloyEstimator._normalize_nonnegative_fractions(wf)
        if wf.size == 0:
            return None

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
        vf = self._normalize_nonnegative_fractions(
            np.array([self._coerce_nonnegative_finite(p.volume_fraction) for p in self.phases], dtype=float)
        )
        wf = self._normalize_nonnegative_fractions(
            np.array([self._coerce_nonnegative_finite(p.weight_fraction) for p in self.phases], dtype=float)
        )
        xf_phase = self._estimate_mole_fractions(self.phases)
        xf_element = self._estimate_element_mole_fractions(self.phases)

        density = float(
            np.sum(
                vf
                * np.array(
                    [self._coerce_positive_finite(p.get("density_g_cm3"), default=1e-6, floor=1e-6) for p in self.phases],
                    dtype=float,
                )
            )
        )
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

        kappa = np.array(
            [self._coerce_positive_finite(p.get("thermal_conductivity_W_mK"), default=1e-8, floor=1e-8) for p in self.phases],
            dtype=float,
        )
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
        tm = np.array(
            [self._coerce_positive_finite(p.get("melting_point_K"), default=1e-8, floor=1e-8) for p in self.phases],
            dtype=float,
        )
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
                exp_v = self._coerce_nonnegative_finite(v, default=np.nan)
                if k in props and np.isfinite(exp_v) and exp_v > 0:
                    err = 100.0 * (props[k] - exp_v) / exp_v
                    print(f"{k}: pred={props[k]:.3f}, exp={exp_v:.3f}, err={err:.2f}%")
