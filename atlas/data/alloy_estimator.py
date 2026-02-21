"""
Alloy property estimator.

Provides a compact, test-friendly API:
- ``AlloyEstimator.from_preset(...)``
- ``AlloyEstimator.custom(...)``
- ``AlloyEstimator.estimate_properties()``
- ``AlloyEstimator.print_report(...)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AlloyPhase:
    name: str
    formula: str
    weight_fraction: float
    properties: Dict[str, float]
    volume_fraction: float = field(default=0.0)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.properties.get(key, default))


class AlloyEstimator:
    def __init__(self, name: str, phases: List[AlloyPhase]):
        self.name = name
        self.phases = phases
        self._normalize_weight_fractions()
        self.convert_wt_to_vol(self.phases)

    @staticmethod
    def convert_wt_to_vol(phases: List[AlloyPhase]):
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
    def from_preset(cls, preset: str) -> "AlloyEstimator":
        key = preset.strip().lower()
        presets = {
            "sac305": cls._preset_sac305,
            "snpb63": cls._preset_snpb63,
            "pure_sn": cls._preset_pure_sn,
        }
        if key not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        return presets[key]()

    @classmethod
    def custom(cls, name: str, phases: List[Dict]) -> "AlloyEstimator":
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
    def _preset_sac305(cls) -> "AlloyEstimator":
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
    def _preset_snpb63(cls) -> "AlloyEstimator":
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
    def _preset_pure_sn(cls) -> "AlloyEstimator":
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

    @staticmethod
    def _safe_reuss(vf: np.ndarray, x: np.ndarray) -> float:
        x = np.maximum(x, 1e-8)
        return float(1.0 / np.sum(vf / x))

    def estimate_properties(self) -> Dict[str, float]:
        vf = np.array([p.volume_fraction for p in self.phases], dtype=float)
        wf = np.array([p.weight_fraction for p in self.phases], dtype=float)

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
        thermal_cond = float(np.exp(np.sum(vf * np.log(kappa))))

        cte = float(np.sum(vf * np.array([p.get("thermal_expansion_1e6_K") for p in self.phases], dtype=float)))
        # Use harmonic averaging for alloy melting point proxy.
        # This is empirically closer to solder liquidus behavior than arithmetic mean.
        tm = np.array([max(p.get("melting_point_K"), 1e-8) for p in self.phases], dtype=float)
        melting = float(1.0 / np.sum(wf / tm))

        pugh = float(K_vrh / max(G_vrh, 1e-8))
        hardness = float(max(0.0, 2.0 * ((G_vrh / max(K_vrh, 1e-8)) ** 2 * G_vrh) ** 0.585 - 3.0))

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
            "ductile": bool(pugh > 1.75),
        }

    def print_report(self, experimental: Optional[Dict[str, float]] = None):
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
