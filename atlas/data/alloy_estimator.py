"""
atlas.data.alloy_estimator — Multi-phase alloy property estimation.

Estimates effective properties of multi-phase alloys from constituent
phase data using standard mixing rules (Voigt-Reuss-Hill, rule of
mixtures, weighted averages).

Example:
    >>> from atlas.data.alloy_estimator import AlloyEstimator
    >>> est = AlloyEstimator.from_preset("SAC305")
    >>> props = est.estimate_properties()
    >>> print(props)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── Phase data ────────────────────────────────────────────────────

@dataclass
class AlloyPhase:
    """A single phase in a multi-phase alloy.

    Attributes:
        name: Human-readable name (e.g., "β-Sn")
        formula: Chemical formula (e.g., "Sn")
        volume_fraction: Volume fraction (0–1). If only weight fractions
            are known, call AlloyEstimator.convert_wt_to_vol().
        properties: Dict of phase properties. Expected keys include:
            - bulk_modulus_GPa: Bulk modulus K (GPa)
            - shear_modulus_GPa: Shear modulus G (GPa)
            - density_g_cm3: Density ρ (g/cm³)
            - melting_point_K: Melting point Tm (K)
            - thermal_conductivity_W_mK: κ (W/m·K)
            - thermal_expansion_1e6_K: CTE α (×10⁻⁶/K)
            - hardness_GPa: Vickers hardness Hv (GPa)
            - youngs_modulus_GPa: Young's modulus E (GPa)
    """
    name: str
    formula: str
    volume_fraction: float = 0.0
    weight_fraction: float = 0.0
    properties: dict = field(default_factory=dict)

    def get(self, key: str, default: float = float("nan")) -> float:
        """Get a property value with fallback."""
        val = self.properties.get(key, default)
        if val is None:
            return default
        return float(val)


# ── Mixing rules ──────────────────────────────────────────────────

def _voigt(phases: list[AlloyPhase], key: str) -> float:
    """Voigt (upper bound): linear volume-weighted average."""
    total = 0.0
    for p in phases:
        v = p.get(key)
        if math.isnan(v):
            return float("nan")
        total += p.volume_fraction * v
    return total


def _reuss(phases: list[AlloyPhase], key: str) -> float:
    """Reuss (lower bound): harmonic volume-weighted average."""
    total = 0.0
    for p in phases:
        v = p.get(key)
        if math.isnan(v) or v == 0:
            return float("nan")
        total += p.volume_fraction / v
    return 1.0 / total if total > 0 else float("nan")


def _vrh(phases: list[AlloyPhase], key: str) -> float:
    """Voigt-Reuss-Hill (VRH) average — best general estimate."""
    voigt = _voigt(phases, key)
    reuss = _reuss(phases, key)
    if math.isnan(voigt) or math.isnan(reuss):
        return float("nan")
    return 0.5 * (voigt + reuss)


def _weighted_avg(phases: list[AlloyPhase], key: str,
                  by: str = "volume") -> float:
    """Simple weighted average by volume or weight fraction."""
    total = 0.0
    for p in phases:
        v = p.get(key)
        if math.isnan(v):
            return float("nan")
        frac = p.volume_fraction if by == "volume" else p.weight_fraction
        total += frac * v
    return total


# ── AlloyEstimator ────────────────────────────────────────────────

class AlloyEstimator:
    """Estimate effective properties of a multi-phase alloy.

    Usage:
        est = AlloyEstimator.from_preset("SAC305")
        props = est.estimate_properties()
        est.print_report()
    """

    def __init__(self, name: str, phases: list[AlloyPhase]):
        self.name = name
        self.phases = phases
        self._normalize_fractions()

    def _normalize_fractions(self):
        """Ensure volume fractions sum to 1."""
        total_vol = sum(p.volume_fraction for p in self.phases)
        total_wt = sum(p.weight_fraction for p in self.phases)

        if total_vol > 0 and abs(total_vol - 1.0) > 0.01:
            for p in self.phases:
                p.volume_fraction /= total_vol

        if total_wt > 0 and abs(total_wt - 1.0) > 0.01:
            for p in self.phases:
                p.weight_fraction /= total_wt

    @classmethod
    def convert_wt_to_vol(cls, phases: list[AlloyPhase]) -> list[AlloyPhase]:
        """Convert weight fractions to volume fractions using densities."""
        vol_parts = []
        for p in phases:
            rho = p.get("density_g_cm3")
            if math.isnan(rho) or rho <= 0:
                raise ValueError(f"Phase {p.name} missing density for wt→vol conversion")
            vol_parts.append(p.weight_fraction / rho)

        total_vol = sum(vol_parts)
        for i, p in enumerate(phases):
            p.volume_fraction = vol_parts[i] / total_vol

        return phases

    def estimate_properties(self) -> dict:
        """Compute all effective alloy properties.

        Returns:
            Dict with estimated property values and bounds.
        """
        phases = self.phases
        result = {}

        # ── Density: simple volume-weighted ──
        result["density_g_cm3"] = _weighted_avg(phases, "density_g_cm3", by="volume")

        # ── Elastic moduli: VRH ──
        result["bulk_modulus_GPa"] = _vrh(phases, "bulk_modulus_GPa")
        result["bulk_modulus_voigt_GPa"] = _voigt(phases, "bulk_modulus_GPa")
        result["bulk_modulus_reuss_GPa"] = _reuss(phases, "bulk_modulus_GPa")

        result["shear_modulus_GPa"] = _vrh(phases, "shear_modulus_GPa")
        result["shear_modulus_voigt_GPa"] = _voigt(phases, "shear_modulus_GPa")
        result["shear_modulus_reuss_GPa"] = _reuss(phases, "shear_modulus_GPa")

        # ── Young's modulus: VRH or from K,G ──
        E_vrh = _vrh(phases, "youngs_modulus_GPa")
        if math.isnan(E_vrh):
            K = result["bulk_modulus_GPa"]
            G = result["shear_modulus_GPa"]
            if not math.isnan(K) and not math.isnan(G) and G > 0:
                E_vrh = 9 * K * G / (3 * K + G)
        result["youngs_modulus_GPa"] = E_vrh

        # ── Poisson's ratio from K, G ──
        K = result["bulk_modulus_GPa"]
        G = result["shear_modulus_GPa"]
        if not math.isnan(K) and not math.isnan(G) and G > 0:
            result["poisson_ratio"] = (3 * K - 2 * G) / (6 * K + 2 * G)
        else:
            result["poisson_ratio"] = float("nan")

        # ── Thermal conductivity: VRH-like ──
        result["thermal_conductivity_W_mK"] = _vrh(phases, "thermal_conductivity_W_mK")

        # ── CTE: volume-weighted ──
        result["thermal_expansion_1e6_K"] = _weighted_avg(
            phases, "thermal_expansion_1e6_K", by="volume"
        )

        # ── Melting point: weighted + solidus/liquidus ──
        tms = [(p.get("melting_point_K"), p.weight_fraction) for p in phases]
        tms = [(t, w) for t, w in tms if not math.isnan(t)]
        if tms:
            result["melting_point_K"] = sum(t * w for t, w in tms) / sum(w for _, w in tms)
            result["solidus_K"] = min(t for t, _ in tms)
            result["liquidus_K"] = max(t for t, _ in tms)
        else:
            result["melting_point_K"] = float("nan")

        # ── Hardness: weight-weighted ──
        result["hardness_GPa"] = _weighted_avg(phases, "hardness_GPa", by="weight")

        # ── Pugh ratio (ductility indicator): K/G ──
        if not math.isnan(K) and not math.isnan(G) and G > 0:
            result["pugh_ratio"] = K / G
            result["ductile"] = result["pugh_ratio"] > 1.75
        else:
            result["pugh_ratio"] = float("nan")
            result["ductile"] = None

        return result

    def print_report(self, experimental: dict | None = None):
        """Print a formatted property report.

        Args:
            experimental: Optional dict of experimental values for comparison.
        """
        props = self.estimate_properties()

        print(f"\n{'═' * 70}")
        print(f"  Alloy Property Report: {self.name}")
        print(f"{'═' * 70}")

        print(f"\n  Phases:")
        for p in self.phases:
            print(f"    {p.name:15s} ({p.formula:10s})  "
                  f"vol={p.volume_fraction*100:5.1f}%  "
                  f"wt={p.weight_fraction*100:5.1f}%")

        print(f"\n  {'Property':35s} {'Estimated':>12s}", end="")
        if experimental:
            print(f" {'Experimental':>12s} {'Error':>8s}", end="")
        print()
        print(f"  {'─' * 35} {'─' * 12}", end="")
        if experimental:
            print(f" {'─' * 12} {'─' * 8}", end="")
        print()

        display = [
            ("density_g_cm3", "Density (g/cm³)", ".2f"),
            ("bulk_modulus_GPa", "Bulk modulus K (GPa)", ".1f"),
            ("shear_modulus_GPa", "Shear modulus G (GPa)", ".1f"),
            ("youngs_modulus_GPa", "Young's modulus E (GPa)", ".1f"),
            ("poisson_ratio", "Poisson's ratio ν", ".3f"),
            ("thermal_conductivity_W_mK", "Thermal conductivity κ (W/m·K)", ".1f"),
            ("thermal_expansion_1e6_K", "CTE α (×10⁻⁶/K)", ".1f"),
            ("melting_point_K", "Melting point Tm (K)", ".0f"),
            ("solidus_K", "Solidus (K)", ".0f"),
            ("liquidus_K", "Liquidus (K)", ".0f"),
            ("hardness_GPa", "Hardness Hv (GPa)", ".2f"),
            ("pugh_ratio", "Pugh ratio K/G", ".2f"),
        ]

        for key, label, fmt in display:
            val = props.get(key, float("nan"))
            if math.isnan(val):
                continue
            val_str = f"{val:{fmt}}"
            print(f"  {label:35s} {val_str:>12s}", end="")

            if experimental and key in experimental:
                exp_val = experimental[key]
                exp_str = f"{exp_val:{fmt}}"
                err = abs(val - exp_val) / exp_val * 100 if exp_val != 0 else 0
                print(f" {exp_str:>12s} {err:>7.1f}%", end="")
            print()

        if props.get("ductile") is not None:
            d_str = "Yes (K/G > 1.75)" if props["ductile"] else "No (K/G < 1.75)"
            print(f"\n  Ductile? {d_str}")

        print(f"\n{'═' * 70}")

    # ── Presets ───────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str) -> "AlloyEstimator":
        """Create estimator from a built-in alloy preset.

        Available presets:
            SAC305, SAC405, SnPb63, pure_Sn, pure_Cu
        """
        presets = cls._get_presets()
        key = name.upper().replace("-", "").replace(" ", "").replace("_", "")
        if key not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        return presets[key]

    @classmethod
    def _get_presets(cls) -> dict[str, "AlloyEstimator"]:
        """Built-in alloy definitions with literature DFT/experimental data."""

        # ── Phase property database (DFT + literature) ────────
        # Sources: JARVIS-DFT, Materials Project, ASM handbook

        beta_sn = AlloyPhase(
            name="β-Sn", formula="Sn",
            volume_fraction=0.0, weight_fraction=0.965,
            properties={
                "bulk_modulus_GPa": 56.3,
                "shear_modulus_GPa": 18.4,
                "youngs_modulus_GPa": 50.0,
                "density_g_cm3": 7.29,
                "melting_point_K": 505.0,
                "thermal_conductivity_W_mK": 66.8,
                "thermal_expansion_1e6_K": 22.0,
                "hardness_GPa": 0.12,
            },
        )

        ag3sn = AlloyPhase(
            name="Ag₃Sn (ε)", formula="Ag3Sn",
            volume_fraction=0.0, weight_fraction=0.038,
            properties={
                "bulk_modulus_GPa": 95.0,
                "shear_modulus_GPa": 32.0,
                "youngs_modulus_GPa": 85.0,
                "density_g_cm3": 9.79,
                "melting_point_K": 753.0,
                "thermal_conductivity_W_mK": 50.0,
                "thermal_expansion_1e6_K": 18.0,
                "hardness_GPa": 2.5,
            },
        )

        cu6sn5 = AlloyPhase(
            name="Cu₆Sn₅ (η)", formula="Cu6Sn5",
            volume_fraction=0.0, weight_fraction=0.012,
            properties={
                "bulk_modulus_GPa": 112.0,
                "shear_modulus_GPa": 42.0,
                "youngs_modulus_GPa": 110.0,
                "density_g_cm3": 8.28,
                "melting_point_K": 688.0,
                "thermal_conductivity_W_mK": 34.1,
                "thermal_expansion_1e6_K": 16.3,
                "hardness_GPa": 6.2,
            },
        )

        cu3sn = AlloyPhase(
            name="Cu₃Sn (ε)", formula="Cu3Sn",
            volume_fraction=0.0, weight_fraction=0.0,
            properties={
                "bulk_modulus_GPa": 130.0,
                "shear_modulus_GPa": 48.0,
                "youngs_modulus_GPa": 130.0,
                "density_g_cm3": 8.90,
                "melting_point_K": 949.0,
                "thermal_conductivity_W_mK": 70.0,
                "thermal_expansion_1e6_K": 19.0,
                "hardness_GPa": 5.5,
            },
        )

        pb = AlloyPhase(
            name="Pb", formula="Pb",
            volume_fraction=0.0, weight_fraction=0.0,
            properties={
                "bulk_modulus_GPa": 45.8,
                "shear_modulus_GPa": 5.6,
                "youngs_modulus_GPa": 16.0,
                "density_g_cm3": 11.34,
                "melting_point_K": 600.0,
                "thermal_conductivity_W_mK": 35.3,
                "thermal_expansion_1e6_K": 28.9,
                "hardness_GPa": 0.044,
            },
        )

        pure_cu = AlloyPhase(
            name="Cu", formula="Cu",
            volume_fraction=1.0, weight_fraction=1.0,
            properties={
                "bulk_modulus_GPa": 137.0,
                "shear_modulus_GPa": 48.3,
                "youngs_modulus_GPa": 130.0,
                "density_g_cm3": 8.96,
                "melting_point_K": 1358.0,
                "thermal_conductivity_W_mK": 401.0,
                "thermal_expansion_1e6_K": 16.5,
                "hardness_GPa": 0.87,
            },
        )

        # ── SAC305: Sn96.5-Ag3.0-Cu0.5 ──
        sac305_phases = [
            AlloyPhase(
                name="β-Sn", formula="Sn",
                weight_fraction=0.965,
                properties=beta_sn.properties,
            ),
            AlloyPhase(
                name="Ag₃Sn", formula="Ag3Sn",
                weight_fraction=0.025,   # ~3.0wt% Ag → ~2.5wt% Ag₃Sn phase
                properties=ag3sn.properties,
            ),
            AlloyPhase(
                name="Cu₆Sn₅", formula="Cu6Sn5",
                weight_fraction=0.010,   # ~0.5wt% Cu → ~1.0wt% Cu₆Sn₅ phase
                properties=cu6sn5.properties,
            ),
        ]
        cls.convert_wt_to_vol(sac305_phases)

        # ── SAC405: Sn95.5-Ag4.0-Cu0.5 ──
        sac405_phases = [
            AlloyPhase(
                name="β-Sn", formula="Sn",
                weight_fraction=0.955,
                properties=beta_sn.properties,
            ),
            AlloyPhase(
                name="Ag₃Sn", formula="Ag3Sn",
                weight_fraction=0.035,
                properties=ag3sn.properties,
            ),
            AlloyPhase(
                name="Cu₆Sn₅", formula="Cu6Sn5",
                weight_fraction=0.010,
                properties=cu6sn5.properties,
            ),
        ]
        cls.convert_wt_to_vol(sac405_phases)

        # ── SnPb63: Sn63-Pb37 (eutectic) ──
        snpb_phases = [
            AlloyPhase(
                name="β-Sn", formula="Sn",
                weight_fraction=0.63,
                properties=beta_sn.properties,
            ),
            AlloyPhase(
                name="Pb", formula="Pb",
                weight_fraction=0.37,
                properties=pb.properties,
            ),
        ]
        cls.convert_wt_to_vol(snpb_phases)

        return {
            "SAC305": cls("SAC305 (Sn96.5-Ag3.0-Cu0.5)", sac305_phases),
            "SAC405": cls("SAC405 (Sn95.5-Ag4.0-Cu0.5)", sac405_phases),
            "SNPB63": cls("SnPb63 (Sn63-Pb37 eutectic)", snpb_phases),
            "PURESN": cls("Pure Sn", [AlloyPhase(
                "β-Sn", "Sn", 1.0, 1.0, beta_sn.properties)]),
            "PURECU": cls("Pure Cu", [pure_cu]),
        }

    @classmethod
    def custom(cls, name: str, phases: list[dict]) -> "AlloyEstimator":
        """Create from a list of dicts.

        Args:
            name: Alloy name
            phases: List of dicts with keys:
                name, formula, weight_fraction, properties

        Example:
            est = AlloyEstimator.custom("MyAlloy", [
                {"name": "Phase A", "formula": "Cu",
                 "weight_fraction": 0.8,
                 "properties": {"bulk_modulus_GPa": 137, "density_g_cm3": 8.96, ...}},
                {"name": "Phase B", "formula": "Sn",
                 "weight_fraction": 0.2,
                 "properties": {"bulk_modulus_GPa": 56.3, "density_g_cm3": 7.29, ...}},
            ])
        """
        alloy_phases = []
        for d in phases:
            alloy_phases.append(AlloyPhase(
                name=d["name"],
                formula=d["formula"],
                volume_fraction=d.get("volume_fraction", 0.0),
                weight_fraction=d.get("weight_fraction", 0.0),
                properties=d.get("properties", {}),
            ))

        # Convert wt → vol if volume fractions not set
        if all(p.volume_fraction == 0 for p in alloy_phases):
            cls.convert_wt_to_vol(alloy_phases)

        return cls(name, alloy_phases)
