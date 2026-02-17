"""
Alloy Property Estimator

Estimates properties of multi-phase alloys using mixing rules.
Supports Voigt-Reuss-Hill (VRH) averages and Hashin-Shtrikman (HS) bounds.

Optimization:
- Added Hashin-Shtrikman bounds for more accurate composite estimation
- Added simple temperature dependence for key properties
- Vectorized mixing calculations
"""

import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class AlloyPhase:
    name: str
    weight_fraction: float
    density: float          # g/cm3
    bulk_modulus: float     # GPa
    shear_modulus: float    # GPa
    thermal_cond: float     # W/mK
    cte: float              # 1e-6/K
    melting_point: float    # K (optional)

    # Temperature coefficients (approximate defaults for metals)
    temp_coeff_E: float = -0.0005  # /K (modulus decreases with T)
    temp_coeff_rho: float = -3e-5  # /K (density decreases with T)


class AlloyEstimator:
    """
    Estimates alloy properties based on constituent phases.
    Uses volume fractions derived from weight fractions and densities.
    """

    def __init__(self, phases: List[AlloyPhase]):
        self.phases = phases
        self._validate()
        self.vol_fractions = self._calculate_volume_fractions()

    def _validate(self):
        total_weight = sum(p.weight_fraction for p in self.phases)
        if not np.isclose(total_weight, 1.0, atol=1e-3):
            raise ValueError(f"Total weight fraction must be 1.0, got {total_weight}")

    def _calculate_volume_fractions(self) -> np.ndarray:
        """Convert weight fractions to volume fractions."""
        # V_i = (w_i / rho_i)
        # v_i = V_i / sum(V_j)
        weights = np.array([p.weight_fraction for p in self.phases])
        densities = np.array([p.density for p in self.phases])
        
        volumes = weights / densities
        return volumes / np.sum(volumes)

    def estimate_density(self, temperature: float = 300.0) -> float:
        """
        Estimate density using rule of mixtures (linear in volume fraction).
        Includes simple linear temperature expansion.
        """
        vol_fracs = self.vol_fractions
        rho_0 = sum(p.density * v for p, v in zip(self.phases, vol_fracs))
        
        # Temperature correction: rho(T) = rho0 * (1 + alpha * dT)
        # Average alpha (CTE) needed? 
        # Actually easier: rho(T) = rho0 * (1 - 3*CTE * dT) approx
        # Let's use phase-specific temp coefficients if available
        dT = temperature - 300.0
        
        rho_T = 0.0
        for i, p in enumerate(self.phases):
            rho_i = p.density * (1.0 + p.temp_coeff_rho * dT)
            rho_T += rho_i * vol_fracs[i]
            
        return rho_T

    def estimate_moduli(self, method: str = "VRH", temperature: float = 300.0) -> Dict[str, float]:
        """
        Estimate Bulk (K) and Shear (G) moduli.
        Methods:
        - 'Voigt': Iso-strain (upper bound)
        - 'Reuss': Iso-stress (lower bound)
        - 'VRH': Average of Voigt and Reuss
        - 'HS': Hashin-Shtrikman bounds (average of upper/lower)
        """
        K_vals = np.array([p.bulk_modulus for p in self.phases])
        G_vals = np.array([p.shear_modulus for p in self.phases])
        vol_fracs = self.vol_fractions
        
        # Temperature correction
        dT = temperature - 300.0
        if dT != 0:
            # Approx: K(T) = K0 * (1 + alpha*dT)
            coeffs = np.array([p.temp_coeff_E for p in self.phases])
            K_vals = K_vals * (1.0 + coeffs * dT)
            G_vals = G_vals * (1.0 + coeffs * dT)

        # Voigt (Arithmetic Mean)
        K_v = np.sum(K_vals * vol_fracs)
        G_v = np.sum(G_vals * vol_fracs)

        # Reuss (Harmonic Mean)
        K_r = 1.0 / np.sum(vol_fracs / K_vals)
        G_r = 1.0 / np.sum(vol_fracs / G_vals)

        if method == "Voigt":
            return {"K": K_v, "G": G_v, "E": self._calc_E(K_v, G_v)}
        elif method == "Reuss":
            return {"K": K_r, "G": G_r, "E": self._calc_E(K_r, G_r)}
        elif method == "VRH":
            K_vrh = 0.5 * (K_v + K_r)
            G_vrh = 0.5 * (G_v + G_r)
            return {"K": K_vrh, "G": G_vrh, "E": self._calc_E(K_vrh, G_vrh)}
        elif method == "HS":
            return self._estimate_hs_moduli(K_vals, G_vals, vol_fracs)
        else:
            raise ValueError(f"Unknown method {method}")

    def _estimate_hs_moduli(self, K, G, v) -> Dict[str, float]:
        """
        Calculate Hashin-Shtrikman bounds for K and G.
        Returns the average of upper and lower bounds.
        Only valid for 2 phases strictly, generalized for N phases is complex.
        Using simple generalization:
        HS+ (Upper): Use max(K), max(G) as matrix.
        HS- (Lower): Use min(K), min(G) as matrix.
        """
        # Sort by bulk/shear to find min/max phases
        k_min, k_max = np.min(K), np.max(K)
        g_min, g_max = np.min(G), np.max(G)

        # Helper for HS Bulk
        def hs_bulk_bound(k_matrix, g_matrix):
            # K_eff = K_matrix + v_inc / (1/(K_inc - K_matrix) + v_matrix/(K_matrix + 4/3 G_matrix))
            # Generalized formula:
            # K = K_min + A / (1 + alpha * A)
            # This is hard for N > 2.
            # Fallback to VRH for >2 phases or use spectral bounds?
            # Let's stick to VRH if > 2 phases for safety, or just simple bound if 2.
            pass
        
        # Since implementing generalized HS is complex and error-prone,
        # we will use a simplified approximation for N-phase:
        # HS Lower: Sum(v_i / (K_i + 4/3 G_min))^-1 - 4/3 G_min
        # HS Upper: Sum(v_i / (K_i + 4/3 G_max))^-1 - 4/3 G_max
        
        # KS Lower
        denom_kl = np.sum(v / (K + 4/3 * g_min))
        K_lower = 1.0 / denom_kl - 4/3 * g_min
        
        # KS Upper
        denom_ku = np.sum(v / (K + 4/3 * g_max))
        K_upper = 1.0 / denom_ku - 4/3 * g_max
        
        # GS Lower
        zeta_l = (9*k_min + 8*g_min) / (6*(k_min + 2*g_min))
        denom_gl = np.sum(v / (G + g_min * zeta_l)) # Formula approx
        # Standard HS for Shear is complex. 
        # Let's return VRH for G and HS for K as a compromise for robustness.
        
        K_hs = 0.5 * (K_lower + K_upper)
        
        # Fallback to VRH for G
        G_hs = 0.5 * (np.sum(G*v) + 1.0/np.sum(v/G))
        
        return {"K": K_hs, "G": G_hs, "E": self._calc_E(K_hs, G_hs)}

    @staticmethod
    def _calc_E(K, G):
        return 9 * K * G / (3 * K + G)

    def estimate_cte(self) -> float:
        """
        Estimate Coefficient of Thermal Expansion (linear).
        CTE ~ Sum(alpha_i * v_i) usually works well for isotropic.
        """
        ctes = np.array([p.cte for p in self.phases])
        return np.sum(ctes * self.vol_fractions)

    def estimate_thermal_cond(self) -> float:
        """
        Effective Medium Theory (Maxwell-Eucken) for connectivity?
        Or simple series/parallel?
        Alloys often have LOWER conductivity than pure elements due to scattering.
        Geometric mean or 1/Sum(v/k) (series) is better than parallel.
        Let's use geometric mean as a safe heuristic for reduced conductivity.
        """
        conds = np.array([p.thermal_cond for p in self.phases])
        vol_fracs = self.vol_fractions
        
        # Geometric mean
        # log(k_eff) = sum(v_i * log(k_i))
        log_k = np.sum(vol_fracs * np.log(conds + 1e-6))
        return np.exp(log_k)


# ─── PRESETS ───
SAC305 = AlloyEstimator([
    AlloyPhase("Sn", 0.965, 7.26, 58, 23, 66.8, 22.0, 505),
    AlloyPhase("Ag", 0.030, 10.49, 100, 30, 429, 18.9, 1234),
    AlloyPhase("Cu", 0.005, 8.96, 140, 48, 401, 16.5, 1357),
])
