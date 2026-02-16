#!/usr/bin/env python3
"""Quick search: automotive thermal module metals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.data.jarvis_client import JARVISClient
from atlas.data.property_estimator import PropertyEstimator
import pandas as pd

client = JARVISClient()
est = PropertyEstimator()
raw = client.load_dft_3d()
df = est.extract_all_properties(raw)

# Stable metals with thermal conductivity data
mask = (
    (df["conductivity_class"] == "metal")
    & (df["ehull"] <= 0.05)
    & (df["kappa_slack"].notna())
    & (df["kappa_slack"] > 5)
)
sub = df[mask].sort_values("kappa_slack", ascending=False)

cols = ["formula", "kappa_slack", "bulk_modulus_kv", "melting_point_est",
        "thermal_expansion_est", "density", "hardness_chen", "pugh_ratio"]

print("=" * 90)
print("  TOP 25 STABLE METALS BY THERMAL CONDUCTIVITY (Slack Model)")
print("  For automotive thermal module applications")
print("=" * 90)
print(f"{'#':>3} {'Formula':>12} {'κ(W/mK)':>9} {'K(GPa)':>8} {'Tm(K)':>7} "
      f"{'CTE(ppm)':>9} {'ρ(g/cc)':>8} {'H(GPa)':>8} {'K/G':>6}")
print("-" * 90)

for i, (_, r) in enumerate(sub[cols].head(25).iterrows()):
    f = r["formula"]
    k = r["kappa_slack"]
    bm = r["bulk_modulus_kv"]
    tm = r["melting_point_est"]
    cte = r["thermal_expansion_est"]
    rho = r["density"]
    h = r["hardness_chen"]
    pg = r["pugh_ratio"]
    print(f"{i+1:3d} {f:>12s} {k:9.1f} {bm:8.0f} {tm:7.0f} "
          f"{cte:9.1f} {rho:8.2f} {h:8.1f} {pg:6.2f}")

# Cu-based
print("\n" + "=" * 90)
print("  Cu-BASED COMPOUNDS")
print("=" * 90)
mask_cu = mask & df["formula"].str.contains("Cu")
sub_cu = df[mask_cu].sort_values("kappa_slack", ascending=False)
for i, (_, r) in enumerate(sub_cu[cols].head(10).iterrows()):
    f = r["formula"]
    k = r["kappa_slack"]
    bm = r["bulk_modulus_kv"]
    tm = r["melting_point_est"]
    cte = r["thermal_expansion_est"]
    rho = r["density"]
    print(f"{i+1:3d} {f:>15s} κ={k:7.1f} K={bm:6.0f} Tm={tm:6.0f} CTE={cte:5.1f} ρ={rho:5.2f}")

# Al-based
print("\n  Al-BASED COMPOUNDS")
mask_al = mask & df["formula"].str.contains("Al")
sub_al = df[mask_al].sort_values("kappa_slack", ascending=False)
for i, (_, r) in enumerate(sub_al[cols].head(10).iterrows()):
    f = r["formula"]
    k = r["kappa_slack"]
    bm = r["bulk_modulus_kv"]
    tm = r["melting_point_est"]
    cte = r["thermal_expansion_est"]
    rho = r["density"]
    print(f"{i+1:3d} {f:>15s} κ={k:7.1f} K={bm:6.0f} Tm={tm:6.0f} CTE={cte:5.1f} ρ={rho:5.2f}")

# W/Mo based (low CTE)
print("\n  W/Mo-BASED COMPOUNDS (LOW CTE)")
mask_wmo = mask & (df["formula"].str.contains("W") | df["formula"].str.contains("Mo"))
sub_wmo = df[mask_wmo].sort_values("kappa_slack", ascending=False)
for i, (_, r) in enumerate(sub_wmo[cols].head(10).iterrows()):
    f = r["formula"]
    k = r["kappa_slack"]
    bm = r["bulk_modulus_kv"]
    tm = r["melting_point_est"]
    cte = r["thermal_expansion_est"]
    rho = r["density"]
    print(f"{i+1:3d} {f:>15s} κ={k:7.1f} K={bm:6.0f} Tm={tm:6.0f} CTE={cte:5.1f} ρ={rho:5.2f}")
