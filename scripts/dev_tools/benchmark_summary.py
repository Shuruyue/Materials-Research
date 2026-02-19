"""Quick Phase 1 results check."""
import json, os

from pathlib import Path
import sys
# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

results_dir = PROJECT_ROOT / "models"
props = ["formation_energy", "band_gap", "bulk_modulus", "shear_modulus"]

print("=" * 80)
print("  PHASE 1 CGCNN RESULTS SUMMARY")
print("=" * 80)
print(f"  {'Property':<20s} {'MAE':>8s} {'Unit':>8s} {'Target':>8s} {'Status':>8s} "
      f"{'R2':>6s} {'Epoch':>8s} {'N_train':>8s} {'Time':>8s}")
print("  " + "-" * 76)

for p in props:
    path = os.path.join(results_dir, f"cgcnn_full_{p}", "results.json")
    if not os.path.exists(path):
        print(f"  {p:<20s}  [NOT FOUND]")
        continue
    r = json.loads(Path(path).read_text())
    mae = r["test_metrics"].get(f"{p}_MAE", float("inf"))
    r2 = r["test_metrics"].get(f"{p}_R2", 0)
    target = r["target_mae"]
    unit = r.get("unit", "")
    passed = "[PASS]" if r["passed"] else "[FAIL]"
    best_ep = r["best_epoch"]
    total_ep = r["total_epochs"]
    n_train = r["n_train"]
    t_min = r.get("training_time_minutes", 0)
    
    # Guess unit from benchmarks
    units = {"formation_energy": "eV/atom", "band_gap": "eV", 
             "bulk_modulus": "GPa", "shear_modulus": "GPa"}
    u = units.get(p, "")
    
    print(f"  {p:<20s} {mae:>8.4f} {u:>8s} {target:>8.4f} {passed:>8s} "
          f"{r2:>6.3f} {best_ep:>4d}/{total_ep:<4d} {n_train:>7d} {t_min:>6.1f}m")

# Also check Phase 2 results if they exist
print()
print("=" * 80)
print("  PHASE 2 EQUIVARIANT GNN RESULTS (if any)")
print("=" * 80)
for p in props:
    path = os.path.join(results_dir, f"equivariant_{p}", "results.json")
    if not os.path.exists(path):
        continue
    r = json.loads(Path(path).read_text())
    mae = r["test_metrics"].get(f"{p}_MAE", float("inf"))
    r2 = r["test_metrics"].get(f"{p}_R2", 0)
    n_train = r["n_train"]
    total_ep = r["total_epochs"]
    t_min = r.get("training_time_minutes", 0)
    units = {"formation_energy": "eV/atom", "band_gap": "eV",
             "bulk_modulus": "GPa", "shear_modulus": "GPa"}
    u = units.get(p, "")
    note = "(quick test)" if n_train < 5000 else "(full)"
    print(f"  {p:<20s} MAE={mae:.4f} {u}, R2={r2:.3f}, "
          f"n={n_train}, epochs={total_ep}, time={t_min:.1f}m {note}")
