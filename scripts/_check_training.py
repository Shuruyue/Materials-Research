"""Quick training health check."""
import json, torch
from pathlib import Path

model_dir = Path("models/equivariant_shear_modulus")

# Check if history exists
h_path = model_dir / "history.json"
if not h_path.exists():
    print("No history.json yet — training may still be loading data.")
    exit()

d = json.loads(h_path.read_text())
n = len(d["train_loss"])
print(f"=== Training Health Check ===")
print(f"Epochs completed: {n}")

if n == 0:
    print("No epochs completed yet.")
    exit()

# Best checkpoint
p = model_dir / "best.pt"
if p.exists():
    ckpt = torch.load(p, weights_only=False)
    print(f"Best checkpoint: epoch={ckpt['epoch']}, val_mae={ckpt['val_mae']:.4f}")

p_ema = model_dir / "best_ema.pt"
if p_ema.exists():
    ckpt_ema = torch.load(p_ema, weights_only=False)
    print(f"Best EMA checkpoint: epoch={ckpt_ema['epoch']}, val_mae={ckpt_ema['val_mae']:.4f}")

# History analysis
losses = d["train_loss"]
val_maes = d["val_mae"]
ema_maes = [x for x in d["ema_val_mae"] if x < 1e10]
lrs = d["lr"]

print(f"\n--- Loss Trend ---")
print(f"  First 5:  {[round(x,4) for x in losses[:5]]}")
print(f"  Last 10:  {[round(x,4) for x in losses[-10:]]}")
print(f"  Ratio (first/last): {losses[0]/losses[-1]:.1f}x improvement")

print(f"\n--- Val MAE Trend ---")
print(f"  First 5:  {[round(x,4) for x in val_maes[:5]]}")
print(f"  Last 10:  {[round(x,4) for x in val_maes[-10:]]}")
print(f"  Best: {min(val_maes):.4f} at epoch {val_maes.index(min(val_maes))+1}")

if ema_maes:
    print(f"\n--- EMA Val MAE ---")
    print(f"  Last 10:  {[round(x,4) for x in d['ema_val_mae'][-10:]]}")
    print(f"  Best: {min(ema_maes):.4f}")

print(f"\n--- Learning Rate ---")
print(f"  Current: {lrs[-1]:.2e}")
print(f"  Initial: {lrs[0]:.2e}")
n_reductions = sum(1 for i in range(1,len(lrs)) if lrs[i] < lrs[i-1])
print(f"  LR reductions: {n_reductions}")

# Health indicators
print(f"\n=== Health Assessment ===")
# 1. Is loss decreasing?
if n >= 10:
    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    if late_avg < early_avg * 0.5:
        print("  ✅ Loss consistently decreasing")
    elif late_avg < early_avg:
        print("  ⚠️ Loss decreasing but slowly — may need more time")
    else:
        print("  ❌ Loss not decreasing — potential problem")

# 2. Val-train gap (overfitting check)
if n >= 10:
    last_loss = losses[-1]
    last_val = val_maes[-1]
    # rough check: if val >> loss, overfitting
    if last_val > last_loss * 5:
        print(f"  ⚠️ Large val/loss gap ({last_val:.3f} vs {last_loss:.3f}) — watch for overfitting")
    else:
        print(f"  ✅ Val/loss ratio healthy ({last_val:.3f} / {last_loss:.3f})")

# 3. NaN check
nan_count = sum(1 for x in losses if x != x)
if nan_count > 0:
    print(f"  ❌ {nan_count} NaN epochs detected")
else:
    print(f"  ✅ No NaN epochs")

# 4. EMA benefit
if ema_maes and len(val_maes) > 0:
    best_reg = min(val_maes)
    best_ema = min(ema_maes)
    if best_ema < best_reg:
        pct = (best_reg - best_ema) / best_reg * 100
        print(f"  ✅ EMA is helping: {best_ema:.4f} vs {best_reg:.4f} ({pct:.1f}% better)")
    else:
        print(f"  ℹ️ EMA not better yet (reg={best_reg:.4f}, ema={best_ema:.4f})")

# 5. Convergence estimate
if n >= 20:
    recent_best = min(val_maes[-20:])
    overall_best = min(val_maes)
    if recent_best <= overall_best * 1.001:
        print(f"  🔄 Still actively improving in last 20 epochs")
    else:
        stale_epochs = n - val_maes.index(min(val_maes)) - 1
        print(f"  ⏳ Plateau: best was {stale_epochs} epochs ago")

print(f"\n  Target: shear_modulus MAE < 7.5 GPa")
print(f"  Current best: {min(val_maes):.4f} GPa")
gap = min(val_maes) - 7.5
if gap <= 0:
    print(f"  🎉 Already below target!")
else:
    print(f"  Gap to target: {gap:.2f} GPa")
