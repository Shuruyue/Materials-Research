#!/usr/bin/env python3
"""
Inspect Phase 2 run artifacts without re-running evaluation.

Shows:
- run metadata (manifest/args/git)
- checkpoint summary (epoch, best metric, parameter count)
- stored metrics (results.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"
FAMILIES = (
    "multitask_lite_e3nn",
    "multitask_std_e3nn",
    "multitask_pro_e3nn",
    "multitask_cgcnn",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _find_run_dir(
    *,
    run_dir: Path | None,
    family: str,
    run_id: str | None,
) -> Path:
    if run_dir is not None:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir

    base_dir = MODELS_ROOT / family
    if not base_dir.exists():
        raise FileNotFoundError(f"Model family directory does not exist: {base_dir}")

    if run_id is not None:
        run_name = run_id if run_id.startswith("run_") else f"run_{run_id}"
        target = base_dir / run_name
        if not target.exists():
            raise FileNotFoundError(f"Run not found: {target}")
        return target

    runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under: {base_dir}")
    return runs[-1]


def _choose_checkpoint(run_dir: Path, preferred: str) -> Path:
    best = run_dir / "best.pt"
    latest = run_dir / "checkpoint.pt"
    if preferred == "best":
        if not best.exists():
            raise FileNotFoundError(f"Missing checkpoint: {best}")
        return best
    if preferred == "checkpoint":
        if not latest.exists():
            raise FileNotFoundError(f"Missing checkpoint: {latest}")
        return latest
    if best.exists():
        return best
    if latest.exists():
        return latest
    raise FileNotFoundError(f"No checkpoint found in {run_dir} (expected best.pt/checkpoint.pt)")


def _extract_head_names(state_dict: dict[str, Any]) -> list[str]:
    names = set()
    for key in state_dict:
        if key.startswith("heads."):
            parts = key.split(".")
            if len(parts) >= 2:
                names.add(parts[1])
    return sorted(names)


def _state_param_count(state_dict: dict[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel())
    return total


def _print_kv(title: str, payload: dict[str, Any] | None) -> None:
    if not payload:
        print(f"{title}: <none>")
        return
    print(title + ":")
    for key in sorted(payload.keys()):
        print(f"  - {key}: {payload[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Phase 2 run artifacts")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit run directory path")
    parser.add_argument("--family", choices=FAMILIES, default="multitask_pro_e3nn")
    parser.add_argument("--run-id", type=str, default=None, help="Run id (with or without 'run_')")
    parser.add_argument(
        "--checkpoint",
        choices=("auto", "best", "checkpoint"),
        default="auto",
        help="Which checkpoint file to inspect",
    )
    parser.add_argument("--show-heads", action="store_true", help="Print all task head names")
    args = parser.parse_args()

    try:
        run_dir = _find_run_dir(run_dir=args.run_dir, family=args.family, run_id=args.run_id)
        ckpt_path = _choose_checkpoint(run_dir, args.checkpoint)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 2

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        print(f"[ERROR] Invalid checkpoint (missing model_state_dict): {ckpt_path}")
        return 2

    manifest = _load_json(run_dir / "run_manifest.json")
    results = _load_json(run_dir / "results.json")
    history = _load_json(run_dir / "history.json")
    task_names = _extract_head_names(state_dict)

    print("=" * 80)
    print("Phase 2 Checkpoint Inspector")
    print("=" * 80)
    print(f"Run dir     : {run_dir}")
    print(f"Checkpoint  : {ckpt_path.name}")
    print(f"Epoch       : {payload.get('epoch', 'n/a')}")
    print(f"Best val MAE: {payload.get('best_val_mae', 'n/a')}")
    print(f"Current MAE : {payload.get('val_mae', 'n/a')}")
    print(f"Task count  : {len(task_names)}")
    print(f"Param count : {_state_param_count(state_dict):,}")

    if args.show_heads:
        print("Task heads  : " + (", ".join(task_names) if task_names else "<none>"))

    if manifest:
        print("-" * 80)
        print("Manifest")
        _print_kv("Args", manifest.get("args") if isinstance(manifest.get("args"), dict) else None)
        _print_kv("Result", manifest.get("result") if isinstance(manifest.get("result"), dict) else None)
        runtime = manifest.get("runtime")
        if isinstance(runtime, dict):
            git = runtime.get("git")
            _print_kv("Git", git if isinstance(git, dict) else None)

    if results:
        print("-" * 80)
        print("results.json")
        _print_kv("Core", {
            "algorithm": results.get("algorithm"),
            "run_id": results.get("run_id"),
            "best_epoch": results.get("best_epoch"),
            "best_val_mae": results.get("best_val_mae"),
            "avg_test_mae": results.get("avg_test_mae"),
            "total_epochs": results.get("total_epochs"),
        })
        hyper = results.get("hyperparameters")
        _print_kv("Hyperparameters", hyper if isinstance(hyper, dict) else None)

    if history and isinstance(history, dict):
        print("-" * 80)
        print("history.json")
        for key in ("train_loss", "val_mae", "weights", "lr"):
            val = history.get(key)
            if isinstance(val, list):
                print(f"  - {key}: length={len(val)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
