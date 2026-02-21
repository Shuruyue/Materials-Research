#!/usr/bin/env python3
"""
Phase 4 baseline: RandomForest topological classifier from composition features.

This is a fast non-GNN baseline for algorithm comparison against TopoGNN.

Usage:
  python scripts/phase4_topology/train_topo_classifier_rf.py
  python scripts/phase4_topology/train_topo_classifier_rf.py --max-samples 8000 --n-estimators 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.config import get_config
from atlas.data.jarvis_client import JARVISClient


MAX_Z = 86


def formula_to_feature(formula: str) -> np.ndarray:
    """Convert formula to element-fraction + atomic-number statistics."""
    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()
    total_atoms = float(sum(amounts.values()))
    if total_atoms <= 0:
        raise ValueError(f"Invalid composition: {formula}")

    fractions = np.zeros(MAX_Z, dtype=np.float32)
    z_vals = []
    w_vals = []

    for symbol, amount in amounts.items():
        z = Element(symbol).Z
        frac = float(amount) / total_atoms
        if 1 <= z <= MAX_Z:
            fractions[z - 1] = frac
            z_vals.append(float(z))
            w_vals.append(frac)

    if not w_vals:
        raise ValueError(f"No valid elements in formula: {formula}")

    z_arr = np.array(z_vals, dtype=np.float32)
    w_arr = np.array(w_vals, dtype=np.float32)
    mean_z = float(np.average(z_arr, weights=w_arr))
    var_z = float(np.average((z_arr - mean_z) ** 2, weights=w_arr))

    extra = np.array(
        [
            float(len(w_vals)),       # number of unique elements
            mean_z,                   # weighted mean Z
            float(np.sqrt(var_z)),    # weighted std Z
            float(z_arr.min()),       # min Z
            float(z_arr.max()),       # max Z
        ],
        dtype=np.float32,
    )
    return np.concatenate([fractions, extra], axis=0)


def prepare_dataset(client: JARVISClient, max_samples: int, seed: int):
    df = client.load_dft_3d()
    df = df[df["spillage"].notna()].copy()
    df = df[df["formula"].notna()].copy()
    df["topo_label"] = (df["spillage"] > 0.5).astype(int)

    n_topo = int(df["topo_label"].sum())
    n_trivial = int(len(df) - n_topo)
    n_each = min(n_topo, n_trivial, max_samples // 2)
    if n_each <= 10:
        raise RuntimeError("Too few labeled samples to train RF baseline.")

    topo = df[df["topo_label"] == 1].sample(n=n_each, random_state=seed)
    trivial = df[df["topo_label"] == 0].sample(n=n_each, random_state=seed)
    balanced = pd.concat([topo, trivial]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    features = []
    labels = []
    formulas = []
    skipped = 0
    for _, row in balanced.iterrows():
        try:
            features.append(formula_to_feature(row["formula"]))
            labels.append(int(row["topo_label"]))
            formulas.append(row["formula"])
        except Exception:
            skipped += 1

    if len(features) < 100:
        raise RuntimeError("Insufficient valid formulas after feature conversion.")

    x = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    formulas = np.array(formulas)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))
    n_train = int(0.8 * len(x))
    n_val = int(0.1 * len(x))

    split = {
        "train": (x[idx[:n_train]], y[idx[:n_train]], formulas[idx[:n_train]]),
        "val": (x[idx[n_train:n_train + n_val]], y[idx[n_train:n_train + n_val]], formulas[idx[n_train:n_train + n_val]]),
        "test": (x[idx[n_train + n_val:]], y[idx[n_train + n_val:]], formulas[idx[n_train + n_val:]]),
        "meta": {
            "n_total": int(len(x)),
            "n_topological": int(y.sum()),
            "n_trivial": int(len(y) - y.sum()),
            "n_skipped": int(skipped),
        },
    }
    return split


def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RandomForest topology baseline")
    parser.add_argument("--max-samples", type=int, default=5000, help="Maximum balanced sample count")
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = get_config()
    model_dir = cfg.paths.models_dir / "topo_classifier_rf"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4 RF Baseline ===")
    print(f"  max_samples:      {args.max_samples}")
    print(f"  n_estimators:     {args.n_estimators}")
    print(f"  max_depth:        {args.max_depth}")
    print(f"  min_samples_leaf: {args.min_samples_leaf}")

    client = JARVISClient()
    data = prepare_dataset(client=client, max_samples=args.max_samples, seed=args.seed)
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["val"]
    x_test, y_test, _ = data["test"]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)

    print("\n=== Validation ===")
    print(f"  Accuracy:  {val_metrics['accuracy']:.3f}")
    print(f"  Precision: {val_metrics['precision']:.3f}")
    print(f"  Recall:    {val_metrics['recall']:.3f}")
    print(f"  F1:        {val_metrics['f1']:.3f}")

    print("\n=== Test ===")
    print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall:    {test_metrics['recall']:.3f}")
    print(f"  F1:        {test_metrics['f1']:.3f}")

    joblib.dump(model, model_dir / "rf_model.joblib")
    report = {
        "algorithm": "random_forest_composition",
        "config": {
            "max_samples": args.max_samples,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "seed": args.seed,
        },
        "data": data["meta"],
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(model_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved model/report to: {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

