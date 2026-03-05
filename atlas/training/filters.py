"""
Outlier Filtering

Statistical outlier detection and removal for crystal property datasets.
Uses sigma-based filtering with CSV export for human review.
"""

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
_SUPPORTED_OUTLIER_METHODS = {"zscore", "modified_zscore"}
_MAD_SCALE = 0.6744897501960817
_EPS = 1e-12


def _is_boolean_like(value) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _extract_scalar(value) -> float | None:
    """Extract finite scalar values from tensor-like objects."""
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            item = value.item()
            if _is_boolean_like(item):
                return None
            scalar = float(item)
        elif hasattr(value, "item"):
            item = value.item()
            if _is_boolean_like(item):
                return None
            scalar = float(item)
        elif isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            item = value.reshape(-1)[0]
            if _is_boolean_like(item):
                return None
            scalar = float(item)
        else:
            if _is_boolean_like(value):
                return None
            scalar = float(value)
    except Exception:
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _compute_sigma(
    values: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray | None, float | None]:
    """Compute standardized residuals and dispersion scale."""
    if values.size < 2:
        return None, None
    if method == "zscore":
        center = float(values.mean())
        scale = float(values.std())
        if scale <= _EPS:
            return None, None
        sigma = (values - center) / scale
        return sigma, scale

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad <= _EPS:
        return None, None
    sigma = _MAD_SCALE * (values - center) / mad
    return sigma, mad


def filter_outliers(
    dataset,
    properties: list[str],
    save_dir: Path | None = None,
    n_sigma: float = 4.0,
    method: str = "zscore",
):
    """
    Remove outliers from a dataset based on z-score thresholding.

    Args:
        dataset: PyG dataset with property attributes.
        properties: List of property names to check.
        save_dir: Directory to save outliers.csv (optional).
        n_sigma: Number of standard deviations for threshold.
        method: Outlier method ("zscore" or "modified_zscore").

    Returns:
        torch.utils.data.Subset with outliers removed.
    """
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError("dataset must support __len__ and __getitem__")
    if isinstance(properties, (str, bytes)) or not isinstance(properties, Sequence):
        raise TypeError("properties must be a sequence of property names")

    threshold = float(n_sigma)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("n_sigma must be finite and > 0")
    method_key = str(method).strip().lower()
    if method_key not in _SUPPORTED_OUTLIER_METHODS:
        raise ValueError(f"Unsupported method '{method}'. Supported: {sorted(_SUPPORTED_OUTLIER_METHODS)}")

    dataset_size = len(dataset)
    if dataset_size <= 0:
        return torch.utils.data.Subset(dataset, [])
    if not properties:
        return torch.utils.data.Subset(dataset, list(range(dataset_size)))

    normalized_properties = []
    for name in properties:
        if not isinstance(name, str):
            raise TypeError("properties must contain only string property names")
        prop = str(name).strip()
        if prop and prop not in normalized_properties:
            normalized_properties.append(prop)
    if not normalized_properties:
        return torch.utils.data.Subset(dataset, list(range(dataset_size)))

    indices_to_keep = set(range(dataset_size))
    outliers_found = []

    for prop in normalized_properties:
        rows: list[tuple[int, float, str]] = []
        for i in range(dataset_size):
            try:
                data = dataset[i]
                raw_value = getattr(data, prop, None)
                val = _extract_scalar(raw_value)
                if val is None:
                    continue
                jid = str(getattr(data, "jid", "unknown"))
                rows.append((i, val, jid))
            except Exception:
                continue

        if not rows:
            continue
        valid_indices = [idx for idx, _, _ in rows]
        values = np.asarray([val for _, val, _ in rows], dtype=float)
        sigma, scale = _compute_sigma(values, method=method_key)
        if sigma is None or scale is None:
            continue

        abs_sigma = np.abs(sigma)
        for j, v in enumerate(values):
            if abs_sigma[j] > threshold:
                idx = valid_indices[j]
                indices_to_keep.discard(idx)
                outliers_found.append({
                    "jid": rows[j][2],
                    "property": prop,
                    "value": float(v),
                    "sigma": float(sigma[j]),
                    "threshold_sigma": float(threshold),
                    "method": method_key,
                    "scale": float(scale),
                })

    if outliers_found:
        logger.info(f"Found {len(outliers_found)} outliers across {len(normalized_properties)} properties")
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / "outliers.csv"
            try:
                import pandas as pd

                df_out = pd.DataFrame(outliers_found)
                df_out.to_csv(out_path, index=False)
            except Exception:
                import csv

                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(outliers_found[0].keys()))
                    writer.writeheader()
                    writer.writerows(outliers_found)
            logger.info(f"Outlier details saved to {save_dir / 'outliers.csv'}")

    return torch.utils.data.Subset(dataset, sorted(indices_to_keep))
