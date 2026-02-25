"""
Outlier Filtering

Statistical outlier detection and removal for crystal property datasets.
Uses sigma-based filtering with CSV export for human review.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def filter_outliers(dataset, properties: list[str], save_dir: Path | None = None,
                    n_sigma: float = 4.0):
    """
    Remove outliers from a dataset based on z-score thresholding.

    Args:
        dataset: PyG dataset with property attributes.
        properties: List of property names to check.
        save_dir: Directory to save outliers.csv (optional).
        n_sigma: Number of standard deviations for threshold.

    Returns:
        torch.utils.data.Subset with outliers removed.
    """
    indices_to_keep = set(range(len(dataset)))
    outliers_found = []

    for prop in properties:
        values = []
        valid_indices = []
        for i in range(len(dataset)):
            try:
                data = dataset[i]
                val = getattr(data, prop).item()
                values.append(val)
                valid_indices.append(i)
            except Exception:
                continue

        if not values:
            continue
        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        if std < 1e-8:
            continue

        for j, v in enumerate(values):
            if abs(v - mean) > n_sigma * std:
                idx = valid_indices[j]
                indices_to_keep.discard(idx)
                outliers_found.append({
                    "jid": getattr(dataset[idx], "jid", "unknown"),
                    "property": prop,
                    "value": v,
                    "mean": mean,
                    "sigma": (v - mean) / std,
                    "threshold_sigma": n_sigma,
                })

    if outliers_found:
        logger.info(f"Found {len(outliers_found)} outliers across {len(properties)} properties")
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            df_out = pd.DataFrame(outliers_found)
            df_out.to_csv(save_dir / "outliers.csv", index=False)
            logger.info(f"Outlier details saved to {save_dir / 'outliers.csv'}")

    return torch.utils.data.Subset(dataset, sorted(indices_to_keep))
