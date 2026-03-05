"""
Target Normalizers

Standardize target values for training neural networks.
Handles single-property and multi-property normalization with
serialization support for checkpoint compatibility.
"""


import math
from collections.abc import Iterable, Mapping

import numpy as np


def _to_finite_float(value) -> float | None:
    """Best-effort extraction of a finite scalar float."""
    try:
        value = value.item() if hasattr(value, "item") else value
    except Exception:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _extract_property(sample, property_name: str):
    if isinstance(sample, Mapping):
        return sample.get(property_name)
    return getattr(sample, property_name, None)


def _iter_dataset(dataset) -> Iterable:
    if dataset is None:
        return ()
    if isinstance(dataset, Mapping):
        return iter(dataset.values())
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        return (dataset[idx] for idx in range(len(dataset)))
    return iter(dataset)


def _normalize_properties(properties: Iterable[str] | None) -> list[str]:
    if properties is None:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in properties:
        if not isinstance(raw, str):
            raise TypeError("properties must contain only string property names")
        name = str(raw).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def _normalize_property_name(property_name: str) -> str:
    if not isinstance(property_name, str):
        raise TypeError("property name must be a string")
    name = str(property_name).strip()
    if not name:
        raise ValueError("property name must be non-empty")
    return name


class TargetNormalizer:
    """Normalize a single property target to zero mean, unit variance."""

    def __init__(self, dataset=None, property_name: str = ""):
        prop = str(property_name).strip()
        if dataset is not None and prop:
            values = []
            for sample in _iter_dataset(dataset):
                try:
                    val = _to_finite_float(_extract_property(sample, prop))
                    if val is not None:
                        values.append(val)
                except Exception:
                    continue
            if values:
                arr = np.asarray(values, dtype=np.float64)
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                self.mean = mean if math.isfinite(mean) else 0.0
                self.std = std if math.isfinite(std) and std > 1e-8 else 1.0
            else:
                self.mean = 0.0
                self.std = 1.0
        else:
            self.mean = 0.0
            self.std = 1.0

    def normalize(self, y):
        return (y - self.mean) / self.std

    def denormalize(self, y):
        return y * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": float(self.mean), "std": float(self.std)}

    def load_state_dict(self, state: dict):
        if not isinstance(state, Mapping):
            raise TypeError("state must be a dict with keys 'mean' and 'std'")
        if "mean" not in state or "std" not in state:
            raise KeyError("state must contain 'mean' and 'std'")
        mean = _to_finite_float(state["mean"])
        std = _to_finite_float(state["std"])
        if mean is None:
            raise ValueError("state['mean'] must be finite")
        if std is None or std <= 1e-8:
            std = 1.0
        self.mean = float(mean)
        self.std = float(std)

    def __repr__(self):
        return f"TargetNormalizer(mean={self.mean:.4f}, std={self.std:.4f})"


class MultiTargetNormalizer:
    """Normalize multiple property targets independently."""

    def __init__(self, dataset=None, properties: list[str] | None = None):
        props = _normalize_properties(properties)
        if dataset is not None and props:
            self.normalizers = {p: TargetNormalizer(dataset, p) for p in props}
        else:
            self.normalizers = {}

    def _get_normalizer(self, prop: str) -> TargetNormalizer:
        normalized_prop = _normalize_property_name(prop)
        if normalized_prop not in self.normalizers:
            available = ", ".join(sorted(self.normalizers)) or "<none>"
            raise KeyError(
                f"unknown property '{normalized_prop}'. available properties: {available}"
            )
        return self.normalizers[normalized_prop]

    def normalize(self, prop: str, y):
        return self._get_normalizer(prop).normalize(y)

    def denormalize(self, prop: str, y):
        return self._get_normalizer(prop).denormalize(y)

    def state_dict(self) -> dict[str, dict]:
        return {p: self.normalizers[p].state_dict() for p in sorted(self.normalizers)}

    def load_state_dict(self, state: Mapping[str, dict]):
        if not isinstance(state, Mapping):
            raise TypeError("state must be a dict mapping property names to states")
        for p in sorted(state):
            s = state[p]
            prop = str(p).strip()
            if not prop:
                raise ValueError("state property names must be non-empty")
            if not isinstance(s, Mapping):
                raise TypeError(
                    f"state[{prop!r}] must be a mapping with keys 'mean' and 'std'"
                )
            if prop in self.normalizers:
                self.normalizers[prop].load_state_dict(s)
            else:
                norm = TargetNormalizer()
                norm.load_state_dict(s)
                self.normalizers[prop] = norm

    def __repr__(self):
        props = ", ".join(sorted(self.normalizers.keys()))
        return f"MultiTargetNormalizer(properties=[{props}])"
