"""
Target Normalizers

Standardize target values for training neural networks.
Handles single-property and multi-property normalization with
serialization support for checkpoint compatibility.
"""


import numpy as np


class TargetNormalizer:
    """Normalize a single property target to zero mean, unit variance."""

    def __init__(self, dataset=None, property_name: str = ""):
        if dataset is not None and property_name:
            values = []
            for i in range(len(dataset)):
                try:
                    val = getattr(dataset[i], property_name).item()
                    if not np.isnan(val):
                        values.append(val)
                except Exception:
                    continue
            arr = np.array(values)
            self.mean = float(arr.mean())
            self.std = float(arr.std()) if arr.std() > 1e-8 else 1.0
        else:
            self.mean = 0.0
            self.std = 1.0

    def normalize(self, y):
        return (y - self.mean) / self.std

    def denormalize(self, y):
        return y * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"]
        self.std = state["std"]

    def __repr__(self):
        return f"TargetNormalizer(mean={self.mean:.4f}, std={self.std:.4f})"


class MultiTargetNormalizer:
    """Normalize multiple property targets independently."""

    def __init__(self, dataset=None, properties: list[str] | None = None):
        if dataset is not None and properties:
            self.normalizers = {
                p: TargetNormalizer(dataset, p) for p in properties
            }
        else:
            self.normalizers = {}

    def normalize(self, prop: str, y):
        return self.normalizers[prop].normalize(y)

    def denormalize(self, prop: str, y):
        return self.normalizers[prop].denormalize(y)

    def state_dict(self) -> dict[str, dict]:
        return {p: n.state_dict() for p, n in self.normalizers.items()}

    def load_state_dict(self, state: dict[str, dict]):
        for p, s in state.items():
            if p in self.normalizers:
                self.normalizers[p].load_state_dict(s)
            else:
                norm = TargetNormalizer()
                norm.load_state_dict(s)
                self.normalizers[p] = norm

    def __repr__(self):
        props = ", ".join(self.normalizers.keys())
        return f"MultiTargetNormalizer(properties=[{props}])"
