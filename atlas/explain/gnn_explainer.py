"""
GNN Explainability

Wraps PyTorch Geometric's GNNExplainer to identify which atoms and bonds
contribute most to property predictions.

Optimization:
- Bond Type Analysis: Aggregates importance by chemical bond (e.g., Bi-Se vs Bi-Bi).
- Academic Plotting: Generates high-quality, static 3D structure plots with importance coloring.
"""

from __future__ import annotations

import logging
from numbers import Integral, Real
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pymatgen.core import Element, Structure

logger = logging.getLogger(__name__)

# Configurations for academic style
plt.rcParams['font.family'] = 'sans-serif' # Arial/Helvetica style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _coerce_positive_int(value: Any, *, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer > 0, got {value!r}")
    if isinstance(value, Integral):
        number = int(value)
    elif isinstance(value, Real):
        scalar = float(value)
        if not np.isfinite(scalar) or not scalar.is_integer():
            raise ValueError(f"{name} must be an integer > 0, got {value!r}")
        number = int(scalar)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer > 0, got {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{name} must be an integer > 0, got {value!r}")
    return number


def _infer_data_device(data: Any) -> torch.device:
    for attr in ("x", "z", "edge_index", "edge_attr"):
        tensor = getattr(data, attr, None)
        if torch.is_tensor(tensor):
            return tensor.device
    return torch.device("cpu")


def _to_1d_importance(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim == 1:
        reduced = mask
    elif mask.ndim == 2:
        # Attribute-level mask -> per-node importance via mean absolute contribution.
        reduced = mask.abs().mean(dim=1)
    else:
        reduced = mask.reshape(mask.shape[0], -1).abs().mean(dim=1)
    out = reduced.detach().cpu().numpy().astype(float, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_atomic_number_array(data) -> np.ndarray:
    if hasattr(data, "z"):
        z = data.z
        z = z.squeeze() if z.dim() > 1 else z
        arr = z.detach().cpu().numpy().astype(int, copy=False)
    elif hasattr(data, "x"):
        x = data.x
        if x.dim() == 1:
            arr = x.detach().cpu().numpy().astype(int, copy=False)
        else:
            arr = x.argmax(dim=1).detach().cpu().numpy().astype(int, copy=False)
    else:
        raise ValueError("Input data must expose node atomic numbers via `z` or `x`.")
    return np.clip(arr, 1, 118)


def _safe_element_symbol(z: int) -> str:
    z_int = int(np.clip(z, 1, 118))
    return Element.from_Z(z_int).symbol


def _safe_atomic_radius(symbol: str) -> float:
    element = Element(symbol)
    radius = element.atomic_radius or element.atomic_radius_calculated
    if radius is None:
        return 1.0
    return float(max(radius, 0.2))


class GNNExplainerWrapper:
    """
    Wrapper for GNNExplainer to analyze property-structure relationships.
    """

    def __init__(self, model: nn.Module, task_name: str = "band_gap"):
        self.model = model
        self.task_name = task_name

    def explain(self, data, n_epochs: int = 200) -> dict[str, np.ndarray]:
        """
        Explain prediction for a single material.
        """
        epochs = _coerce_positive_int(n_epochs, name="n_epochs")
        try:
            from torch_geometric.explain import Explainer, GNNExplainer

            explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainer(epochs=epochs),
                explanation_type="model",
                node_mask_type="attributes", # or "object" if no node attrs
                edge_mask_type="object",
            )

            # Create a batch index if not present
            if not hasattr(data, 'batch') or data.batch is None:
                num_nodes = int(getattr(data, "num_nodes", 0))
                if num_nodes <= 0:
                    raise ValueError("Input data must expose positive `num_nodes`")
                batch = torch.zeros(num_nodes, dtype=torch.long, device=_infer_data_device(data))
            else:
                batch = data.batch
            if getattr(data, "edge_index", None) is None:
                raise ValueError("Input data must expose edge_index for explainability.")
            if getattr(data, "x", None) is None:
                raise ValueError("Input data must expose node features `x` for explainability.")

            edge_attr = getattr(data, "edge_attr", None)
            explanation = explainer(
                data.x,
                data.edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )

            node_mask = explanation.node_mask
            if node_mask is None:
                node_importance = np.zeros(int(data.num_nodes), dtype=float)
            else:
                node_importance = _to_1d_importance(node_mask)
            edge_mask = explanation.edge_mask
            if edge_mask is None:
                edge_importance = np.zeros(data.edge_index.size(1), dtype=float)
            else:
                edge_importance = np.nan_to_num(
                    edge_mask.detach().cpu().numpy().astype(float, copy=False),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            if node_importance.shape[0] != int(data.num_nodes):
                raise RuntimeError(
                    f"node_importance length mismatch: {node_importance.shape[0]} vs {int(data.num_nodes)}"
                )
            if edge_importance.shape[0] != int(data.edge_index.size(1)):
                raise RuntimeError(
                    f"edge_importance length mismatch: {edge_importance.shape[0]} vs {int(data.edge_index.size(1))}"
                )

            return {
                "node_importance": node_importance,
                "edge_importance": edge_importance,
                "node_z": _safe_atomic_number_array(data),
                "edge_index": data.edge_index.detach().cpu().numpy(),
            }

        except ImportError:
            raise ImportError("torch_geometric.explain is required. Install PyG >= 2.3.0") from None

    def aggregate_bond_importance(
        self,
        dataset,
        n_samples: int = 100
    ) -> dict[str, float]:
        """
        Aggregates edge importance scores by bond type (e.g. "Bi-Se": 0.85).
        Useful for identifying chemically significant interactions.
        """
        bond_scores = {}
        bond_counts = {}

        max_samples = _coerce_positive_int(n_samples, name="n_samples")

        for i, data in enumerate(dataset):
            if i >= max_samples:
                break

            res = self.explain(data, n_epochs=50) # Faster for aggregation
            edge_imp = res["edge_importance"]
            edge_index = res["edge_index"]
            node_z = res["node_z"]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                continue

            # Iterate over edges
            for e_idx, importance in enumerate(edge_imp):
                src, dst = edge_index[:, e_idx]
                src = int(src)
                dst = int(dst)
                if src >= node_z.shape[0] or dst >= node_z.shape[0]:
                    continue
                z1 = int(node_z[src])
                z2 = int(node_z[dst])

                # Sort to ensure Bi-Se == Se-Bi
                el1 = _safe_element_symbol(z1)
                el2 = _safe_element_symbol(z2)
                bond_name = "-".join(sorted([el1, el2]))

                if bond_name not in bond_scores:
                    bond_scores[bond_name] = 0.0
                    bond_counts[bond_name] = 0

                bond_scores[bond_name] +=  float(importance)
                bond_counts[bond_name] += 1

        # Average
        avg_scores = {k: v / max(bond_counts[k], 1) for k, v in bond_scores.items()}
        return avg_scores

    def plot_structure_importance(
        self,
        structure: Structure,
        explanation: dict[str, np.ndarray],
        save_path: str | None = None,
        view_angle: tuple[int, int] = (15, 45)
    ):
        """
        Plot 3D crystal structure with atoms and bonds colored by importance.
        Designed for publication (static, clean).
        """
        node_imp = np.asarray(explanation["node_importance"], dtype=float).reshape(-1)
        if node_imp.shape[0] != len(structure):
            raise ValueError(
                f"node_importance length mismatch: got {node_imp.shape[0]}, expected {len(structure)}"
            )
        node_imp = np.nan_to_num(node_imp, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalize node importance to 0-1 for colormap
        if node_imp.max() > node_imp.min():
            norm_node_imp = (node_imp - node_imp.min()) / (node_imp.max() - node_imp.min())
        else:
            norm_node_imp = node_imp
        if len(view_angle) != 2:
            raise ValueError("view_angle must be a tuple of (elev, azim)")
        elev, azim = float(view_angle[0]), float(view_angle[1])
        if not np.isfinite(elev) or not np.isfinite(azim):
            raise ValueError("view_angle entries must be finite")

        # Setup 3D plot
        fig = plt.figure(figsize=(6, 6), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Plot Atoms
        coords = structure.cart_coords
        species = [str(s.specie) for s in structure]

        # Use a colormap for importance (e.g., Reds)
        cmap = plt.get_cmap('Reds')
        colors = cmap(norm_node_imp)

        # Atom sizes based on species (basic approximation)
        sizes = [100.0 * _safe_atomic_radius(s) for s in species]

        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=colors, s=sizes, edgecolors='k', linewidth=0.5, alpha=0.9,
            depthshade=True
        )

        # Plot Bonds (draw if < 3.0 A)
        # Re-computing neighbors here for visualization
        # In a real pipeline, reuse the GNN's edge_index to map importance strictly
        neighbors = structure.get_all_neighbors(3.0)

        # Basic bond plotting (gray lines) - fully mapping GNN edges to 3D lines is complex
        # without index mapping. Here we draw geometric bonds for context.
        for i, nbrs in enumerate(neighbors):
            start = coords[i]
            for nbr in nbrs:
                end = nbr.coords
                # Dist check
                if nbr.nn_distance < 3.0:
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        c='gray', alpha=0.3, linewidth=1.0
                    )

        # Clean up axes
        ax.set_axis_off()

        # Title
        formula = structure.composition.reduced_formula
        plt.title(f"{formula} - Feature Importance", fontsize=12, fontweight='bold')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.05)
        cbar.set_label("Relative Importance", fontsize=10)

        # Adjust view
        ax.view_init(elev, azim)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        return fig
