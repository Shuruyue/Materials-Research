"""
GNN Explainability

Wraps PyTorch Geometric's GNNExplainer and Captum to identify
which atoms and bonds contribute most to each property prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import numpy as np


class GNNExplainerWrapper:
    """
    Wrapper for GNNExplainer to analyze property-structure relationships.

    For each material, identifies the most important subgraph
    (atoms + bonds) contributing to a specific property prediction.

    Args:
        model: trained GNN model
        task_name: which prediction head to explain (for multi-task models)
    """

    def __init__(self, model: nn.Module, task_name: str = "band_gap"):
        self.model = model
        self.task_name = task_name

    def explain(
        self,
        data,
        n_epochs: int = 200,
    ) -> Dict[str, np.ndarray]:
        """
        Explain prediction for a single material.

        Args:
            data: PyG Data object (single crystal)
            n_epochs: optimization epochs for GNNExplainer

        Returns:
            Dict with:
                "node_importance": (N,) importance score per atom
                "edge_importance": (E,) importance score per bond
                "prediction": predicted property value
        """
        try:
            from torch_geometric.explain import Explainer, GNNExplainer

            explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainer(epochs=n_epochs),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
            )

            explanation = explainer(
                data.x,
                data.edge_index,
                edge_attr=data.edge_attr,
                batch=torch.zeros(data.num_nodes, dtype=torch.long),
            )

            return {
                "node_importance": explanation.node_mask.detach().cpu().numpy(),
                "edge_importance": explanation.edge_mask.detach().cpu().numpy(),
            }

        except ImportError:
            raise ImportError(
                "torch_geometric.explain is required. "
                "Install PyG >= 2.3.0"
            )

    def explain_batch(
        self,
        dataset,
        n_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Explain predictions for multiple materials and aggregate.

        Returns element-level importance averaged over the dataset.
        """
        element_importance = {}

        for i, data in enumerate(dataset):
            if i >= n_samples:
                break

            result = self.explain(data)
            node_imp = result["node_importance"]

            # Map importance back to element types
            for j in range(data.num_nodes):
                elem_idx = data.x[j].argmax().item()
                if elem_idx not in element_importance:
                    element_importance[elem_idx] = []
                avg_imp = node_imp[j].mean() if node_imp[j].ndim > 0 else node_imp[j]
                element_importance[elem_idx].append(float(avg_imp))

        # Average per element
        avg_importance = {
            idx: np.mean(vals) for idx, vals in element_importance.items()
        }
        return avg_importance
