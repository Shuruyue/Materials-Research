"""
Latent Space Analysis

Analyzes the learned representation space of the GNN encoder:
- t-SNE / UMAP visualization colored by properties
- Property-property correlation in latent space
- Cluster analysis for materials families
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path


class LatentSpaceAnalyzer:
    """
    Analyzes the latent space of a trained GNN encoder.

    Extracts graph-level embeddings and performs dimensionality reduction
    and correlation analysis.

    Args:
        model: trained GNN model (must have .encode() method)
        device: computation device
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def extract_embeddings(
        self,
        loader,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract latent embeddings for all materials in a dataloader.

        Args:
            loader: PyG DataLoader
            properties: list of property names to extract as metadata

        Returns:
            Dict with:
                "embeddings": (N, embed_dim)
                "property_name": (N,) for each requested property
        """
        self.model.eval()
        all_embeddings = []
        all_properties = {p: [] for p in (properties or [])}

        for batch in loader:
            batch = batch.to(self.device)
            emb = self.model.encode(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            all_embeddings.append(emb.cpu().numpy())

            for p in (properties or []):
                if hasattr(batch, p):
                    all_properties[p].append(getattr(batch, p).cpu().numpy())

        result = {
            "embeddings": np.concatenate(all_embeddings, axis=0),
        }
        for p, vals in all_properties.items():
            if vals:
                result[p] = np.concatenate(vals, axis=0)

        return result

    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Reduce embedding dimensionality for visualization.

        Args:
            embeddings: (N, D) high-dimensional embeddings
            method: "umap" or "tsne"
            n_components: target dimensions (2 or 3)

        Returns:
            (N, n_components) reduced coordinates
        """
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method == "umap":
            try:
                from umap import UMAP
                reducer = UMAP(n_components=n_components, random_state=42)
            except ImportError:
                from sklearn.manifold import TSNE
                print("UMAP not installed, falling back to t-SNE")
                reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(embeddings)

    def plot_latent_space(
        self,
        embeddings_2d: np.ndarray,
        color_by: np.ndarray,
        color_label: str = "Property",
        save_path: Optional[Path] = None,
        title: str = "Materials Property Space",
    ):
        """
        Plot 2D latent space colored by a property.

        Args:
            embeddings_2d: (N, 2) coordinates
            color_by: (N,) property values for coloring
            color_label: label for colorbar
            save_path: path to save figure
            title: plot title
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=color_by,
            cmap="viridis",
            s=5,
            alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax, label=color_label)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.close(fig)

    def property_correlation_matrix(
        self,
        embeddings: np.ndarray,
        properties: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute correlation between latent embedding dimensions
        and physical properties.

        Returns:
            (n_properties, embed_dim) correlation matrix
        """
        from scipy.stats import pearsonr

        prop_names = list(properties.keys())
        n_props = len(prop_names)
        n_dims = embeddings.shape[1]

        corr = np.zeros((n_props, n_dims))
        for i, name in enumerate(prop_names):
            for j in range(n_dims):
                r, _ = pearsonr(properties[name].flatten(), embeddings[:, j])
                corr[i, j] = r

        return corr
