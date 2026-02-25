"""
Latent Space Analysis

Analyzes the learned representation space of the GNN encoder to reveal
hidden structure-property relationships.

Optimization:
- Clustering: Automatic grouping of materials (K-Means/DBSCAN) to find families.
- Academic Plotting: Generates publication-ready scatter plots with Seaborn styles.
- Correlation Analysis: Quantifies feature importance in the latent space.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

# Configure seaborn for academic style
sns.set_theme(style="white", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'


class LatentSpaceAnalyzer:
    """
    Analyzes the latent space of a trained GNN encoder.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def extract_embeddings(
        self,
        loader,
        properties: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Extract latent embeddings for all materials in a dataloader.
        """
        self.model.eval()
        all_embeddings = []
        all_properties = {p: [] for p in (properties or [])}

        for batch in loader:
            batch = batch.to(self.device)
            # Assuming model has an 'encode' method that returns graph embedding
            if hasattr(self.model, 'encode'):
                emb = self.model.encode(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
            else:
                # Fallback for simple models: use forward but stop before head?
                # This is tricky without knowing model structure.
                # Assuming standard GNN interface.
                emb = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch
                )

            all_embeddings.append(emb.cpu().numpy())

            for p in (properties or []):
                if hasattr(batch, p):
                    val = getattr(batch, p)
                    if val.dim() > 1: val = val.squeeze()
                    all_properties[p].append(val.cpu().numpy())

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
        """
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
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

    def perform_clustering(
        self,
        embeddings: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 5
    ) -> np.ndarray:
        """
        Cluster materials in the latent space.
        """
        if method == "kmeans":
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == "dbscan":
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return clusterer.fit_predict(embeddings)

    def plot_latent_space(
        self,
        embeddings_2d: np.ndarray,
        color_by: np.ndarray,
        color_label: str = "Property",
        discrete: bool = False,
        save_path: Path | None = None,
        title: str | None = None,
    ):
        """
        Plot 2D latent space colored by a property.
        Academic style: clean background, clear labels, minimal chartjunk.
        """
        df = pd.DataFrame({
            "Dim 1": embeddings_2d[:, 0],
            "Dim 2": embeddings_2d[:, 1],
            "Value": color_by
        })

        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

        # Plot
        if discrete:
            # Categorical colormap
            sns.scatterplot(
                data=df, x="Dim 1", y="Dim 2", hue="Value",
                palette="deep", s=30, alpha=0.8, edgecolor="w", linewidth=0.2, ax=ax
            )
            ax.legend(title=color_label, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            # Continuous colormap (viridis)
            points = ax.scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=color_by, cmap="viridis", s=30, alpha=0.7,
                edgecolors="w", linewidth=0.2
            )
            cbar = plt.colorbar(points, ax=ax, label=color_label, pad=0.02)
            cbar.outline.set_linewidth(0.5)

        # Aesthetics
        ax.set_title(title if title else f"Latent Space by {color_label}")
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")

        # Remove top/right spines
        sns.despine(trim=True)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved figure: {save_path}")
            plt.close(fig)
        else:
            return fig

    def analyze_clusters(
        self,
        embeddings: np.ndarray,
        properties: dict[str, np.ndarray],
        n_clusters: int = 5
    ) -> pd.DataFrame:
        """
        Perform clustering and statistically analyze each cluster's properties.
        Returns a DataFrame summarizing cluster characteristics.
        """
        labels = self.perform_clustering(embeddings, n_clusters=n_clusters)

        df = pd.DataFrame(properties)
        df["Cluster"] = labels

        # Calculate mean/std for each property per cluster
        summary = df.groupby("Cluster").agg(["mean", "std", "count"])
        return summary
