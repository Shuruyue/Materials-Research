"""
Latent Space Analysis

Analyzes the learned representation space of the GNN encoder to reveal
hidden structure-property relationships.

Optimization:
- Clustering: Automatic grouping of materials (K-Means/DBSCAN) to find families.
- Academic Plotting: Generates publication-ready scatter plots with Seaborn styles.
- Correlation Analysis: Quantifies feature importance in the latent space.
"""

from __future__ import annotations

import logging
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

# Configure seaborn for academic style
sns.set_theme(style="white", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

logger = logging.getLogger(__name__)


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


def _as_2d_finite_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must have positive shape in both dimensions, got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _as_1d_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _normalize_device(device: str) -> str:
    value = str(device).strip().lower()
    if value.startswith("cuda"):
        if torch.cuda.is_available():
            return value
        logger.warning("CUDA requested for latent analysis but unavailable; falling back to CPU.")
        return "cpu"
    if value == "cpu":
        return "cpu"
    logger.warning("Unknown latent-analysis device '%s'; falling back to CPU.", device)
    return "cpu"


def _coerce_embedding_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("embedding", "embeddings", "latent", "features", "mean"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    raise TypeError("Model output must be Tensor or dict containing an embedding Tensor.")


class LatentSpaceAnalyzer:
    """
    Analyzes the latent space of a trained GNN encoder.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device = _normalize_device(device)
        self.model = model.to(self.device)

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
        all_embeddings: list[np.ndarray] = []
        property_names = [str(p) for p in (properties or [])]
        all_properties = {p: [] for p in property_names}
        collected_rows = 0

        for batch in loader:
            batch = batch.to(self.device)
            edge_attr = getattr(batch, "edge_attr", None)
            batch_index = getattr(batch, "batch", None)
            if batch_index is None:
                num_nodes = int(getattr(batch, "num_nodes", 0))
                if num_nodes > 0:
                    batch_index = torch.zeros(num_nodes, dtype=torch.long, device=batch.x.device)
            # Assuming model has an 'encode' method that returns graph embedding
            if hasattr(self.model, 'encode'):
                output = self.model.encode(
                    batch.x,
                    batch.edge_index,
                    edge_attr,
                    batch_index,
                )
            else:
                output = self.model(
                    batch.x,
                    batch.edge_index,
                    edge_attr,
                    batch_index
                )
            emb = _coerce_embedding_tensor(output)
            emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            emb_np = emb.detach().cpu().numpy()
            if emb_np.ndim == 1:
                emb_np = emb_np.reshape(1, -1)
            emb_np = _as_2d_finite_array(emb_np, name="embedding batch")
            all_embeddings.append(emb_np)
            batch_rows = int(emb_np.shape[0])
            collected_rows += batch_rows

            for p in property_names:
                if hasattr(batch, p):
                    val = getattr(batch, p)
                    if torch.is_tensor(val):
                        if val.dim() > 1:
                            val = val.squeeze()
                        val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                        np_val = val.detach().cpu().numpy().reshape(-1)
                    else:
                        np_val = np.asarray(val).reshape(-1)
                    if np_val.shape[0] == batch_rows:
                        all_properties[p].append(np_val)
                    elif np_val.shape[0] == 1:
                        all_properties[p].append(np.repeat(np_val, batch_rows))
                    else:
                        logger.warning(
                            "Property %s length mismatch for batch: got %d, expected %d; filling NaN.",
                            p,
                            np_val.shape[0],
                            batch_rows,
                        )
                        all_properties[p].append(np.full(batch_rows, np.nan, dtype=float))
                else:
                    all_properties[p].append(np.full(batch_rows, np.nan, dtype=float))

        if not all_embeddings:
            raise ValueError("Loader produced no batches; cannot extract latent embeddings.")

        embeddings = np.concatenate(all_embeddings, axis=0)
        result = {"embeddings": embeddings}
        for p, vals in all_properties.items():
            if vals:
                merged = np.concatenate(vals, axis=0)
                if merged.shape[0] != collected_rows:
                    raise RuntimeError(
                        f"Property {p!r} length mismatch after concatenation: {merged.shape[0]} vs {collected_rows}"
                    )
                result[p] = merged

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
        embeddings_2d = _as_2d_finite_array(embeddings, name="embeddings")
        method_name = str(method).strip().lower()
        n_samples = int(embeddings_2d.shape[0])
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for dimensionality reduction.")
        components = _coerce_positive_int(n_components, name="n_components")

        if method_name == "tsne":
            from sklearn.manifold import TSNE

            perplexity = max(1.0, min(30.0, float(n_samples - 1)))
            reducer = TSNE(n_components=components, random_state=42, perplexity=perplexity)
        elif method_name == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=min(components, embeddings_2d.shape[1], n_samples))
        elif method_name == "umap":
            try:
                from umap import UMAP

                if n_samples <= 2:
                    from sklearn.decomposition import PCA

                    logger.warning("UMAP requires local neighborhoods; falling back to PCA for <=2 samples.")
                    reducer = PCA(n_components=min(components, embeddings_2d.shape[1], n_samples))
                else:
                    n_neighbors = min(15, n_samples - 1)
                    reducer = UMAP(n_components=components, n_neighbors=n_neighbors, random_state=42)
            except ImportError:
                from sklearn.manifold import TSNE
                logger.warning("UMAP not installed; falling back to t-SNE")
                perplexity = max(1.0, min(30.0, float(n_samples - 1)))
                reducer = TSNE(n_components=components, random_state=42, perplexity=perplexity)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(embeddings_2d)

    def perform_clustering(
        self,
        embeddings: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 5
    ) -> np.ndarray:
        """
        Cluster materials in the latent space.
        """
        embeddings_2d = _as_2d_finite_array(embeddings, name="embeddings")
        method_name = str(method).strip().lower()
        n_samples = int(embeddings_2d.shape[0])
        if n_samples < 1:
            raise ValueError("Need at least 1 sample for clustering.")

        if method_name == "kmeans":
            from sklearn.cluster import KMeans

            clusters = _coerce_positive_int(n_clusters, name="n_clusters")
            if clusters > n_samples:
                raise ValueError("n_clusters cannot exceed number of samples")
            clusterer = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        elif method_name == "dbscan":
            from sklearn.cluster import DBSCAN

            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return clusterer.fit_predict(embeddings_2d)

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
        embedding_points = _as_2d_finite_array(embeddings_2d, name="embeddings_2d")
        if embedding_points.shape[1] != 2:
            raise ValueError("embeddings_2d must have shape (N, 2)")
        color_values = _as_1d_array(color_by, name="color_by")
        if color_values.shape[0] != embedding_points.shape[0]:
            raise ValueError("color_by length must match number of points in embeddings_2d")

        df = pd.DataFrame({
            "Dim 1": embedding_points[:, 0],
            "Dim 2": embedding_points[:, 1],
            "Value": color_values
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
                embedding_points[:, 0], embedding_points[:, 1],
                c=np.nan_to_num(color_values, nan=0.0, posinf=0.0, neginf=0.0),
                cmap="viridis", s=30, alpha=0.7,
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
            logger.info("Saved latent-space figure to %s", save_path)
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
        embeddings_2d = _as_2d_finite_array(embeddings, name="embeddings")
        clusters = _coerce_positive_int(n_clusters, name="n_clusters")
        n_rows = int(embeddings_2d.shape[0])
        normalized_properties: dict[str, np.ndarray] = {}
        for key, value in properties.items():
            property_values = _as_1d_array(value, name=f"properties[{key}]")
            if property_values.shape[0] != n_rows:
                raise ValueError(
                    f"Property length mismatch for {key!r}: got {property_values.shape[0]}, expected {n_rows}"
                )
            normalized_properties[str(key)] = property_values
        labels = self.perform_clustering(embeddings_2d, n_clusters=clusters)

        df = pd.DataFrame(normalized_properties)
        df["Cluster"] = labels

        # Calculate mean/std for each property per cluster
        summary = df.groupby("Cluster").agg(["mean", "std", "count"])
        return summary
