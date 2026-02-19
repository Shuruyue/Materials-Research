"""
ATLAS Explain Module

Interpretability analysis for crystal GNN models:
- GNNExplainer: identify important substructures per property
- Latent space: t-SNE/UMAP visualization of materials property space
"""

from atlas.explain.gnn_explainer import GNNExplainerWrapper

try:
    from atlas.explain.latent_analysis import LatentSpaceAnalyzer
except ImportError:
    LatentSpaceAnalyzer = None  # seaborn not installed

__all__ = [
    "GNNExplainerWrapper",
    "LatentSpaceAnalyzer",
]
