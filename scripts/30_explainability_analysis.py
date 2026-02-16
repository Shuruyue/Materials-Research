"""
Phase 5: Explainability & Latent Space Analysis Script

After training, runs:
1. GNNExplainer → per-property important substructures
2. Integrated Gradients → atom/bond attribution
3. Latent space → t-SNE/UMAP visualization
4. Multi-task transfer → gradient alignment analysis
5. Property-property correlation in learned representations

Usage:
    python scripts/30_explainability_analysis.py --model-dir models/equivariant_formation_energy
    python scripts/30_explainability_analysis.py --multi-task --model-dir models/multitask_equivariant
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.config import get_config
from atlas.data.crystal_dataset import CrystalPropertyDataset, DEFAULT_PROPERTIES
from atlas.explain.gnn_explainer import GNNExplainerWrapper
from atlas.explain.latent_analysis import LatentSpaceAnalyzer
from atlas.training.metrics import scalar_metrics


ELEMENT_NAMES = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]


def run_gnn_explainer(model, dataset, properties, save_dir, n_samples=100):
    """Run GNNExplainer for each property and save element importance."""
    print("\n  ── GNNExplainer Analysis ──")
    results = {}

    for prop in properties:
        print(f"    Property: {prop} ({n_samples} samples)...")
        explainer = GNNExplainerWrapper(model, task_name=prop)
        importance = explainer.explain_batch(dataset, n_samples=n_samples)

        # Map element indices to names
        named_importance = {}
        for idx, score in sorted(importance.items(), key=lambda x: -x[1]):
            if idx < len(ELEMENT_NAMES):
                named_importance[ELEMENT_NAMES[idx]] = float(score)

        results[prop] = named_importance

        # Print top-10 elements
        top10 = list(named_importance.items())[:10]
        for elem, score in top10:
            bar = "█" * int(score * 50)
            print(f"      {elem:>3s}: {score:.4f} {bar}")

    with open(save_dir / "element_importance.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved to {save_dir / 'element_importance.json'}")
    return results


def run_latent_analysis(model, loader, properties, save_dir, method="tsne"):
    """Extract embeddings and create latent space visualizations."""
    print(f"\n  ── Latent Space Analysis ({method.upper()}) ──")

    analyzer = LatentSpaceAnalyzer(model, device="cuda" if torch.cuda.is_available() else "cpu")
    data = analyzer.extract_embeddings(loader, properties=properties)

    embeddings = data["embeddings"]
    print(f"    Embedding shape: {embeddings.shape}")

    # Dimensionality reduction
    coords_2d = analyzer.reduce_dimensions(embeddings, method=method)

    # Plot for each property
    for prop in properties:
        if prop in data:
            save_path = save_dir / f"latent_{prop}_{method}.png"
            analyzer.plot_latent_space(
                coords_2d,
                color_by=data[prop],
                color_label=prop,
                save_path=save_path,
                title=f"Materials Property Space — {prop}",
            )
            print(f"    Saved: {save_path}")

    # Property-property correlation in latent space
    prop_data = {p: data[p] for p in properties if p in data}
    if len(prop_data) >= 2:
        corr = analyzer.property_correlation_matrix(embeddings, prop_data)
        np.save(save_dir / "property_latent_correlation.npy", corr)
        print(f"    Correlation matrix shape: {corr.shape}")
        print(f"    Saved to {save_dir / 'property_latent_correlation.npy'}")

    return data


@torch.no_grad()
def gradient_alignment_analysis(model, loader, properties, save_dir, device="cuda"):
    """
    Analyze gradient alignment between task heads (multi-task transfer).

    For each pair of properties, compute cosine similarity of gradients
    w.r.t. shared encoder parameters → positive/negative transfer signal.
    """
    print("\n  ── Gradient Alignment Analysis ──")
    if not hasattr(model, "heads"):
        print("    Skipping (single-task model)")
        return {}

    model.eval()
    model.train()  # Need gradients

    # Get one batch
    batch = next(iter(loader)).to(device)

    # Determine edge features
    if hasattr(batch, "edge_vec"):
        edge_feats = batch.edge_vec
    else:
        edge_feats = batch.edge_attr

    # Get shared encoder embedding
    embedding = model.encoder.encode(batch.x, batch.edge_index, edge_feats, batch.batch)

    # Compute per-task gradients w.r.t. encoder
    task_grads = {}
    for prop in properties:
        if prop not in model.heads or not hasattr(batch, prop):
            continue

        target = getattr(batch, prop).view(-1, 1)
        pred = model.heads[prop](embedding)
        loss = torch.nn.functional.mse_loss(pred, target)

        # Gradient of loss w.r.t. encoder parameters
        encoder_params = list(model.encoder.parameters())
        grads = torch.autograd.grad(loss, encoder_params, retain_graph=True, allow_unused=True)
        flat_grad = torch.cat([g.flatten() for g in grads if g is not None])
        task_grads[prop] = flat_grad

    # Cosine similarity matrix
    names = list(task_grads.keys())
    n = len(names)
    cosine_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            gi = task_grads[names[i]]
            gj = task_grads[names[j]]
            cos = torch.nn.functional.cosine_similarity(gi.unsqueeze(0), gj.unsqueeze(0)).item()
            cosine_matrix[i, j] = cos

    print(f"\n    Gradient Cosine Similarity (positive = positive transfer):")
    print(f"    {'':>20s}", end="")
    for name in names:
        print(f" {name[:8]:>10s}", end="")
    print()
    for i, n1 in enumerate(names):
        print(f"    {n1:>20s}", end="")
        for j in range(len(names)):
            val = cosine_matrix[i, j]
            marker = "+" if val > 0 else "-"
            print(f" {val:>9.3f}{marker}", end="")
        print()

    results = {
        "task_names": names,
        "cosine_similarity": cosine_matrix.tolist(),
    }
    with open(save_dir / "gradient_alignment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n    Saved to {save_dir / 'gradient_alignment.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Explainability Analysis")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--multi-task", action="store_true",
                        help="Model is multi-task (has multiple heads)")
    parser.add_argument("--n-explain", type=int, default=100,
                        help="Number of samples for GNNExplainer")
    parser.add_argument("--method", type=str, default="tsne",
                        choices=["tsne", "umap"])
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples for analysis (speed)")
    args = parser.parse_args()

    config = get_config()
    model_dir = Path(args.model_dir)
    save_dir = model_dir / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     Phase 5: Explainability & Latent Space Analysis            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Load model
    print(f"\n  Loading model from {model_dir}...")
    checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)

    properties = DEFAULT_PROPERTIES if args.multi_task else [
        checkpoint.get("property", "formation_energy")
    ]

    # Load test data
    print(f"  Loading test data (max {args.max_samples} samples)...")
    test_ds = CrystalPropertyDataset(
        properties=properties,
        max_samples=args.max_samples,
        split="test",
    )
    test_ds.prepare()
    test_loader = test_ds.to_pyg_loader(batch_size=64, shuffle=False)

    # The model must be reconstructed based on what was trained.
    # For now, provide guidance:
    print("\n  ⚠️  Model reconstruction: please set up model before running.")
    print("      (This script provides analysis functions; integrate with your model)")
    print("      Example usage:")
    print("        model = EquivariantGNN(...).to(device)")
    print("        model.load_state_dict(checkpoint['model_state_dict'])")
    print("        run_latent_analysis(model, test_loader, properties, save_dir)")

    print(f"\n  Analysis directory: {save_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
