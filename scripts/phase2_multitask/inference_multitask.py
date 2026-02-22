"""
Phase 2: Multi-task inference CLI for CIF files.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atlas.data.crystal_dataset import DEFAULT_PROPERTIES
from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.models.utils import load_phase2_model
from jarvis.core.atoms import Atoms

PHASE2_MODEL_FAMILIES = (
    "multitask_lite_e3nn",
    "multitask_std_e3nn",
    "multitask_pro_e3nn",
    "multitask_cgcnn",
)


class AtlasMultitaskPredictor:
    def __init__(self, model_dir: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(model_dir)
        self.builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12, compute_3body=True)

        ckpt_path = self.model_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = self.model_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No model found in {self.model_dir}")

        print(f"[INFO] Loading model from {ckpt_path.name}...")
        self.model, self.normalizer = load_phase2_model(str(ckpt_path), self.device)
        self.tasks = list(getattr(self.model, "task_names", DEFAULT_PROPERTIES))
        encoder = getattr(self.model, "encoder", None)
        # e3nn uses edge_vec; CGCNN uses edge_attr
        self._use_edge_vec = bool(hasattr(encoder, "sh_irreps"))
        print(f"[INFO] Model loaded. Tasks: {self.tasks}")

    def _process_atoms(self, atoms):
        structure = atoms.pymatgen_converter()
        return self.builder.structure_to_pyg(structure)

    def _denormalize(self, prop, value):
        if self.normalizer is None:
            return value
        if not hasattr(self.normalizer, "normalizers"):
            return value
        if prop not in self.normalizer.normalizers:
            return value
        return self.normalizer.denormalize(prop, value)

    def predict_one(self, atoms):
        data = self._process_atoms(atoms)
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(self.device)
        edge_features = data.edge_vec if self._use_edge_vec else data.edge_attr

        with torch.no_grad():
            preds = self.model(data.x, data.edge_index, edge_features, data.batch)

        result = {}
        for prop in self.tasks:
            if prop not in preds:
                continue
            value = self._denormalize(prop, preds[prop]).item()
            result[prop] = float(value)
        return result

    def predict_batch(self, cif_files):
        rows = []
        print(f"Dataset size: {len(cif_files)}")

        for cif_path in cif_files:
            try:
                atoms = Atoms.from_cif(str(cif_path))
                record = {"file": cif_path.name}
                record.update(self.predict_one(atoms))
                rows.append(record)
            except Exception as exc:
                print(f"Failed {cif_path.name}: {exc}")

        return pd.DataFrame(rows)


def _detect_latest_model_dir(project_root: Path, family: str) -> Path:
    base_dir = project_root / "models" / family
    if not base_dir.exists():
        raise FileNotFoundError(f"Model family directory not found: {base_dir}")
    runs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )
    if not runs:
        raise FileNotFoundError(f"No trained Multi-Task models found in {base_dir}")
    print(f"[INFO] Using latest run: {runs[-1].name}")
    return runs[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Atlas Phase 2 Multi-Task Inference")
    parser.add_argument("--cif", type=Path, help="Path to a single CIF file")
    parser.add_argument("--dir", type=Path, help="Directory containing CIF files")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Explicit model run directory (overrides --family auto-detect)",
    )
    parser.add_argument(
        "--family",
        choices=PHASE2_MODEL_FAMILIES,
        default="multitask_pro_e3nn",
        help="Model family for auto-selecting latest run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.xlsx"),
        help="Output file path (.xlsx or .csv)",
    )
    parser.add_argument("--test-random", action="store_true", help="Reserved placeholder")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    try:
        model_dir = args.model_dir if args.model_dir is not None else _detect_latest_model_dir(project_root, args.family)
        predictor = AtlasMultitaskPredictor(model_dir)
    except Exception as exc:
        print(f"Failed to initialize predictor: {exc}")
        return 2

    if args.cif:
        atoms = Atoms.from_cif(str(args.cif))
        predictions = predictor.predict_one(atoms)
        print(f"\nPredictions for {args.cif.name}:")
        for prop in predictor.tasks:
            if prop in predictions:
                print(f"  {prop:20s}: {predictions[prop]:.4f}")
        return 0

    if args.dir:
        cifs = list(args.dir.glob("*.cif"))
        if not cifs:
            print("No CIF files found.")
            return 2

        df = predictor.predict_batch(cifs)
        if args.output.suffix.lower() == ".xlsx":
            try:
                df.to_excel(args.output, index=False)
                print(f"Saved to {args.output}")
            except Exception:
                csv_output = args.output.with_suffix(".csv")
                df.to_csv(csv_output, index=False)
                print(f"Saved to {csv_output}")
        else:
            df.to_csv(args.output, index=False)
            print(f"Saved to {args.output}")

        if not df.empty:
            print("\n" + df.head().to_markdown(index=False, tablefmt="grid"))
        return 0

    if args.test_random:
        print("Random validation sampling is not implemented in this CLI yet.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
