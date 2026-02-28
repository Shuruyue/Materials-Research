import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGLoader

from atlas.data.crystal_dataset import CrystalPropertyDataset
from atlas.models.utils import load_phase1_model
from atlas.training.metrics import scalar_metrics

# Configuration
PROPERTY_NAME = "formation_energy"
MODEL_PATH = PROJECT_ROOT / "models/cgcnn_full_formation_energy/best.pt"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"





def main():
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # 1. Load Data (Test Set)
    print("Loading test dataset...")
    test_ds = CrystalPropertyDataset(
        properties=[PROPERTY_NAME],
        split="test",
    ).prepare()

    # Filter outliers (using same strategy as training: 4.0 sigma)
    # Note: Strictly speaking we should filter test set too if we did during training for evaluation consistency
    # But usually we evaluate on raw test set. Let's filter to be consistent with 'Clean Data' concept.
    values = []
    for data in test_ds:
        if hasattr(data, PROPERTY_NAME):
            values.append(getattr(data, PROPERTY_NAME).item())
    arr = np.array(values)
    mean, std = arr.mean(), arr.std()
    mask = np.abs(arr - mean) <= 4.0 * std
    indices = np.where(mask)[0].tolist()
    from torch.utils.data import Subset
    test_ds = Subset(test_ds, indices)
    print(f"Test set size (filtered): {len(test_ds)}")

    loader = PyGLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Rebuild Model
    print("Rebuilding model...")
    model, normalizer = load_phase1_model(MODEL_PATH, device=DEVICE)
    if normalizer is None:
        print("Normalizer missing in checkpoint. Using identity.")
    else:
        print(f"Normalizer loaded: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    # 3. Evaluate
    print("Evaluating...")
    all_pred = []
    all_target = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = getattr(batch, PROPERTY_NAME).view(-1, 1)

            pred_denorm = normalizer.denormalize(pred) if normalizer is not None else pred

            all_pred.append(pred_denorm.cpu())
            all_target.append(target.cpu())

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)

    metrics = scalar_metrics(pred, target, prefix=PROPERTY_NAME)

    print("\n" + "="*50)
    print("FINAL RESULTS (Recovered from best.pt)")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"\nBest Epoch: {checkpoint['epoch']}")
    if isinstance(checkpoint, dict) and "val_mae" in checkpoint:
        print(f"Saved Validation MAE: {checkpoint['val_mae']:.4f}")

if __name__ == "__main__":
    main()
