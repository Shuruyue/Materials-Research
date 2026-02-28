
import os
import sys

# Add LiFlow to path
ROOT_DIR = os.path.abspath(r"references/recisic/liflow")
sys.path.append(ROOT_DIR)

from liflow.model.modules import FlowModule


def inspect_ckpt():
    ckpt_path = os.path.join(ROOT_DIR, "ckpt", "P_LGPS.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading {ckpt_path}...")
    try:
        model = FlowModule.load_from_checkpoint(ckpt_path, map_location="cpu")
        print("Model loaded.")
        print("Config Model Num Elements:", model.cfg.model.num_elements)

        # Check atom_embedding
        if hasattr(model.model, "atom_embedding"):
            embedding = model.model.atom_embedding
            print("Embedding num_embeddings:", embedding.num_embeddings)
            print("Embedding shape:", embedding.weight.shape)
        else:
             print("model.model has no atom_embedding")


    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_ckpt()
