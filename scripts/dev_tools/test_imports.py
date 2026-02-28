import sys
from importlib import import_module
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Testing imports...")
try:
    import torch
    print(f"Torch: {torch.__version__}")
    import torch_geometric
    print(f"PyG: {torch_geometric.__version__}")
    import_module("atlas.models.cgcnn")
    print("CGCNN imported successfully")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Error: {e}")

try:
    import_module("e3nn")
    print("e3nn imported")
except ImportError:
    print("e3nn MISSING")

try:
    import_module("botorch")
    print("botorch imported")
except ImportError:
    print("botorch MISSING")
