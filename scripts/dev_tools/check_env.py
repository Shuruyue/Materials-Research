import sys
from pathlib import Path
# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("--- Detailed Environment Check ---")
try:
    import torch
    print(f"Torch: {torch.__version__} (CUDA: {torch.version.cuda})")
    print(f"CUDA Available: {torch.cuda.is_available()}")
except ImportError:
    print("Torch MISSING")

try:
    import torch_scatter
    print(f"torch_scatter: {torch_scatter.__version__}")
except ImportError:
    print("torch_scatter MISSING")

try:
    import torch_sparse
    print(f"torch_sparse: {torch_sparse.__version__}")
except ImportError:
    print("torch_sparse MISSING")

try:
    import e3nn
    from e3nn import o3
    print(f"e3nn: {e3nn.__version__} (o3 imported)")
except ImportError:
    print("e3nn MISSING or o3 failed")

try:
    import mace
    print(f"mace: {mace.__version__}")
except ImportError:
    print("mace MISSING")

try:
    import botorch
    print(f"botorch: {botorch.__version__}")
except ImportError:
    print("botorch MISSING")

try:
    import gpytorch
    print(f"gpytorch: {gpytorch.__version__}")
except ImportError:
    print("gpytorch MISSING")
