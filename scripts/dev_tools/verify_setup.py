"""
ATLAS Environment Verification Script (verify_setup.py)

Combines dependency checks and module verification into a single tool.
Usage: python scripts/dev_tools/verify_setup.py
"""

import sys
import logging
from pathlib import Path
import importlib.util

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ATLAS_Verify")

def check_package(name, version_attr="__version__"):
    """Check if a package is installed and print its version."""
    try:
        if name == "e3nn.o3":
            from e3nn import o3
            logger.info(f"[OK] e3nn.o3 available")
            return True
        
        module = __import__(name)
        ver = getattr(module, version_attr, "unknown")
        logger.info(f"[OK] {name:<15} : {ver}")
        return True
    except ImportError:
        logger.error(f"[FAIL] {name:<15} : MISSING")
        return False
    except Exception as e:
        logger.error(f"[FAIL] {name:<15} : Error - {e}")
        return False

def verify_modules():
    """Attempt to instantiate core ATLAS modules to ensure code integrity."""
    logger.info("\n--- Verifying ATLAS Phase 2 Modules ---")
    
    # 1. Models
    try:
        from atlas.models import M3GNet, MultiTaskGNN, EvidentialHead, CrystalGraphBuilder
        logger.info("[OK] [Import] atlas.models successful")
        
        # Instantiate M3GNet
        model = M3GNet(n_species=86, embed_dim=16)
        logger.info("[OK] [Class] M3GNet instantiated")
        
        # Instantiate Head
        head = EvidentialHead(embed_dim=16)
        logger.info("[OK] [Class] EvidentialHead instantiated")
        
        # Instantiate GraphBuilder (critical 3-body check)
        builder = CrystalGraphBuilder(compute_3body=True)
        logger.info("[OK] [Class] CrystalGraphBuilder (3-body) instantiated")
        
    except Exception as e:
        logger.error(f"[FAIL] [Models] Verification failed: {e}")
        return False

    # 2. Training (Losses)
    try:
        from atlas.training.losses import EvidentialLoss
        loss = EvidentialLoss()
        logger.info("[OK] [Class] EvidentialLoss instantiated")
    except Exception as e:
        logger.error(f"[FAIL] [Losses] Verification failed: {e}")
        return False

    # 3. Active Learning
    try:
        from atlas.active_learning import DiscoveryController
        logger.info("[OK] [Import] Active Learning modules found")
    except Exception as e:
        logger.error(f"[FAIL] [AL] Verification failed: {e}")
        return False
        
    return True

def main():
    print("=" * 60)
    print("   ATLAS PROJECT DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Dependency Check
    logger.info("Checking Core Dependencies...")
    deps = [
        "torch", 
        "torch_geometric", 
        "torch_scatter", 
        "torch_sparse", 
        "e3nn", 
        "mace", 
        "botorch", 
        "gpytorch"
    ]
    
    all_deps_ok = all(check_package(d) for d in deps)
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        logger.info(f"[OK] CUDA Available   : {torch.version.cuda}")
        logger.info(f"[OK] GPU Device       : {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("[WARN] CUDA NOT Available (Running on CPU)")
    
    # 2. Module Check
    modules_ok = verify_modules()
    
    print("\n" + "=" * 60)
    if all_deps_ok and modules_ok:
        print("   [PASS] ALL SYSTEMS GO! Environment is healthy.")
        sys.exit(0)
    else:
        print("   [FAIL] ISSUES DETECTED. See logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
