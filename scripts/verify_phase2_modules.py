
"""
Verification Script for Phase 2 Optimization
"""
import sys
import logging
import torch

# Configure check
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase2Verification")

def verify_modules():
    logger.info("Verifying ATLAS Phase 2 Modules...")
    
    # 1. Models
    try:
        from atlas.models import M3GNet, MultiTaskGNN, EvidentialHead, CrystalGraphBuilder
        logger.info("[Pass] Models import successful")
        
        # Instantiate
        model = M3GNet(n_species=86, embed_dim=16)
        logger.info("[Pass] M3GNet instantiation successful")
        
        head = EvidentialHead(embed_dim=16)
        logger.info("[Pass] EvidentialHead instantiation successful")
        
        builder = CrystalGraphBuilder(compute_3body=True)
        logger.info("[Pass] CrystalGraphBuilder (3-body) instantiation successful")
        
    except Exception as e:
        logger.error(f"[Fail] Models verification failed: {e}")
        return False

    # 2. Training (Losses)
    try:
        from atlas.training.losses import EvidentialLoss, MultiTaskLoss
        loss = EvidentialLoss()
        logger.info("[Pass] EvidentialLoss instantiation successful")
    except Exception as e:
        logger.error(f"[Fail] Training verification failed: {e}")
        return False

    # 3. Active Learning
    try:
        from atlas.active_learning import DiscoveryController, expected_improvement
        logger.info("[Pass] Active Learning import successful")
    except Exception as e:
        logger.error(f"[Fail] Active Learning verification failed: {e}")
        return False

    # 4. Thermo
    try:
        from atlas.thermo import PhaseStabilityAnalyst
        analyst = PhaseStabilityAnalyst()
        logger.info("[Pass] PhaseStabilityAnalyst instantiation successful")
    except Exception as e:
        logger.error(f"[Fail] Thermo verification failed: {e}")
        return False
        
    logger.info("\nALL SYSTEMS GO! Phase 2 Optimization Verified.")
    return True

if __name__ == "__main__":
    success = verify_modules()
    sys.exit(0 if success else 1)
