import logging
from typing import Dict, Any, Optional
import openmm
import openmm.app as app

from atlas.utils.registry import RELAXERS

try:
    # Full native import of the Assimilated Atomate2 OpenMM base makers
    import atlas.third_party.atomate2.openmm.jobs.core as atomate2_jobs
    import atlas.third_party.atomate2.openmm.jobs.base as atomate2_base
    from atlas.third_party.atomate2.openmm.utils import PymatgenTrajectoryReporter as OriginalPmtReporter
except ImportError as e:
    logging.warning(f"Could not import assimilated Atomate2 OpenMM modules: {e}")
    atomate2_jobs = None
    atomate2_base = None
    OriginalPmtReporter = None

logger = logging.getLogger(__name__)

@RELAXERS.register("atomate2_native")
class NativeAtomate2OpenMMEngine:
    """
    Full architecture assimilation of the Atomate2 OpenMM loop.
    Rather than rewriting the OpenMM simulation loop, we employ Atomate2's `NPTMaker` 
    and `NVTMaker` natively, inheriting their robust Barostat and Thermostat setups.
    """
    def __init__(self, temperature: float = 300.0, step_size: float = 1.0, ensemble: str = "nvt"):
        self.temperature = temperature
        self.step_size = step_size
        self.ensemble = ensemble.lower()
        
        if atomate2_jobs is None or atomate2_base is None:
            raise RuntimeError("Assimilated Atomate2 components are missing. Cannot build native engine.")
            
    def run_simulation(self, interchange_data, steps: int = 1000):
        """
        Runs the simulation using the exact Atomate2 Makers.
        Requires an OpenMM Interchange/System object which is standardized by Atomate2.
        """
        try:
            if self.ensemble == "npt":
                maker = atomate2_jobs.NPTMaker(
                    temperature=self.temperature,
                    step_size=self.step_size,
                    n_steps=steps
                )
            elif self.ensemble == "nvt":
                maker = atomate2_jobs.NVTMaker(
                    temperature=self.temperature,
                    step_size=self.step_size,
                    n_steps=steps
                )
            else:
                 maker = atomate2_jobs.EnergyMinimizationMaker()
                 
            logger.info(f"Instantiated Native Atomate2 {maker.__class__.__name__}")
            # The maker runs the complete simulation internally.
            response = maker.make(interchange_data)
            return response
            
        except Exception as e:
            logger.error(f"Native Atomate2 OpenMM Simulation Failed: {e}")
            raise
