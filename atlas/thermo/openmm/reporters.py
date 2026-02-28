import openmm.unit as unit
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory


class PymatgenTrajectoryReporter:
    """
    Reporter that creates a pymatgen Trajectory from an OpenMM simulation.
    Adapted from materialsproject/atomate2.
    """
    def __init__(self,
                 reportInterval: int,
                 structure: Structure,
                 enforcePeriodicBox: bool = True):
        self._reportInterval = reportInterval
        self._structure = structure
        self._enforcePeriodicBox = enforcePeriodicBox

        self.coords = []
        self.time_steps = []
        self.energies = []
        self.forces = []

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, True, True, True, self._enforcePeriodicBox)

    def report(self, simulation, state):
        # Extract Positions
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        self.coords.append(positions)

        # Extract Time
        time_ps = state.getTime().value_in_unit(unit.picosecond)
        self.time_steps.append(time_ps)

        # Extract Energy (kJ/mol -> eV)
        pe_ev = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) / 96.4853
        self.energies.append(pe_ev)

        # Extract Forces (kJ/nm/mol -> eV/A)
        # 1 kJ/mol/nm = 0.01036427 eV/A
        forces_eva = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer) * 0.01036427
        self.forces.append(forces_eva)

    def get_trajectory(self) -> Trajectory:
        """
        Convert gathered data into a pymatgen Trajectory object.
        """
        # Ensure base structure has proper properties
        traj = Trajectory(
            species=self._structure.species,
            coords=self.coords,
            time_step=self.time_steps,
            lattice=self._structure.lattice,
            frame_properties=[
                {
                    "energy": e,
                    "forces": f
                } for e, f in zip(self.energies, self.forces, strict=False)
            ]
        )
        return traj
