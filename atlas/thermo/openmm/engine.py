import openmm
import openmm.app as app
import openmm.unit as unit
from typing import Optional, Any
import ase
from atlas.utils.structure import ase_to_pymatgen
from atlas.thermo.openmm.reporters import PymatgenTrajectoryReporter
from atlas.utils.registry import RELAXERS

@RELAXERS.register("atomate2_openmm")
class OpenMMEngine:
    """
    Wrapper for OpenMM to perform molecular dynamics simulations.
    Designed to integrate with ATLAS MACE potentials via openmm-ml/torch.
    """
    
    def __init__(self, 
                 temperature: float = 300.0,
                 friction: float = 1.0,
                 step_size: float = 1.0):
        """
        Initialize the OpenMM Engine.

        Args:
            temperature (float): Temperature in Kelvin.
            friction (float): Friction coefficient in 1/ps.
            step_size (float): Time step in fs.
        """
        self.temperature = temperature * unit.kelvin
        self.friction = friction / unit.picosecond
        self.step_size = step_size * unit.femtoseconds
        self.simulation: Optional[app.Simulation] = None

    def setup_system(self, atoms: ase.Atoms, forcefield_path: Optional[str] = None):
        """
        Setup OpenMM System from ASE Atoms.
        
        Args:
            atoms (ase.Atoms): Structure to simulate.
            forcefield_path (str): Path to forcefield XML (if using classical FF).
                                   If None, uses a simple Lennard-Jones potential for testing.
        """
        self.atoms = atoms
        
        # Convert ASE to OpenMM Topology
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("RES", chain)
        
        # Box vectors
        if atoms.pbc.any():
            box = atoms.get_cell()
            a = box[0] * unit.angstrom
            b = box[1] * unit.angstrom
            c = box[2] * unit.angstrom
            
            # Set on Topology (for createSystem)
            topology.setUnitCellDimensions(unit.Quantity([a.value_in_unit(unit.nanometer)[0], 
                                                          b.value_in_unit(unit.nanometer)[1], 
                                                          c.value_in_unit(unit.nanometer)[2]], unit.nanometer))
            
            # We also set on Manual System later if needed
            vectors = (a, b, c)
        else:
            vectors = None

        # Add atoms to Topology
        masses = atoms.get_masses()
        for i, atom in enumerate(atoms):
            element = app.Element.getBySymbol(atom.symbol)
            topology.addAtom(atom.symbol, element, residue)

        # Logic to create System
        self.system = None
        
        if forcefield_path == "mace":
             try:
                 from openmmml import MLPotential
                 # Download/Load MACE model
                 # Use 'mace-mpa-0-medium' for Materials Project coverage (supports Cu, etc.)
                 potential = MLPotential('mace-mpa-0-medium')
                 # Create System from Topology
                 self.system = potential.createSystem(topology)
                 print("MACE Potential attached via openmm-ml.")
             except ImportError:
                  print("Warning: openmm-ml not found. Falling back to LJ.")
             except Exception as e:
                  print(f"Failed to initialize MACE: {e}. Falling back to LJ.")

        if self.system is None:
            # Manual creation (LJ fallback or manual FF)
            self.system = openmm.System()
            if vectors:
                self.system.setDefaultPeriodicBoxVectors(*vectors)
            
            # Add particles to System
            for masse in masses:
                self.system.addParticle(masse * unit.dalton)
            
            if forcefield_path and forcefield_path != "mace":
                 # Placeholder for XML loading
                 pass
            else:
                 self._add_lj_force(atoms)

        # Integrator
        self.integrator = openmm.LangevinMiddleIntegrator(
            self.temperature,
            self.friction,
            self.step_size
        )
        
        # Simulation
        self.simulation = app.Simulation(topology, self.system, self.integrator)
        
        # Set positions
        positions = atoms.get_positions() * unit.angstrom
        self.simulation.context.setPositions(positions)
        
        # Set velocities
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

        print(f"OpenMM System Configured. Particles: {self.system.getNumParticles()}")

    def _add_lj_force(self, atoms):
        """Add simple Lennard-Jones Force for testing (Argon-like)."""
        force = openmm.NonbondedForce()
        force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        # Reduce cutoff to 5.0 A to be safe with smaller boxes (min box >= 10.0 A)
        force.setCutoffDistance(5.0 * unit.angstrom)
        for _ in range(len(atoms)):
            # Charge, Sigma, Epsilon
            force.addParticle(0.0, 3.4 * unit.angstrom, 0.2 * unit.kilocalories_per_mole)
        self.system.addForce(force)

    def run(self, steps: int, trajectory_interval: int = 100) -> Any:
        """
        Run simulation for N steps and return a Pymatgen Trajectory.
        
        Args:
            steps (int): Number of steps to simulate.
            trajectory_interval (int): How often to record a frame.
        Returns:
            trajectory (pymatgen.core.trajectory.Trajectory): Recorded MD trajectory.
        """
        if not self.simulation:
            raise RuntimeError("Simulation not initialized. Call setup_system first.")
            
        print(f"Running OpenMM simulation for {steps} steps...")
        
        # Attach Atomate2-style Reporter
        pmg_struct = ase_to_pymatgen(self.atoms)
        pmg_reporter = PymatgenTrajectoryReporter(
            reportInterval=trajectory_interval,
            structure=pmg_struct
        )
        self.simulation.reporters.append(pmg_reporter)
        
        # Run
        self.simulation.step(steps)
        
        # Get final state summary
        state = self.simulation.context.getState(getEnergy=True)
        pe_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        pot_energy_ev = pe_kj / 96.4853
        print(f"Simulation Done. Final Potential Energy: {pot_energy_ev:.4f} eV")
        
        # Return elegant PyMatgen Trajectory
        return pmg_reporter.get_trajectory()
