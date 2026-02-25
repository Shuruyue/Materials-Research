

import numpy as np

from atlas.discovery.alchemy.calculator import AlchemicalMACECalculator


class CompositionOptimizer:
    """
    Optimizes the alchemical weights (composition) of a structure using gradient descent.
    Enforces constraints such that weights for species at the same site sum to 1.0.
    """

    def __init__(self, calculator: AlchemicalMACECalculator, learning_rate: float = 0.01):
        self.calc = calculator
        self.lr = learning_rate
        self.constraints = self._infer_constraints()
        print(f"DEBUG: Inferred constraints: {self.constraints}")

    def _infer_constraints(self) -> dict[int, list[int]]:
        """
        Groups weight indices by atom index.
        Returns: {atom_index: [weight_index_0, weight_index_1, ...]}
        """
        # Access AlchemyManager internals
        mgr = self.calc.alchemy_manager

        # mgr.alchemical_pairs is the source of truth for weight ordering.
        # It matches the alchemical_weights tensor order.
        # alchemical_pairs[i] corresponds to weights[i]

        atom_to_weights = {}

        for weight_idx, pairs in enumerate(mgr.alchemical_pairs):
            # Each 'pairs' list corresponds to one weight channel.
            # Usually it contains tuples like (atom_index, atomic_number).
            # If multiple pairs share the same weight, they are coupled.
            # We assume here that we want to constrain weights at a specific PHYSICAL SITE.

            # Let's extract the atom indices controlled by this weight
            atom_indices = set()
            for pair in pairs:
                if hasattr(pair, "atom_index"):
                    atom_indices.add(pair.atom_index)
                else:
                    atom_indices.add(pair[0])

            for atom_id in atom_indices:
                if atom_id not in atom_to_weights:
                    atom_to_weights[atom_id] = []
                atom_to_weights[atom_id].append(weight_idx)

        # Filter out atoms with only 1 weight (no competition? or vacancy?)
        # If there is only 1 species, maybe we want to optimize vacancy?
        # i.e. w <= 1.0.
        # For now, let's strictly handle Sum(w) = 1 groups (2+ species).
        # And for single species, keep w in [0, 1].

        return atom_to_weights

    def step(self):
        """
        Perform one optimization step.
        """
        # 1. Get Gradients
        # This assumes calculate_alchemical_grad=True was used in the last calc,
        # OR we trigger a new calculation.

        # We trigger a new calculation to be safe and get fresh gradients
        self.calc.calculate_alchemical_grad = True

        # We need to call get_potential_energy to trigger calculate()
        # But we don't want to move atoms, so just call it.
        atoms = self.calc.atoms
        if atoms is None or atoms.calc is None:
             # Ensure linkage
             if atoms is not None:
                 atoms.calc = self.calc

        energy = atoms.get_potential_energy()
        grad = self.calc.results["alchemical_grad"] # numpy array

        # 2. Update Weights
        current_weights = self.calc.alchemy_manager.alchemical_weights.detach().cpu().numpy()
        new_weights = current_weights - self.lr * grad

        # 3. Project Constraints
        new_weights = self._project_weights(new_weights)

        # 4. Update Calculator
        self.calc.set_alchemical_weights(new_weights)
        self.calc.calculate_alchemical_grad = False # Reset

        return energy, new_weights

    def _project_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Project weights onto the simplex (Sum w = 1) for each site,
        and clip to [0, 1].
        """
        # We need to handle this carefully.
        # Use simple projection: Clip negative, then Normalize groups.

        # 1. Global Clip [0, 1]
        weights = np.clip(weights, 0.0, 1.0)

        # 2. Group Normalization
        for _atom_idx, weight_indices in self.constraints.items():
            if len(weight_indices) > 1:
                # Sum of these weights should be 1.0
                subset = weights[weight_indices]
                total = np.sum(subset)

                if total > 1e-6:
                    weights[weight_indices] = subset / total
                else:
                    # If all zero, reset to uniform? Or keep zero?
                    # Keep as is, or warn.
                    pass
            elif len(weight_indices) == 1:
                # Single species. Unconstrained optimization in [0, 1]?
                # Usually implies vacancy optimization if we let it drift.
                # If we want it fixed 1.0, it shouldn't be in alchemical_weights optimization?
                # Let's assume [0, 1] optimization (vacancy).
                pass

        return weights

    def run(self, steps=50, verbose=True):
        traj = []
        for i in range(steps):
            e, w = self.step()
            traj.append({"energy": e, "weights": w.copy()})
            if verbose:
                print(f"Step {i}: Energy={e:.4f}, Weights={w}")
        return traj
