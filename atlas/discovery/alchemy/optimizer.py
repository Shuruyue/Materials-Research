import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from atlas.discovery.alchemy.calculator import AlchemicalMACECalculator

logger = logging.getLogger(__name__)


def _is_boolean_like(value: object) -> bool:
    return isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"}


def _coerce_non_negative_int(value: object, *, name: str) -> int:
    if _is_boolean_like(value):
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)):
        scalar = float(value)
        if not np.isfinite(scalar) or not scalar.is_integer():
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
        number = int(scalar)
    else:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}") from exc
    if number < 0:
        raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    return number


class CompositionOptimizer:
    """
    Optimizes the alchemical weights (composition) of a structure using gradient descent.
    Enforces constraints such that weights for species at the same site sum to 1.0.
    """

    def __init__(self, calculator: "AlchemicalMACECalculator", learning_rate: float = 0.01):
        self.calc = calculator
        if _is_boolean_like(learning_rate):
            raise ValueError("learning_rate must be a finite positive scalar")
        lr = float(learning_rate)
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("learning_rate must be a finite positive scalar")
        self.lr = lr
        self.constraints = self._infer_constraints()
        logger.debug("Inferred composition constraints: %s", self.constraints)

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

        atom_to_weights: dict[int, list[int]] = {}

        for weight_idx, pairs in enumerate(mgr.alchemical_pairs):
            # Each 'pairs' list corresponds to one weight channel.
            # Usually it contains tuples like (atom_index, atomic_number).
            # If multiple pairs share the same weight, they are coupled.
            # We assume here that we want to constrain weights at a specific PHYSICAL SITE.

            # Let's extract the atom indices controlled by this weight
            atom_indices = set()
            for pair in pairs:
                if hasattr(pair, "atom_index"):
                    atom_indices.add(int(pair.atom_index))
                else:
                    atom_indices.add(int(pair[0]))

            for atom_id in atom_indices:
                if atom_id not in atom_to_weights:
                    atom_to_weights[atom_id] = []
                atom_to_weights[atom_id].append(weight_idx)

        # Filter out atoms with only 1 weight (no competition? or vacancy?)
        # If there is only 1 species, maybe we want to optimize vacancy?
        # i.e. w <= 1.0.
        # For now, let's strictly handle Sum(w) = 1 groups (2+ species).
        # And for single species, keep w in [0, 1].

        for atom_id in list(atom_to_weights.keys()):
            atom_to_weights[atom_id] = sorted(set(atom_to_weights[atom_id]))
        return atom_to_weights

    @staticmethod
    def _project_to_simplex(vector: np.ndarray) -> np.ndarray:
        """
        Euclidean projection onto probability simplex.

        Reference:
        - Duchi et al. (2008), "Efficient Projections onto the l1-Ball..." (ICML)
        """
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return v.astype(np.float32)

        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        index = np.arange(1, v.size + 1)
        cond = u - cssv / index > 0.0
        if not np.any(cond):
            return np.full(v.shape, 1.0 / float(v.size), dtype=np.float32)
        rho = int(index[cond][-1])
        theta = cssv[rho - 1] / float(rho)
        projected = np.maximum(v - theta, 0.0)
        return projected.astype(np.float32)

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
        if atoms is None:
            raise RuntimeError("Calculator has no attached atoms.")
        if atoms.calc is not self.calc:
            atoms.calc = self.calc

        try:
            energy = float(atoms.get_potential_energy())
            grad = np.asarray(self.calc.results["alchemical_grad"], dtype=np.float32).reshape(-1)
        finally:
            self.calc.calculate_alchemical_grad = False

        # 2. Update Weights
        current_weights = self.calc.alchemy_manager.alchemical_weights.detach().cpu().numpy().astype(np.float32)
        if grad.shape != current_weights.shape:
            raise ValueError(
                f"Gradient shape mismatch: expected {current_weights.shape}, got {grad.shape}"
            )
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        new_weights = current_weights - self.lr * grad

        # 3. Project Constraints
        new_weights = self._project_weights(new_weights)

        # 4. Update Calculator
        self.calc.set_alchemical_weights(new_weights)

        return energy, new_weights

    def _project_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Project weights onto the simplex (Sum w = 1) for each site,
        and clip to [0, 1].
        """
        # We need to handle this carefully.
        # Use simple projection: Clip negative, then Normalize groups.

        projected = np.asarray(weights, dtype=np.float32).reshape(-1)
        expected = int(self.calc.alchemy_manager.alchemical_weights.numel())
        if projected.size != expected:
            raise ValueError(f"weights size mismatch: got {projected.size}, expected {expected}")
        projected = np.nan_to_num(projected, nan=0.0, posinf=1.0, neginf=0.0)
        projected = np.clip(projected, 0.0, 1.0)

        # 2. Group Normalization
        for _atom_idx, weight_indices in self.constraints.items():
            if not weight_indices:
                continue
            if min(weight_indices) < 0 or max(weight_indices) >= projected.size:
                raise ValueError(
                    f"constraint contains out-of-range weight index for atom {_atom_idx}: {weight_indices}"
                )
            if len(weight_indices) > 1:
                # Sum of these weights should be 1.0
                subset = projected[weight_indices]
                projected[weight_indices] = self._project_to_simplex(subset)
            elif len(weight_indices) == 1:
                # Single species. Unconstrained optimization in [0, 1]?
                # Usually implies vacancy optimization if we let it drift.
                # If we want it fixed 1.0, it shouldn't be in alchemical_weights optimization?
                # Let's assume [0, 1] optimization (vacancy).
                projected[weight_indices] = np.clip(projected[weight_indices], 0.0, 1.0)

        return projected

    def run(self, steps=50, verbose=True):
        total_steps = _coerce_non_negative_int(steps, name="steps")
        traj = []
        for i in range(total_steps):
            e, w = self.step()
            traj.append({"energy": e, "weights": w.copy()})
            if verbose:
                print(f"Step {i}: Energy={e:.4f}, Weights={w}")
        return traj
