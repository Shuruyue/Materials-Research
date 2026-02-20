# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from nequip.data import AtomicDataDict


class NonPeriodicCellTransform(torch.nn.Module):
    """Replace cell with identity matrix when all PBCs are False.

    This transform is useful for non-periodic systems where cell information should not affect neighborlist construction or other periodic-dependent calculations.
    For each frame where all PBC values are False, the cell is replaced with a dummy identity matrix.
    """

    def __init__(self):
        super().__init__()

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # check if cell and pbc are present
        if AtomicDataDict.CELL_KEY not in data or AtomicDataDict.PBC_KEY not in data:
            return data

        cell = data[AtomicDataDict.CELL_KEY]
        pbc = data[AtomicDataDict.PBC_KEY]

        # (n_frames, 3) -> (n_frames,)
        is_non_periodic = ~pbc.any(dim=1)

        # replace cell with identity matrix for non-periodic frames
        if is_non_periodic.any():
            identity = torch.eye(3, dtype=cell.dtype, device=cell.device)
            # for each non-periodic frame, set cell to identity
            for i in range(pbc.shape[0]):
                if is_non_periodic[i]:
                    cell[i] = identity
            data[AtomicDataDict.CELL_KEY] = cell

        return data
