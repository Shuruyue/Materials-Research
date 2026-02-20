
import ast
from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple, Union

import ase
import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.util.jit import compile_mode
from mace import modules, tools
from mace.calculators import mace_mp
from mace.data.neighborhood import get_neighborhood
from mace.modules import RealAgnosticResidualInteractionBlock, ScaleShiftMACE
from mace.modules.utils import get_edge_vectors_and_lengths, get_symmetric_displacement
from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    utils,
)
from mace.tools.scatter import scatter_sum

# Define AlchemicalPair as a named tuple for clarity
AlchemicalPair = namedtuple("AlchemicalPair", ["atom_index", "atomic_number"])


class AlchemyManager(torch.nn.Module):
    """
    Manages alchemical weights and constructs the alchemical graph for MACE.
    Allows for continuous interpolation between atomic species.
    """

    def __init__(
        self,
        atoms: ase.Atoms,
        alchemical_pairs: Sequence[Sequence[Tuple[int, int]]],
        alchemical_weights: torch.Tensor,
        z_table: AtomicNumberTable,
        r_max: float,
    ):
        """
        Initialize the AlchemyManager.

        Args:
            atoms: Base ASE atoms object.
            alchemical_pairs: List of lists, where each inner list contains tuples of
                              (atom_index, atomic_number) representing the possible species
                              for a specific site.
            alchemical_weights: Tensor of weights determining the mixing ratio of species.
            z_table: AtomicNumberTable from MACE.
            r_max: Cutoff radius for graph construction.
        """
        super().__init__()
        self.alchemical_weights = torch.nn.Parameter(alchemical_weights)
        self.alchemical_pairs = alchemical_pairs
        self.r_max = r_max

        # --- Process Alchemical Indices ---
        # 1-based indexing for alchemical weights (0 is reserved for non-alchemical/fixed atoms)
        alchemical_atom_indices = []
        alchemical_atomic_numbers = []
        alchemical_weight_indices = []

        for weight_idx, pairs in enumerate(alchemical_pairs):
            for pair in pairs:
                # Support both simple tuples (index, Z) and namedtuples/objects
                if hasattr(pair, "atom_index"):
                    idx = pair.atom_index
                    z = pair.atomic_number
                else:
                    idx = pair[0]
                    z = pair[1]
                    
                alchemical_atom_indices.append(idx)
                alchemical_atomic_numbers.append(z)
                # weight_idx + 1 because 0 is for fixed atoms
                alchemical_weight_indices.append(weight_idx + 1)

        # Identify fixed (non-alchemical) atoms
        non_alchemical_atom_indices = [
            i for i in range(len(atoms)) if i not in alchemical_atom_indices
        ]
        non_alchemical_atomic_numbers = atoms.get_atomic_numbers()[
            non_alchemical_atom_indices
        ].tolist()
        non_alchemical_weight_indices = [0] * len(non_alchemical_atom_indices)

        # Merge and Sort
        self.atom_indices = np.array(alchemical_atom_indices + non_alchemical_atom_indices)
        self.atomic_numbers = np.array(alchemical_atomic_numbers + non_alchemical_atomic_numbers)
        self.weight_indices = np.array(alchemical_weight_indices + non_alchemical_weight_indices)

        sort_idx = np.argsort(self.atom_indices)
        self.atom_indices = self.atom_indices[sort_idx]
        self.atomic_numbers = self.atomic_numbers[sort_idx]
        self.weight_indices = self.weight_indices[sort_idx]

        # --- Map Original Indices to Alchemical Data ---
        # self.original_to_alchemical_index[original_atom_idx, weight_channel] = internal_index
        # -1 indicates no mapping
        self.original_to_alchemical_index = np.full(
            (len(atoms), len(alchemical_pairs) + 1), -1, dtype=int
        )
        for i, (atom_idx, weight_idx) in enumerate(zip(self.atom_indices, self.weight_indices)):
            self.original_to_alchemical_index[atom_idx, weight_idx] = i

        # Boolean mask: Is this original atom involved in alchemy?
        # Check if any non-zero weight channel has a mapping
        self.is_original_atom_alchemical = np.any(
            self.original_to_alchemical_index[:, 1:] != -1, axis=1
        )

        # --- Precompute Node Features ---
        z_indices = atomic_numbers_to_indices(self.atomic_numbers, z_table=z_table)
        node_attrs = to_one_hot(
            torch.tensor(z_indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        self.register_buffer("node_attrs", node_attrs)
        self.pbc = atoms.get_pbc()

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Construct the alchemical graph.

        Args:
            positions: Atomic positions [N_atoms, 3].
            cell: Unit cell [3, 3].

        Returns:
            Batch dictionary compatible with MACE models.
        """
        # 1. Build Standard Neighbor List (on original atoms)
        # get_neighborhood might return extra values in newer MACE versions
        orig_edge_index, shifts, unit_shifts, *rest = get_neighborhood(
            positions=positions.detach().cpu().numpy(),
            cutoff=self.r_max,
            pbc=self.pbc,
            cell=cell.detach().cpu().numpy(),
        )

        # 2. Expand Edges for Alchemical Species
        # We need to create edges between all possible species pairs, weighted appropriately.
        edge_index = []
        orig_edge_loc = [] # To track which original edge generated this expanded edge
        edge_weight_indices = []

        is_alchemical = self.is_original_atom_alchemical[orig_edge_index]
        
        # Scenario masks
        src_non_dst_non = ~is_alchemical[0] & ~is_alchemical[1]
        src_non_dst_alch = ~is_alchemical[0] & is_alchemical[1]
        src_alch_dst_non = is_alchemical[0] & ~is_alchemical[1]
        src_alch_dst_alch = is_alchemical[0] & is_alchemical[1]

        # Case A: Fixed -> Fixed
        # 1-to-1 mapping, weight is 1.0 (index 0)
        _orig_edge_index = orig_edge_index[:, src_non_dst_non]
        edge_index.append(self.original_to_alchemical_index[_orig_edge_index, 0])
        orig_edge_loc.append(np.where(src_non_dst_non)[0])
        edge_weight_indices.append(np.zeros_like(_orig_edge_index[0]))

        # Case B: Fixed -> Alchemical
        # 1-to-Many mapping. Connect fixed src to ALL species of dst.
        _src, _dst = orig_edge_index[:, src_non_dst_alch]
        _orig_edge_loc_b = np.where(src_non_dst_alch)[0]
        
        _src = self.original_to_alchemical_index[_src, 0] # Fixed
        _dst = self.original_to_alchemical_index[_dst, :] # All channels
        
        _dst_mask = _dst != -1
        _dst = _dst[_dst_mask] # Flatten valid destinations
        
        _repeat = _dst_mask.sum(axis=1) # How many species per dst atom
        _src = np.repeat(_src, _repeat)
        
        edge_index.append(np.stack((_src, _dst), axis=0))
        orig_edge_loc.append(np.repeat(_orig_edge_loc_b, _repeat))
        edge_weight_indices.append(np.zeros_like(_src)) # Controlled by node weights, edges are unweighted here? Or implicity 1.

        # Case C: Alchemical -> Fixed
        # Many-to-1 mapping. Connect ALL species of src to fixed dst.
        # Edge weight is determined by the species of src.
        _src, _dst = orig_edge_index[:, src_alch_dst_non]
        _orig_edge_loc_c = np.where(src_alch_dst_non)[0]
        
        _src = self.original_to_alchemical_index[_src, :]
        _dst = self.original_to_alchemical_index[_dst, 0]
        
        _src_mask = _src != -1
        _src = _src[_src_mask]
        
        _repeat = _src_mask.sum(axis=1)
        _dst = np.repeat(_dst, _repeat)
        
        edge_index.append(np.stack((_src, _dst), axis=0))
        orig_edge_loc.append(np.repeat(_orig_edge_loc_c, _repeat))
        # Weight index corresponds to the alchemical channel of source
        edge_weight_indices.append(np.where(_src_mask)[1])

        # Case D: Alchemical -> Alchemical
        # Many-to-Many. Connect valid (src_specie, dst_specie) pairs.
        # Weight depends on src_specie (standard MACE message passing logic).
        _orig_edge_index = orig_edge_index[:, src_alch_dst_alch]
        _orig_edge_loc_d = np.where(src_alch_dst_alch)[0]
        
        _alch_edge_index = self.original_to_alchemical_index[_orig_edge_index, :] # [2, N_edges, N_channels]
        
        # We need a cross product of src channels and dst channels ideally?
        # WAIT: MACE usually sums over neighbors. 
        # The key is: Message = Interaction(Node_src, Edge) * Weight_src
        # So we just need to list all valid nodes.
        
        # In Recisic's implementation:
        # It seems they map edges 1-to-1 for active channels?
        # Let's check original code logic:
        # _idx = np.where((_alch_edge_index != -1).all(axis=0)) 
        # This implies they occupy the SAME alchemical channel index? That seems restrictive if we mix different sites.
        # Actually, `self.original_to_alchemical_index` has shape [N_atoms, N_weights+1].
        # weight_idx corresponds to `alchemical_pairs` list index. 
        # So if site i is pair 0, and site j is pair 1, they likely DON'T share columns.
    
        # Logic fix: Original code might assume singular alchemical channel per system or strict alignment?
        # Let's trust the ported logic for now but keep an eye on it.
        # "pair according to alchemical indices" -> This implies we only connect if they share the weight channel?
        # Ah, looking at `AlchemyManager.__init__`:
        # weight_idx is `enumerate(alchemical_pairs)`.
        # So each "site" (or group of coupled sites) has its own weight_idx.
        # If site A and site B are different alchemical groups, they are effectively "Fixed" to each other in terms of channel?
        # No, `original_to_alchemical_index` is -1 everywhere except the valid channel.
        # So `_alch_edge_index[:, _idx[0], _idx[1]]` works if they match in the 2nd dim (channel).
        # This implies alchemical interaction ONLY happens within the SAME weight group?
        # If I have site A (Sr/Ca) and site B (Ti/Zr), they have different weight indices.
        # Then `(_alch_edge_index != -1).all(axis=0)` would be empty?
        # This suggests Recisic's code might act like "Fixed->Fixed" between different alchemical groups?
        # Let's re-read carefully: "Both alchemical... pair according to alchemical indices"
        # If site A is channel 1, site B is channel 2.
        # src_alch_dst_alch is True.
        # `_alch_edge_index` shape: [2, N_edges, N_channels].
        # For a specific edge (A, B): A has valid entry at col 1, B at col 2.
        # They never match columns. So `all(axis=0)` is False.
        # That means NO edges between different alchemical groups? That would be a bug or specific design.
        
        # Correction: The original code supports shared weights (e.g. multiple sites controlled by same lambda).
        # But if they are distinct...
        # Wait, if they are distint, we need Cross-Interaction.
        # The current implementation might be missing cross-group interactions if interpreted this way.
        # HOWEVER, let's look at Case B/C again.
        # They handle Non-Alchemical interaction.
        # If A and B are both Alchemical but different groups, they are "Fixed" relative to each other's lambda?
        # No, they are both flagged `is_original_atom_alchemical`.
        
        # For safety, I will preserve the exact logic of Recisic first to ensure reproduction.
        # Then we can debug.
        _idx = np.where((_alch_edge_index != -1).all(axis=0))
        edge_index.append(_alch_edge_index[:, _idx[0], _idx[1]])
        orig_edge_loc.append(_orig_edge_loc_d[_idx[0]])
        edge_weight_indices.append(np.zeros_like(_idx[0]))

        # 3. Assemble
        edge_index = np.concatenate(edge_index, axis=1)
        orig_edge_loc = np.concatenate(orig_edge_loc)
        edge_weight_indices = np.concatenate(edge_weight_indices)

        # Convert
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        shifts = torch.tensor(shifts[orig_edge_loc], dtype=torch.float32)
        unit_shifts = torch.tensor(unit_shifts[orig_edge_loc], dtype=torch.float32)

        # 4. Weights
        # Pad with 1.0 at index 0 (for fixed atoms/edges)
        weights = F.pad(self.alchemical_weights, (1, 0), "constant", 1.0)
        node_weights = weights[self.weight_indices]
        edge_weights = weights[edge_weight_indices]

        # 5. Batch
        batch = torch_geometric.data.Data(
            num_nodes=len(self.atom_indices),
            edge_index=edge_index,
            node_attrs=self.node_attrs,
            positions=positions[self.atom_indices],
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell,
            node_weights=node_weights,
            edge_weights=edge_weights,
            node_atom_indices=torch.tensor(self.atom_indices, dtype=torch.long),
            atomic_numbers=torch.tensor(self.atomic_numbers, dtype=torch.long), # Required by newer MACE
        )
        
        # MACE model expects a batch, even if size 1
        # Use simpler construction than DataLoader to avoid overhead
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
        batch.ptr = torch.tensor([0, batch.num_nodes], dtype=torch.long)
        
        return batch


def get_outputs_alchemical(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    node_weights: torch.Tensor,
    edge_weights: torch.Tensor,
    retain_graph: bool = False,
    create_graph: bool = False,
    compute_force: bool = True,
    compute_stress: bool = False,
    compute_alchemical_grad: bool = False,
) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Computes forces, stress, and alchemical gradients.
    Modified from mace.modules.utils.get_outputs.
    """
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    
    if not compute_force:
        return None, None, None, None, None

    inputs = [positions]
    if compute_stress:
        inputs.append(displacement)
    if compute_alchemical_grad:
        inputs.extend([node_weights, edge_weights])

    gradients = torch.autograd.grad(
        outputs=[energy],
        inputs=inputs,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )

    forces = gradients[0]
    stress = torch.zeros_like(displacement)
    virials = gradients[1] if compute_stress else None
    
    node_grad, edge_grad = None, None
    if compute_alchemical_grad:
        node_grad, edge_grad = gradients[-2], gradients[-1]

    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.einsum(
            "zi,zi->z",
            cell[:, 0, :],
            torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        ).unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)

    if forces is not None:
        forces = -1 * forces
    if virials is not None:
        virials = -1 * virials
        
    return forces, virials, stress, node_grad, edge_grad


@compile_mode("script")
class AlchemicalResidualInteractionBlock(RealAgnosticResidualInteractionBlock):
    """
    Alchemical version of the Interaction Block.
    Scales message passing weights by alchemical edge weights.
    """
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,  # [Alchemical Extra]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        
        tp_weights = self.conv_tp_weights(edge_feats)
        
        # --- Alchemical Scaling ---
        # Scale the weights of the tensor product by the species weight
        tp_weights = tp_weights * edge_weights[:, None]
        
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )
        
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )
        message = self.linear(message) / self.avg_num_neighbors
        
        return (self.reshape(message), sc)


@compile_mode("script")
class AlchemicalModel(ScaleShiftMACE):
    """
    Alchemical MACE M odel.
    Allows for continuous interpolation of atomic energies and interactions.
    """
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        retain_graph: bool = False,
        create_graph: bool = False,
        compute_force: bool = True,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_alchemical_grad: bool = False,
        map_to_original_atoms: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        # Setup gradients
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        
        if compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # 1. Atomic Energies (weighted by species weight)
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        node_e0 = node_e0 * data["node_weights"].view(-1, 1)  # Alchemical Scaling with correct broadcast
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        )

        # 2. Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            shifts=data["shifts"],
            edge_index=data["edge_index"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths, node_attrs=data["node_attrs"], edge_index=data["edge_index"], atomic_numbers=data["atomic_numbers"])
        if isinstance(edge_feats, tuple):
             edge_feats = edge_feats[0]


        # 3. Interactions
        node_es_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                edge_weights=data["edge_weights"],  # Alchemical Prop
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_es_list.append(readout(node_feats).squeeze(-1))

        # 4. Sum Interactions and Scale
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        # Use head 0 (default) for single-task MACE models
        head_idx = torch.tensor(0, device=node_inter_es.device, dtype=torch.long)
        node_inter_es = self.scale_shift(node_inter_es, head=head_idx)
        node_inter_es = node_inter_es * data["node_weights"]  # Alchemical Scaling

        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=0, dim_size=num_graphs
        )

        # Total Energy
        total_energy = e0 + inter_e

        
        if total_energy.numel() != 1:
             if total_energy.shape == (1, 1):
                 total_energy = total_energy.squeeze()
        node_energy = node_e0 + node_inter_es

        # 5. Gradients (Forces, Stress, and Alchemical Gradients)
        forces, virials, stress, node_grad, edge_grad = get_outputs_alchemical(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            node_weights=data["node_weights"],
            edge_weights=data["edge_weights"],
            retain_graph=retain_graph,
            create_graph=create_graph,
            compute_force=compute_force,
            compute_stress=compute_stress,
            compute_alchemical_grad=compute_alchemical_grad,
        )

        # Map back to original atoms if requested
        # Because we split alchemical atoms into multiple "virtual" species nodes,
        # we need to sum their forces/energies back to the single real atom.
        if map_to_original_atoms:
            node_index = data["node_atom_indices"]
            node_energy = scatter_sum(src=node_energy, dim=0, index=node_index)
            if compute_force and forces is not None:
                forces = scatter_sum(src=forces, dim=0, index=node_index)

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_grad": node_grad,
            "edge_grad": edge_grad,
        }


def load_alchemical_model(
    model_size: str = "medium",
    device: str = "cpu",
    default_dtype: str = "float32",
) -> AlchemicalModel:
    """
    Factory function to load a pre-trained MACE model wrapped in AlchemicalModel.

    Args:
        model_size: 'small', 'medium' (or 'large' if supported)
        device: 'cpu' or 'cuda'
        default_dtype: 'float32' or 'float64'
    """
    # 1. Load Standard Pre-trained MACE
    pretrained_mace = mace_mp(model=model_size, device=device, default_dtype=default_dtype).models[0]
    
    # Extract params dynamically to ensure match
    atomic_energies = pretrained_mace.atomic_energies_fn.atomic_energies.detach().clone()
    z_table = utils.AtomicNumberTable([int(z) for z in pretrained_mace.atomic_numbers])
    scale = pretrained_mace.scale_shift.scale.detach().clone()
    shift = pretrained_mace.scale_shift.shift.detach().clone()

    # 2. Build Alchemical Instance using EXACT config from pretrained model
    # Infer max_ell from hidden_irreps lmax
    inter_0 = pretrained_mace.interactions[0]
    hidden_irreps = inter_0.hidden_irreps
    max_ell = hidden_irreps.lmax
    
    # Infer num_bessel from edge_feats_irreps (which comes from radial embedding)
    # The output dimension of radial embedding is num_bessel (or num_radial)
    # inter_0.edge_feats_irreps should be "Nx0e" where N is num_bessel
    num_bessel = inter_0.edge_feats_irreps.dim
    
    # Infer cutoff power directly
    radial_ref = pretrained_mace.radial_embedding
    num_polynomial_cutoff = float(radial_ref.cutoff_fn.p) # Ensure float/int conversion

    # Correlation is not exposed in the block, usually 3 for MACE-MP
    correlation = 3
    
    # Gate is also not exposed, usually silu
    gate = torch.nn.functional.silu

    # Radial MLP and Type are also often hidden
    radial_MLP = getattr(pretrained_mace, 'radial_MLP', [64, 64, 64])
    radial_type = getattr(pretrained_mace, 'radial_type', "bessel")

    model = AlchemicalModel(
        r_max=pretrained_mace.r_max.item(),
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=int(inter_0.edge_attrs_irreps.lmax), # Must match edge attributes, not hidden features
        interaction_cls=AlchemicalResidualInteractionBlock,
        interaction_cls_first=AlchemicalResidualInteractionBlock,
        num_interactions=pretrained_mace.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps("16x0e"), # Standard readout
        atomic_energies=atomic_energies,
        avg_num_neighbors=inter_0.avg_num_neighbors, 
        atomic_numbers=z_table.zs,
        correlation=correlation,
        gate=gate,
        radial_MLP=radial_MLP,
        radial_type=radial_type,
        atomic_inter_scale=scale,
        atomic_inter_shift=shift,
    )

    # 3. Transfer Weights
    # Load state dict with strict=False to allow for version differences (e.g. extra buffers like weights_max_zeroed in newer e3nn)
    model.load_state_dict(pretrained_mace.state_dict(), strict=False)
    
    # Transfer average num neighbors explicitly if needed
    for i in range(int(model.num_interactions)):
        model.interactions[i].avg_num_neighbors = pretrained_mace.interactions[i].avg_num_neighbors
        
    model.to(device)
    return model
