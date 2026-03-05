
import numpy as np
import pytest
import torch

from atlas.models.graph_builder import gaussian_expansion


def test_gaussian_expansion_standalone():
    """Test the standalone gaussian_expansion function."""
    distances = np.array([0.0, 1.0, 2.0, 5.0])
    cutoff = 5.0
    n_gaussians = 10

    expanded = gaussian_expansion(distances, cutoff, n_gaussians)

    assert expanded.shape == (4, 10)
    assert expanded.dtype == np.float32
    # Check bounds (0 to 1 for Gaussian)
    assert np.all(expanded >= 0.0)
    assert np.all(expanded <= 1.0)

    # Check center (0 dist should peak at first gaussian which is near 0)
    # The first center is at 0.0
    assert expanded[0, 0] > 0.9  # Should be close to 1.0

def test_topology_classifier_integration():
    """Test that topology.classifier.CrystalGraphBuilder uses the shared function."""
    from atlas.topology import CrystalGraphBuilder

    builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12)
    distances = np.array([0.0, 2.5, 5.0])

    # This calls _gaussian_expansion which calls the shared function
    expanded = builder._gaussian_expansion(distances, n_gaussians=20)

    assert expanded.shape == (3, 20)
    assert expanded.dtype == np.float32

def test_models_graph_builder_integration():
    """Test that models.graph_builder.CrystalGraphBuilder uses the shared function."""
    from atlas.models.graph_builder import CrystalGraphBuilder as ModelGraphBuilder

    builder = ModelGraphBuilder(cutoff=5.0)
    distances = np.array([0.0, 2.5, 5.0])

    expanded = builder._gaussian_expansion(distances, n_gaussians=20)

    assert expanded.shape == (3, 20)


def test_gaussian_expansion_single_basis_and_non_finite_distances():
    distances = np.array([0.0, np.nan, np.inf, -np.inf], dtype=float)
    expanded = gaussian_expansion(distances, cutoff=5.0, n_gaussians=1)
    assert expanded.shape == (4, 1)
    assert expanded.dtype == np.float32
    assert np.isfinite(expanded).all()


def test_gaussian_expansion_invalid_parameters_raise():
    with pytest.raises(ValueError, match="n_gaussians must be > 0"):
        gaussian_expansion(np.array([1.0]), cutoff=5.0, n_gaussians=0)
    with pytest.raises(ValueError, match="cutoff must be a finite value > 0"):
        gaussian_expansion(np.array([1.0]), cutoff=float("nan"), n_gaussians=10)
    with pytest.raises(ValueError, match="n_gaussians must be integer-valued"):
        gaussian_expansion(np.array([1.0]), cutoff=5.0, n_gaussians=1.5)
    with pytest.raises(ValueError, match="n_gaussians must be integer-valued, not boolean"):
        gaussian_expansion(np.array([1.0]), cutoff=5.0, n_gaussians=True)


def test_graph_builder_validates_init_params():
    from atlas.models.graph_builder import CrystalGraphBuilder

    with pytest.raises(ValueError, match="cutoff must be a finite value > 0"):
        CrystalGraphBuilder(cutoff=0.0)
    with pytest.raises(ValueError, match="max_neighbors must be > 0"):
        CrystalGraphBuilder(max_neighbors=0)
    with pytest.raises(ValueError, match="max_neighbors must be integer-valued"):
        CrystalGraphBuilder(max_neighbors=2.2)
    with pytest.raises(ValueError, match="max_neighbors must be integer-valued, not boolean"):
        CrystalGraphBuilder(max_neighbors=False)
    with pytest.raises(ValueError, match="cutoff must be a finite value > 0, not boolean"):
        CrystalGraphBuilder(cutoff=True)
    with pytest.raises(ValueError, match="compute_3body must be boolean-like"):
        CrystalGraphBuilder(compute_3body="maybe")


def test_graph_builder_coerces_compute_3body_flag():
    from atlas.models.graph_builder import CrystalGraphBuilder

    assert CrystalGraphBuilder(compute_3body="false").compute_3body is False
    assert CrystalGraphBuilder(compute_3body=1).compute_3body is True


def test_graph_builder_no_neighbors_falls_back_to_per_node_self_loops():
    from pymatgen.core import Lattice, Structure

    from atlas.models.graph_builder import CrystalGraphBuilder

    structure = Structure(
        lattice=Lattice.cubic(20.0),
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.0], [0.45, 0.45, 0.45]],
    )
    builder = CrystalGraphBuilder(cutoff=0.5, max_neighbors=1)
    graph = builder.structure_to_graph(structure)
    edge_index = graph["edge_index"]
    assert edge_index.shape[1] == len(structure)
    assert np.array_equal(edge_index[0], edge_index[1])


def test_graph_builder_empty_structure_raises():
    from pymatgen.core import Lattice, Structure

    from atlas.models.graph_builder import CrystalGraphBuilder

    empty = Structure(lattice=Lattice.cubic(3.0), species=[], coords=[])
    with pytest.raises(ValueError, match="empty structure"):
        CrystalGraphBuilder().structure_to_graph(empty)


def test_structure_to_pyg_validates_property_targets():
    from pymatgen.core import Lattice, Structure

    from atlas.models.graph_builder import CrystalGraphBuilder

    structure = Structure(
        lattice=Lattice.cubic(4.0),
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12)

    data = builder.structure_to_pyg(
        structure,
        band_gap=1.2,
        elastic_target=[1.0, 2.0, 3.0],
        tensor_target=torch.tensor([4.0, 5.0]),
    )
    assert torch.isfinite(data.band_gap).all()
    assert data.elastic_target.shape == (3,)
    assert data.tensor_target.shape == (2,)

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        builder.structure_to_pyg(structure, band_gap=float("nan"))
