
import numpy as np
import torch
import pytest
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
    from atlas.topology.classifier import CrystalGraphBuilder
    
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
