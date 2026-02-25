"""
ATLAS Test Configuration

Shared fixtures and utilities for test suite.
"""

import sys
from pathlib import Path

import pytest

# Ensure atlas package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def sample_cif_path():
    """Return path to a sample CIF file for testing (if available)."""
    cif_dir = PROJECT_ROOT / "data" / "raw"
    cif_files = list(cif_dir.glob("*.cif")) if cif_dir.exists() else []
    if cif_files:
        return cif_files[0]
    return None


@pytest.fixture
def device():
    """Return available device (cuda or cpu)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def pytest_collection_modifyitems(items):
    """
    Auto-classify tests by folder:
    - tests/integration/** -> integration
    - others -> unit
    """
    for item in items:
        node = item.nodeid.replace("\\", "/")
        if "tests/integration/" in node:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
