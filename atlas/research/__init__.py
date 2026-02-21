"""
Research method registry for switchable project methodologies.
"""

from atlas.research.method_registry import (
    MethodSpec,
    METHODS,
    get_method,
    list_methods,
    recommended_method_order,
)
from atlas.research.workflow_reproducible_graph import (
    IterationSnapshot,
    RunManifest,
    WorkflowReproducibleGraph,
)

__all__ = [
    "MethodSpec",
    "METHODS",
    "get_method",
    "list_methods",
    "recommended_method_order",
    "IterationSnapshot",
    "RunManifest",
    "WorkflowReproducibleGraph",
]
