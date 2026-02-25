"""
Switchable research methods for ATLAS.

Methods are intentionally lightweight descriptors so pipelines can choose
one strategy and keep alternatives as ready fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MethodSpec:
    """A high-level methodology profile for experiments."""

    key: str
    name: str
    summary: str
    strengths: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)


class MethodRegistry:
    def __init__(self):
        self._methods: dict[str, MethodSpec] = {}

    def register(self, spec: MethodSpec):
        self._methods[spec.key] = spec

    def get(self, key: str) -> MethodSpec:
        if key not in self._methods:
            available = ", ".join(sorted(self._methods.keys()))
            raise KeyError(f"Unknown method '{key}'. Available: {available}")
        return self._methods[key]

    def list_keys(self) -> list[str]:
        return sorted(self._methods.keys())

    def list_all(self) -> list[MethodSpec]:
        return [self._methods[k] for k in self.list_keys()]


METHODS = MethodRegistry()

# Recommended default for this project stage:
# graph/equivariant first, descriptor baseline and physics-screened fallback.
METHODS.register(
    MethodSpec(
        key="graph_equivariant",
        name="Graph Equivariant",
        summary="GNN/equivariant models for crystal property and potential learning.",
        strengths=[
            "Strong performance on structure-dependent properties",
            "Naturally supports forces/stress when model head is configured",
            "Good fit for inorganic and semiconductor crystal tasks",
        ],
        tradeoffs=[
            "Higher dependency and compute cost",
            "Needs robust graph preprocessing and model version control",
        ],
    )
)

METHODS.register(
    MethodSpec(
        key="descriptor_tabular",
        name="Descriptor + Tabular",
        summary="DScribe/matminer descriptors with conventional regressors.",
        strengths=[
            "Fast to train, easier ablations and interpretability",
            "Good low-data baseline for thesis experiments",
        ],
        tradeoffs=[
            "Feature engineering sensitive",
            "Usually lower ceiling than strong equivariant models",
        ],
    )
)

METHODS.register(
    MethodSpec(
        key="physics_screened_graph",
        name="Physics-Screened Graph",
        summary="Apply chemical/defect priors (e.g., SMACT, SnB-like heuristics) before GNN.",
        strengths=[
            "Improves candidate validity before expensive model runs",
            "Useful for doping/defect and semiconductor screening",
        ],
        tradeoffs=[
            "Requires domain priors and rule maintenance",
            "Might filter out novel edge cases if too strict",
        ],
    )
)

METHODS.register(
    MethodSpec(
        key="workflow_reproducible_graph",
        name="Workflow Reproducible Graph",
        summary="Graph-first discovery workflow with reproducibility manifests and benchmark-ready traces.",
        strengths=[
            "Run-level metadata and iteration trace for thesis-grade reproducibility",
            "Compatible with current graph/equivariant model stack",
            "Easy handoff to benchmark/reporting pipelines",
        ],
        tradeoffs=[
            "Adds workflow bookkeeping overhead",
            "Requires discipline in profile and experiment naming",
        ],
    )
)

METHODS.register(
    MethodSpec(
        key="gp_active_learning",
        name="GP Active Learning",
        summary="Gaussian-process surrogate acquisition over candidate-level descriptors.",
        strengths=[
            "Balances exploration and exploitation in limited-label settings",
            "Useful for early-stage search over inorganic/semiconductor candidates",
        ],
        tradeoffs=[
            "Surrogate quality depends on feature design",
            "Can be unstable with very sparse history",
        ],
    )
)

METHODS.register(
    MethodSpec(
        key="descriptor_microstructure",
        name="Descriptor Microstructure",
        summary="Descriptor-heavy branch for low-data and microstructure-oriented experiments.",
        strengths=[
            "Fast baseline cycle time",
            "Pairs well with matminer/dscribe/pymks style features",
        ],
        tradeoffs=[
            "Lower ceiling on complex structure-dependent properties",
            "Requires stronger feature curation",
        ],
    )
)


def get_method(key: str) -> MethodSpec:
    return METHODS.get(key)


def list_methods() -> list[MethodSpec]:
    return METHODS.list_all()


def recommended_method_order(primary: str = "workflow_reproducible_graph") -> list[str]:
    """Return a stable ordered list with selected primary first."""
    keys = METHODS.list_keys()
    if primary in keys:
        keys.remove(primary)
        return [primary, *keys]
    return keys
