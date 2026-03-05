"""
Switchable research methods for ATLAS.

Methods are intentionally lightweight descriptors so pipelines can choose
one strategy and keep alternatives as ready fallbacks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


def _normalize_lookup_key(
    value: object,
    *,
    field_name: str = "key",
    lowercase: bool = False,
) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    key = value.strip()
    if not key:
        raise ValueError(f"{field_name} must be a non-empty string")
    if lowercase:
        key = key.lower()
    return key


@dataclass(frozen=True)
class MethodSpec:
    """A high-level methodology profile for experiments."""

    key: str
    name: str
    summary: str
    strengths: tuple[str, ...] = field(default_factory=tuple)
    tradeoffs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        key = _normalize_lookup_key(
            self.key,
            field_name="MethodSpec.key",
            lowercase=True,
        )
        name = _normalize_lookup_key(self.name, field_name="MethodSpec.name")
        summary = _normalize_lookup_key(self.summary, field_name="MethodSpec.summary")

        strengths = tuple(self._normalize_text_items(self.strengths, "strengths"))
        tradeoffs = tuple(self._normalize_text_items(self.tradeoffs, "tradeoffs"))

        object.__setattr__(self, "key", key)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "strengths", strengths)
        object.__setattr__(self, "tradeoffs", tradeoffs)

    @staticmethod
    def _normalize_text_items(values: Iterable[str], field_name: str) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, str):
                raise TypeError(
                    f"MethodSpec.{field_name} entries must be strings, got {type(value).__name__}"
                )
            text = value.strip()
            if not text:
                raise ValueError(f"MethodSpec.{field_name} entries must be non-empty")
            if text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized


class MethodRegistry:
    def __init__(self) -> None:
        self._methods: dict[str, MethodSpec] = {}

    def register(self, spec: MethodSpec, *, replace: bool = False) -> None:
        if not isinstance(spec, MethodSpec):
            raise TypeError(f"spec must be MethodSpec, got {type(spec)!r}")
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean")
        if spec.key in self._methods and not replace:
            raise ValueError(
                f"Method '{spec.key}' already registered. Use replace=True to overwrite."
            )
        self._methods[spec.key] = spec

    def get(self, key: str) -> MethodSpec:
        normalized_key = _normalize_lookup_key(
            key,
            field_name="method key",
            lowercase=True,
        )
        if normalized_key not in self._methods:
            available = ", ".join(sorted(self._methods.keys()))
            raise KeyError(f"Unknown method '{normalized_key}'. Available: {available}")
        return self._methods[normalized_key]

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
        strengths=(
            "Strong performance on structure-dependent properties",
            "Naturally supports forces/stress when model head is configured",
            "Good fit for inorganic and semiconductor crystal tasks",
        ),
        tradeoffs=(
            "Higher dependency and compute cost",
            "Needs robust graph preprocessing and model version control",
        ),
    )
)

METHODS.register(
    MethodSpec(
        key="descriptor_tabular",
        name="Descriptor + Tabular",
        summary="DScribe/matminer descriptors with conventional regressors.",
        strengths=(
            "Fast to train, easier ablations and interpretability",
            "Good low-data baseline for thesis experiments",
        ),
        tradeoffs=(
            "Feature engineering sensitive",
            "Usually lower ceiling than strong equivariant models",
        ),
    )
)

METHODS.register(
    MethodSpec(
        key="physics_screened_graph",
        name="Physics-Screened Graph",
        summary="Apply chemical/defect priors (e.g., SMACT, SnB-like heuristics) before GNN.",
        strengths=(
            "Improves candidate validity before expensive model runs",
            "Useful for doping/defect and semiconductor screening",
        ),
        tradeoffs=(
            "Requires domain priors and rule maintenance",
            "Might filter out novel edge cases if too strict",
        ),
    )
)

METHODS.register(
    MethodSpec(
        key="workflow_reproducible_graph",
        name="Workflow Reproducible Graph",
        summary="Graph-first discovery workflow with reproducibility manifests and benchmark-ready traces.",
        strengths=(
            "Run-level metadata and iteration trace for thesis-grade reproducibility",
            "Compatible with current graph/equivariant model stack",
            "Easy handoff to benchmark/reporting pipelines",
        ),
        tradeoffs=(
            "Adds workflow bookkeeping overhead",
            "Requires discipline in profile and experiment naming",
        ),
    )
)

METHODS.register(
    MethodSpec(
        key="gp_active_learning",
        name="GP Active Learning",
        summary="Gaussian-process surrogate acquisition over candidate-level descriptors.",
        strengths=(
            "Balances exploration and exploitation in limited-label settings",
            "Useful for early-stage search over inorganic/semiconductor candidates",
        ),
        tradeoffs=(
            "Surrogate quality depends on feature design",
            "Can be unstable with very sparse history",
        ),
    )
)

METHODS.register(
    MethodSpec(
        key="descriptor_microstructure",
        name="Descriptor Microstructure",
        summary="Descriptor-heavy branch for low-data and microstructure-oriented experiments.",
        strengths=(
            "Fast baseline cycle time",
            "Pairs well with matminer/dscribe/pymks style features",
        ),
        tradeoffs=(
            "Lower ceiling on complex structure-dependent properties",
            "Requires stronger feature curation",
        ),
    )
)


def get_method(key: str) -> MethodSpec:
    return METHODS.get(key)


def list_methods() -> list[MethodSpec]:
    return METHODS.list_all()


def recommended_method_order(primary: str = "workflow_reproducible_graph") -> list[str]:
    """Return a stable ordered list with selected primary first."""
    keys = METHODS.list_keys()
    primary_key = _normalize_lookup_key(primary, field_name="primary", lowercase=True)
    if primary_key in keys:
        keys.remove(primary_key)
        return [primary_key, *keys]
    return keys
