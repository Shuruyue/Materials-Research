"""
Algorithmic synthesizability evaluator.

References / 参考文献:
- McDermott et al. (2021), graph-based reaction-network formalism:
  https://doi.org/10.1038/s41467-021-23339-x
- Martins (1984), multi-objective shortest path (MOSP):
  https://doi.org/10.1016/0377-2217(84)90077-8
- Feillet et al. (2004), label-setting dominance pruning for constrained paths:
  https://doi.org/10.1002/NET.20033
- Deb et al. (2002), non-dominated sorting and diversity ranking:
  https://ieeexplore.ieee.org/document/996017
- Ahmadi-Javid (2012), entropic risk measure:
  https://doi.org/10.1016/j.ejor.2011.11.016
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

try:
    import rustworkx as rx
except ImportError:  # pragma: no cover - optional dependency
    rx = None

from atlas.utils.registry import EVALUATORS

logger = logging.getLogger(__name__)

_SOURCE_NODE = "__ELEMENTS__"


def _coerce_int(value: object, default: int, *, min_value: int | None = None) -> int:
    fallback = int(default)
    if isinstance(value, bool):
        out = fallback
    else:
        try:
            numeric = float(value)
        except Exception:
            out = fallback
        else:
            if not math.isfinite(numeric) or not numeric.is_integer():
                out = fallback
            else:
                out = int(numeric)
    if min_value is not None:
        out = max(int(min_value), out)
    return int(out)


def _coerce_float(
    value: object,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        out = float(default)
    if min_value is not None:
        out = max(float(min_value), out)
    if max_value is not None:
        out = min(float(max_value), out)
    return float(out)


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "y", "on"}:
            return True
        if key in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


@dataclass(slots=True, frozen=True)
class _ReactionStep:
    src: str
    dst: str
    delta_g: float
    activation: float
    uncertainty: float
    provenance: str = "heuristic"


@dataclass(slots=True)
class _PathCandidate:
    nodes: tuple[str, ...]
    steps: tuple[_ReactionStep, ...]
    objectives: tuple[float, float, float, float]
    scalar_penalty: float = math.inf

    def as_string(self) -> str:
        return " -> ".join(self.nodes)


@EVALUATORS.register("rustworkx_pathfinder")
class SynthesisPathfinder:
    """
    Multi-objective synthesis-path evaluator.

    Core algorithm:
    1) Build candidate synthesis-state graph from formula decomposition.
    2) Enumerate simple paths with constraints (bounded-length RCSP approximation).
    3) Rank feasible paths by Pareto front + augmented Tchebycheff scalarization.

    核心流程:
    1) 由化学式分解构建候选合成状态图。
    2) 在约束条件下枚举简单路径（受限路径问题近似）。
    3) 使用 Pareto 前沿与增广 Tchebycheff 标量化进行排序。
    """

    def __init__(
        self,
        precursor_db: str = "jarvis_default",
        max_pathways: int = 5,
        max_steps: int = 6,
        max_subset_nodes: int = 96,
        max_path_expansions: int = 120000,
        max_rank_candidates: int = 2048,
        max_total_activation: float = 2.6,
        max_total_delta_g: float = 0.20,
        score_threshold: float = 0.18,
        risk_aversion: float = 3.0,
        objective_weights: dict[str, float] | None = None,
        allow_jump_edges: bool = True,
        use_adaptive_threshold: bool = True,
        threshold_min: float = 0.12,
        threshold_max: float = 0.55,
        threshold_uncertainty_weight: float = 0.35,
        threshold_step_weight: float = 0.03,
        energy_prior_center: float = -0.50,
        energy_prior_temperature: float = 0.22,
    ):
        self.precursor_db = str(precursor_db)
        self.max_pathways = _coerce_int(max_pathways, 5, min_value=1)
        self.max_steps = _coerce_int(max_steps, 6, min_value=2)
        self.max_subset_nodes = _coerce_int(max_subset_nodes, 96, min_value=8)
        self.max_path_expansions = _coerce_int(max_path_expansions, 120000, min_value=100)
        self.max_rank_candidates = _coerce_int(max_rank_candidates, 2048, min_value=16)
        self.max_total_activation = _coerce_float(max_total_activation, 2.6)
        self.max_total_delta_g = _coerce_float(max_total_delta_g, 0.20)
        self.score_threshold = _coerce_float(score_threshold, 0.18, min_value=0.0, max_value=1.0)
        self.risk_aversion = _coerce_float(risk_aversion, 3.0, min_value=1e-6)
        self.allow_jump_edges = _coerce_bool(allow_jump_edges, True)
        self.use_adaptive_threshold = _coerce_bool(use_adaptive_threshold, True)
        tmin = _coerce_float(threshold_min, 0.12, min_value=0.0, max_value=1.0)
        tmax = _coerce_float(threshold_max, 0.55, min_value=0.0, max_value=1.0)
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        self.threshold_min = tmin
        self.threshold_max = tmax
        self.threshold_uncertainty_weight = _coerce_float(threshold_uncertainty_weight, 0.35, min_value=0.0)
        self.threshold_step_weight = _coerce_float(threshold_step_weight, 0.03, min_value=0.0)
        self.energy_prior_center = _coerce_float(energy_prior_center, -0.50)
        self.energy_prior_temperature = _coerce_float(energy_prior_temperature, 0.22, min_value=1e-6)

        self.objective_weights = self._normalize_weights(objective_weights)
        self._route_library = self._load_route_library(self.precursor_db)

        if rx is None:
            logger.info(
                "rustworkx is not installed; falling back to pure-Python path search. "
                "Algorithm remains available."
            )

    @staticmethod
    def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
        defaults = {
            "thermo": 0.45,
            "activation": 0.25,
            "steps": 0.15,
            "risk": 0.15,
        }
        if not weights:
            return defaults

        out = defaults.copy()
        for key, value in weights.items():
            if key not in defaults:
                continue
            out[key] = _coerce_float(value, defaults[key], min_value=0.0)

        total = sum(out.values())
        if total <= 0:
            return defaults
        return {k: v / total for k, v in out.items()}

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            out = float(value)
            if math.isnan(out) or math.isinf(out):
                return default
            return out
        except Exception:
            return default

    @staticmethod
    def _format_count(count: float) -> str:
        nearest = round(count)
        if abs(count - nearest) < 1e-9:
            return "" if nearest == 1 else str(int(nearest))
        txt = f"{count:.3f}".rstrip("0").rstrip(".")
        return txt

    @classmethod
    def _parse_formula(cls, formula: str) -> tuple[list[str], dict[str, float]]:
        tokens = re.findall(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", str(formula))
        if not tokens:
            return [], {}

        order: list[str] = []
        amounts: dict[str, float] = {}
        for elem, amount_txt in tokens:
            amount = float(amount_txt) if amount_txt else 1.0
            if elem not in amounts:
                order.append(elem)
                amounts[elem] = 0.0
            amounts[elem] += amount
        return order, amounts

    @classmethod
    def _formula_signature(cls, formula: str) -> tuple:
        order, amounts = cls._parse_formula(formula)
        if not order:
            return (str(formula).strip(),)
        key = []
        for elem in sorted(amounts):
            key.append((elem, round(float(amounts[elem]), 8)))
        return tuple(key)

    @classmethod
    def _matches_target_formula(cls, node_formula: str, target_formula: str) -> bool:
        return cls._formula_signature(node_formula) == cls._formula_signature(target_formula)

    @classmethod
    def _subset_formula(
        cls,
        subset: tuple[str, ...],
        full_order: list[str],
        full_amounts: dict[str, float],
    ) -> str:
        in_subset = set(subset)
        parts: list[str] = []
        for elem in full_order:
            if elem not in in_subset:
                continue
            count = full_amounts.get(elem, 0.0)
            if count <= 0:
                continue
            parts.append(f"{elem}{cls._format_count(count)}")
        return "".join(parts)

    @staticmethod
    def _choose_subsets(elements: list[str], max_subset_nodes: int) -> list[tuple[str, ...]]:
        n = len(elements)
        if n <= 0:
            return []

        all_subsets: list[tuple[str, ...]] = []
        for size in range(1, n + 1):
            all_subsets.extend(combinations(elements, size))

        if len(all_subsets) <= max_subset_nodes:
            return all_subsets

        selected: list[tuple[str, ...]] = []
        selected_set: set[tuple[str, ...]] = set()
        wanted_sizes = [1, 2, max(1, n - 1), n]
        for size in wanted_sizes:
            for subset in combinations(elements, size):
                if subset in selected_set:
                    continue
                selected.append(subset)
                selected_set.add(subset)
                if len(selected) >= max_subset_nodes:
                    return selected

        for subset in all_subsets:
            if subset in selected_set:
                continue
            selected.append(subset)
            selected_set.add(subset)
            if len(selected) >= max_subset_nodes:
                break
        return selected

    def _load_route_library(self, precursor_db: str) -> dict[str, list[dict]]:
        key = str(precursor_db).strip()
        if key in {"", "jarvis_default", "default"}:
            return {}

        path = Path(key)
        if not path.exists():
            logger.warning(
                "precursor_db path %s does not exist; using heuristic route generator.",
                key,
            )
            return {}

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse precursor_db JSON at %s: %s", key, exc)
            return {}

        routes = payload.get("routes") if isinstance(payload, dict) else None
        if not isinstance(routes, dict):
            logger.warning(
                "precursor_db JSON at %s has no 'routes' object; using heuristic route generator.",
                key,
            )
            return {}

        normalized: dict[str, list[dict]] = {}
        for formula, entries in routes.items():
            if not isinstance(entries, list):
                continue
            normalized[str(formula)] = [e for e in entries if isinstance(e, dict)]
        return normalized

    def _edge_params(
        self,
        size_src: int,
        size_dst: int,
        size_total: int,
        candidate_energy: float,
        to_target: bool,
    ) -> tuple[float, float, float]:
        # Potential-based decomposition to preserve global energetic trend.
        # 基于势函数的分解：在局部步进里保持全局能量趋势。
        jump = max(1, size_dst - size_src)
        frac_src = 0.0 if size_total <= 0 else size_src / size_total
        frac_dst = 1.0 if size_total <= 0 else size_dst / size_total

        phi_src = candidate_energy * (frac_src ** 1.20)
        phi_dst = candidate_energy * (frac_dst ** 1.20)

        mixing_penalty = 0.06 * max(0, jump - 1) ** 2 + 0.03 * max(0, size_dst - 2)
        delta_g = (phi_dst - phi_src) + mixing_penalty
        if to_target:
            delta_g -= 0.04

        activation = 0.10 + 0.16 * jump + 0.05 * max(0, size_dst - 2)
        uncertainty = 0.03 + 0.018 * jump + 0.01 * min(3.0, abs(candidate_energy))
        return delta_g, activation, uncertainty

    def _build_heuristic_graph(
        self,
        formula: str,
        candidate_energy: float,
    ) -> tuple[dict[str, list[_ReactionStep]], str]:
        order, amounts = self._parse_formula(formula)
        if not order:
            return {}, formula

        subsets = self._choose_subsets(order, self.max_subset_nodes)
        subset_to_node: dict[tuple[str, ...], str] = {}
        node_to_subset: dict[str, tuple[str, ...]] = {}

        for subset in subsets:
            node = self._subset_formula(subset, order, amounts)
            if not node:
                continue
            subset_to_node[subset] = node
            node_to_subset[node] = subset

        target_subset = tuple(order)
        target_node = subset_to_node.get(target_subset)
        if not target_node:
            target_node = formula
            subset_to_node[target_subset] = target_node
            node_to_subset[target_node] = target_subset

        adj: dict[str, list[_ReactionStep]] = {_SOURCE_NODE: []}
        for node in node_to_subset:
            adj.setdefault(node, [])

        for subset, node in subset_to_node.items():
            if len(subset) != 1:
                continue
            delta_g, activation, uncertainty = self._edge_params(
                size_src=0,
                size_dst=1,
                size_total=len(order),
                candidate_energy=candidate_energy,
                to_target=len(order) == 1,
            )
            adj[_SOURCE_NODE].append(
                _ReactionStep(
                    src=_SOURCE_NODE,
                    dst=node,
                    delta_g=delta_g,
                    activation=activation,
                    uncertainty=uncertainty,
                    provenance="element_assembly",
                )
            )

        for subset_a, node_a in subset_to_node.items():
            size_a = len(subset_a)
            if size_a == len(order):
                continue

            for subset_b, node_b in subset_to_node.items():
                size_b = len(subset_b)
                if size_b <= size_a:
                    continue
                if not set(subset_a).issubset(subset_b):
                    continue
                jump = size_b - size_a
                if not self.allow_jump_edges and jump != 1:
                    continue
                if jump > 2 and len(order) > 4:
                    continue

                delta_g, activation, uncertainty = self._edge_params(
                    size_src=size_a,
                    size_dst=size_b,
                    size_total=len(order),
                    candidate_energy=candidate_energy,
                    to_target=(node_b == target_node),
                )
                adj[node_a].append(
                    _ReactionStep(
                        src=node_a,
                        dst=node_b,
                        delta_g=delta_g,
                        activation=activation,
                        uncertainty=uncertainty,
                        provenance="subset_extension",
                    )
                )

        for steps in adj.values():
            steps.sort(key=lambda e: (e.dst, e.activation, e.delta_g))
        return adj, target_node

    @classmethod
    def _route_dict_to_path(cls, route: dict) -> _PathCandidate | None:
        nodes = route.get("nodes")
        if not isinstance(nodes, list) or len(nodes) < 2:
            return None

        steps_raw = route.get("steps")
        if isinstance(steps_raw, list) and steps_raw:
            steps: list[_ReactionStep] = []
            for item in steps_raw:
                if not isinstance(item, dict):
                    return None
                src = str(item.get("src", "")).strip()
                dst = str(item.get("dst", "")).strip()
                if not src or not dst:
                    return None
                steps.append(
                    _ReactionStep(
                        src=src,
                        dst=dst,
                        delta_g=cls._safe_float(item.get("delta_g"), 0.0),
                        activation=abs(cls._safe_float(item.get("activation"), 0.0)),
                        uncertainty=max(0.0, cls._safe_float(item.get("uncertainty"), 0.05)),
                        provenance=str(item.get("provenance", "library")),
                    )
                )
        else:
            delta_gs = route.get("delta_g", [])
            activations = route.get("activation", [])
            uncertainties = route.get("uncertainty", [])
            if not isinstance(delta_gs, list) or not isinstance(activations, list):
                return None
            if len(nodes) - 1 != len(delta_gs) or len(nodes) - 1 != len(activations):
                return None
            steps = []
            for idx in range(len(nodes) - 1):
                unc = 0.05
                if isinstance(uncertainties, list) and idx < len(uncertainties):
                    unc = max(0.0, cls._safe_float(uncertainties[idx], 0.05))
                steps.append(
                    _ReactionStep(
                        src=str(nodes[idx]),
                        dst=str(nodes[idx + 1]),
                        delta_g=cls._safe_float(delta_gs[idx], 0.0),
                        activation=abs(cls._safe_float(activations[idx], 0.0)),
                        uncertainty=unc,
                        provenance="library",
                    )
                )

        node_tuple = tuple(str(v) for v in nodes)
        objective_placeholder = (0.0, 0.0, 0.0, 0.0)
        return _PathCandidate(nodes=node_tuple, steps=tuple(steps), objectives=objective_placeholder)

    def _library_paths_for_formula(self, formula: str) -> list[_PathCandidate]:
        payloads = self._route_library.get(str(formula), [])
        out: list[_PathCandidate] = []
        for payload in payloads:
            parsed = self._route_dict_to_path(payload)
            if parsed is not None:
                out.append(parsed)
        return out

    @staticmethod
    def _label_dominates(a: tuple[float, float, float, float], b: tuple[float, float, float, float], eps: float = 1e-12) -> bool:
        return (
            a[0] <= b[0] + eps
            and a[1] <= b[1] + eps
            and a[2] <= b[2] + eps
            and a[3] <= b[3] + eps
            and (a[0] < b[0] - eps or a[1] < b[1] - eps or a[2] < b[2] - eps or a[3] < b[3] - eps)
        )

    @classmethod
    def _register_non_dominated_label(
        cls,
        labels: dict[str, list[tuple[float, float, float, float]]],
        node: str,
        label: tuple[float, float, float, float],
    ) -> bool:
        bucket = labels.setdefault(node, [])
        for incumbent in bucket:
            if cls._label_dominates(incumbent, label):
                return False

        survivors: list[tuple[float, float, float, float]] = []
        for incumbent in bucket:
            if not cls._label_dominates(label, incumbent):
                survivors.append(incumbent)
        survivors.append(label)
        labels[node] = survivors
        return True

    def _enumerate_paths(
        self,
        adj: dict[str, list[_ReactionStep]],
        source: str,
        target: str,
        max_steps: int,
    ) -> list[_PathCandidate]:
        if source not in adj or target not in adj:
            return []

        out: list[_PathCandidate] = []
        labels: dict[str, list[tuple[float, float, float, float]]] = {}
        stack: list[
            tuple[
                str,
                tuple[str, ...],
                tuple[_ReactionStep, ...],
                float,
                float,
                float,
            ]
        ] = [(source, (source,), tuple(), 0.0, 0.0, 0.0)]
        expansions = 0

        while stack:
            node, nodes, steps, total_delta_g, total_activation, cumulative_loss = stack.pop()
            if node == target and steps:
                out.append(_PathCandidate(nodes=nodes, steps=steps, objectives=(0.0, 0.0, 0.0, 0.0)))
                if len(out) >= self.max_rank_candidates:
                    break
                continue

            if len(steps) >= max_steps:
                continue

            if expansions >= self.max_path_expansions:
                logger.warning(
                    "Path expansion budget reached (%d); truncating enumeration.",
                    self.max_path_expansions,
                )
                break

            for edge in reversed(adj.get(node, [])):
                if edge.dst in nodes:
                    continue

                next_delta_g = total_delta_g + edge.delta_g
                next_activation = total_activation + edge.activation
                if next_activation > self.max_total_activation + 0.8:
                    continue
                if next_delta_g > self.max_total_delta_g + 0.8:
                    continue

                next_steps = len(steps) + 1
                edge_loss = edge.activation + max(0.0, edge.delta_g) + edge.uncertainty
                next_loss = cumulative_loss + edge_loss
                label = (
                    float(next_delta_g),
                    float(next_activation),
                    float(next_steps),
                    float(next_loss),
                )
                if not self._register_non_dominated_label(labels, edge.dst, label):
                    continue

                expansions += 1
                next_nodes = nodes + (edge.dst,)
                next_edge_seq = steps + (edge,)
                stack.append((edge.dst, next_nodes, next_edge_seq, next_delta_g, next_activation, next_loss))

        return out

    def _path_objectives(self, path: _PathCandidate) -> tuple[float, float, float, float]:
        total_delta_g = sum(step.delta_g for step in path.steps)
        total_activation = sum(step.activation for step in path.steps)
        num_steps = float(len(path.steps))

        losses = [
            step.activation + max(0.0, step.delta_g) + step.uncertainty
            for step in path.steps
        ]
        if not losses:
            entropic_risk = 0.0
        else:
            # Entropic risk regularization (Ahmadi-Javid, 2012).
            # 熵风险正则项，用于惩罚高尾部损失路径。
            gamma = self.risk_aversion
            m = max(losses)
            shifted = [math.exp(gamma * (loss - m)) for loss in losses]
            entropic_risk = m + math.log(sum(shifted) / len(shifted)) / gamma

        return (
            float(total_delta_g),
            float(total_activation),
            float(num_steps),
            float(entropic_risk),
        )

    def _annotate_objectives(self, paths: list[_PathCandidate]) -> list[_PathCandidate]:
        out: list[_PathCandidate] = []
        for path in paths:
            out.append(
                _PathCandidate(
                    nodes=path.nodes,
                    steps=path.steps,
                    objectives=self._path_objectives(path),
                    scalar_penalty=path.scalar_penalty,
                )
            )
        return out

    def _is_feasible(self, obj: tuple[float, float, float, float]) -> bool:
        total_delta_g, total_activation, num_steps, _ = obj
        if total_activation > self.max_total_activation:
            return False
        if total_delta_g > self.max_total_delta_g:
            return False
        return not num_steps > float(self.max_steps)

    @staticmethod
    def _dominates(a: tuple[float, ...], b: tuple[float, ...], eps: float = 1e-12) -> bool:
        le_all = all(x <= y + eps for x, y in zip(a, b, strict=True))
        lt_any = any(x < y - eps for x, y in zip(a, b, strict=True))
        return le_all and lt_any

    @classmethod
    def _pareto_front(cls, paths: list[_PathCandidate]) -> list[_PathCandidate]:
        front: list[_PathCandidate] = []
        for cand in paths:
            dominated = False
            survivors: list[_PathCandidate] = []
            for incumbent in front:
                if cls._dominates(incumbent.objectives, cand.objectives):
                    dominated = True
                    break
                if not cls._dominates(cand.objectives, incumbent.objectives):
                    survivors.append(incumbent)
            if not dominated:
                survivors.append(cand)
                front = survivors
        return front

    def _scalarize(self, path: _PathCandidate, ideals: tuple[float, ...], nadirs: tuple[float, ...]) -> float:
        normalized: list[float] = []
        for idx, value in enumerate(path.objectives):
            denom = max(1e-12, nadirs[idx] - ideals[idx])
            normalized.append((value - ideals[idx]) / denom)

        weights = self.objective_weights
        weighted = [
            weights["thermo"] * normalized[0],
            weights["activation"] * normalized[1],
            weights["steps"] * normalized[2],
            weights["risk"] * normalized[3],
        ]

        # Augmented Tchebycheff scalarization for Pareto ranking.
        # 参考多目标优化中的增广 Tchebycheff 规范。
        return max(weighted) + 0.05 * sum(weighted)

    def _rank_paths(self, paths: list[_PathCandidate]) -> list[_PathCandidate]:
        if not paths:
            return []

        ideals = tuple(min(p.objectives[i] for p in paths) for i in range(4))
        nadirs = tuple(max(p.objectives[i] for p in paths) for i in range(4))

        scored: list[_PathCandidate] = []
        for path in paths:
            scored.append(
                _PathCandidate(
                    nodes=path.nodes,
                    steps=path.steps,
                    objectives=path.objectives,
                    scalar_penalty=self._scalarize(path, ideals, nadirs),
                )
            )

        scored.sort(key=lambda p: (p.scalar_penalty, p.objectives[3], p.objectives[2], p.objectives[0]))
        return scored

    def _energy_prior(self, candidate_energy: float) -> float:
        # Monotone prior p(E): lower formation energy => higher prior probability.
        # 单调先验：形成能越低，先验可合成概率越高。
        x = (candidate_energy - self.energy_prior_center) / self.energy_prior_temperature
        if not math.isfinite(x):
            return 0.5
        # Numerically stable logistic: avoids overflow when |x| is large.
        # 数值稳定 sigmoid，避免极端能量下的 exp 溢出。
        if x >= 0:
            z = math.exp(-x)
            return z / (1.0 + z)
        z = math.exp(x)
        return 1.0 / (1.0 + z)

    def _score_from_best(self, best: _PathCandidate, candidate_energy: float) -> float:
        thermo = self._safe_float(best.objectives[0], 0.0)
        activation = self._safe_float(best.objectives[1], 0.0)
        steps = self._safe_float(best.objectives[2], 0.0)
        risk = self._safe_float(best.objectives[3], 0.0)

        stability_term = 1.0 - math.exp(-max(0.0, -thermo))
        kinetic_term = math.exp(-0.8 * max(0.0, activation))
        complexity_term = math.exp(-0.35 * max(0.0, steps - 1.0))
        risk_term = math.exp(-max(0.0, risk))
        prior = self._energy_prior(candidate_energy)

        score = stability_term * kinetic_term * complexity_term * risk_term * prior
        if not math.isfinite(score):
            return 0.0
        return float(max(0.0, min(1.0, score)))

    def _decision_threshold(self, best: _PathCandidate) -> float:
        if not self.use_adaptive_threshold:
            return float(max(0.0, min(1.0, self.score_threshold)))

        mean_uncertainty = 0.0
        if best.steps:
            mean_uncertainty = sum(step.uncertainty for step in best.steps) / len(best.steps)
        step_excess = max(0.0, best.objectives[2] - 1.0)

        threshold = (
            self.score_threshold
            + self.threshold_uncertainty_weight * mean_uncertainty
            + self.threshold_step_weight * step_excess
        )
        threshold = max(self.threshold_min, threshold)
        threshold = min(self.threshold_max, threshold)
        return float(max(0.0, min(1.0, threshold)))

    def _empty_result(self, graph_nodes: int = 0, graph_edges: int = 0) -> dict:
        return {
            "synthesizable": False,
            "score": 0.0,
            "pathway": [],
            "graph_nodes": int(graph_nodes),
            "graph_edges": int(graph_edges),
            "pathway_count": 0,
            "pareto_front_size": 0,
            "metrics": {
                "total_delta_g": 0.0,
                "total_activation": 0.0,
                "num_steps": 0.0,
                "entropic_risk": 0.0,
                "scalar_penalty": None,
                "decision_threshold": float(max(0.0, min(1.0, self.score_threshold))),
            },
        }

    def evaluate(
        self,
        candidate_formula: str,
        candidate_energy: float | None,
        pathways: list[dict] | None = None,
    ) -> dict:
        """
        Evaluate synthesizability via constrained path ranking.

        Parameters
        ----------
        candidate_formula:
            Target composition formula.
        candidate_energy:
            Predicted formation energy per atom.
        pathways:
            Optional explicit pathways (dict payloads) to override heuristic graph.

        Returns
        -------
        dict with `synthesizable`, `score`, and ranked pathway summaries.
        """

        formula = str(candidate_formula).strip()
        if not formula:
            return self._empty_result()

        energy = self._safe_float(candidate_energy, 0.0)
        explicit_pathways_supplied = pathways is not None

        custom_paths: list[_PathCandidate] = []
        if pathways:
            for route in pathways:
                if not isinstance(route, dict):
                    continue
                parsed = self._route_dict_to_path(route)
                if parsed is not None:
                    custom_paths.append(parsed)

        if not custom_paths:
            custom_paths.extend(self._library_paths_for_formula(formula))

        # Enforce target consistency: only keep routes whose terminal node matches target formula.
        # 目标一致性过滤：仅保留终点等于目标化学式的路径。
        custom_paths = [
            p for p in custom_paths
            if p.nodes and self._matches_target_formula(p.nodes[-1], formula)
        ]
        if explicit_pathways_supplied and not custom_paths:
            return self._empty_result(graph_nodes=0, graph_edges=0)

        graph_nodes = 0
        graph_edges = 0
        if custom_paths:
            annotated = self._annotate_objectives(custom_paths)
            graph_nodes = len({n for p in custom_paths for n in p.nodes})
            graph_edges = sum(len(p.steps) for p in custom_paths)
        else:
            adj, target = self._build_heuristic_graph(formula, energy)
            graph_nodes = len(adj)
            graph_edges = sum(len(v) for v in adj.values())
            if not adj or target not in adj:
                return self._empty_result(graph_nodes=graph_nodes, graph_edges=graph_edges)
            enumerated = self._enumerate_paths(adj, source=_SOURCE_NODE, target=target, max_steps=self.max_steps)
            annotated = self._annotate_objectives(enumerated)

        feasible = [p for p in annotated if self._is_feasible(p.objectives)]
        if not feasible:
            return self._empty_result(graph_nodes=graph_nodes, graph_edges=graph_edges)

        if len(feasible) > self.max_rank_candidates:
            feasible = sorted(
                feasible,
                key=lambda p: (p.objectives[0], p.objectives[1], p.objectives[2], p.objectives[3]),
            )[: self.max_rank_candidates]

        pareto = self._pareto_front(feasible)
        ranked = self._rank_paths(pareto)
        if not ranked:
            return self._empty_result(graph_nodes=graph_nodes, graph_edges=graph_edges)

        best = ranked[0]
        score = self._score_from_best(best, energy)
        decision_threshold = self._decision_threshold(best)
        synthesizable = bool(score >= decision_threshold)

        pathway_strings = [p.as_string() for p in ranked[: self.max_pathways]]
        return {
            "synthesizable": synthesizable,
            "score": score,
            "pathway": pathway_strings,
            "graph_nodes": int(graph_nodes),
            "graph_edges": int(graph_edges),
            "pathway_count": len(ranked),
            "pareto_front_size": len(pareto),
            "metrics": {
                "total_delta_g": float(best.objectives[0]),
                "total_activation": float(best.objectives[1]),
                "num_steps": float(best.objectives[2]),
                "entropic_risk": float(best.objectives[3]),
                "scalar_penalty": float(best.scalar_penalty),
                "decision_threshold": float(decision_threshold),
                "best_path_provenance": list({step.provenance for step in best.steps}),
            },
        }
