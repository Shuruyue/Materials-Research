"""
Native reaction-network evaluator with algorithmic ranking.

References / 参考文献:
- McDermott, Dwaraknath, Persson (2021), graph-based reaction network:
  https://doi.org/10.1038/s41467-021-23339-x
- Yen (1971), K-shortest loopless paths:
  https://www.jstor.org/stable/2629312
- Deb et al. (2002), NSGA-II non-dominated sorting:
  https://ieeexplore.ieee.org/document/996017
- Ahmadi-Javid (2012), entropic risk measure:
  https://doi.org/10.1016/j.ejor.2011.11.016
"""

from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import dataclass
from importlib import import_module

import numpy as np

from atlas.utils.registry import EVALUATORS

logger = logging.getLogger(__name__)

# Lazy-initialized native reaction-network symbols.
ReactionNetwork = None
PathwaySolver = None
Softplus = None
WeightedSum = None
PathwaySet = None
BasicReaction = None
get_computed_rxn = None
Composition = None
HAS_NATIVE_RXN_NETWORK = False
NATIVE_IMPORT_ERROR: Exception | None = None


def _bootstrap_native_rxn_network() -> bool:
    """
    Wire vendored `atlas.third_party.rxn_network` as top-level `rxn_network`.
    将 vendored 包映射到顶层命名空间，确保原生导入链可用。
    """

    global ReactionNetwork
    global PathwaySolver
    global Softplus
    global WeightedSum
    global PathwaySet
    global BasicReaction
    global get_computed_rxn
    global Composition
    global NATIVE_IMPORT_ERROR

    try:
        vendored_pkg = import_module("atlas.third_party.rxn_network")
        sys.modules.setdefault("rxn_network", vendored_pkg)

        ReactionNetwork = import_module("rxn_network.network.network").ReactionNetwork
        PathwaySolver = import_module("rxn_network.pathways.solver").PathwaySolver
        Softplus = import_module("rxn_network.costs.functions").Softplus
        WeightedSum = import_module("rxn_network.costs.functions").WeightedSum
        PathwaySet = import_module("rxn_network.pathways.pathway_set").PathwaySet
        BasicReaction = import_module("rxn_network.reactions.basic").BasicReaction
        get_computed_rxn = import_module("rxn_network.enumerators.utils").get_computed_rxn
        Composition = import_module("rxn_network.core").Composition
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        NATIVE_IMPORT_ERROR = exc
        logger.warning(f"Could not import vendored rxn_network stack: {exc}")
        return False


HAS_NATIVE_RXN_NETWORK = _bootstrap_native_rxn_network()


@dataclass(slots=True)
class _PathMetrics:
    average_cost: float
    driving_force: float
    num_steps: float
    uncertainty: float
    intermediate_complexity: float
    entropic_risk: float
    risk_excess: float

    def as_dict(self) -> dict[str, float]:
        return {
            "average_cost": float(self.average_cost),
            "driving_force": float(self.driving_force),
            "num_steps": float(self.num_steps),
            "uncertainty": float(self.uncertainty),
            "intermediate_complexity": float(self.intermediate_complexity),
            "entropic_risk": float(self.entropic_risk),
            "risk_excess": float(self.risk_excess),
        }


@EVALUATORS.register("rxn_network_native")
class NativeReactionNetworkEvaluator:
    """
    Multi-objective synthesis evaluator backed by vendored reaction-network.

    Core algorithm:
    1) Build or consume candidate pathways.
    2) Rank pathways by Pareto fronts (NSGA-II style non-dominated sorting).
    3) Break ties by crowding distance and risk-regularized scalar penalty.

    核心流程:
    1) 构建或读取候选反应路径。
    2) 用 Pareto 非支配排序进行主排序。
    3) 用 crowding distance + 风险正则化标量罚项做次排序。
    """

    def __init__(
        self,
        cost_function: str = "softplus",
        max_num_pathways: int = 5,
        k_shortest_paths: int = 25,
        precursors: list[str] | None = None,
        use_balanced_solver: bool = True,
        max_num_combos: int = 4,
        find_intermediate_rxns: bool = True,
        intermediate_rxn_energy_cutoff: float = 0.0,
        use_basic_enumerator: bool = True,
        use_minimize_enumerator: bool = False,
        filter_interdependent: bool = True,
        chunk_size: int = 100000,
        risk_aversion: float = 2.0,
        objective_weights: dict[str, float] | None = None,
        uncertainty_default: float = 0.05,
        fallback_mode: str = "conservative",
        require_native: bool = False,
    ):
        self.cost_function = self._canonicalize_cost_function(cost_function)
        self.max_num_pathways = max(1, int(max_num_pathways))
        self.k_shortest_paths = max(self.max_num_pathways, int(k_shortest_paths))
        self.precursors = list(precursors) if precursors else []

        self.use_balanced_solver = bool(use_balanced_solver)
        self.max_num_combos = int(max_num_combos)
        self.find_intermediate_rxns = bool(find_intermediate_rxns)
        self.intermediate_rxn_energy_cutoff = float(intermediate_rxn_energy_cutoff)
        self.use_basic_enumerator = bool(use_basic_enumerator)
        self.use_minimize_enumerator = bool(use_minimize_enumerator)
        self.filter_interdependent = bool(filter_interdependent)
        self.chunk_size = int(chunk_size)

        self.risk_aversion = max(1e-8, float(risk_aversion))
        self.uncertainty_default = max(0.0, float(uncertainty_default))
        self.fallback_mode = str(fallback_mode).strip().lower()
        self.objective_weights = self._normalize_weights(objective_weights)

        self._entries = None
        self._reaction_set = None
        self._network = None
        self._pathways = None
        self._cached_cost_function_obj = None

        self.native_available = bool(HAS_NATIVE_RXN_NETWORK)
        if require_native and not self.native_available:
            raise RuntimeError(
                "rxn_network native stack is unavailable; "
                f"import error: {NATIVE_IMPORT_ERROR!r}"
            )

    @staticmethod
    def _canonicalize_cost_function(name: str) -> str:
        key = str(name).strip().lower().replace(" ", "_")
        # Backward-compatible alias kept from previous placeholder implementation.
        # 兼容旧配置别名：soft_mish -> softplus。
        if key in {"soft_mish", "softplus", "soft_plus"}:
            return "softplus"
        if key in {"weighted_sum", "linear"}:
            return "weighted_sum"
        return "softplus"

    @staticmethod
    def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
        defaults = {
            "average_cost": 0.30,
            "driving_force": 0.20,  # maximize
            "num_steps": 0.15,
            "uncertainty": 0.10,
            "intermediate_complexity": 0.10,
            "entropic_risk": 0.15,
        }
        if not weights:
            return defaults

        merged = defaults.copy()
        for key, value in weights.items():
            try:
                merged[str(key)] = max(0.0, float(value))
            except Exception:
                continue
        total = sum(merged.values())
        if total <= 0:
            return defaults
        return {k: v / total for k, v in merged.items()}

    def set_context(
        self,
        *,
        entries: object | None = None,
        reaction_set: object | None = None,
        network: object | None = None,
        pathways: object | None = None,
        precursors: list[str] | None = None,
    ) -> None:
        """Set reusable context for repeated evaluations."""
        if entries is not None:
            self._entries = entries
        if reaction_set is not None:
            self._reaction_set = reaction_set
        if network is not None:
            self._network = network
        if pathways is not None:
            self._pathways = pathways
        if precursors is not None:
            self.precursors = list(precursors)

    def clear_context(self) -> None:
        self._entries = None
        self._reaction_set = None
        self._network = None
        self._pathways = None

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _to_path_list(pathways: object | None) -> list[object]:
        if pathways is None:
            return []
        if hasattr(pathways, "paths"):
            return list(pathways.paths)
        if isinstance(pathways, list):
            return pathways
        try:
            return list(pathways)  # type: ignore[arg-type]
        except Exception:
            return []

    @staticmethod
    def _infer_precursors_from_formula(formula: str) -> list[str]:
        # Lightweight parser fallback if no explicit precursor list is provided.
        # 在未指定前驱体时，采用元素符号的轻量解析作为回退。
        tokens = re.findall(r"[A-Z][a-z]?", str(formula))
        return sorted(set(tokens))

    @staticmethod
    def _canonical_formula(value: object) -> str:
        if value is None:
            return ""
        if hasattr(value, "reduced_formula"):
            try:
                return str(value.reduced_formula)
            except Exception:
                pass
        text = str(value)
        if Composition is not None:
            try:
                return str(Composition(text).reduced_formula)
            except Exception:
                pass
        return text.replace(" ", "")

    def _path_formulas(self, path: object) -> set[str]:
        formulas: set[str] = set()

        products = getattr(path, "products", None)
        if products is not None:
            for item in products:
                f = self._canonical_formula(item)
                if f:
                    formulas.add(f)

        reactions = getattr(path, "reactions", [])
        for rxn in reactions:
            rxn_products = getattr(rxn, "products", None)
            if rxn_products is None:
                continue
            for item in rxn_products:
                f = self._canonical_formula(item)
                if f:
                    formulas.add(f)
        return formulas

    def _path_produces_target(self, path: object, target_formula: str) -> bool:
        target = self._canonical_formula(target_formula)
        if not target:
            return False
        formulas = self._path_formulas(path)
        if formulas:
            return target in formulas
        # If no symbolic formula info is available, reject for safety.
        # 若路径无任何可判别产物信息，按保守策略拒绝。
        return False

    def _filter_target_compatible_paths(self, paths: list[object], target_formula: str) -> list[object]:
        out: list[object] = []
        for p in paths:
            try:
                if self._path_produces_target(p, target_formula):
                    out.append(p)
            except Exception:
                continue
        return out

    def _resolve_cost_function(self):
        if self._cached_cost_function_obj is not None:
            return self._cached_cost_function_obj
        if not self.native_available:
            return None
        try:
            if self.cost_function == "weighted_sum":
                self._cached_cost_function_obj = WeightedSum(
                    params=["energy_per_atom"],
                    weights=[1.0],
                )
            else:
                self._cached_cost_function_obj = Softplus()
        except Exception as exc:  # pragma: no cover - optional runtime deps
            logger.warning(f"Failed to build cost function '{self.cost_function}': {exc}")
            self._cached_cost_function_obj = None
        return self._cached_cost_function_obj

    def _build_network_pathways(
        self,
        candidate_formula: str,
        reaction_set: object | None,
        network: object | None,
        precursors: list[str],
    ) -> list[object]:
        if not self.native_available:
            return []

        if network is None:
            if reaction_set is None:
                return []
            cost_obj = self._resolve_cost_function()
            if cost_obj is None:
                return []
            network = ReactionNetwork(reaction_set, cost_function=cost_obj)

        try:
            if getattr(network, "graph", None) is None:
                network.build()
            network.set_precursors(precursors)
            paths = network.find_pathways([candidate_formula], k=self.k_shortest_paths)
            return list(paths)
        except Exception as exc:
            logger.warning(f"Native reaction-network pathfinding failed: {exc}")
            return []

    def _balance_pathways(
        self,
        *,
        pathways: list[object],
        candidate_formula: str,
        entries: object | None,
        precursors: list[str],
    ) -> list[object]:
        if not self.native_available or not self.use_balanced_solver:
            return pathways
        if entries is None or not pathways or not precursors:
            return pathways

        try:
            net_rxn = BasicReaction.balance(
                [Composition(p) for p in precursors],
                [Composition(candidate_formula)],
            )
            if not net_rxn.balanced:
                return pathways
            net_computed = get_computed_rxn(net_rxn, entries, chempots=None)
            cost_obj = self._resolve_cost_function()
            if cost_obj is None:
                return pathways
            solver = PathwaySolver(
                pathways=PathwaySet.from_paths(pathways),
                entries=entries,
                cost_function=cost_obj,
                chunk_size=self.chunk_size,
            )
            solved = solver.solve(
                net_rxn=net_computed,
                max_num_combos=self.max_num_combos,
                find_intermediate_rxns=self.find_intermediate_rxns,
                intermediate_rxn_energy_cutoff=self.intermediate_rxn_energy_cutoff,
                use_basic_enumerator=self.use_basic_enumerator,
                use_minimize_enumerator=self.use_minimize_enumerator,
                filter_interdependent=self.filter_interdependent,
            )
            solved_paths = list(solved.paths)
            return solved_paths or pathways
        except Exception as exc:
            logger.warning(f"Pathway balancing failed; using unbalanced paths. Error: {exc}")
            return pathways

    @staticmethod
    def _path_step_costs(path: object) -> np.ndarray:
        costs = getattr(path, "costs", None)
        if isinstance(costs, list) and costs:
            return np.asarray([float(c) for c in costs], dtype=float)
        reactions = getattr(path, "reactions", [])
        if reactions:
            vals = []
            for rxn in reactions:
                vals.append(float(getattr(rxn, "energy_per_atom", 0.0)))
            return np.asarray(vals, dtype=float)
        return np.asarray([0.0], dtype=float)

    @staticmethod
    def _entropic_risk(step_costs: np.ndarray, risk_aversion: float) -> float:
        if step_costs.size == 0:
            return 0.0
        eta = max(float(risk_aversion), 1e-8)
        if eta <= 1e-6:
            return float(np.mean(step_costs))
        shifted = step_costs - float(np.min(step_costs))
        return float(np.min(step_costs) + np.log(np.mean(np.exp(eta * shifted))) / eta)

    def _extract_metrics(self, path: object) -> _PathMetrics:
        step_costs = self._path_step_costs(path)
        steps = max(1.0, float(len(getattr(path, "reactions", []))))
        avg_cost = self._safe_float(getattr(path, "average_cost", None), float(np.mean(step_costs)))
        driving_force = -self._safe_float(getattr(path, "energy_per_atom", 0.0), 0.0)

        reactions = getattr(path, "reactions", [])
        if reactions:
            uq = []
            for rxn in reactions:
                if hasattr(rxn, "energy_uncertainty_per_atom"):
                    uq.append(abs(self._safe_float(getattr(rxn, "energy_uncertainty_per_atom", 0.0), 0.0)))
            uncertainty = float(np.mean(uq)) if uq else self.uncertainty_default
        else:
            uncertainty = self.uncertainty_default

        intermediates = getattr(path, "intermediates", None)
        intermediate_complexity = float(len(intermediates)) if intermediates is not None else 0.0
        entropic_risk = self._entropic_risk(step_costs, self.risk_aversion)
        risk_excess = max(0.0, float(entropic_risk - avg_cost))
        return _PathMetrics(
            average_cost=float(avg_cost),
            driving_force=float(driving_force),
            num_steps=float(steps),
            uncertainty=float(uncertainty),
            intermediate_complexity=float(intermediate_complexity),
            entropic_risk=float(entropic_risk),
            risk_excess=float(risk_excess),
        )

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> bool:
        return bool(np.all(a <= b + eps) and np.any(a < b - eps))

    @classmethod
    def _non_dominated_sort(cls, points: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        """
        Fast non-dominated sorting (NSGA-II style), O(MN^2).
        快速非支配排序，复杂度 O(MN^2)。
        """
        n = points.shape[0]
        if n == 0:
            return [], np.zeros(0, dtype=int)

        dominates_list: list[list[int]] = [[] for _ in range(n)]
        dominated_count = np.zeros(n, dtype=int)
        rank = np.full(n, fill_value=n, dtype=int)
        first_front: list[int] = []

        for p in range(n):
            for q in range(p + 1, n):
                if cls._dominates(points[p], points[q]):
                    dominates_list[p].append(q)
                    dominated_count[q] += 1
                elif cls._dominates(points[q], points[p]):
                    dominates_list[q].append(p)
                    dominated_count[p] += 1
            if dominated_count[p] == 0:
                rank[p] = 0
                first_front.append(p)

        fronts: list[list[int]] = []
        current = first_front
        level = 0
        while current:
            fronts.append(current)
            next_front: list[int] = []
            for p in current:
                for q in dominates_list[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        rank[q] = level + 1
                        next_front.append(q)
            current = next_front
            level += 1
        return fronts, rank

    @staticmethod
    def _crowding_distance(points: np.ndarray, front: list[int]) -> dict[int, float]:
        if len(front) == 0:
            return {}
        if len(front) <= 2:
            return {idx: float("inf") for idx in front}

        m = points.shape[1]
        dist = {idx: 0.0 for idx in front}
        front_points = points[front]
        for dim in range(m):
            order = np.argsort(front_points[:, dim])
            sorted_idx = [front[i] for i in order]
            dist[sorted_idx[0]] = float("inf")
            dist[sorted_idx[-1]] = float("inf")
            low = float(front_points[order[0], dim])
            high = float(front_points[order[-1], dim])
            span = high - low
            if span <= 1e-12:
                continue
            for pos in range(1, len(sorted_idx) - 1):
                left = float(front_points[order[pos - 1], dim])
                right = float(front_points[order[pos + 1], dim])
                if math.isfinite(dist[sorted_idx[pos]]):
                    dist[sorted_idx[pos]] += (right - left) / span
        return dist

    @staticmethod
    def _minmax_normalize(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax - vmin <= 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - vmin) / (vmax - vmin)

    def _scalar_penalty(self, metrics: list[_PathMetrics]) -> np.ndarray:
        avg_cost = self._minmax_normalize(np.asarray([m.average_cost for m in metrics], dtype=float))
        driving = self._minmax_normalize(np.asarray([m.driving_force for m in metrics], dtype=float))
        steps = self._minmax_normalize(np.asarray([m.num_steps for m in metrics], dtype=float))
        uq = self._minmax_normalize(np.asarray([m.uncertainty for m in metrics], dtype=float))
        complexity = self._minmax_normalize(np.asarray([m.intermediate_complexity for m in metrics], dtype=float))
        # Use excess risk above mean cost to avoid double-counting pure cost signal.
        # 使用“超额风险”而非总风险，避免与平均成本重复计分。
        risk_excess = self._minmax_normalize(np.asarray([m.risk_excess for m in metrics], dtype=float))

        w = self.objective_weights
        # Minimize cost/risk/steps/uncertainty/complexity and maximize driving force.
        # 最小化成本/风险/步数/不确定度/复杂度，同时最大化驱动力。
        return (
            w["average_cost"] * avg_cost
            + w["entropic_risk"] * risk_excess
            + w["num_steps"] * steps
            + w["uncertainty"] * uq
            + w["intermediate_complexity"] * complexity
            - w["driving_force"] * driving
        )

    def _rank_pathways(self, paths: list[object]) -> tuple[list[int], list[_PathMetrics], dict[int, float], np.ndarray]:
        metrics = [self._extract_metrics(p) for p in paths]

        # Pareto objectives are all converted to minimization coordinates.
        # 将目标统一映射到“最小化”坐标系以执行 Pareto 排序。
        points = np.asarray(
            [
                [
                    m.average_cost,
                    -m.driving_force,
                    m.num_steps,
                    m.uncertainty,
                    m.intermediate_complexity,
                    m.risk_excess,
                ]
                for m in metrics
            ],
            dtype=float,
        )

        fronts, rank = self._non_dominated_sort(points)
        crowding: dict[int, float] = {i: 0.0 for i in range(len(paths))}
        for front in fronts:
            crowding.update(self._crowding_distance(points, front))

        penalty = self._scalar_penalty(metrics)

        crowding_values = np.asarray([crowding[i] for i in range(len(paths))], dtype=float)
        finite = np.isfinite(crowding_values)
        if np.any(finite):
            c_norm = self._minmax_normalize(crowding_values[finite])
            crowding_values[finite] = c_norm
        crowding_values[~finite] = 1.0

        order = sorted(
            range(len(paths)),
            key=lambda idx: (rank[idx], -crowding_values[idx], penalty[idx]),
        )
        crowding_map = {i: float(crowding_values[i]) for i in range(len(paths))}
        return order, metrics, crowding_map, penalty

    @staticmethod
    def _format_pathway(path: object) -> str:
        reactions = getattr(path, "reactions", [])
        if not reactions:
            return "empty_pathway"
        return " | ".join(str(rxn) for rxn in reactions)

    def _fallback_result(self, candidate_formula: str, candidate_energy: float | None) -> dict:
        energy = self._safe_float(candidate_energy, default=0.0)
        if self.fallback_mode == "energy_prior":
            # Weak prior only; not claimed as pathway-validated synthesizability.
            # 仅弱先验，不代表路径求解意义上的可合成性。
            prior = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, energy))))
            return {
                "synthesizable": False,
                "score": float(prior),
                "pathway": [],
                "pathway_count": 0,
                "candidate": candidate_formula,
                "mode": "fallback_energy_prior",
                "native_available": self.native_available,
            }
        return {
            "synthesizable": False,
            "score": 0.0,
            "pathway": [],
            "pathway_count": 0,
            "candidate": candidate_formula,
            "mode": "fallback_conservative",
            "native_available": self.native_available,
        }

    def evaluate(
        self,
        candidate_formula: str,
        candidate_energy: float | None,
        *,
        entries: object | None = None,
        reaction_set: object | None = None,
        network: object | None = None,
        pathways: object | None = None,
        precursors: list[str] | None = None,
    ) -> dict:
        """
        Evaluate candidate synthesizability through pathway ranking.
        通过路径排序评估候选材料可合成性。

        Inputs can be supplied either via `set_context(...)` or per-call kwargs.
        """

        resolved_entries = entries if entries is not None else self._entries
        resolved_reaction_set = reaction_set if reaction_set is not None else self._reaction_set
        resolved_network = network if network is not None else self._network
        resolved_pathways = pathways if pathways is not None else self._pathways
        resolved_precursors = list(precursors) if precursors is not None else list(self.precursors)
        if not resolved_precursors:
            resolved_precursors = self._infer_precursors_from_formula(candidate_formula)

        path_list = self._to_path_list(resolved_pathways)
        if not path_list:
            path_list = self._build_network_pathways(
                candidate_formula,
                resolved_reaction_set,
                resolved_network,
                resolved_precursors,
            )
        if path_list:
            path_list = self._balance_pathways(
                pathways=path_list,
                candidate_formula=candidate_formula,
                entries=resolved_entries,
                precursors=resolved_precursors,
            )
            path_list = self._filter_target_compatible_paths(path_list, candidate_formula)

        if not path_list:
            return self._fallback_result(candidate_formula, candidate_energy)

        order, metrics, crowding_map, penalties = self._rank_pathways(path_list)
        ordered = order[: self.max_num_pathways]
        best_idx = ordered[0]
        best_path = path_list[best_idx]
        best_penalty = float(max(penalties[best_idx], -1.0))
        best_rank_pos = float(order.index(best_idx))
        # Smooth bounded utility score in (0, 1].
        score = float(1.0 / (1.0 + best_rank_pos + max(0.0, best_penalty)))
        synthesizable = bool(len(ordered) > 0)

        top_notes = [self._format_pathway(path_list[i]) for i in ordered]
        best_metrics = metrics[best_idx].as_dict()
        best_metrics["crowding"] = float(crowding_map.get(best_idx, 0.0))
        best_metrics["scalar_penalty"] = float(penalties[best_idx])

        logger.info(
            "Native pathway ranking done for %s: paths=%d, score=%.4f, best_rank=%d",
            candidate_formula,
            len(path_list),
            score,
            int(best_rank_pos),
        )

        return {
            "synthesizable": synthesizable,
            "score": score,
            "pathway": top_notes,
            "pathway_count": int(len(path_list)),
            "candidate": candidate_formula,
            "metrics": best_metrics,
            "algorithm": {
                "ranking": "pareto_nsga2_like",
                "risk": "entropic",
                "cost_function": self.cost_function,
                "balanced_solver": bool(self.use_balanced_solver),
                "native_available": self.native_available,
            },
            "energy_observation": self._safe_float(candidate_energy, 0.0),
            "precursors": resolved_precursors,
            "best_path_repr": self._format_pathway(best_path),
        }
