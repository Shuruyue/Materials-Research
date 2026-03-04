"""
Active Learning Discovery Controller

The brain of ATLAS — orchestrates the closed-loop discovery cycle:
    Generate -> Relax -> Classify -> Score -> Select -> Loop

Optimization:
- Checkpoint & Resume: Automatically continues from last saved iteration
- Robust Error Handling: Isolates failures in external tools (MACE/GNN)
- Duplicate Prevention: Tracks known formulas across sessions
"""

import gc
import json
import logging
import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from atlas.active_learning.acquisition import (
    DISCOVERY_ACQUISITION_STRATEGIES,
    normalize_acquisition_strategy,
    schedule_ucb_kappa,
    score_acquisition,
)
from atlas.active_learning.generator import StructureGenerator
from atlas.active_learning.gp_surrogate import GPSurrogateAcquirer
from atlas.active_learning.objective_space import (
    clip01 as clip01_value,
)
from atlas.active_learning.objective_space import (
    collect_history_objective_points,
    feasibility_mask_from_points,
    infer_objective_dimension,
    objective_points_from_map,
    objective_points_from_terms,
)
from atlas.active_learning.objective_space import (
    safe_float as safe_float_value,
)
from atlas.active_learning.pareto_utils import (
    crowding_distance,
    hypervolume,
    hypervolume_2d,
    mc_hv_improvements_shared,
    non_dominated_sort,
    pareto_front,
    pareto_rank_score,
)
from atlas.active_learning.policy_engine import PolicyEngine
from atlas.active_learning.policy_state import ActiveLearningPolicyConfig, PolicyState
from atlas.config import get_config
from atlas.models.prediction_utils import extract_mean_and_std
from atlas.research.workflow_reproducible_graph import IterationSnapshot, WorkflowReproducibleGraph
from atlas.utils.registry import EvaluatorFactory, ModelFactory, RelaxerFactory
from atlas.utils.reproducibility import set_global_seed

try:
    import rustworkx as rx
except ImportError:
    rx = None

try:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Element
except Exception:  # pragma: no cover - optional runtime dependency
    StructureMatcher = None
    Element = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A material candidate with all computed properties."""
    structure: object                    # pymatgen Structure (not serialized directly)
    formula: str = ""
    method: str = ""
    parent: str = ""
    mutations: str = ""

    # Scores
    topo_probability: float = 0.0
    stability_score: float = 0.0
    heuristic_topo_score: float = 0.0
    novelty_score: float = 0.0
    synthesis_score: float = 0.0
    synthesis_feasibility: float = 0.0
    ood_score: float = 0.0
    acquisition_value: float = 0.0

    # Properties
    energy_per_atom: float | None = None
    energy_mean: float | None = None
    energy_std: float | None = None
    calibrated_mean: float | None = None
    calibrated_std: float | None = None
    conformal_radius: float = 0.0
    risk_score: float = 0.0
    estimated_cost: float = 0.0
    gain_per_cost: float = 0.0
    reject_reason: str = ""
    relaxed_structure: object = None
    converged: bool = False

    # Metadata
    iteration: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for JSON storage (structure handling omitted for brevity)."""
        d = asdict(self)
        if hasattr(d['structure'], 'as_dict'):
             d['structure'] = d['structure'].as_dict()
        if hasattr(d['relaxed_structure'], 'as_dict'):
             d['relaxed_structure'] = d['relaxed_structure'].as_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize from JSON."""
        from pymatgen.core import Structure
        if isinstance(data.get('structure'), dict):
            data['structure'] = Structure.from_dict(data['structure'])
        if isinstance(data.get('relaxed_structure'), dict):
            data['relaxed_structure'] = Structure.from_dict(data['relaxed_structure'])
        return cls(**data)


class DiscoveryController:
    """
    Active learning controller for topological material discovery.
    Supports resuming from checkpoints.
    """

    def __init__(
        self,
        generator: StructureGenerator,
        relaxer=None,
        classifier_model=None,
        graph_builder=None,
        w_topo: float = 0.4,
        w_stability: float = 0.3,
        w_heuristic: float = 0.15,
        w_novelty: float = 0.15,
        acquisition_strategy: str = "hybrid",
        acquisition_kappa: float = 2.0,
        acquisition_best_f: float = -0.5,
        acquisition_jitter: float = 0.01,
        policy_name: str = "legacy",
        risk_mode: str = "soft",
        cost_aware: bool = False,
        calibration_window: int = 128,
        policy_engine: PolicyEngine | None = None,
        results_dir: str | Path | None = None,
    ):
        self.generator = generator
        self.relaxer = relaxer
        self.classifier = classifier_model
        self.graph_builder = graph_builder


        self.cfg = get_config()
        self.profile = self.cfg.profile
        self.policy_config = ActiveLearningPolicyConfig.from_profile(
            self.profile,
            policy_name=policy_name,
            risk_mode=risk_mode,
            cost_aware=cost_aware,
            calibration_window=calibration_window,
        )
        self.policy_state = PolicyState()
        self.policy_engine = policy_engine

        logger.info(f"Initializing DiscoveryController with profile: {self.profile}")

        # Config-driven Dynamic Loading (Factory Pattern)
        if self.relaxer is None and hasattr(self.profile, "relaxer_name"):
            try:
                self.relaxer = RelaxerFactory.create(self.profile.relaxer_name)
                logger.info(f"Dynamically loaded relaxer: {self.profile.relaxer_name}")
            except Exception as e:
                logger.warning(f"Could not load auto-relaxer {self.profile.relaxer_name}: {e}")

        if self.classifier is None and hasattr(self.profile, "model_name"):
             try:
                 self.classifier = ModelFactory.create(self.profile.model_name)
                 logger.info(f"Dynamically loaded classifier model: {self.profile.model_name}")
             except Exception as e:
                 logger.warning(f"Could not load auto-classifier {self.profile.model_name}: {e}")

        self.synthesizability_evaluator = None
        if hasattr(self.profile, "evaluator_name"):
            try:
                self.synthesizability_evaluator = EvaluatorFactory.create(self.profile.evaluator_name)
                logger.info(f"Dynamically loaded synthesizability evaluator: {self.profile.evaluator_name}")
            except Exception:
                logger.info("No synthesis evaluator attached.")

        # Hyperparams
        self.weights = {
            "topo": w_topo,
            "stability": w_stability,
            "heuristic": w_heuristic,
            "novelty": w_novelty,
        }
        strategy_key = str(acquisition_strategy).strip().lower().replace(" ", "_")
        if strategy_key in {"hybrid", "stability"}:
            self.acquisition_strategy = strategy_key
        else:
            self.acquisition_strategy = normalize_acquisition_strategy(strategy_key)
        if self.acquisition_strategy not in DISCOVERY_ACQUISITION_STRATEGIES:
            available = ", ".join(sorted(DISCOVERY_ACQUISITION_STRATEGIES))
            raise ValueError(
                f"Unsupported acquisition strategy: {acquisition_strategy!r}. Available: {available}"
            )
        self.acquisition_kappa = float(acquisition_kappa)
        self.acquisition_best_f = float(acquisition_best_f)
        self.acquisition_jitter = float(acquisition_jitter)
        self.acquisition_kappa_schedule = str(
            getattr(self.profile, "acquisition_kappa_schedule", "gp_ucb")
        )
        self.acquisition_kappa_min = float(getattr(self.profile, "acquisition_kappa_min", 1.0))
        self.acquisition_kappa_decay = float(getattr(self.profile, "acquisition_kappa_decay", 0.08))
        self.acquisition_ucb_delta = float(getattr(self.profile, "acquisition_ucb_delta", 0.1))
        self.acquisition_ucb_dimension = int(getattr(self.profile, "acquisition_ucb_dimension", 1))
        self.use_constrained_acquisition = bool(
            getattr(self.profile, "use_constrained_acquisition", True)
        )
        self.use_noisy_ei = bool(getattr(self.profile, "use_noisy_ei", True))
        self.noisy_ei_mc_samples = int(getattr(self.profile, "noisy_ei_mc_samples", 128))
        self.batch_diversity_strength = float(getattr(self.profile, "batch_diversity_strength", 0.15))
        self.batch_diversity_sigma = float(getattr(self.profile, "batch_diversity_sigma", 1.0))
        self.batch_diversity_space = str(getattr(self.profile, "batch_diversity_space", "chemistry"))
        self.dynamic_best_f = bool(getattr(self.profile, "dynamic_best_f", True))
        self.dynamic_best_f_quantile = float(getattr(self.profile, "dynamic_best_f_quantile", 0.0))
        self.max_observation_history = int(getattr(self.profile, "max_observation_history", 5000))
        self.use_pareto_rank_bonus = bool(getattr(self.profile, "use_pareto_rank_bonus", True))
        self.pareto_rank_bonus_weight = float(getattr(self.profile, "pareto_rank_bonus_weight", 0.2))
        self.use_pareto_hv_bonus = bool(getattr(self.profile, "use_pareto_hv_bonus", True))
        self.pareto_hv_bonus_weight = float(getattr(self.profile, "pareto_hv_bonus_weight", 0.25))
        self.pareto_feasibility_threshold = float(getattr(self.profile, "pareto_feasibility_threshold", 0.05))
        self.use_hv_batch_greedy = bool(getattr(self.profile, "use_hv_batch_greedy", True))
        self.hv_batch_weight = float(getattr(self.profile, "hv_batch_weight", 0.35))
        self.hv_mc_samples = int(getattr(self.profile, "hv_mc_samples", 4096))
        self.hv_mc_seed = int(getattr(self.profile, "hv_mc_seed", int(self.cfg.train.seed) + 97))
        self.hv_use_shared_samples = bool(getattr(self.profile, "hv_use_shared_samples", True))
        self.hv_candidate_pool_limit = int(getattr(self.profile, "hv_candidate_pool_limit", 96))
        self.hv_chunk_size = int(getattr(self.profile, "hv_chunk_size", 512))
        self.experimental_algorithms_enabled = bool(getattr(self.profile, "experimental_algorithms_enabled", True))
        self.use_synthesis_objective = bool(
            getattr(self.profile, "use_synthesis_objective", self.synthesizability_evaluator is not None)
        )
        self.synthesis_objective_weight = float(getattr(self.profile, "synthesis_objective_weight", 0.15))
        self.synthesis_eval_topk = int(getattr(self.profile, "synthesis_eval_topk", 128))
        self.synthesis_gate_topo_threshold = float(getattr(self.profile, "synthesis_gate_topo_threshold", 0.20))
        self.synthesis_score_floor = float(getattr(self.profile, "synthesis_score_floor", 0.0))
        self.synthesis_eval_strategy = str(getattr(self.profile, "synthesis_eval_strategy", "hybrid_topk_uncertain"))
        self.synthesis_uncertainty_weight = float(getattr(self.profile, "synthesis_uncertainty_weight", 0.35))
        self.synthesis_uncertainty_decay = float(getattr(self.profile, "synthesis_uncertainty_decay", 0.8))
        self.synthesis_time_budget_sec = float(getattr(self.profile, "synthesis_time_budget_sec", 1.25))
        self.synthesis_cache_max_size = int(getattr(self.profile, "synthesis_cache_max_size", 50000))
        self.pareto_joint_feasibility = bool(getattr(self.profile, "pareto_joint_feasibility", True))
        self.pareto_synthesis_feasibility_threshold = float(
            getattr(self.profile, "pareto_synthesis_feasibility_threshold", 0.10)
        )
        self.use_ood_penalty = bool(getattr(self.profile, "use_ood_penalty", True))
        self.ood_penalty_weight = float(getattr(self.profile, "ood_penalty_weight", 0.15))
        self.ood_history_min_points = int(getattr(self.profile, "ood_history_min_points", 24))
        self.ood_quantile = float(getattr(self.profile, "ood_quantile", 0.95))
        self.ood_space = str(getattr(self.profile, "ood_space", "chemistry"))
        self.max_pathway_annotations_per_iter = int(getattr(self.profile, "max_pathway_annotations_per_iter", 8))
        self.enable_structure_dedup = bool(getattr(self.profile, "enable_structure_dedup", True))
        self.relax_timeout_sec = float(self.policy_config.relax_timeout_sec)
        self.relax_max_retries = int(self.policy_config.relax_max_retries)
        self.relax_circuit_breaker_failures = int(self.policy_config.relax_circuit_breaker_failures)
        self.relax_circuit_breaker_cooldown_iters = int(self.policy_config.relax_circuit_breaker_cooldown_iters)
        if not self.experimental_algorithms_enabled:
            # Conservative mode for production reliability.
            # 生产保守模式：关闭实验性策略，保留确定性核心流程。
            self.use_synthesis_objective = False
            self.hv_use_shared_samples = False
        self.structure_matcher_ltol = float(getattr(self.profile, "structure_matcher_ltol", 0.2))
        self.structure_matcher_stol = float(getattr(self.profile, "structure_matcher_stol", 0.3))
        self.structure_matcher_angle_tol = float(getattr(self.profile, "structure_matcher_angle_tol", 5.0))
        self._structure_matcher = None
        if self.enable_structure_dedup and StructureMatcher is not None:
            try:
                self._structure_matcher = StructureMatcher(
                    ltol=self.structure_matcher_ltol,
                    stol=self.structure_matcher_stol,
                    angle_tol=self.structure_matcher_angle_tol,
                    primitive_cell=True,
                    scale=True,
                )
            except Exception as exc:
                logger.warning(f"Failed to initialize StructureMatcher; fallback to formula dedup. Error: {exc}")
                self._structure_matcher = None
        self._acquisition_generator = torch.Generator().manual_seed(int(self.cfg.train.seed) + 17)
        self.method_key = getattr(self.profile, "method_key", "graph_equivariant")
        self.fallback_methods = tuple(getattr(self.profile, "fallback_methods", ()))
        self.use_gp_active = self.method_key == "gp_active_learning"
        logger.info(
            (
                "Acquisition strategy: %s "
                "(kappa=%.3f, schedule=%s, best_f=%.3f, jitter=%.4f, constrained=%s, "
                "noisy_ei=%s, diversity_lambda=%.3f, diversity_space=%s, "
                "dynamic_best_f=%s, pareto_rank_bonus=%s, pareto_hv_bonus=%s, "
                "hv_batch_greedy=%s, synthesis_objective=%s, "
                "ood_penalty=%s, structure_dedup=%s)"
            ),
            self.acquisition_strategy,
            self.acquisition_kappa,
            self.acquisition_kappa_schedule,
            self.acquisition_best_f,
            self.acquisition_jitter,
            self.use_constrained_acquisition,
            self.use_noisy_ei,
            self.batch_diversity_strength,
            self.batch_diversity_space,
            self.dynamic_best_f,
            self.use_pareto_rank_bonus,
            self.use_pareto_hv_bonus,
            self.use_hv_batch_greedy,
            self.use_synthesis_objective,
            self.use_ood_penalty,
            self.enable_structure_dedup,
        )
        logger.info(
            "Policy engine: name=%s risk_mode=%s cost_aware=%s calibration_window=%s",
            self.policy_config.policy_name,
            self.policy_config.risk_mode,
            self.policy_config.cost_aware,
            self.policy_config.calibration_window,
        )

        if results_dir is not None:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = self.cfg.paths.data_dir / "discovery_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.workflow = None
        if self.method_key in {"workflow_reproducible_graph", "gp_active_learning"}:
            self.workflow = WorkflowReproducibleGraph(self.results_dir / "workflow_runs")
        self.gp_acquirer = GPSurrogateAcquirer() if self.use_gp_active else None

        # State
        self.iteration = 0
        self.known_formulas: set[str] = set()
        self.known_structures_by_formula: dict[str, list[object]] = {}
        self.all_candidates: list[Candidate] = []
        self.top_candidates: list[Candidate] = []
        self._synthesis_cache: OrderedDict[tuple[str, float], dict] = OrderedDict()
        self._last_relax_stats: dict[str, object] = {"total": 0, "success": 0, "failures": 0, "buckets": {}}

        self._policy_state_path = self.results_dir / "policy_state.json"
        self._load_policy_state()
        if self.policy_engine is None:
            self.policy_engine = PolicyEngine(config=self.policy_config, state=self.policy_state)
        else:
            # Keep caller-provided engine state/config synchronized with runtime config.
            self.policy_engine.config = self.policy_config
            self.policy_engine.state = self.policy_state

        # Auto-load previous state
        self._load_checkpoint()

    @staticmethod
    def _safe_reduced_formula(structure: object, fallback: str = "") -> str:
        try:
            return str(structure.composition.reduced_formula)
        except Exception:
            return str(fallback)

    def _register_structure_in_bucket(self, structure: object, buckets: dict[str, list[object]]) -> None:
        if structure is None:
            return
        formula = self._safe_reduced_formula(structure)
        if not formula:
            return
        buckets.setdefault(formula, []).append(structure)

    def _register_known_structure(self, structure: object, formula_hint: str = "") -> None:
        if structure is None and not formula_hint:
            return
        formula = self._safe_reduced_formula(structure, fallback=formula_hint)
        if formula:
            self.known_formulas.add(formula)
        if structure is None:
            return
        # Structure-level dedup for polymorph preservation.
        # Ref: pymatgen StructureMatcher (Ong et al., 2013), https://pymatgen.org/
        self._register_structure_in_bucket(structure, self.known_structures_by_formula)

    def _structures_match(self, a: object, b: object) -> bool:
        if a is None or b is None:
            return False
        if self._structure_matcher is None:
            return self._safe_reduced_formula(a) == self._safe_reduced_formula(b)
        try:
            return bool(self._structure_matcher.fit(a, b))
        except Exception:
            return self._safe_reduced_formula(a) == self._safe_reduced_formula(b)

    def _is_duplicate_structure(
        self,
        structure: object,
        *,
        extra_buckets: dict[str, list[object]] | None = None,
    ) -> bool:
        if structure is None:
            return False
        formula = self._safe_reduced_formula(structure)
        if not formula:
            return False

        if not self.enable_structure_dedup:
            return formula in self.known_formulas

        known_bucket = self.known_structures_by_formula.get(formula, [])
        for known_structure in known_bucket:
            if self._structures_match(structure, known_structure):
                return True

        if extra_buckets is not None:
            for known_structure in extra_buckets.get(formula, []):
                if self._structures_match(structure, known_structure):
                    return True

        return False

    def _current_best_f(self, observed_mean: torch.Tensor | None) -> float:
        """
        Dynamic incumbent for EI/NEI in minimization setting.

        Reference:
        - Jones et al. (1998), EGO incumbent update:
          https://link.springer.com/article/10.1023/A:1008306431147
        """
        fallback = float(self.acquisition_best_f)
        if not self.dynamic_best_f or observed_mean is None or observed_mean.numel() == 0:
            return fallback

        q = min(max(float(self.dynamic_best_f_quantile), 0.0), 1.0)
        if q <= 0.0:
            return float(torch.min(observed_mean).item())
        return float(torch.quantile(observed_mean, q).item())

    def _load_policy_state(self) -> None:
        if not self._policy_state_path.exists():
            return
        try:
            with open(self._policy_state_path, encoding="utf-8") as fp:
                payload = json.load(fp)
            state_payload = payload.get("state") if isinstance(payload, dict) else payload
            self.policy_state = PolicyState.from_dict(state_payload if isinstance(state_payload, dict) else {})
            logger.info(
                "Loaded policy state: iter=%s, calibration_points=%s",
                self.policy_state.iteration,
                self.policy_state.calibration_points,
            )
        except Exception as exc:
            logger.warning("Failed to load policy state: %s", exc)

    def _save_policy_state(self) -> None:
        payload = {
            "updated_at": time.time(),
            "config": self.policy_config.to_dict(),
            "state": self.policy_state.to_dict(),
        }
        try:
            with open(self._policy_state_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to persist policy state: %s", exc)

    def _load_checkpoint(self):
        """Scan results directory and restore state."""
        files = sorted(self.results_dir.glob("iteration_*.json"))
        if not files:
            logger.info("No checkpoints found. Starting fresh discovery.")
            return

        logger.info(f"Found {len(files)} checkpoints. Restoring state...")

        # Load all history to populate dedup state and incumbent observations.
        for f in files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    iter_num = data.get("iteration", 0)
                    self.iteration = max(self.iteration, iter_num)

                    for cand_dict in data.get("candidates", []):
                        formula = cand_dict.get("formula", "")
                        if formula:
                            self.known_formulas.add(str(formula))
                        # Rebuild structure buckets from checkpointed winners when available.
                        # This keeps polymorph-aware dedup active across resume.
                        try:
                            cand = Candidate.from_dict(dict(cand_dict))
                            ref_structure = cand.relaxed_structure or cand.structure
                            self._register_known_structure(ref_structure, formula_hint=cand.formula)
                            self.all_candidates.append(cand)
                        except Exception:
                            # Checkpoints may have partial candidate payloads.
                            continue

            except Exception as e:
                logger.warning(f"Failed to load checkpoint {f}: {e}")

        # Load top candidates from the VERY LAST iteration to seed the generator
        last_file = files[-1]
        try:
            with open(last_file) as fp:
                data = json.load(fp)
                top_cands = [Candidate.from_dict(d) for d in data.get("candidates", [])]

                # Add good ones to generator logic
                seeds = [c.relaxed_structure or c.structure for c in top_cands if c.acquisition_value > 0.3]
                if seeds:
                    self.generator.add_seeds(seeds)
                    logger.info(f"Restored {len(seeds)} seeds from last iteration.")
        except Exception:
            pass

        if len(self.all_candidates) > self.max_observation_history:
            self.all_candidates = self.all_candidates[-self.max_observation_history :]

        logger.info(
            "Resuming from Iteration %s. Known formulas: %s, known structure buckets: %s",
            self.iteration,
            len(self.known_formulas),
            len(self.known_structures_by_formula),
        )

    def run_discovery_loop(
        self,
        n_iterations: int = 10,
        n_candidates_per_iter: int = 50,
        n_select_top: int = 10,
    ) -> list[Candidate]:
        """
        Run the discovery loop, resuming if necessary.
        Total iterations = current_iteration + n_iterations.
        """
        start_iter = self.iteration + 1
        end_iter = self.iteration + n_iterations
        status = "completed"
        seed_meta = set_global_seed(
            self.cfg.train.seed,
            deterministic=self.cfg.train.deterministic,
        )

        if self.workflow is not None:
            self.workflow.start(
                extra_metrics={
                    "requested_iterations": n_iterations,
                    "candidates_per_iteration": n_candidates_per_iter,
                    "select_top": n_select_top,
                    "resume_from_iteration": self.iteration,
                    "seed_metadata": seed_meta,
                    "acquisition_strategy": self.acquisition_strategy,
                    "acquisition_kappa": self.acquisition_kappa,
                    "acquisition_best_f": self.acquisition_best_f,
                    "acquisition_jitter": self.acquisition_jitter,
                    "acquisition_kappa_schedule": self.acquisition_kappa_schedule,
                    "use_constrained_acquisition": self.use_constrained_acquisition,
                    "use_noisy_ei": self.use_noisy_ei,
                    "noisy_ei_mc_samples": self.noisy_ei_mc_samples,
                    "batch_diversity_strength": self.batch_diversity_strength,
                    "batch_diversity_sigma": self.batch_diversity_sigma,
                    "batch_diversity_space": self.batch_diversity_space,
                    "dynamic_best_f": self.dynamic_best_f,
                    "dynamic_best_f_quantile": self.dynamic_best_f_quantile,
                    "use_pareto_rank_bonus": self.use_pareto_rank_bonus,
                    "pareto_rank_bonus_weight": self.pareto_rank_bonus_weight,
                    "use_pareto_hv_bonus": self.use_pareto_hv_bonus,
                    "pareto_hv_bonus_weight": self.pareto_hv_bonus_weight,
                    "pareto_feasibility_threshold": self.pareto_feasibility_threshold,
                    "use_hv_batch_greedy": self.use_hv_batch_greedy,
                    "hv_batch_weight": self.hv_batch_weight,
                    "hv_mc_samples": self.hv_mc_samples,
                    "hv_use_shared_samples": self.hv_use_shared_samples,
                    "hv_candidate_pool_limit": self.hv_candidate_pool_limit,
                    "hv_chunk_size": self.hv_chunk_size,
                    "experimental_algorithms_enabled": self.experimental_algorithms_enabled,
                    "use_synthesis_objective": self.use_synthesis_objective,
                    "synthesis_objective_weight": self.synthesis_objective_weight,
                    "synthesis_eval_topk": self.synthesis_eval_topk,
                    "synthesis_gate_topo_threshold": self.synthesis_gate_topo_threshold,
                    "synthesis_eval_strategy": self.synthesis_eval_strategy,
                    "synthesis_uncertainty_weight": self.synthesis_uncertainty_weight,
                    "synthesis_uncertainty_decay": self.synthesis_uncertainty_decay,
                    "synthesis_time_budget_sec": self.synthesis_time_budget_sec,
                    "synthesis_cache_max_size": self.synthesis_cache_max_size,
                    "pareto_joint_feasibility": self.pareto_joint_feasibility,
                    "pareto_synthesis_feasibility_threshold": self.pareto_synthesis_feasibility_threshold,
                    "use_ood_penalty": self.use_ood_penalty,
                    "ood_penalty_weight": self.ood_penalty_weight,
                    "ood_history_min_points": self.ood_history_min_points,
                    "ood_quantile": self.ood_quantile,
                    "ood_space": self.ood_space,
                    "max_pathway_annotations_per_iter": self.max_pathway_annotations_per_iter,
                    "enable_structure_dedup": self.enable_structure_dedup,
                    "policy_name": self.policy_config.policy_name,
                    "policy_risk_mode": self.policy_config.risk_mode,
                    "policy_cost_aware": self.policy_config.cost_aware,
                    "policy_calibration_window": self.policy_config.calibration_window,
                    "relax_timeout_sec": self.relax_timeout_sec,
                    "relax_max_retries": self.relax_max_retries,
                    "relax_circuit_breaker_failures": self.relax_circuit_breaker_failures,
                    "relax_circuit_breaker_cooldown_iters": self.relax_circuit_breaker_cooldown_iters,
                }
            )

        print("\n" + "=" * 70)
        print(f"  ATLAS Discovery Engine (Iter {start_iter} -> {end_iter})")
        print("=" * 70)

        for it in range(start_iter, end_iter + 1):
            self.iteration = it
            t0 = time.time()
            stage_timings = {}

            print(f"\n{'─' * 50}")
            print(f"  Iteration {it}/{end_iter}")
            print(f"{'─' * 50}")

            # 1. Generate
            print(f"  [1/4] Generating {n_candidates_per_iter} candidates...")
            t_gen = time.time()
            try:
                methods = ["substitute", "strain"] if it > 3 else ["substitute"]
                raw_candidates = self.generator.generate_batch(n_candidates_per_iter, methods=methods)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raw_candidates = []
            stage_timings["generate"] = time.time() - t_gen

            # Phase 1: structure-aware dedup (polymorph-safe) before expensive relaxation.
            staged_buckets: dict[str, list[object]] = {}
            new_raw: list[dict] = []
            for r in raw_candidates:
                struct = r.get("structure")
                if self._is_duplicate_structure(struct, extra_buckets=staged_buckets):
                    continue
                new_raw.append(r)
                self._register_structure_in_bucket(struct, staged_buckets)
            print(f"        Generated {len(raw_candidates)}, New Unique: {len(new_raw)}")

            if not new_raw:
                logger.warning("No new unique candidates. Trying fallback seeds...")
                status = "stalled_no_new_candidates"
                if self.workflow is not None:
                    self.workflow.record_iteration(
                        IterationSnapshot(
                            iteration=it,
                            generated=len(raw_candidates),
                            unique=0,
                            relaxed=0,
                            selected=0,
                            duration_sec=time.time() - t0,
                            stage_timings_sec=stage_timings,
                            seed_pool_size=len(self.generator.seeds),
                            status=status,
                            notes="no unique candidate after dedup",
                        )
                    )
                break

            # 2. Relax
            print(f"  [2/4] Relaxing {len(new_raw)} structures...")
            t_relax = time.time()
            candidates = self._relax_candidates(new_raw)
            stage_timings["relax"] = time.time() - t_relax

            # 3. Classify
            print("  [3/4] Classifying candidates...")
            t_classify = time.time()
            candidates = self._classify_candidates(candidates)
            stage_timings["classify"] = time.time() - t_classify

            # 4. Select
            t_select = time.time()
            candidates = self._score_and_select(candidates, n_select_top)
            stage_timings["select"] = time.time() - t_select

            # Save & Feedback
            selected = candidates[:n_select_top]
            t_save = time.time()
            self._save_iteration(it, selected)
            stage_timings["save"] = time.time() - t_save

            best_structs = [
                c.relaxed_structure or c.structure
                for c in selected
                if c.acquisition_value > 0.25
            ]
            t_feedback = time.time()
            self.generator.add_seeds(best_structs)
            stage_timings["seed_feedback"] = time.time() - t_feedback

            dt = time.time() - t0
            self._print_iteration_summary(selected, dt)
            self.policy_state.iteration = int(self.iteration)
            self._save_policy_state()

            if self.workflow is not None:
                relax_buckets = self._last_relax_stats.get("buckets", {})
                notes = ""
                if isinstance(relax_buckets, dict) and relax_buckets:
                    notes = f"relax_failures={relax_buckets}"
                self.workflow.record_iteration(
                    IterationSnapshot(
                        iteration=it,
                        generated=len(raw_candidates),
                        unique=len(new_raw),
                        relaxed=len(candidates),
                        selected=len(selected),
                        duration_sec=dt,
                        stage_timings_sec=stage_timings,
                        seed_pool_size=len(self.generator.seeds),
                        notes=notes,
                    )
                )

        if self.workflow is not None:
            self.workflow.finalize(
                status=status,
                extra_metrics={
                    "iterations_completed": self.iteration,
                    "known_formulas": len(self.known_formulas),
                    "top_candidates": len(self.top_candidates),
                },
            )

        return self.top_candidates

    def _call_relax_with_timeout(self, structure: object, *, steps: int = 100) -> dict:
        timeout_sec = max(0.0, float(getattr(self, "relax_timeout_sec", 0.0)))
        if self.relaxer is None:
            return {}
        if timeout_sec <= 0.0:
            return self.relaxer.relax_structure(structure, steps=steps)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.relaxer.relax_structure, structure, steps)
            try:
                return future.result(timeout=timeout_sec)
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"relaxation timed out after {timeout_sec:.2f}s") from exc

    @staticmethod
    def _bucket_error(exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            return "timeout"
        msg = str(exc).strip().lower()
        if "cuda out of memory" in msg or "out of memory" in msg:
            return "oom"
        if "keyboardinterrupt" in msg:
            return "interrupt"
        return "exception"

    def _relax_candidates(self, raw_candidates: list[dict]) -> list[Candidate]:
        candidates = []
        buckets: dict[str, int] = {}
        success_count = 0
        fail_count = 0
        for rc in raw_candidates:
            struct = rc["structure"]
            formula = struct.composition.reduced_formula

            # Default fallback values
            energy = None
            converged = False
            relaxed = struct
            stability = 0.5

            if self.relaxer:
                if int(getattr(self.policy_state, "relax_circuit_open_until_iter", 0)) >= int(self.iteration):
                    fail_count += 1
                    buckets["circuit_open"] = buckets.get("circuit_open", 0) + 1
                    stability = 0.0
                else:
                    retries = max(0, int(getattr(self, "relax_max_retries", 0)))
                    relax_ok = False
                    last_exc: Exception | None = None
                    for _attempt in range(retries + 1):
                        try:
                            res = self._call_relax_with_timeout(struct, steps=100)
                            relaxed = res.get("relaxed_structure", struct)
                            energy = res.get("energy_per_atom")
                            converged = res.get("converged", False)
                            stability = self.relaxer.score_stability(relaxed)
                            relax_ok = True
                            success_count += 1
                            self.policy_state.relax_consecutive_failures = 0
                            break
                        except Exception as exc:
                            last_exc = exc
                            code = self._bucket_error(exc)
                            buckets[code] = buckets.get(code, 0) + 1
                    if not relax_ok:
                        fail_count += 1
                        stability = 0.0
                        self.policy_state.relax_consecutive_failures += 1
                        if self.policy_state.relax_consecutive_failures >= int(
                            getattr(self, "relax_circuit_breaker_failures", 8)
                        ):
                            cooldown = int(getattr(self, "relax_circuit_breaker_cooldown_iters", 2))
                            self.policy_state.relax_circuit_open_until_iter = int(self.iteration) + cooldown
                            logger.warning(
                                "Relaxation circuit breaker opened until iteration %s",
                                self.policy_state.relax_circuit_open_until_iter,
                            )
                            self.policy_state.relax_consecutive_failures = 0
                        if last_exc is not None:
                            logger.debug("Relaxation error for %s: %s", formula, last_exc)

            cand = Candidate(
                structure=struct,
                formula=formula,
                method=rc.get("method", "unknown"),
                parent=rc.get("parent", ""),
                mutations=rc.get("mutations", ""),
                heuristic_topo_score=rc.get("topo_score", 0.0),
                stability_score=stability,
                energy_per_atom=energy,
                relaxed_structure=relaxed,
                converged=converged,
                iteration=self.iteration,
                timestamp=time.time(),
            )
            candidates.append(cand)
        self._last_relax_stats = {
            "total": len(raw_candidates),
            "success": success_count,
            "failures": fail_count,
            "buckets": buckets,
        }
        return candidates

    def _classify_candidates(self, candidates):
        """Use GNN classifier to predict topological probability for each candidate."""
        if not self.classifier or not self.graph_builder:
            # Fallback to heuristic
            for c in candidates:
                c.topo_probability = c.heuristic_topo_score
            return candidates

        self.classifier.eval()
        device = next(self.classifier.parameters()).device

        # Batch processing would be better, but loop is safer for now
        for c in candidates:
            try:
                struct = c.relaxed_structure or c.structure
                graph = self.graph_builder.structure_to_graph(struct)

                with torch.no_grad():
                    node_feats = torch.as_tensor(graph["node_features"], dtype=torch.float32, device=device)
                    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device)
                    edge_features = torch.as_tensor(graph["edge_features"], dtype=torch.float32, device=device)
                    batch_idx = torch.zeros(node_feats.size(0), dtype=torch.long, device=device)

                    # Predict using the model (Surrogate)
                    pred = self.classifier(
                        node_feats,
                        edge_index,
                        edge_features,
                        batch_idx,
                    )

                    # Handle Multi-Task Dictionary Output
                    if isinstance(pred, dict):
                        # 1. Topology (Band Gap or Classification)
                        if "band_gap" in pred:
                            bg_mean, _ = extract_mean_and_std(pred["band_gap"])
                            mu = float(bg_mean.reshape(-1)[0].item())
                            c.topo_probability = 1.0 if mu < 0.1 else 0.0

                        # 2. Stability (Formation Energy)
                        if "formation_energy" in pred:
                            fe_mean, fe_std = extract_mean_and_std(pred["formation_energy"])
                            mu_fe = float(fe_mean.reshape(-1)[0].item())
                            c.energy_mean = mu_fe
                            if fe_std is not None:
                                std_fe = float(fe_std.reshape(-1)[0].item())
                                lcb = mu_fe - 2.0 * std_fe
                                c.stability_score = max(0.0, min(1.0, -lcb))
                                c.energy_mean = mu_fe
                                c.energy_std = std_fe
                            else:
                                c.energy_std = None
                                c.stability_score = max(0.0, min(1.0, -mu_fe))

                    else:
                        # Old binary classifier behavior
                        if pred.shape[-1] == 1:
                            c.topo_probability = torch.sigmoid(pred).item()
                        elif pred.shape[-1] == 2:
                             c.topo_probability = torch.softmax(pred, dim=-1)[0, 1].item()

            except Exception as e:
                logger.warning(f"Classification failed for {c.formula}: {e}")
                c.topo_probability = 0.0

        # Optimization 2 & 3: Force aggressive memory cleanup after batch
        if "cuda" in str(device):
            torch.cuda.empty_cache()
        gc.collect()

        return candidates

    def _stability_component(self, candidate: Candidate) -> float:
        """
        Build the stability term used in composite acquisition score.

        `hybrid` preserves legacy behavior:
        - use EI(formation_energy) when uncertainty is available
        - fall back to deterministic stability_score otherwise
        """
        strategy = self.acquisition_strategy
        has_energy_uq = candidate.energy_mean is not None and candidate.energy_std is not None
        kappa_t = self._current_acquisition_kappa()
        obs_mean, obs_std = self._historical_energy_observations()
        best_f_t = self._current_best_f(obs_mean)

        if strategy == "stability":
            return float(candidate.stability_score)

        if not has_energy_uq:
            return float(candidate.stability_score)

        mean = torch.tensor([float(candidate.energy_mean)], dtype=torch.float32)
        std = torch.tensor([float(candidate.energy_std)], dtype=torch.float32).clamp(min=1e-9)

        if strategy == "hybrid":
            if self.use_noisy_ei and obs_mean is not None:
                # Noisy EI with Monte Carlo incumbent samples.
                # Ref: Letham et al. (2019), https://arxiv.org/abs/1706.07094
                log_nei = score_acquisition(
                    mean,
                    std,
                    strategy="log_nei",
                    best_f=best_f_t,
                    maximize=False,
                    observed_mean=obs_mean,
                    observed_std=obs_std,
                    nei_mc_samples=self.noisy_ei_mc_samples,
                    jitter=self.acquisition_jitter,
                    kappa=kappa_t,
                    kappa_schedule="fixed",
                    iteration=self.iteration,
                    generator=self._acquisition_generator,
                )
                return float(torch.exp(log_nei).item())

            # LogEI is numerically more stable than EI in deep negative tails.
            # Ref: Ament et al. (2023), https://arxiv.org/abs/2310.20708
            log_ei = score_acquisition(
                mean,
                std,
                strategy="log_ei",
                best_f=best_f_t,
                maximize=False,
                jitter=self.acquisition_jitter,
                kappa=kappa_t,
                kappa_schedule="fixed",
                iteration=self.iteration,
                generator=self._acquisition_generator,
            )
            return float(torch.exp(log_ei).item())

        acq = score_acquisition(
            mean,
            std,
            strategy=strategy,
            best_f=best_f_t,
            maximize=False,
            jitter=self.acquisition_jitter,
            kappa=kappa_t,
            kappa_schedule=getattr(self, "acquisition_kappa_schedule", "fixed"),
            iteration=getattr(self, "iteration", 1),
            ucb_dimension=getattr(self, "acquisition_ucb_dimension", 1),
            ucb_delta=getattr(self, "acquisition_ucb_delta", 0.1),
            kappa_min=getattr(self, "acquisition_kappa_min", 1.0),
            kappa_decay=getattr(self, "acquisition_kappa_decay", 0.08),
            observed_mean=obs_mean if strategy in {"nei", "log_nei"} else None,
            observed_std=obs_std if strategy in {"nei", "log_nei"} else None,
            nei_mc_samples=getattr(self, "noisy_ei_mc_samples", 128),
            generator=self._acquisition_generator,
        )
        if strategy in {"log_ei", "log_pi", "log_nei"}:
            return float(torch.exp(acq).item())
        return float(acq.item())

    def _current_acquisition_kappa(self) -> float:
        """
        Compute time-varying UCB kappa_t.

        Ref:
        - Srinivas et al. (2010), GP-UCB: https://arxiv.org/abs/0912.3995
        """
        return schedule_ucb_kappa(
            iteration=getattr(self, "iteration", 1),
            base_kappa=getattr(self, "acquisition_kappa", 2.0),
            mode=getattr(self, "acquisition_kappa_schedule", "fixed"),
            dimension=getattr(self, "acquisition_ucb_dimension", 1),
            delta=getattr(self, "acquisition_ucb_delta", 0.1),
            kappa_min=getattr(self, "acquisition_kappa_min", 1.0),
            decay=getattr(self, "acquisition_kappa_decay", 0.08),
        )

    def _historical_energy_observations(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        means: list[float] = []
        stds: list[float] = []
        history = self.all_candidates if self.all_candidates else self.top_candidates
        for c in history:
            if c.energy_mean is None:
                continue
            means.append(float(c.energy_mean))
            stds.append(float(c.energy_std) if c.energy_std is not None else 0.0)
        if not means:
            return None, None
        return (
            torch.tensor(means, dtype=torch.float32),
            torch.tensor(stds, dtype=torch.float32),
        )

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        return safe_float_value(value, default=default)

    @staticmethod
    def _clip01(x: float) -> float:
        return clip01_value(x)

    def _candidate_energy_value(self, candidate: Candidate) -> float:
        if candidate.energy_mean is not None:
            return self._safe_float(candidate.energy_mean, 0.0)
        if candidate.energy_per_atom is not None:
            return self._safe_float(candidate.energy_per_atom, 0.0)
        return 0.0

    def _synthesis_cache_key(self, candidate: Candidate) -> tuple[str, float]:
        formula = str(getattr(candidate, "formula", "")).strip()
        energy = round(self._candidate_energy_value(candidate), 6)
        return formula, float(energy)

    def _evaluate_synthesizability(self, candidate: Candidate) -> dict:
        """
        Cached synthesis evaluator wrapper.

        The evaluator output is normalized to:
        {"synthesizable": bool, "score": float, "pathway": list[str]}
        """
        if self.synthesizability_evaluator is None:
            return {"synthesizable": False, "score": 0.0, "pathway": []}

        key = self._synthesis_cache_key(candidate)
        cached = self._synthesis_cache.get(key)
        if cached is not None:
            # LRU touch.
            self._synthesis_cache.move_to_end(key)
            return dict(cached)

        formula = key[0]
        if not formula:
            out = {"synthesizable": False, "score": 0.0, "pathway": []}
            self._synthesis_cache[key] = out
            return dict(out)

        energy = self._candidate_energy_value(candidate)
        try:
            raw = self.synthesizability_evaluator.evaluate(formula, energy)
        except Exception as exc:
            logger.warning("Synthesis evaluator failed for %s: %s", formula, exc)
            raw = {}

        score = self._clip01(self._safe_float(raw.get("score", 0.0), 0.0))
        score = max(score, self._clip01(self._safe_float(getattr(self, "synthesis_score_floor", 0.0), 0.0)))
        synthesizable = bool(raw.get("synthesizable", False))
        pathways = raw.get("pathway", [])
        if not isinstance(pathways, list):
            pathways = []
        out = {
            "synthesizable": synthesizable,
            "score": score,
            "pathway": [str(p) for p in pathways[:3]],
        }
        self._synthesis_cache[key] = out
        max_size = max(1, int(getattr(self, "synthesis_cache_max_size", 50000)))
        while len(self._synthesis_cache) > max_size:
            self._synthesis_cache.popitem(last=False)
        return dict(out)

    def _apply_synthesis_objective(self, candidates: list[Candidate]) -> list[float]:
        """
        Integrate pathway-level synthesizability into candidate utilities.

        - Evaluates only top-K provisional candidates for tractability.
        - Uses cached evaluator results to avoid duplicate calls.
        """
        if not candidates:
            return []

        # Reset stale values each iteration.
        for c in candidates:
            c.synthesis_score = 0.0
            c.synthesis_feasibility = 0.0

        enabled = bool(getattr(self, "use_synthesis_objective", False))
        if not enabled or self.synthesizability_evaluator is None:
            return [0.0 for _ in candidates]

        topo_gate = float(getattr(self, "synthesis_gate_topo_threshold", 0.20))
        topk = max(1, int(getattr(self, "synthesis_eval_topk", 128)))
        strategy = str(getattr(self, "synthesis_eval_strategy", "hybrid_topk_uncertain")).strip().lower()

        # Candidate prioritization for synthesis evaluation budget.
        # 候选优先级：兼顾 exploitation（高分）与 exploration（高不确定性）。
        score_order = sorted(
            range(len(candidates)),
            key=lambda idx: float(candidates[idx].acquisition_value),
            reverse=True,
        )
        uncertainty_weight = max(0.0, float(getattr(self, "synthesis_uncertainty_weight", 0.35)))
        uncertainty_order = sorted(
            range(len(candidates)),
            key=lambda idx: float(max(0.0, self._safe_float(getattr(candidates[idx], "energy_std", 0.0), 0.0))),
            reverse=True,
        )

        selected: list[int] = []
        seen: set[int] = set()
        if strategy in {"topk", "top_k"}:
            for idx in score_order:
                if idx in seen:
                    continue
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= topk:
                    break
        else:
            n_uncertain = int(round(topk * uncertainty_weight))
            n_uncertain = min(max(0, n_uncertain), topk)
            n_top = topk - n_uncertain
            for idx in score_order:
                if idx in seen:
                    continue
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= n_top:
                    break
            for idx in uncertainty_order:
                if idx in seen:
                    continue
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= topk:
                    break
            for idx in score_order:
                if idx in seen:
                    continue
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= topk:
                    break
        selected_set = set(selected[: min(topk, len(selected))])
        eval_budget_sec = max(0.0, float(getattr(self, "synthesis_time_budget_sec", 0.0)))
        t0 = time.perf_counter()
        decay = max(0.0, float(getattr(self, "synthesis_uncertainty_decay", 0.0)))
        timed_out = False

        for idx, c in enumerate(candidates):
            if eval_budget_sec > 0.0 and (time.perf_counter() - t0) > eval_budget_sec:
                timed_out = True
                break
            if idx not in selected_set:
                continue
            topo = self._clip01(self._safe_float(getattr(c, "topo_probability", 0.0), 0.0))
            if topo < topo_gate:
                continue
            result = self._evaluate_synthesizability(c)
            score = self._clip01(self._safe_float(result.get("score", 0.0), 0.0))
            if decay > 0.0:
                energy_std = max(0.0, self._safe_float(getattr(c, "energy_std", 0.0), 0.0))
                score *= float(math.exp(-decay * energy_std))
            c.synthesis_score = self._clip01(score)
            c.synthesis_feasibility = 1.0 if bool(result.get("synthesizable", False)) else 0.0
        if timed_out:
            logger.info(
                "Synthesis evaluation hit time budget %.3fs; remaining candidates deferred.",
                eval_budget_sec,
            )
        return [self._clip01(float(c.synthesis_score)) for c in candidates]

    @staticmethod
    def _candidate_feature_vector(candidate: Candidate) -> np.ndarray:
        energy = candidate.energy_mean
        if energy is None:
            energy = candidate.energy_per_atom
        energy_std = candidate.energy_std if candidate.energy_std is not None else 0.0
        return np.array(
            [
                float(candidate.topo_probability),
                float(candidate.stability_score),
                float(candidate.heuristic_topo_score),
                float(candidate.novelty_score),
                float(getattr(candidate, "synthesis_score", 0.0)),
                float(energy if energy is not None else 0.0),
                float(energy_std),
            ],
            dtype=float,
        )

    def _candidate_diversity_vector(self, candidate: Candidate) -> np.ndarray:
        """
        Chemical-space descriptor used by local penalization.

        Reference:
        - Ward et al. (2016), compositional descriptors for materials:
          https://doi.org/10.1038/npjcompumats.2016.28
        """
        struct = candidate.relaxed_structure or candidate.structure
        if struct is None or not hasattr(struct, "composition"):
            return self._candidate_feature_vector(candidate)

        comp = struct.composition
        try:
            amount_dict = comp.get_el_amt_dict()
        except Exception:
            return self._candidate_feature_vector(candidate)

        total = float(sum(amount_dict.values()))
        if total <= 0:
            return self._candidate_feature_vector(candidate)

        z_vals: list[float] = []
        frac_vals: list[float] = []
        for el, amt in amount_dict.items():
            if amt <= 0:
                continue
            if Element is None:
                continue
            try:
                z = float(Element(el).Z)
            except Exception:
                continue
            frac = float(amt / total)
            z_vals.append(z)
            frac_vals.append(frac)

        if not z_vals:
            return self._candidate_feature_vector(candidate)

        z_arr = np.asarray(z_vals, dtype=float)
        f_arr = np.asarray(frac_vals, dtype=float)
        z_mean = float(np.sum(f_arr * z_arr))
        z_var = float(np.sum(f_arr * (z_arr - z_mean) ** 2))
        entropy = float(-np.sum(f_arr * np.log(np.clip(f_arr, 1e-12, 1.0))))
        n_elem = float(len(z_vals))
        max_frac = float(np.max(f_arr))

        volume_per_atom = 0.0
        try:
            n_sites = max(1, int(struct.num_sites))
            volume_per_atom = float(struct.volume) / float(n_sites)
        except Exception:
            volume_per_atom = 0.0

        return np.array(
            [z_mean, math.sqrt(max(0.0, z_var)), entropy, n_elem, max_frac, volume_per_atom],
            dtype=float,
        )

    def _candidate_ood_vector(self, candidate: Candidate) -> np.ndarray:
        """
        Feature space used by OOD penalty.

        - `chemistry`: composition/structure descriptor.
        - `performance`: predicted objective descriptor.
        """
        space = str(getattr(self, "ood_space", "chemistry")).strip().lower()
        if space in {"performance", "objective", "acquisition"}:
            return self._candidate_feature_vector(candidate)
        return self._candidate_diversity_vector(candidate)

    def _historical_ood_matrix(self) -> np.ndarray:
        """
        Build historical matrix for OOD calibration.
        """
        history = self.all_candidates if self.all_candidates else self.top_candidates
        vectors: list[np.ndarray] = []
        for c in history:
            try:
                v = np.asarray(self._candidate_ood_vector(c), dtype=float).reshape(-1)
            except Exception:
                continue
            if v.size == 0 or not np.all(np.isfinite(v)):
                continue
            vectors.append(v)
        if not vectors:
            return np.zeros((0, 0), dtype=float)
        dim = int(min(v.shape[0] for v in vectors))
        if dim <= 0:
            return np.zeros((0, 0), dtype=float)
        return np.vstack([v[:dim] for v in vectors])

    def _estimate_ood_scores(self, candidates: list[Candidate]) -> list[float]:
        """
        Estimate OOD (out-of-distribution) scores in [0, 1].

        Method:
        - Robust standardization via median/IQR.
        - Radius threshold by historical distance quantile.
        - Smooth excess-distance map: `score = 1 - exp(-excess_ratio)`.

        References:
        - Huber (1981), Robust Statistics.
        - Rousseeuw & Croux (1993), robust scale estimators.
        """
        if not candidates:
            return []
        if not bool(getattr(self, "use_ood_penalty", False)):
            return [0.0 for _ in candidates]

        hist = self._historical_ood_matrix()
        min_points = max(1, int(getattr(self, "ood_history_min_points", 24)))
        if hist.shape[0] < min_points or hist.shape[1] <= 0:
            return [0.0 for _ in candidates]

        center = np.median(hist, axis=0)
        q25 = np.percentile(hist, 25.0, axis=0)
        q75 = np.percentile(hist, 75.0, axis=0)
        scale = q75 - q25
        # Fallback when IQR degenerates on low-variance dimensions.
        # 低方差维度回退到标准差，避免尺度塌陷。
        std_fallback = np.std(hist, axis=0)
        scale = np.where(scale < 1e-8, std_fallback, scale)
        scale = np.where(scale < 1e-8, 1.0, scale)

        hist_z = (hist - center.reshape(1, -1)) / scale.reshape(1, -1)
        hist_dist = np.sqrt(np.mean(hist_z * hist_z, axis=1))
        q = min(max(float(getattr(self, "ood_quantile", 0.95)), 0.50), 0.999)
        threshold = float(np.quantile(hist_dist, q))
        threshold = max(threshold, 1e-6)

        dim = int(hist.shape[1])
        out: list[float] = []
        for c in candidates:
            try:
                v = np.asarray(self._candidate_ood_vector(c), dtype=float).reshape(-1)
            except Exception:
                out.append(0.0)
                continue
            if v.size <= 0:
                out.append(0.0)
                continue
            if v.size < dim:
                padded = np.zeros(dim, dtype=float)
                padded[: v.size] = v
                v_use = padded
            else:
                v_use = v[:dim]
            if not np.all(np.isfinite(v_use)):
                out.append(0.0)
                continue
            z = (v_use - center) / scale
            dist = float(math.sqrt(float(np.mean(z * z))))
            excess = max(0.0, dist - threshold)
            ratio = excess / threshold
            out.append(self._clip01(1.0 - math.exp(-ratio)))
        return out

    def _apply_ood_penalty(self, candidates: list[Candidate]) -> None:
        """
        Apply multiplicative OOD penalty to final acquisition utility.

        utility' = utility * (1 - w * ood_score), with w in [0, 1].
        """
        if not candidates:
            return
        penalty_weight = min(max(float(getattr(self, "ood_penalty_weight", 0.0)), 0.0), 1.0)
        if penalty_weight <= 0.0 or not bool(getattr(self, "use_ood_penalty", False)):
            for c in candidates:
                c.ood_score = 0.0
            return

        ood_scores = self._estimate_ood_scores(candidates)
        for idx, c in enumerate(candidates):
            score = self._clip01(self._safe_float(ood_scores[idx] if idx < len(ood_scores) else 0.0, 0.0))
            c.ood_score = score
            c.acquisition_value = float(c.acquisition_value) * (1.0 - penalty_weight * score)

    @staticmethod
    def _pareto_front(points: np.ndarray) -> np.ndarray:
        return pareto_front(points)

    @staticmethod
    def _non_dominated_sort(points: np.ndarray) -> list[np.ndarray]:
        """
        Non-dominated sorting (maximization) used in NSGA-II style ranking.

        Reference:
        - Deb et al. (2002), NSGA-II:
          https://doi.org/10.1109/4235.996017
        """
        return non_dominated_sort(points)

    @staticmethod
    def _crowding_distance(points: np.ndarray) -> np.ndarray:
        """
        Crowding-distance metric for front diversity in objective space.

        Reference:
        - Deb et al. (2002), NSGA-II:
          https://doi.org/10.1109/4235.996017
        """
        return crowding_distance(points)

    @classmethod
    def _pareto_rank_score(cls, points: np.ndarray) -> np.ndarray:
        """
        Convert Pareto fronts + crowding into normalized ranking score [0, 1].
        """
        return pareto_rank_score(points)

    def _apply_pareto_rank_bonus(
        self,
        candidates: list[Candidate],
        topo_terms: list[float],
        stability_terms: list[float],
        synthesis_terms: list[float] | None = None,
    ) -> None:
        """
        Inject Pareto rank signal (front rank + crowding) into scalar utility.
        """
        if not bool(getattr(self, "use_pareto_rank_bonus", False)):
            return
        if not candidates:
            return

        rank_weight = min(max(float(getattr(self, "pareto_rank_bonus_weight", 0.0)), 0.0), 1.0)
        if rank_weight <= 0.0:
            return

        points = objective_points_from_terms(topo_terms, stability_terms, synthesis_terms)
        if points.size == 0:
            return

        feas_threshold = float(getattr(self, "pareto_feasibility_threshold", 0.0))
        feas_mask = points[:, 0] >= feas_threshold
        feasible_points = points[feas_mask]
        rank_scores = np.zeros(points.shape[0], dtype=float)
        if feasible_points.size > 0:
            rank_scores[feas_mask] = self._pareto_rank_score(feasible_points)

        for idx, c in enumerate(candidates):
            c.acquisition_value = (1.0 - rank_weight) * float(c.acquisition_value) + rank_weight * float(
                rank_scores[idx]
            )

    @classmethod
    def _hypervolume_2d(cls, points: np.ndarray, reference: np.ndarray) -> float:
        """
        Exact 2D dominated hypervolume for maximization objectives.

        Reference:
        - Daulton et al. (2020), differentiable EHVI:
          https://arxiv.org/abs/2006.05078
        """
        return hypervolume_2d(points, reference)

    def _hypervolume(self, points: np.ndarray, reference: np.ndarray) -> float:
        """
        Dominated hypervolume for maximization objectives.

        - Exact in 2D via geometric sweep.
        - Monte Carlo approximation in >=3D (qEHVI/qNEHVI-style integration bridge).
        """
        return hypervolume(
            points,
            reference,
            hv_mc_samples=int(getattr(self, "hv_mc_samples", 4096)),
            hv_mc_seed=int(getattr(self, "hv_mc_seed", 12345)),
            iteration=int(getattr(self, "iteration", 1)),
            hv_chunk_size=int(getattr(self, "hv_chunk_size", 512)),
        )

    def _sample_hv_probes(
        self,
        reference: np.ndarray,
        upper: np.ndarray,
        *,
        dim: int,
    ) -> np.ndarray:
        # Kept for backward compatibility with internal callers.
        samples = max(256, int(getattr(self, "hv_mc_samples", 4096)))
        seed = int(getattr(self, "hv_mc_seed", 12345)) + int(getattr(self, "iteration", 1))
        rng = np.random.default_rng(seed)
        u = rng.random((samples, dim))
        side = (upper - reference).reshape(1, -1)
        return reference.reshape(1, -1) + u * side

    def _mc_hv_improvements_shared(
        self,
        hist_arr: np.ndarray,
        cand_points: np.ndarray,
        ref: np.ndarray,
        feas_threshold: float,
    ) -> np.ndarray:
        """
        Shared-sample Monte Carlo HV improvements for candidate pool.

        Uses one probe set for all candidates to reduce repeated MC variance and cost.
        """
        return mc_hv_improvements_shared(
            hist_arr,
            cand_points,
            ref,
            feas_threshold=feas_threshold,
            hv_mc_samples=int(getattr(self, "hv_mc_samples", 4096)),
            hv_mc_seed=int(getattr(self, "hv_mc_seed", 12345)),
            iteration=int(getattr(self, "iteration", 1)),
            hv_chunk_size=int(getattr(self, "hv_chunk_size", 512)),
        )

    def _apply_pareto_hv_bonus(
        self,
        candidates: list[Candidate],
        topo_terms: list[float],
        stability_terms: list[float],
        synthesis_terms: list[float] | None = None,
    ) -> None:
        """
        Add deterministic EHVI-style bonus on top of scalar acquisition score.

        This is a low-cost bridge toward EHVI/qNEHVI-style decision policies,
        while preserving the existing controller interface.

        References:
        - Daulton et al. (2020), qEHVI: https://arxiv.org/abs/2006.05078
        - Daulton et al. (2021), qNEHVI: https://arxiv.org/abs/2105.08195
        """
        if not bool(getattr(self, "use_pareto_hv_bonus", False)):
            return
        if not candidates:
            return

        hv_weight = min(max(float(getattr(self, "pareto_hv_bonus_weight", 0.0)), 0.0), 1.0)
        if hv_weight <= 0.0:
            return
        feas_threshold = float(getattr(self, "pareto_feasibility_threshold", 0.0))
        use_joint_feas = bool(getattr(self, "pareto_joint_feasibility", False))
        syn_feas_threshold = float(getattr(self, "pareto_synthesis_feasibility_threshold", 0.10))

        use_synthesis = synthesis_terms is not None
        obj_dim = 3 if use_synthesis else 2
        hist_arr = collect_history_objective_points(
            self.all_candidates,
            obj_dim=obj_dim,
            topo_threshold=feas_threshold,
            use_joint_synthesis=use_joint_feas and use_synthesis,
            synthesis_threshold=syn_feas_threshold,
        )
        cand_points = objective_points_from_terms(topo_terms, stability_terms, synthesis_terms)
        if cand_points.size == 0:
            return

        merged = np.vstack([hist_arr, cand_points]) if hist_arr.size else cand_points

        ref = np.min(merged, axis=0) - 1e-3
        candidate_feas_mask = feasibility_mask_from_points(
            cand_points,
            topo_threshold=feas_threshold,
            use_joint_synthesis=use_joint_feas and use_synthesis,
            synthesis_threshold=syn_feas_threshold,
        )

        pool_limit = max(8, int(getattr(self, "hv_candidate_pool_limit", 96)))
        base_vals = np.asarray([float(c.acquisition_value) for c in candidates], dtype=float)
        feasible_indices = np.flatnonzero(candidate_feas_mask)
        if feasible_indices.size > pool_limit:
            order = np.argsort(base_vals[feasible_indices])[::-1]
            keep = feasible_indices[order[:pool_limit]]
            candidate_feas_mask = np.zeros_like(candidate_feas_mask, dtype=bool)
            candidate_feas_mask[keep] = True

        filtered = cand_points.copy()
        filtered[~candidate_feas_mask] = ref.reshape(1, -1)
        if bool(getattr(self, "hv_use_shared_samples", True)):
            improvements = self._mc_hv_improvements_shared(hist_arr, filtered, ref, feas_threshold)
        else:
            baseline = self._hypervolume(hist_arr, ref)
            improvements = np.zeros(len(candidates), dtype=float)
            for idx, point in enumerate(filtered):
                if not candidate_feas_mask[idx]:
                    continue
                augmented = (
                    point.reshape(1, cand_points.shape[1])
                    if hist_arr.size == 0
                    else np.vstack([hist_arr, point.reshape(1, cand_points.shape[1])])
                )
                hv_after = self._hypervolume(augmented, ref)
                improvements[idx] = max(0.0, hv_after - baseline)

        max_imp = float(np.max(improvements))
        if max_imp <= 0.0:
            return
        normalized_imp = improvements / max_imp
        for idx, c in enumerate(candidates):
            c.acquisition_value = (1.0 - hv_weight) * float(c.acquisition_value) + hv_weight * float(
                normalized_imp[idx]
            )

    def _select_top_diverse(
        self,
        candidates: list[Candidate],
        n_top: int,
        *,
        objective_map: dict[int, np.ndarray] | None = None,
    ) -> list[Candidate]:
        """
        Greedy diversity-aware batch selection.

        Combines:
        - Local penalization in feature space.
        - Optional greedy hypervolume-gain bonus (qEHVI-inspired).

        Reference:
        - Gonzalez et al. (2016), Batch BO via Local Penalization:
          https://proceedings.mlr.press/v51/gonzalez16a.html
        - Daulton et al. (2020), qEHVI:
          https://arxiv.org/abs/2006.05078
        """
        if n_top <= 0:
            return []
        if len(candidates) <= n_top:
            return list(candidates)

        lam = max(0.0, float(getattr(self, "batch_diversity_strength", 0.0)))
        use_hv = bool(getattr(self, "use_hv_batch_greedy", False)) and objective_map is not None
        if lam <= 0.0 and not use_hv:
            return list(candidates[:n_top])

        sigma = max(1e-6, float(getattr(self, "batch_diversity_sigma", 1.0)))
        base = np.asarray([float(c.acquisition_value) for c in candidates], dtype=float)
        similarity_matrix = None
        max_similarity = None
        if lam > 0.0:
            diversity_space = str(getattr(self, "batch_diversity_space", "chemistry")).strip().lower()
            if diversity_space == "performance":
                features = np.vstack([self._candidate_feature_vector(c) for c in candidates])
            else:
                features = np.vstack([self._candidate_diversity_vector(c) for c in candidates])
            feat_std = features.std(axis=0, keepdims=True)
            feat_std[feat_std < 1e-8] = 1.0
            features = (features - features.mean(axis=0, keepdims=True)) / feat_std
            if features.shape[0] > 0:
                # Precompute Gaussian similarity once; later rounds only do max-updates.
                sq_norm = np.sum(features * features, axis=1, keepdims=True)
                d2 = np.maximum(0.0, sq_norm + sq_norm.T - 2.0 * (features @ features.T))
                similarity_matrix = np.exp(-d2 / (2.0 * sigma * sigma))
        else:
            features = None

        scale = max(1e-6, float(np.std(base)))
        penalty_weight = lam * scale
        hv_weight = max(0.0, float(getattr(self, "hv_batch_weight", 0.0))) * scale

        selected_idx: list[int] = [int(np.argmax(base))]
        selected_mask = np.zeros(len(candidates), dtype=bool)
        selected_mask[selected_idx[0]] = True
        if similarity_matrix is not None:
            max_similarity = similarity_matrix[selected_idx[0]].copy()

        hist_arr = np.zeros((0, 2), dtype=float)
        ref = None
        feas_threshold = float(getattr(self, "pareto_feasibility_threshold", 0.0))
        candidate_obj = None
        obj_dim = 2
        if use_hv:
            obj_dim = infer_objective_dimension(objective_map, default=2)
            hist_arr = collect_history_objective_points(
                self.all_candidates,
                obj_dim=obj_dim,
                topo_threshold=feas_threshold,
            )
            candidate_obj = objective_points_from_map(candidates, objective_map, obj_dim=obj_dim)
            merged = candidate_obj if hist_arr.size == 0 else np.vstack([hist_arr, candidate_obj])
            ref = np.min(merged, axis=0) - 1e-3

        while len(selected_idx) < n_top:
            best_i = -1
            current_hv = 0.0
            selected_obj: list[np.ndarray] = []
            current_arr = hist_arr
            use_joint_feas = bool(getattr(self, "pareto_joint_feasibility", False))
            syn_feas_threshold = float(getattr(self, "pareto_synthesis_feasibility_threshold", 0.10))
            if use_hv and ref is not None:
                obj_dim = int(ref.shape[0])
                selected_obj = [
                    (
                        candidate_obj[idx]
                        if candidate_obj is not None
                        else np.zeros(obj_dim, dtype=float)
                    ).reshape(obj_dim)
                    for idx in selected_idx
                ]
                if selected_obj:
                    selected_stack = np.vstack(selected_obj)
                    current_arr = selected_stack if current_arr.size == 0 else np.vstack([current_arr, selected_stack])
                current_hv = self._hypervolume(current_arr, ref)

            hv_gain_cache: dict[int, float] = {}
            hv_gain_array = np.zeros(len(candidates), dtype=float)
            used_shared_hv_gains = False
            max_hv_gain = 0.0
            if use_hv and ref is not None and candidate_obj is not None:
                obj_dim = int(ref.shape[0])
                if bool(getattr(self, "hv_use_shared_samples", True)):
                    hv_gain_array = self._mc_hv_improvements_shared(
                        current_arr,
                        candidate_obj,
                        ref,
                        feas_threshold,
                    )
                    if use_joint_feas and obj_dim >= 3:
                        hv_gain_array[candidate_obj[:, 2] < syn_feas_threshold] = 0.0
                    hv_gain_array[selected_mask] = 0.0
                    used_shared_hv_gains = True
                    max_hv_gain = float(np.max(hv_gain_array))
                else:
                    for i in range(len(candidates)):
                        if selected_mask[i]:
                            continue
                        p = candidate_obj[i].reshape(obj_dim)
                        if p[0] < feas_threshold:
                            hv_gain_cache[i] = 0.0
                            continue
                        if use_joint_feas and obj_dim >= 3 and p[2] < syn_feas_threshold:
                            hv_gain_cache[i] = 0.0
                            continue
                        augmented = (
                            p.reshape(1, obj_dim)
                            if current_arr.size == 0
                            else np.vstack([current_arr, p.reshape(1, obj_dim)])
                        )
                        gain = max(0.0, self._hypervolume(augmented, ref) - current_hv)
                        hv_gain_cache[i] = gain
                        if gain > max_hv_gain:
                            max_hv_gain = gain

            candidate_indices = np.flatnonzero(~selected_mask)
            if candidate_indices.size == 0:
                break

            penalties = np.zeros(candidate_indices.shape[0], dtype=float)
            if max_similarity is not None and penalty_weight > 0.0:
                penalties = max_similarity[candidate_indices]

            hv_bonus = np.zeros(candidate_indices.shape[0], dtype=float)
            if use_hv and hv_weight > 0.0 and max_hv_gain > 0.0:
                if used_shared_hv_gains:
                    gains = hv_gain_array[candidate_indices]
                else:
                    gains = np.asarray([hv_gain_cache.get(int(i), 0.0) for i in candidate_indices], dtype=float)
                hv_bonus = hv_weight * (gains / max_hv_gain)

            adjusted = base[candidate_indices] - penalty_weight * penalties + hv_bonus
            best_pos = int(np.argmax(adjusted))
            best_i = int(candidate_indices[best_pos])
            if best_i < 0:
                break
            selected_idx.append(best_i)
            selected_mask[best_i] = True
            if max_similarity is not None and similarity_matrix is not None:
                max_similarity = np.maximum(max_similarity, similarity_matrix[best_i])

        return [candidates[i] for i in selected_idx]

    def _score_and_select(self, candidates: list[Candidate], n_top: int) -> list[Candidate]:
        engine = getattr(self, "policy_engine", None)
        if engine is not None:
            return engine.score_and_select(self, candidates, n_top)
        return self._score_and_select_legacy(candidates, n_top)

    def _score_and_select_legacy(self, candidates: list[Candidate], n_top: int) -> list[Candidate]:
        topo_terms: list[float] = []
        stability_terms: list[float] = []
        synthesis_terms: list[float] = []
        objective_map: dict[int, np.ndarray] = {}
        for c in candidates:
            # Novelty is structure-aware to avoid collapsing polymorphs by formula.
            candidate_structure = c.relaxed_structure or c.structure
            is_new = not self._is_duplicate_structure(candidate_structure)
            c.novelty_score = 1.0 if is_new else 0.0
            c.ood_score = 0.0

            stability_term = max(0.0, float(self._stability_component(c)))
            topo_prob = max(0.0, min(1.0, float(c.topo_probability)))
            topo_terms.append(topo_prob)
            stability_terms.append(stability_term)
            synthesis_terms.append(0.0)
            c.synthesis_score = 0.0
            c.synthesis_feasibility = 0.0

            if self.use_constrained_acquisition:
                # Expected Improvement with Constraints (EIC) style composition:
                # utility ~ objective_utility * feasibility_probability.
                # Ref: Gardner et al. (2014), https://proceedings.mlr.press/v32/gardner14.html
                constrained_stability = stability_term * topo_prob
                c.acquisition_value = (
                    self.weights["topo"] * topo_prob +
                    self.weights["stability"] * constrained_stability +
                    self.weights["heuristic"] * c.heuristic_topo_score +
                    self.weights["novelty"] * c.novelty_score
                )
            else:
                c.acquisition_value = (
                    self.weights["topo"] * topo_prob +
                    self.weights["stability"] * stability_term +
                    self.weights["heuristic"] * c.heuristic_topo_score +
                    self.weights["novelty"] * c.novelty_score
                )

        if self.gp_acquirer is not None:
            gp_utility = self.gp_acquirer.suggest_constrained_utility(candidates)
            if gp_utility is None:
                gp_utility = self.gp_acquirer.suggest_ucb(candidates)
            if gp_utility is not None:
                blend = self.gp_acquirer.config.blend_ratio
                for c, gp_score in zip(candidates, gp_utility, strict=False):
                    c.acquisition_value = (1.0 - blend) * c.acquisition_value + blend * float(gp_score)

        synthesis_scores = self._apply_synthesis_objective(candidates)
        synthesis_weight = min(max(float(getattr(self, "synthesis_objective_weight", 0.0)), 0.0), 1.0)
        use_synthesis = bool(getattr(self, "use_synthesis_objective", False)) and synthesis_weight > 0.0
        for idx, c in enumerate(candidates):
            synth = self._clip01(self._safe_float(synthesis_scores[idx] if idx < len(synthesis_scores) else 0.0, 0.0))
            if self.use_constrained_acquisition:
                synth = synth * self._clip01(self._safe_float(c.topo_probability, 0.0))
            synthesis_terms[idx] = synth
            if use_synthesis:
                c.acquisition_value = (1.0 - synthesis_weight) * float(c.acquisition_value) + synthesis_weight * synth

            if use_synthesis:
                objective_map[id(c)] = np.array([topo_terms[idx], stability_terms[idx], synthesis_terms[idx]], dtype=float)
            else:
                objective_map[id(c)] = np.array([topo_terms[idx], stability_terms[idx]], dtype=float)

        self._apply_pareto_rank_bonus(
            candidates,
            topo_terms,
            stability_terms,
            synthesis_terms if use_synthesis else None,
        )
        self._apply_pareto_hv_bonus(
            candidates,
            topo_terms,
            stability_terms,
            synthesis_terms if use_synthesis else None,
        )
        self._apply_ood_penalty(candidates)
        candidates.sort(key=lambda x: x.acquisition_value, reverse=True)
        top = self._select_top_diverse(candidates, n_top, objective_map=objective_map)
        return self._finalize_ranked_candidates(candidates, top)

    def _finalize_ranked_candidates(
        self,
        candidates: list[Candidate],
        top: list[Candidate],
    ) -> list[Candidate]:
        selected_ids = {id(c) for c in top}
        remaining = [c for c in candidates if id(c) not in selected_ids]
        ranked = top + remaining

        # Register winners
        for c in top:
            self._register_known_structure(c.relaxed_structure or c.structure, formula_hint=c.formula)
            self.top_candidates.append(c)

        self.all_candidates.extend(candidates)
        if len(self.all_candidates) > self.max_observation_history:
            self.all_candidates = self.all_candidates[-self.max_observation_history :]

        if self.gp_acquirer is not None:
            # Update surrogate on all evaluated points to avoid top-k censoring bias.
            # Ref: Letham et al. (2019), noisy BO under observation uncertainty.
            self.gp_acquirer.update(candidates)

        # Optional: Attempt to find a synthesis pathway using reaction-network logic
        if rx is not None:
            self._analyze_pathways(top)

        return ranked

    def _analyze_pathways(self, candidates: list[Candidate]):
        """
        Uses rustworkx (inspired by materialsproject/reaction-network)
        to model basic synthesis pathways for top candidates.
        """
        logger.info("Analyzing synthesis pathways for top candidates...")
        # Create a basic graph network
        graph = rx.PyDiGraph()

        # In a full implementation, we would extract available precursors
        # from Materials Project/JARVIS and link them to the target candidates
        # via mass-balanced edges. Here we simulate the network builder.
        precursor_idx = graph.add_node("__ELEMENTS__")
        max_annotations = int(getattr(self, "max_pathway_annotations_per_iter", 8))
        annotation_limit = len(candidates) if max_annotations <= 0 else max(0, max_annotations)
        annotated = 0

        for cand in candidates:
             # Add target node
             target_idx = graph.add_node(cand.formula)

             if self.synthesizability_evaluator:
                 if annotated >= annotation_limit:
                     continue
                 eval_result = self._evaluate_synthesizability(cand)
                 cand.synthesis_score = self._clip01(self._safe_float(eval_result.get("score", 0.0), 0.0))
                 cand.synthesis_feasibility = 1.0 if bool(eval_result.get("synthesizable", False)) else 0.0
                 if eval_result["synthesizable"]:
                     pathways = eval_result.get("pathway", [])
                     if pathways:
                         cand.mutations += " " + str(pathways[0])
                 annotated += 1
             elif cand.energy_per_atom and cand.energy_per_atom < -1.0:
                 graph.add_edge(precursor_idx, target_idx, "Exothermic Formation")
                 cand.mutations += " [Synthesizable via direct elements]"

        if self.synthesizability_evaluator and annotation_limit < len(candidates):
            logger.info(
                "Pathway annotations capped at %s this iteration (candidates=%s).",
                annotation_limit,
                len(candidates),
            )
        logger.info(f"Built pathway graph with {graph.num_nodes()} nodes and {graph.num_edges()} edges.")

    def _save_iteration(self, iteration: int, top_candidates: list[Candidate]):
        data = {
            "iteration": iteration,
            "timestamp": time.time(),
            "n_candidates": len(top_candidates),
            "candidates": [c.to_dict() for c in top_candidates]
        }
        fpath = self.results_dir / f"iteration_{iteration:03d}.json"
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_final_report(self, filename: str = "final_report.json"):
        """
        Persist a final summary after discovery loop completes.
        """
        unique_formulas = sorted({c.formula for c in self.top_candidates})
        top_ranked = sorted(self.top_candidates, key=lambda c: c.acquisition_value, reverse=True)
        payload = {
            "timestamp": time.time(),
            "iterations_completed": self.iteration,
            "n_top_candidates": len(self.top_candidates),
            "n_unique_formulas": len(unique_formulas),
            "unique_formulas": unique_formulas,
            "top_candidates": [c.to_dict() for c in top_ranked[:50]],
            "weights": self.weights,
            "method_key": self.method_key,
            "fallback_methods": list(self.fallback_methods),
            "policy": {
                "name": self.policy_config.policy_name,
                "risk_mode": self.policy_config.risk_mode,
                "cost_aware": self.policy_config.cost_aware,
                "calibration_window": self.policy_config.calibration_window,
                "ood_combination": self.policy_config.ood_combination,
                "ood_gate_threshold": self.policy_config.ood_gate_threshold,
                "conformal_gate_threshold": self.policy_config.conformal_gate_threshold,
                "conformal_alpha": self.policy_config.conformal_alpha,
                "conformal_penalty_weight": self.policy_config.conformal_penalty_weight,
                "policy_ood_penalty_weight": self.policy_config.ood_penalty_weight,
                "max_conformal_radius": self.policy_config.max_conformal_radius,
                "diversity_novelty_boost": self.policy_config.diversity_novelty_boost,
                "cost_eps": self.policy_config.cost_eps,
                "relax_cost_base": self.policy_config.relax_cost_base,
                "classify_cost_base": self.policy_config.classify_cost_base,
                "calibration_min_points": self.policy_config.calibration_min_points,
                "relax_timeout_sec": self.policy_config.relax_timeout_sec,
                "relax_max_retries": self.policy_config.relax_max_retries,
                "relax_circuit_breaker_failures": self.policy_config.relax_circuit_breaker_failures,
                "relax_circuit_breaker_cooldown_iters": self.policy_config.relax_circuit_breaker_cooldown_iters,
            },
            "policy_state": self.policy_state.to_dict(),
            "acquisition": {
                "strategy": self.acquisition_strategy,
                "kappa": self.acquisition_kappa,
                "kappa_schedule": self.acquisition_kappa_schedule,
                "best_f": self.acquisition_best_f,
                "jitter": self.acquisition_jitter,
                "constrained": self.use_constrained_acquisition,
                "use_noisy_ei": self.use_noisy_ei,
                "noisy_ei_mc_samples": self.noisy_ei_mc_samples,
                "batch_diversity_strength": self.batch_diversity_strength,
                "batch_diversity_sigma": self.batch_diversity_sigma,
                "batch_diversity_space": self.batch_diversity_space,
                "dynamic_best_f": self.dynamic_best_f,
                "dynamic_best_f_quantile": self.dynamic_best_f_quantile,
                "use_pareto_rank_bonus": self.use_pareto_rank_bonus,
                "pareto_rank_bonus_weight": self.pareto_rank_bonus_weight,
                "use_pareto_hv_bonus": self.use_pareto_hv_bonus,
                "pareto_hv_bonus_weight": self.pareto_hv_bonus_weight,
                "pareto_feasibility_threshold": self.pareto_feasibility_threshold,
                "use_hv_batch_greedy": self.use_hv_batch_greedy,
                "hv_batch_weight": self.hv_batch_weight,
                "hv_mc_samples": self.hv_mc_samples,
                "hv_chunk_size": self.hv_chunk_size,
                "hv_use_shared_samples": self.hv_use_shared_samples,
                "hv_candidate_pool_limit": self.hv_candidate_pool_limit,
                "experimental_algorithms_enabled": self.experimental_algorithms_enabled,
                "use_synthesis_objective": self.use_synthesis_objective,
                "synthesis_objective_weight": self.synthesis_objective_weight,
                "synthesis_eval_topk": self.synthesis_eval_topk,
                "synthesis_gate_topo_threshold": self.synthesis_gate_topo_threshold,
                "synthesis_eval_strategy": self.synthesis_eval_strategy,
                "synthesis_uncertainty_weight": self.synthesis_uncertainty_weight,
                "synthesis_uncertainty_decay": self.synthesis_uncertainty_decay,
                "synthesis_time_budget_sec": self.synthesis_time_budget_sec,
                "synthesis_cache_max_size": self.synthesis_cache_max_size,
                "pareto_joint_feasibility": self.pareto_joint_feasibility,
                "pareto_synthesis_feasibility_threshold": self.pareto_synthesis_feasibility_threshold,
                "use_ood_penalty": self.use_ood_penalty,
                "ood_penalty_weight": self.ood_penalty_weight,
                "ood_history_min_points": self.ood_history_min_points,
                "ood_quantile": self.ood_quantile,
                "ood_space": self.ood_space,
                "max_pathway_annotations_per_iter": self.max_pathway_annotations_per_iter,
                "enable_structure_dedup": self.enable_structure_dedup,
            },
        }
        if self.gp_acquirer is not None:
            payload["gp_history_size"] = self.gp_acquirer.history_size
            payload["gp_objective_history_size"] = getattr(
                self.gp_acquirer, "objective_history_size", self.gp_acquirer.history_size
            )
            payload["gp_feasibility_history_size"] = getattr(
                self.gp_acquirer, "feasibility_history_size", self.gp_acquirer.history_size
            )
        fpath = self.results_dir / filename
        with open(fpath, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"Final discovery report saved to {fpath}")

    def _print_iteration_summary(self, top: list[Candidate], dt: float):
        print(f"\n  Top candidates ({dt:.1f}s):")
        print(f"  {'Rank':>4} {'Formula':>12} {'Method':>10} {'Topo%':>6} {'Stab':>5} {'Acq':>5}")
        print(f"  {'----':>4} {'-------':>12} {'------':>10} {'-----':>6} {'----':>5} {'---':>5}")
        for i, c in enumerate(top[:5]):
            print(f"  {i+1:4d} {c.formula:>12s} {c.method:>10s} "
                  f"{c.topo_probability:6.2f} {c.stability_score:5.2f} {c.acquisition_value:5.2f}")
