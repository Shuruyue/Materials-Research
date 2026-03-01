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
import time
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
from atlas.config import get_config
from atlas.models.prediction_utils import extract_mean_and_std
from atlas.research.workflow_reproducible_graph import IterationSnapshot, WorkflowReproducibleGraph
from atlas.utils.registry import EvaluatorFactory, ModelFactory, RelaxerFactory
from atlas.utils.reproducibility import set_global_seed

try:
    import rustworkx as rx
except ImportError:
    rx = None

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
    acquisition_value: float = 0.0

    # Properties
    energy_per_atom: float | None = None
    energy_mean: float | None = None
    energy_std: float | None = None
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
        results_dir: str | Path | None = None,
    ):
        self.generator = generator
        self.relaxer = relaxer
        self.classifier = classifier_model
        self.graph_builder = graph_builder


        self.cfg = get_config()
        self.profile = self.cfg.profile

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
        self._acquisition_generator = torch.Generator().manual_seed(int(self.cfg.train.seed) + 17)
        self.method_key = getattr(self.profile, "method_key", "graph_equivariant")
        self.fallback_methods = tuple(getattr(self.profile, "fallback_methods", ()))
        self.use_gp_active = self.method_key == "gp_active_learning"
        logger.info(
            (
                "Acquisition strategy: %s "
                "(kappa=%.3f, schedule=%s, best_f=%.3f, jitter=%.4f, constrained=%s, "
                "noisy_ei=%s, diversity_lambda=%.3f)"
            ),
            self.acquisition_strategy,
            self.acquisition_kappa,
            self.acquisition_kappa_schedule,
            self.acquisition_best_f,
            self.acquisition_jitter,
            self.use_constrained_acquisition,
            self.use_noisy_ei,
            self.batch_diversity_strength,
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
        self.all_candidates: list[Candidate] = []
        self.top_candidates: list[Candidate] = []

        # Auto-load previous state
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Scan results directory and restore state."""
        files = sorted(self.results_dir.glob("iteration_*.json"))
        if not files:
            logger.info("No checkpoints found. Starting fresh discovery.")
            return

        logger.info(f"Found {len(files)} checkpoints. Restoring state...")

        # Load all history to populate known formulas
        for f in files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    iter_num = data.get("iteration", 0)
                    self.iteration = max(self.iteration, iter_num)

                    for cand_dict in data.get("candidates", []):
                        formula = cand_dict.get("formula")
                        if formula:
                            self.known_formulas.add(formula)
                        # We don't load full objects to RAM to save space,
                        # just formulas for novelty check.

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

        logger.info(f"Resuming from Iteration {self.iteration}. Known formulas: {len(self.known_formulas)}")

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

            # Filter duplicates immediately
            new_raw = [
                r for r in raw_candidates
                if r["structure"].composition.reduced_formula not in self.known_formulas
            ]
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

            if self.workflow is not None:
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

    def _relax_candidates(self, raw_candidates: list[dict]) -> list[Candidate]:
        candidates = []
        for rc in raw_candidates:
            struct = rc["structure"]
            formula = struct.composition.reduced_formula

            # Default fallback values
            energy = None
            converged = False
            relaxed = struct
            stability = 0.5

            if self.relaxer:
                try:
                    # TODO: Add explicit timeout here if feasible
                    res = self.relaxer.relax_structure(struct, steps=100) # steps limit acts as timeout
                    relaxed = res.get("relaxed_structure", struct)
                    energy = res.get("energy_per_atom")
                    converged = res.get("converged", False)
                    stability = self.relaxer.score_stability(relaxed)
                except Exception as e:
                    logger.debug(f"Relaxation error for {formula}: {e}")
                    stability = 0.0 # Penalize failed relaxation

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
                    best_f=self.acquisition_best_f,
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
                best_f=self.acquisition_best_f,
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
            best_f=self.acquisition_best_f,
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
        for c in self.top_candidates:
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
                float(energy if energy is not None else 0.0),
                float(energy_std),
            ],
            dtype=float,
        )

    def _select_top_diverse(self, candidates: list[Candidate], n_top: int) -> list[Candidate]:
        """
        Greedy diversity-aware batch selection (local-penalization style).

        Reference:
        - Gonzalez et al. (2016), Batch BO via Local Penalization:
          https://proceedings.mlr.press/v51/gonzalez16a.html
        """
        if n_top <= 0:
            return []
        if len(candidates) <= n_top:
            return list(candidates)

        lam = max(0.0, float(getattr(self, "batch_diversity_strength", 0.0)))
        if lam <= 0.0:
            return list(candidates[:n_top])

        sigma = max(1e-6, float(getattr(self, "batch_diversity_sigma", 1.0)))
        base = np.asarray([float(c.acquisition_value) for c in candidates], dtype=float)
        features = np.vstack([self._candidate_feature_vector(c) for c in candidates])
        feat_std = features.std(axis=0, keepdims=True)
        feat_std[feat_std < 1e-8] = 1.0
        features = (features - features.mean(axis=0, keepdims=True)) / feat_std

        scale = max(1e-6, float(np.std(base)))
        penalty_weight = lam * scale

        selected_idx: list[int] = [int(np.argmax(base))]
        selected_mask = np.zeros(len(candidates), dtype=bool)
        selected_mask[selected_idx[0]] = True

        while len(selected_idx) < n_top:
            best_i = -1
            best_adj = -float("inf")
            chosen_features = features[selected_idx]
            for i in range(len(candidates)):
                if selected_mask[i]:
                    continue
                d2 = np.sum((chosen_features - features[i]) ** 2, axis=1)
                penalty = float(np.exp(-d2 / (2.0 * sigma * sigma)).max())
                adjusted = float(base[i] - penalty_weight * penalty)
                if adjusted > best_adj:
                    best_adj = adjusted
                    best_i = i
            if best_i < 0:
                break
            selected_idx.append(best_i)
            selected_mask[best_i] = True

        return [candidates[i] for i in selected_idx]

    def _score_and_select(self, candidates: list[Candidate], n_top: int) -> list[Candidate]:
        for c in candidates:
            # Novelty check (already screened, but good for scoring)
            is_new = c.formula not in self.known_formulas
            c.novelty_score = 1.0 if is_new else 0.0

            stability_term = max(0.0, float(self._stability_component(c)))
            topo_prob = max(0.0, min(1.0, float(c.topo_probability)))

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

        candidates.sort(key=lambda x: x.acquisition_value, reverse=True)
        top = self._select_top_diverse(candidates, n_top)

        selected_ids = {id(c) for c in top}
        remaining = [c for c in candidates if id(c) not in selected_ids]
        ranked = top + remaining

        # Register winners
        for c in top:
            self.known_formulas.add(c.formula)
            self.top_candidates.append(c)

        if self.gp_acquirer is not None:
            self.gp_acquirer.update(top)

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

        for cand in candidates:
             # Add target node
             target_idx = graph.add_node(cand.formula)

             if self.synthesizability_evaluator:
                 eval_result = self.synthesizability_evaluator.evaluate(cand.formula, cand.energy_per_atom)
                 if eval_result["synthesizable"]:
                     cand.mutations += " " + eval_result["pathway"][0]
             elif cand.energy_per_atom and cand.energy_per_atom < -1.0:
                 graph.add_edge(0, target_idx, "Exothermic Formation")
                 cand.mutations += " [Synthesizable via direct elements]"

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
