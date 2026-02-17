"""
Active Learning Discovery Controller

The brain of ATLAS — orchestrates the closed-loop discovery cycle:
    Generate -> Relax -> Classify -> Score -> Select -> Loop

Optimization:
- Checkpoint & Resume: Automatically continues from last saved iteration
- Robust Error Handling: Isolates failures in external tools (MACE/GNN)
- Duplicate Prevention: Tracks known formulas across sessions
"""

import json
import time
import shutil
import logging
import torch
from pathlib import Path
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, field, asdict

from atlas.config import get_config
from atlas.active_learning.generator import StructureGenerator
from atlas.active_learning.acquisition import expected_improvement, upper_confidence_bound

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    energy_per_atom: Optional[float] = None
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
    ):
        self.generator = generator
        self.relaxer = relaxer
        self.classifier = classifier_model
        self.graph_builder = graph_builder
        
        # Hyperparams
        self.weights = {
            "topo": w_topo,
            "stability": w_stability,
            "heuristic": w_heuristic,
            "novelty": w_novelty,
        }

        self.cfg = get_config()
        self.results_dir = self.cfg.paths.data_dir / "discovery_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.iteration = 0
        self.known_formulas: Set[str] = set()
        self.all_candidates: List[Candidate] = []
        self.top_candidates: List[Candidate] = []
        
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
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    iter_num = data.get("iteration", 0)
                    self.iteration = max(self.iteration, iter_num)
                    
                    for cand_dict in data.get("candidates", []):
                        self.known_formulas.add(cand_dict.get("formula"))
                        # We don't load full objects to RAM to save space, 
                        # just formulas for novelty check.
                        
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {f}: {e}")

        # Load top candidates from the VERY LAST iteration to seed the generator
        last_file = files[-1]
        try:
            with open(last_file, 'r') as fp:
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
    ) -> List[Candidate]:
        """
        Run the discovery loop, resuming if necessary.
        Total iterations = current_iteration + n_iterations.
        """
        start_iter = self.iteration + 1
        end_iter = self.iteration + n_iterations
        
        print("\n" + "=" * 70)
        print(f"  ATLAS Discovery Engine (Iter {start_iter} -> {end_iter})")
        print("=" * 70)

        for it in range(start_iter, end_iter + 1):
            self.iteration = it
            t0 = time.time()
            
            print(f"\n{'─' * 50}")
            print(f"  Iteration {it}/{end_iter}")
            print(f"{'─' * 50}")

            # 1. Generate
            print(f"  [1/4] Generating {n_candidates_per_iter} candidates...")
            try:
                # Dynamically adjust generation methods based on iteration?
                # Early iters: exploration (random sub). Late: exploitation (strain).
                methods = ["substitute", "strain"] if it > 3 else ["substitute"]
                
                raw_candidates = self.generator.generate_batch(n_candidates_per_iter, methods=methods)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raw_candidates = []

            # Filter duplicates immediately
            new_raw = [r for r in raw_candidates if r["structure"].composition.reduced_formula not in self.known_formulas]
            print(f"        Generated {len(raw_candidates)}, New Unique: {len(new_raw)}")
            
            if not new_raw:
                logger.warning("No new unique candidates. Trying fallback seeds...")
                # Logic to inject random seeds from JARVIS could go here
                break

            # 2. Relax
            print(f"  [2/4] Relaxing {len(new_raw)} structures...")
            candidates = self._relax_candidates(new_raw)

            # 3. Classify
            print(f"  [3/4] Classifying candidates...")
            candidates = self._classify_candidates(candidates)

            # 4. Select
            candidates = self._score_and_select(candidates, n_select_top)

            # Save & Feedback
            self._save_iteration(it, candidates[:n_select_top])
            
            # Add top winners as seeds for next round
            best_structs = [
                c.relaxed_structure or c.structure 
                for c in candidates[:n_select_top]
                if c.acquisition_value > 0.25 # Threshold
            ]
            self.generator.add_seeds(best_structs)
            
            dt = time.time() - t0
            self._print_iteration_summary(candidates[:n_select_top], dt)

        return self.top_candidates

    def _relax_candidates(self, raw_candidates: list[dict]) -> List[Candidate]:
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

        if not self.classifier or not self.graph_builder:
            # Fallback to heuristic
            for c in candidates: c.topo_probability = c.heuristic_topo_score
            return candidates

        self.classifier.eval()
        device = next(self.classifier.parameters()).device
        
        # Batch processing would be better, but loop is safer for now
        for c in candidates:
            try:
                struct = c.relaxed_structure or c.structure
                graph = self.graph_builder.structure_to_graph(struct)
                
                with torch.no_grad():
                    node_feats = graph["node_features"].to(device)
                    if node_feats.dim() == 2: node_feats = node_feats.unsqueeze(0)
                        
                    # Predict using the model (Surrogate)
                    pred = self.classifier(
                        node_feats,
                        graph["edge_index"].to(device),
                        graph["edge_features"].to(device)
                    )
                    
                    # Handle Multi-Task Dictionary Output
                    if isinstance(pred, dict):
                        # 1. Topology (Band Gap or Classification)
                        if "band_gap" in pred: # Regression proxy for topology?
                            bg = pred["band_gap"]
                            if isinstance(bg, dict): # Evidential
                                mu = bg["gamma"].item()
                                std = bg["total_std"].item()
                                # Small band gap (<0.1) often indicates topology? 
                                # Or if we predicted "is_topological" directly
                                c.topo_probability = 1.0 if mu < 0.1 else 0.0 # Placeholder
                            else:
                                c.topo_probability = 1.0 if bg.item() < 0.1 else 0.0

                        # 2. Stability (Formation Energy)
                        if "formation_energy" in pred:
                            fe = pred["formation_energy"]
                            if isinstance(fe, dict): # Evidential
                                mu_fe = fe["gamma"].item()
                                std_fe = fe["total_std"].item()
                                
                                # Use LCB for stability (Minimize Energy) -> Maximize -Energy
                                # But here we just store the score. 
                                # Stability Score = UCB of Stability (to be safe? or LCB to be optimistic?)
                                # Optimistic exploration: LCB (Lower bound of energy is very stable)
                                # LCB = mu - kappa * std
                                lcb = mu_fe - 2.0 * std_fe
                                
                                # Normalize score: < -0.5 eV/atom is good
                                # 0.0 if > 0, 1.0 if < -1.0
                                c.stability_score = max(0.0, min(1.0, -lcb))
                                
                                # Store raw for acquisition
                                c.energy_mean = mu_fe
                                c.energy_std = std_fe
                            else:
                                c.stability_score = max(0.0, min(1.0, -fe.item()))
                                
                    else:
                        # Old binary classifier behavior
                        if pred.shape[-1] == 1:
                            c.topo_probability = torch.sigmoid(pred).item()
                        elif pred.shape[-1] == 2:
                             c.topo_probability = torch.softmax(pred, dim=-1)[0, 1].item()

            except Exception as e:
                # logger.warning(f"Prediction failed: {e}")
                c.topo_probability = 0.0 # Penalize GNN failure
        
        return candidates

    def _score_and_select(self, candidates: List[Candidate], n_top: int) -> List[Candidate]:
        for c in candidates:
            # Novelty check (already screened, but good for scoring)
            is_new = c.formula not in self.known_formulas
            c.novelty_score = 1.0 if is_new else 0.0
            
            if hasattr(c, "energy_mean") and hasattr(c, "energy_std"):
                # Bayesian Acquisition (EI) for stability
                # Target: Minimize Energy (Find stable phases)
                # Best so far? Hard to define globally, assume -0.5 eV/atom
                ei_val = expected_improvement(
                    c.energy_mean, 
                    c.energy_std, 
                    best_f=-0.5, # Target
                    maximize=False # Minimize energy
                ).item()
                
                # Hybrid Score: Topo (Exploitation) + Stability (Exploration/EI)
                c.acquisition_value = (
                    self.weights["topo"] * c.topo_probability + 
                    self.weights["stability"] * (ei_val * 10.0) + # Scale EI to be comparable
                    self.weights["heuristic"] * c.heuristic_topo_score +
                    self.weights["novelty"] * c.novelty_score
                )
            else:
                c.acquisition_value = (
                    self.weights["topo"] * c.topo_probability +
                    self.weights["stability"] * c.stability_score +
                    self.weights["heuristic"] * c.heuristic_topo_score +
                    self.weights["novelty"] * c.novelty_score
                )
        
        candidates.sort(key=lambda x: x.acquisition_value, reverse=True)
        
        # Register winners
        top = candidates[:n_top]
        for c in top:
            self.known_formulas.add(c.formula)
            self.top_candidates.append(c)
            
        return candidates

    def _save_iteration(self, iteration: int, top_candidates: List[Candidate]):
        data = {
            "iteration": iteration,
            "timestamp": time.time(),
            "n_candidates": len(top_candidates),
            "candidates": [c.to_dict() for c in top_candidates]
        }
        fpath = self.results_dir / f"iteration_{iteration:03d}.json"
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _print_iteration_summary(self, top: List[Candidate], dt: float):
        print(f"\n  Top candidates ({dt:.1f}s):")
        print(f"  {'Rank':>4} {'Formula':>12} {'Method':>10} {'Topo%':>6} {'Stab':>5} {'Acq':>5}")
        print(f"  {'----':>4} {'-------':>12} {'------':>10} {'-----':>6} {'----':>5} {'---':>5}")
        for i, c in enumerate(top[:5]):
            print(f"  {i+1:4d} {c.formula:>12s} {c.method:>10s} "
                  f"{c.topo_probability:6.2f} {c.stability_score:5.2f} {c.acquisition_value:5.2f}")
