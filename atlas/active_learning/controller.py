"""
Active Learning Discovery Controller

The brain of ATLAS — orchestrates the closed-loop discovery cycle:
    Generate -> Relax -> Classify -> Score -> Select -> Loop

Optimization:
- Improved error handling (logging instead of silent failures)
- Better type hinting
- Robust candidate validation
"""

import json
import time
import numpy as np
import logging
import torch
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from atlas.config import get_config
from atlas.active_learning.generator import StructureGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A material candidate with all computed properties."""
    structure: object                    # pymatgen Structure
    formula: str = ""
    method: str = ""                     # how it was generated
    parent: str = ""                     # parent material
    mutations: str = ""                  # what mutations were applied

    # Scores
    topo_probability: float = 0.0        # from GNN classifier
    stability_score: float = 0.0         # from MACE relaxation
    heuristic_topo_score: float = 0.0    # from domain knowledge
    novelty_score: float = 0.0           # distance from known materials
    acquisition_value: float = 0.0       # combined score for selection

    # Computed properties
    energy_per_atom: Optional[float] = None
    relaxed_structure: object = None
    converged: bool = False

    # Metadata
    iteration: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "formula": self.formula,
            "method": self.method,
            "parent": self.parent,
            "mutations": self.mutations,
            "topo_probability": self.topo_probability,
            "stability_score": self.stability_score,
            "heuristic_topo_score": self.heuristic_topo_score,
            "novelty_score": self.novelty_score,
            "acquisition_value": self.acquisition_value,
            "energy_per_atom": self.energy_per_atom,
            "converged": self.converged,
            "iteration": self.iteration,
        }


class DiscoveryController:
    """
    Active learning controller for topological material discovery.
    """

    def __init__(
        self,
        generator: StructureGenerator,
        relaxer=None,
        classifier_model=None,
        graph_builder=None,
        # Acquisition function weights
        w_topo: float = 0.4,
        w_stability: float = 0.3,
        w_heuristic: float = 0.15,
        w_novelty: float = 0.15,
    ):
        self.generator = generator
        self.relaxer = relaxer
        self.classifier = classifier_model
        self.graph_builder = graph_builder

        self.w_topo = w_topo
        self.w_stability = w_stability
        self.w_heuristic = w_heuristic
        self.w_novelty = w_novelty

        self.cfg = get_config()
        self.results_dir = self.cfg.paths.data_dir / "discovery_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Discovery state
        self.all_candidates: List[Candidate] = []
        self.top_candidates: List[Candidate] = []
        self.iteration = 0
        self.known_formulas: set[str] = set()

    def run_discovery_loop(
        self,
        n_iterations: int = 10,
        n_candidates_per_iter: int = 50,
        n_select_top: int = 10,
    ) -> List[Candidate]:
        """
        Run the full discovery loop.
        """
        print("\n" + "=" * 70)
        print("  ATLAS Discovery Engine — Active Learning Loop")
        print("=" * 70)
        print(f"  Iterations:          {n_iterations}")
        print(f"  Candidates/iter:     {n_candidates_per_iter}")

        for it in range(1, n_iterations + 1):
            self.iteration = it
            t0 = time.time()

            print(f"\n{'─' * 50}")
            print(f"  Iteration {it}/{n_iterations}")
            print(f"{'─' * 50}")

            # Step 1: Generate
            print(f"  [1/4] Generating {n_candidates_per_iter} candidates...")
            try:
                raw_candidates = self.generator.generate_batch(n_candidates_per_iter)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raw_candidates = []
                
            print(f"        Generated {len(raw_candidates)} valid structures")
            
            if not raw_candidates:
                logger.warning("No candidates generated! Stopping early.")
                break

            # Step 2: Relax
            print(f"  [2/4] Relaxing with MACE potential...")
            candidates = self._relax_candidates(raw_candidates)
            print(f"        Relaxed {len(candidates)} structures")

            # Step 3: Classify
            print(f"  [3/4] Classifying topological properties...")
            candidates = self._classify_candidates(candidates)

            # Step 4: Score & Select
            print(f"  [4/4] Computing acquisition function & selecting top...")
            candidates = self._score_and_select(candidates, n_select_top)

            dt = time.time() - t0
            self._print_iteration_summary(candidates[:n_select_top], dt)

            # Save results
            self._save_iteration(it, candidates[:n_select_top])

            # Feed back top candidates as new seeds
            top_structs = [
                c.relaxed_structure or c.structure 
                for c in candidates[:n_select_top]
                if c.acquisition_value > 0.3
            ]
            if top_structs:
                self.generator.add_seeds(top_structs)

        self._print_final_summary()
        return self.top_candidates

    def _relax_candidates(self, raw_candidates: list[dict]) -> List[Candidate]:
        """Relax candidate structures with MACE."""
        candidates = []

        for rc in raw_candidates:
            struct = rc["structure"]

            if self.relaxer is not None:
                try:
                    result = self.relaxer.relax_structure(struct, steps=100)
                    stability = self.relaxer.score_stability(struct)
                except Exception as e:
                    logger.warning(f"Relaxation failed for {rc.get('formula', 'unknown')}: {e}")
                    result = {"relaxed_structure": struct, "energy_per_atom": None, "converged": False}
                    stability = 0.0
            else:
                result = {
                    "relaxed_structure": struct,
                    "energy_per_atom": None,
                    "converged": False,
                }
                stability = 0.5  # Neutral stability if no relaxer

            cand = Candidate(
                structure=struct,
                formula=struct.composition.reduced_formula,
                method=rc.get("method", "unknown"),
                parent=rc.get("parent", ""),
                mutations=rc.get("mutations", ""),
                heuristic_topo_score=rc.get("topo_score", 0.0),
                stability_score=stability,
                energy_per_atom=result.get("energy_per_atom"),
                relaxed_structure=result.get("relaxed_structure"),
                converged=result.get("converged", False),
                iteration=self.iteration,
                timestamp=time.time(),
            )

            candidates.append(cand)

        return candidates

    def _classify_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Run topological GNN classifier on candidates."""
        if self.classifier is None or self.graph_builder is None:
            # No classifier — use heuristic score as proxy
            for c in candidates:
                c.topo_probability = c.heuristic_topo_score
            return candidates

        self.classifier.eval()
        device = next(self.classifier.parameters()).device

        valid_candidates = []
        for c in candidates:
            try:
                struct = c.relaxed_structure or c.structure
                graph = self.graph_builder.structure_to_graph(struct)

                with torch.no_grad():
                    # Handle both simple graph and batch inputs
                    node_feats = graph["node_features"].to(device)
                    if node_feats.dim() == 2:
                        node_feats = node_feats.unsqueeze(0)
                        
                    prob = self.classifier.predict_proba(
                        node_feats,
                        graph["edge_index"].to(device),
                        graph["edge_features"].to(device),
                    )
                    c.topo_probability = prob.item()
                    valid_candidates.append(c)
            except Exception as e:
                logger.warning(f"Classification failed for {c.formula}: {e}")
                c.topo_probability = c.heuristic_topo_score
                valid_candidates.append(c)

        return valid_candidates

    def _score_and_select(
        self, candidates: List[Candidate], n_top: int
    ) -> List[Candidate]:
        """Compute acquisition function and rank candidates."""
        for c in candidates:
            # Novelty: is this formula new?
            c.novelty_score = 1.0 if c.formula not in self.known_formulas else 0.2

            # Multi-objective acquisition function
            c.acquisition_value = (
                self.w_topo * c.topo_probability
                + self.w_stability * c.stability_score
                + self.w_heuristic * c.heuristic_topo_score
                + self.w_novelty * c.novelty_score
            )

        # Sort by acquisition value
        candidates.sort(key=lambda x: x.acquisition_value, reverse=True)

        # Track top candidates globally
        for c in candidates[:n_top]:
            self.known_formulas.add(c.formula)
            self.all_candidates.append(c)

            self.top_candidates.append(c)
            self.top_candidates.sort(
                key=lambda x: x.acquisition_value, reverse=True
            )
            self.top_candidates = self.top_candidates[:50]  # keep top 50 global

        return candidates

    def _print_iteration_summary(self, top: List[Candidate], dt: float):
        """Print summary of one iteration."""
        print(f"\n  Top candidates this iteration ({dt:.1f}s):")
        print(f"  {'Rank':>4} {'Formula':>15} {'Method':>10} "
              f"{'P(topo)':>8} {'Stab':>6} {'Acq':>6}")
        print(f"  {'----':>4} {'-------':>15} {'------':>10} "
              f"{'-------':>8} {'----':>6} {'---':>6}")
        for i, c in enumerate(top[:10]):
            print(f"  {i+1:4d} {c.formula:>15s} {c.method:>10s} "
                  f"{c.topo_probability:8.3f} {c.stability_score:6.3f} "
                  f"{c.acquisition_value:6.3f}")

    def _print_final_summary(self):
        """Print final discovery summary."""
        print(f"\n{'=' * 70}")
        print(f"  ATLAS Discovery — Final Summary")
        print(f"{'=' * 70}")
        print(f"  Total iterations:     {self.iteration}")
        print(f"  Total candidates:     {len(self.all_candidates)}")
        print(f"  Unique formulas:      {len(self.known_formulas)}")
        print(f"  Top candidates:       {len(self.top_candidates)}")

    def _save_iteration(self, iteration: int, top_candidates: List[Candidate]):
        """Save iteration results to JSON."""
        results = {
            "iteration": iteration,
            "timestamp": time.time(),
            "n_candidates": len(top_candidates),
            "candidates": [c.to_dict() for c in top_candidates],
        }

        out_file = self.results_dir / f"iteration_{iteration:03d}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def save_final_report(self, filename: str = "discovery_report.json"):
        """Save the final discovery report."""
        report = {
            "total_iterations": self.iteration,
            "total_candidates_evaluated": len(self.all_candidates),
            "unique_formulas": len(self.known_formulas),
            "top_candidates": [c.to_dict() for c in self.top_candidates],
        }
        out_file = self.results_dir / filename
        with open(out_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to: {out_file}")
