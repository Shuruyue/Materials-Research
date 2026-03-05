"""Policy engine for discovery candidate scoring and selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atlas.active_learning.policy_state import ActiveLearningPolicyConfig, PolicyState


@dataclass
class PolicyEngine:
    """Decision-only engine (no I/O) for legacy and CMOEIC policies."""

    config: ActiveLearningPolicyConfig
    state: PolicyState

    @staticmethod
    def _coerce_positive_int(value: object, default: int = 1) -> int:
        if isinstance(value, bool):
            return max(1, int(default))
        if isinstance(value, (int, np.integer)):
            return max(1, int(value))
        if isinstance(value, (float, np.floating)):
            out_f = float(value)
            if not np.isfinite(out_f) or not out_f.is_integer():
                return max(1, int(default))
            return max(1, int(out_f))
        try:
            out = int(value)
        except Exception:
            return max(1, int(default))
        return max(1, out)

    def score_and_select(self, controller, candidates: list[object], n_top: int):
        n_top_i = self._coerce_positive_int(n_top, default=1)
        policy_name = str(getattr(self.config, "policy_name", "legacy")).strip().lower()
        if policy_name == "legacy":
            return controller._score_and_select_legacy(candidates, n_top_i)
        if policy_name == "cmoeic":
            return self._score_and_select_cmoeic(controller, candidates, n_top_i)
        raise ValueError(f"Unsupported policy: {policy_name}")

    @staticmethod
    def _safe_num(value: object, default: float = 0.0) -> float:
        try:
            out = float(value)
            if np.isfinite(out):
                return out
        except Exception:
            pass
        return float(default)

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _calibrated_energy_stats(self, candidate: object) -> tuple[float, float, float]:
        raw_mean = self._safe_num(
            getattr(candidate, "energy_mean", None),
            self._safe_num(getattr(candidate, "energy_per_atom", 0.0), 0.0),
        )
        raw_std = abs(self._safe_num(getattr(candidate, "energy_std", 0.0), 0.0))
        calibrated_std = raw_std * float(self.state.std_scale)
        conformal_radius = calibrated_std * float(self.state.conformal_scale)
        return float(raw_mean), float(calibrated_std), float(conformal_radius)

    def _base_utility(
        self,
        *,
        topo_prob: float,
        stability_term: float,
        novelty_score: float,
        estimated_cost: float,
    ) -> float:
        objective = max(0.0, stability_term)
        diversity = 1.0 + float(self.config.diversity_novelty_boost) * novelty_score
        denom = max(estimated_cost + float(self.config.cost_eps), float(self.config.cost_eps))
        return float((objective * diversity) / denom)

    def _estimate_cost(self, candidate: object, controller) -> float:
        struct = getattr(candidate, "relaxed_structure", None) or getattr(candidate, "structure", None)
        n_sites = 0
        try:
            n_sites = int(getattr(struct, "num_sites", 0))
        except Exception:
            n_sites = 0
        n_sites = max(1, n_sites)

        relax_cost = float(self.config.relax_cost_base) * (1.0 + (n_sites / 32.0))
        classify_cost = float(self.config.classify_cost_base) * (1.0 + (n_sites / 64.0))
        if getattr(controller, "relaxer", None) is None:
            relax_cost *= 0.35
        return max(self.config.cost_eps, relax_cost + classify_cost)

    def _risk_gate(self, risk_ood: float, risk_conf: float) -> bool:
        ood_hit = risk_ood >= float(self.config.ood_gate_threshold)
        conf_hit = risk_conf >= float(self.config.conformal_gate_threshold)
        if str(self.config.ood_combination).lower() == "or":
            return bool(ood_hit or conf_hit)
        return bool(ood_hit and conf_hit)

    def _score_and_select_cmoeic(self, controller, candidates: list[object], n_top: int):
        if not candidates:
            return []

        self.state.update_calibration(controller.all_candidates, cfg=self.config)

        topo_terms: list[float] = []
        stability_terms: list[float] = []
        synthesis_terms: list[float] = []
        objective_map: dict[int, np.ndarray] = {}

        # Stage 1: base objective + cost + calibration-aware uncertainty fields.
        for cand in candidates:
            candidate_structure = getattr(cand, "relaxed_structure", None) or getattr(cand, "structure", None)
            is_new = not controller._is_duplicate_structure(candidate_structure)
            cand.novelty_score = 1.0 if is_new else 0.0
            cand.reject_reason = ""

            topo_prob = self._clamp01(self._safe_num(getattr(cand, "topo_probability", 0.0), 0.0))
            stability_term = max(0.0, float(controller._stability_component(cand)))
            raw_mean, calibrated_std, conformal_radius = self._calibrated_energy_stats(cand)

            cand.calibrated_mean = raw_mean
            cand.calibrated_std = calibrated_std
            cand.conformal_radius = conformal_radius

            estimated_cost = self._estimate_cost(cand, controller) if bool(self.config.cost_aware) else 1.0
            cand.estimated_cost = estimated_cost
            cand.acquisition_value = self._base_utility(
                topo_prob=topo_prob,
                stability_term=stability_term,
                novelty_score=float(cand.novelty_score),
                estimated_cost=estimated_cost,
            )

            topo_terms.append(topo_prob)
            stability_terms.append(stability_term)
            synthesis_terms.append(0.0)
            cand.synthesis_score = 0.0
            cand.synthesis_feasibility = 0.0

        synthesis_scores = controller._apply_synthesis_objective(candidates)
        use_synthesis = bool(getattr(controller, "use_synthesis_objective", False))
        for idx, cand in enumerate(candidates):
            synth = float(np.clip(self._safe_num(synthesis_scores[idx] if idx < len(synthesis_scores) else 0.0, 0.0), 0.0, 1.0))
            synth = synth * float(np.clip(self._safe_num(getattr(cand, "topo_probability", 0.0), 0.0), 0.0, 1.0))
            synthesis_terms[idx] = synth

            # Feasibility mixes topo and synthesis (when available).
            topo_prob = topo_terms[idx]
            feasibility = topo_prob * (synth if use_synthesis else 1.0)
            base = max(0.0, float(cand.acquisition_value))
            cand.acquisition_value = float(base * max(1e-6, feasibility))

            if use_synthesis:
                objective_map[id(cand)] = np.array([topo_terms[idx], stability_terms[idx], synthesis_terms[idx]], dtype=float)
            else:
                objective_map[id(cand)] = np.array([topo_terms[idx], stability_terms[idx]], dtype=float)

        # OOD risk in the currently selected controller space.
        ood_scores = controller._estimate_ood_scores(candidates)
        max_conformal_radius = max(float(self.config.max_conformal_radius), float(self.config.cost_eps), 1e-9)
        for idx, cand in enumerate(candidates):
            risk_ood = self._clamp01(self._safe_num(ood_scores[idx] if idx < len(ood_scores) else 0.0, 0.0))
            conf_ratio = self._clamp01(float(getattr(cand, "conformal_radius", 0.0)) / max_conformal_radius)
            risk_penalty = (
                float(self.config.ood_penalty_weight) * risk_ood
                + float(self.config.conformal_penalty_weight) * conf_ratio
            )
            risk_score = self._clamp01(risk_penalty)
            cand.ood_score = risk_ood
            cand.risk_score = risk_score

            gate_hit = self._risk_gate(risk_ood, conf_ratio)
            mode = str(self.config.risk_mode).lower()
            if mode == "hard" and gate_hit:
                cand.reject_reason = "risk_gate"
                cand.acquisition_value = -1e9
            elif mode == "hybrid" and gate_hit:
                cand.reject_reason = "risk_gate"
                cand.acquisition_value = float(cand.acquisition_value) * 0.5 - risk_penalty
            else:
                cand.acquisition_value = float(cand.acquisition_value) - risk_penalty

            cost = max(float(getattr(cand, "estimated_cost", 1.0)), float(self.config.cost_eps))
            cand.gain_per_cost = float(cand.acquisition_value) / cost

        # Reuse controller Pareto/HV/diversity machinery on calibrated scores.
        controller._apply_pareto_rank_bonus(
            candidates,
            topo_terms,
            stability_terms,
            synthesis_terms if use_synthesis else None,
        )
        controller._apply_pareto_hv_bonus(
            candidates,
            topo_terms,
            stability_terms,
            synthesis_terms if use_synthesis else None,
        )

        candidates.sort(key=lambda x: float(getattr(x, "acquisition_value", 0.0)), reverse=True)
        top = controller._select_top_diverse(candidates, n_top, objective_map=objective_map)
        ranked = controller._finalize_ranked_candidates(candidates, top)

        self.state.iteration = int(getattr(controller, "iteration", self.state.iteration))
        return ranked
