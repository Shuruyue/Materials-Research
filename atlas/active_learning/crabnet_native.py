from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Preferred path used by older assimilated snapshots.
    from atlas.third_party.crabnet.crabnet_ import CrabNet as OriginalCrabNet
except Exception:
    try:
        # Fallback to current in-repo CrabNet implementation.
        from atlas.third_party.crabnet.kingcrab import CrabNet as OriginalCrabNet
    except Exception as e:  # pragma: no cover - optional runtime dependency
        logging.warning(f"Could not import assimilated CrabNet: {e}")
        OriginalCrabNet = None

from atlas.utils.registry import MODELS

logger = logging.getLogger(__name__)

_INT_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_SIMPLEX_TRANSFORMS = {"none", "escort", "clr_softmax"}
_UQ_HEAD_MODES = {"auto", "log_std", "softplus_std", "log_var", "deterministic"}


@MODELS.register("crabnet_native")
class NativeCrabnetScreener(nn.Module):
    """
    Native CrabNet wrapper with simplex-aware fraction preprocessing and UQ helpers.

    Algorithmic upgrades:
    1) Simplex transforms for composition vectors:
       - Aitchison power transform ("escort")
       - CLR + softmax smoothing ("clr_softmax")
    2) Explicit uncertainty-head semantics (log-std / log-var / softplus-std).
    3) Optional ensemble + MC-dropout epistemic decomposition.
    4) Global and grouped conformal temperature calibration.

    References:
    - Madani et al. (2021), CrabNet: https://arxiv.org/abs/2109.08203
    - Aitchison (1982), compositional geometry:
      https://doi.org/10.1111/j.2517-6161.1982.tb01195.x
    - Egozcue et al. (2003), ILR transform:
      https://doi.org/10.1023/A:1023818214614
    - Gal & Ghahramani (2016), MC Dropout:
      https://arxiv.org/abs/1612.01474
    - Kendall & Gal (2017), aleatoric/epistemic decomposition:
      https://arxiv.org/abs/1703.04977
    - Lakshminarayanan et al. (2017), Deep Ensembles:
      https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa85bce38a7-Abstract.html
    - Romano et al. (2019), conformalized uncertainty ideas:
      https://arxiv.org/abs/1905.03222
    """

    def __init__(
        self,
        compute_device: str = "cpu",
        d_model: int = 512,
        heads: int = 4,
        d_ffn: int = 2048,
        N: int = 3,
        pe_resolution: int = 5000,
        f_prop: str = "num_atoms",
        out_dims: int = 3,
        simplex_transform: str = "escort",
        simplex_blend: float = 0.25,
        escort_power: float = 0.85,
        clr_temperature: float = 1.0,
        uncertainty_min_std: float = 1e-6,
        uncertainty_temperature: float = 1.0,
        uncertainty_head_mode: str = "auto",
        mean_dims: int = 1,
        return_distribution: bool = True,
        ensemble_size: int = 1,
        mc_dropout_samples: int = 0,
        grouped_calibration: bool = False,
        grouped_calibration_default: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        if OriginalCrabNet is None:
            raise RuntimeError("Assimilated CrabNet source is missing from atlas/third_party/crabnet.")

        simplex_key = str(simplex_transform).strip().lower()
        if simplex_key not in _SIMPLEX_TRANSFORMS:
            supported = ", ".join(sorted(_SIMPLEX_TRANSFORMS))
            raise ValueError(f"Unknown simplex_transform: {simplex_transform!r}. Supported: {supported}")

        self.simplex_transform = simplex_key
        self.simplex_blend = float(min(max(simplex_blend, 0.0), 1.0))
        self.escort_power = float(max(escort_power, 1e-6))
        self.clr_temperature = float(max(clr_temperature, 1e-6))
        self.uncertainty_min_std = float(max(uncertainty_min_std, 0.0))
        head_mode = str(uncertainty_head_mode).strip().lower()
        if head_mode not in _UQ_HEAD_MODES:
            supported = ", ".join(sorted(_UQ_HEAD_MODES))
            raise ValueError(f"Unknown uncertainty_head_mode: {uncertainty_head_mode!r}. Supported: {supported}")
        self.uncertainty_head_mode = head_mode
        self.mean_dims = int(max(mean_dims, 1))
        self.register_buffer(
            "uncertainty_temperature",
            torch.tensor(float(max(uncertainty_temperature, 1e-6)), dtype=torch.float32),
        )
        self.return_distribution = bool(return_distribution)
        self.ensemble_size = int(max(ensemble_size, 1))
        self.mc_dropout_samples = int(max(mc_dropout_samples, 0))
        self.grouped_calibration = bool(grouped_calibration)
        self.grouped_calibration_default = float(max(grouped_calibration_default, 1e-6))
        self.group_temperature_table: dict[int, float] = {}
        self._member_init_kwargs = dict(
            compute_device=compute_device,
            out_dims=out_dims,
            d_model=d_model,
            N=N,
            heads=heads,
            d_ffn=d_ffn,
            pe_resolution=pe_resolution,
            f_prop=f_prop,
            **kwargs,
        )

        logger.info(
            (
                "Instantiating native CrabNet: d_model=%s, N=%s, out_dims=%s, "
                "simplex_transform=%s, simplex_blend=%.3f, uq_head=%s, ensemble=%s"
            ),
            d_model,
            N,
            out_dims,
            self.simplex_transform,
            self.simplex_blend,
            self.uncertainty_head_mode,
            self.ensemble_size,
        )

        # Native instantiation from upstream architecture.
        self.engine = self._build_engine()
        self._ensemble_members = nn.ModuleList([self.engine])
        for _ in range(1, self.ensemble_size):
            self._ensemble_members.append(copy.deepcopy(self.engine))

    def _build_engine(self):
        return OriginalCrabNet(**self._member_init_kwargs)

    @property
    def ensemble_members(self) -> nn.ModuleList:
        return self._ensemble_members

    @staticmethod
    def _is_integer_tensor(x: torch.Tensor) -> bool:
        return x.dtype in _INT_DTYPES

    def _normalize_input_order(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve argument order robustly.

        Historical wrappers used both signatures:
        - forward(src, frac)
        - forward(frac, src)
        """
        if self._is_integer_tensor(a) and not self._is_integer_tensor(b):
            src = a.to(dtype=torch.long)
            frac = b.to(dtype=torch.float32)
            return src, frac
        if self._is_integer_tensor(b) and not self._is_integer_tensor(a):
            src = b.to(dtype=torch.long)
            frac = a.to(dtype=torch.float32)
            return src, frac

        # Heuristic fallback for ambiguous dtypes.
        if torch.max(a).item() > 1.5 and torch.max(b).item() <= 1.5:
            return a.to(dtype=torch.long), b.to(dtype=torch.float32)
        if torch.max(b).item() > 1.5 and torch.max(a).item() <= 1.5:
            return b.to(dtype=torch.long), a.to(dtype=torch.float32)
        return a.to(dtype=torch.long), b.to(dtype=torch.float32)

    @staticmethod
    def _renormalize_simplex(frac: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        frac = torch.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
        frac = torch.clamp(frac, min=0.0)
        frac = frac * mask.to(dtype=frac.dtype)
        mass = frac.sum(dim=1, keepdim=True)
        safe_mass = torch.clamp(mass, min=eps)
        frac = frac / safe_mass
        # If row has no active elements, keep row as zeros.
        has_mass = (mass > eps).to(dtype=frac.dtype)
        return frac * has_mass

    def _sanitize_fractions(self, src: torch.Tensor, frac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src = src.to(dtype=torch.long)
        frac = frac.to(dtype=torch.float32)
        mask = src != 0
        frac = self._renormalize_simplex(frac, mask)
        return src, frac

    def _apply_simplex_transform(self, frac: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.simplex_transform == "none" or self.simplex_blend <= 0.0:
            return frac

        base = frac
        transformed = base

        if self.simplex_transform == "escort":
            # Aitchison power transform on the simplex (aka escort family).
            # p_i -> p_i^q / sum_j p_j^q, q in (0, +inf)
            transformed = torch.pow(torch.clamp(base, min=1e-12), self.escort_power)
            transformed = self._renormalize_simplex(transformed, mask)

        elif self.simplex_transform == "clr_softmax":
            # CLR-inspired smoothing:
            # clr(p)_i = log p_i - mean_j log p_j, then map back via softmax.
            safe = torch.clamp(base, min=1e-12)
            logp = torch.log(safe)
            m = mask.to(dtype=base.dtype)
            denom = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
            clr_mean = (logp * m).sum(dim=1, keepdim=True) / denom
            clr = (logp - clr_mean).masked_fill(~mask, float("-inf"))
            transformed = torch.softmax(clr / self.clr_temperature, dim=1)
            transformed = torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
            transformed = self._renormalize_simplex(transformed, mask)

        blended = (1.0 - self.simplex_blend) * base + self.simplex_blend * transformed
        return self._renormalize_simplex(blended, mask)

    @torch.no_grad()
    def optimize_escort_power(
        self,
        frac: torch.Tensor,
        src: torch.Tensor | None = None,
        *,
        target_entropy_ratio: float = 0.75,
        q_min: float = 0.2,
        q_max: float = 2.0,
        q_steps: int = 25,
    ) -> float:
        """
        Tune escort power by matching mean normalized entropy on the simplex.

        This provides a data-driven q for the Aitchison power transform family.
        """
        target = float(min(max(target_entropy_ratio, 1e-3), 0.999))
        q_lo = float(max(q_min, 1e-3))
        q_hi = float(max(q_max, q_lo + 1e-3))
        steps = int(max(q_steps, 3))

        if src is None:
            mask = frac > 0
        else:
            mask = src != 0
        base = self._renormalize_simplex(frac.to(dtype=torch.float32), mask)

        def _entropy_ratio(x: torch.Tensor) -> float:
            safe = torch.clamp(x, min=1e-12)
            h = -(safe * torch.log(safe)).sum(dim=1)
            n_active = torch.clamp(mask.sum(dim=1).to(dtype=torch.float32), min=1.0)
            h_max = torch.log(torch.clamp(n_active, min=1.0))
            ratio = h / torch.clamp(h_max, min=1e-12)
            ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
            return float(ratio.mean().item())

        q_grid = torch.linspace(q_lo, q_hi, steps)
        best_q = self.escort_power
        best_err = float("inf")
        for q in q_grid:
            transformed = torch.pow(torch.clamp(base, min=1e-12), float(q.item()))
            transformed = self._renormalize_simplex(transformed, mask)
            ratio = _entropy_ratio(transformed)
            err = abs(ratio - target)
            if err < best_err:
                best_err = err
                best_q = float(q.item())

        self.escort_power = float(max(best_q, 1e-6))
        return self.escort_power

    def _extract_mean_std(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        if output.dim() == 1:
            output = output.unsqueeze(-1)
        feature_dim = int(output.shape[-1])

        mode = self.uncertainty_head_mode
        if mode == "auto":
            mode = "deterministic" if feature_dim < 2 else "log_std"

        if mode == "deterministic":
            return None

        m = min(self.mean_dims, feature_dim // 2 if feature_dim >= 2 else 1)
        if feature_dim < 2 * m:
            return None

        mean_raw = output[..., :m]
        scale_raw = output[..., m : 2 * m]
        if mode == "log_std":
            std = torch.exp(scale_raw)
        elif mode == "log_var":
            std = torch.exp(0.5 * scale_raw)
        elif mode == "softplus_std":
            std = F.softplus(scale_raw)
        else:
            raise ValueError(f"Unsupported uncertainty head mode: {mode}")
        std = torch.clamp(std, min=self.uncertainty_min_std)
        return mean_raw, std

    def _group_ids_from_src(self, src: torch.Tensor) -> torch.Tensor:
        return (src != 0).sum(dim=1).to(dtype=torch.long)

    def _apply_uncertainty_calibration(self, std: torch.Tensor, src: torch.Tensor | None = None) -> torch.Tensor:
        temp = self.uncertainty_temperature.to(device=std.device, dtype=std.dtype)
        out = std * temp
        if src is None or not self.grouped_calibration:
            return out

        group_ids = self._group_ids_from_src(src)
        group_scale = torch.full(
            (group_ids.shape[0], 1),
            float(self.grouped_calibration_default),
            dtype=std.dtype,
            device=std.device,
        )
        for g, scale in self.group_temperature_table.items():
            group_scale[group_ids == int(g)] = float(scale)
        return out * group_scale

    @staticmethod
    def _set_dropout_train(module: nn.Module) -> None:
        for child in module.modules():
            if isinstance(child, nn.Dropout):
                child.train()

    def _forward_member_once(
        self,
        member: nn.Module,
        src: torch.Tensor,
        frac: torch.Tensor,
    ):
        output = member(src, frac)
        if not torch.is_tensor(output):
            return output

        parsed = self._extract_mean_std(output)
        if parsed is None:
            return output

        mean, std = parsed
        std = self._apply_uncertainty_calibration(std, src=src)
        return mean, std, output

    def _forward_member_distribution(
        self,
        member: nn.Module,
        src: torch.Tensor,
        frac: torch.Tensor,
        mc_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns per-member (mean, aleatoric_var, epistemic_var).
        """
        if mc_samples < 2:
            out = self._forward_member_once(member, src, frac)
            if torch.is_tensor(out):
                mean = out if out.dim() > 1 else out.unsqueeze(-1)
                zero = torch.zeros_like(mean)
                return mean, zero, zero
            mean, std, _ = out
            return mean, std * std, torch.zeros_like(std)

        was_training = member.training
        member.eval()
        self._set_dropout_train(member)

        means = []
        vars_alea = []
        with torch.no_grad():
            for _ in range(mc_samples):
                out = member(src, frac)
                if not torch.is_tensor(out):
                    raise TypeError("MC-dropout path requires tensor outputs.")
                parsed = self._extract_mean_std(out)
                if parsed is None:
                    mean = out if out.dim() > 1 else out.unsqueeze(-1)
                    std = torch.zeros_like(mean)
                else:
                    mean, std = parsed
                    std = self._apply_uncertainty_calibration(std, src=src)
                means.append(mean)
                vars_alea.append(std * std)

        if was_training:
            member.train()
        else:
            member.eval()

        mean_stack = torch.stack(means, dim=0)
        alea_stack = torch.stack(vars_alea, dim=0)
        mean = mean_stack.mean(dim=0)
        epistemic_var = mean_stack.var(dim=0, unbiased=False)
        aleatoric_var = alea_stack.mean(dim=0)
        return mean, aleatoric_var, epistemic_var

    def _forward_distribution(
        self,
        src: torch.Tensor,
        frac: torch.Tensor,
        mc_samples: int,
    ) -> dict[str, torch.Tensor]:
        member_means = []
        member_alea = []
        member_epi = []

        for member in self.ensemble_members:
            mean, alea_var, epi_var = self._forward_member_distribution(member, src, frac, mc_samples)
            member_means.append(mean)
            member_alea.append(alea_var)
            member_epi.append(epi_var)

        mean_stack = torch.stack(member_means, dim=0)
        alea_stack = torch.stack(member_alea, dim=0)
        epi_stack = torch.stack(member_epi, dim=0)

        mean = mean_stack.mean(dim=0)
        inter_member_epi = mean_stack.var(dim=0, unbiased=False)
        aleatoric_var = alea_stack.mean(dim=0)
        epistemic_var = epi_stack.mean(dim=0) + inter_member_epi
        total_std = torch.sqrt(torch.clamp(aleatoric_var + epistemic_var, min=0.0))

        return {
            "mean": mean,
            "mu": mean,
            "aleatoric_std": torch.sqrt(torch.clamp(aleatoric_var, min=0.0)),
            "epistemic_std": torch.sqrt(torch.clamp(epistemic_var, min=0.0)),
            "total_std": total_std,
            "std": total_std,
            "sigma": total_std,
            "ensemble_size": torch.tensor(len(self.ensemble_members), dtype=torch.int64, device=mean.device),
            "mc_samples": torch.tensor(int(mc_samples), dtype=torch.int64, device=mean.device),
        }

    @torch.no_grad()
    def calibrate_uncertainty_temperature(
        self,
        y_true: torch.Tensor,
        y_pred_mean: torch.Tensor,
        y_pred_std: torch.Tensor,
        quantile: float = 0.9,
    ) -> float:
        """
        Robust scalar calibration for predictive std.

        Uses a conformal-style scaling statistic:
        t = Quantile_q( |y - mu| / sigma )
        sigma_cal = t * sigma
        """
        q = float(min(max(quantile, 1e-3), 0.999))
        y_true = y_true.reshape(-1).to(dtype=torch.float32)
        y_pred_mean = y_pred_mean.reshape(-1).to(dtype=torch.float32)
        y_pred_std = torch.clamp(y_pred_std.reshape(-1).to(dtype=torch.float32), min=1e-9)
        score = torch.abs(y_true - y_pred_mean) / y_pred_std
        t = float(torch.quantile(score, q).item())
        t = max(t, 1e-6)
        self.uncertainty_temperature.data = torch.tensor(t, dtype=self.uncertainty_temperature.dtype)
        return t

    @torch.no_grad()
    def calibrate_uncertainty_temperature_grouped(
        self,
        y_true: torch.Tensor,
        y_pred_mean: torch.Tensor,
        y_pred_std: torch.Tensor,
        src: torch.Tensor,
        *,
        quantile: float = 0.9,
        min_group_size: int = 16,
    ) -> dict[int, float]:
        """
        Grouped conformal calibration by composition complexity (number of elements).
        """
        q = float(min(max(quantile, 1e-3), 0.999))
        group_ids = self._group_ids_from_src(src).reshape(-1)
        y_true = y_true.reshape(-1).to(dtype=torch.float32)
        y_pred_mean = y_pred_mean.reshape(-1).to(dtype=torch.float32)
        y_pred_std = torch.clamp(y_pred_std.reshape(-1).to(dtype=torch.float32), min=1e-9)
        score = torch.abs(y_true - y_pred_mean) / y_pred_std

        table: dict[int, float] = {}
        unique_groups = torch.unique(group_ids)
        for g in unique_groups:
            mask = group_ids == g
            if int(mask.sum().item()) < int(min_group_size):
                continue
            t = float(torch.quantile(score[mask], q).item())
            table[int(g.item())] = max(t, 1e-6)

        if table:
            self.group_temperature_table = table
            self.grouped_calibration = True
        return dict(self.group_temperature_table)

    def predict_distribution(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        mc_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        src, frac = self._normalize_input_order(a, b)
        src, frac = self._sanitize_fractions(src, frac)
        mask = src != 0
        frac = self._apply_simplex_transform(frac, mask)
        samples = self.mc_dropout_samples if mc_samples is None else int(max(mc_samples, 0))
        return self._forward_distribution(src, frac, samples)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        """
        Forward pass with robust input-order handling.

        Supports both:
        - forward(src, frac)
        - forward(frac, src)
        """
        src, frac = self._normalize_input_order(a, b)
        src, frac = self._sanitize_fractions(src, frac)
        mask = src != 0
        frac = self._apply_simplex_transform(frac, mask)

        if self.return_distribution:
            return self._forward_distribution(src, frac, mc_samples=0)

        out = self._forward_member_once(self.engine, src, frac)
        if torch.is_tensor(out):
            return out
        mean, _, _ = out
        return mean
