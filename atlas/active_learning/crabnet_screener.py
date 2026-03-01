"""CrabNet-inspired composition screener with simplex-aware algorithms.

Algorithmic upgrades (math-first):
1) Continuous fractional Fourier encoding (no integer bin quantization).
2) Simplex-aware composition transforms:
   - Aitchison power / escort transform
   - CLR-softmax transform
   - ILR-softmax transform (Helmert orthonormal basis)
3) Fraction-weighted barycentric pooling (set-style aggregation).
4) Optional uncertainty head + ensemble/MC-dropout epistemic decomposition.
5) Training objective helper with Gaussian NLL for UQ-consistent fitting.

References:
- CrabNet, Wang et al. (2021): https://www.nature.com/articles/s41524-021-00545-1
- Aitchison simplex geometry (1982): https://doi.org/10.1111/j.2517-6161.1982.tb01195.x
- ILR / log-ratio analysis (Egozcue et al., 2003): https://doi.org/10.1023/A:1023818214614
- Deep Sets invariance (Zaheer et al., 2017): https://arxiv.org/abs/1703.06114
- Aleatoric uncertainty in deep models (Kendall & Gal, 2017): https://arxiv.org/abs/1703.04977
- Deep Ensembles (Lakshminarayanan et al., 2017): https://arxiv.org/abs/1612.01474
- MC Dropout as Bayesian approximation (Gal & Ghahramani, 2015): https://arxiv.org/abs/1506.02142
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from atlas.utils.registry import MODELS

_SIMPLEX_TRANSFORMS = {"none", "escort", "clr_softmax", "ilr_softmax"}
_UQ_MODES = {"none", "log_std", "log_var", "softplus_std"}


class ResidualNetwork(nn.Module):
    """Feed-forward residual network."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layer_dims: list[int]):
        super().__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False) if (dims[i] != dims[i + 1]) else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea: torch.Tensor) -> torch.Tensor:
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)


class ContinuousFractionalEncoder(nn.Module):
    """Continuous sinusoidal encoder for composition fractions.

    The original integer-indexed encoder discretizes x via round(x * resolution),
    which loses smoothness. This module computes sinusoidal phases directly from
    continuous fractions for better local sensitivity.
    """

    def __init__(self, d_model: int, resolution: int = 5000, log10: bool = False, base: float = 50.0):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = int(max(resolution, 2))
        self.log10 = bool(log10)
        self.base = float(max(base, 1.0001))

        idx = torch.arange(self.d_model, dtype=torch.float32)
        scale = torch.pow(torch.tensor(self.base, dtype=torch.float32), 2 * idx / max(self.d_model, 1))
        self.register_buffer("scale", scale)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        if self.log10:
            safe = torch.clamp(x, min=1.0 / self.resolution)
            x = 0.0025 * torch.square(torch.log2(safe))
            x = torch.clamp(x, max=1.0)
        return torch.clamp(x, min=1.0 / self.resolution, max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        phase_base = x * float(self.resolution - 1)
        phase = phase_base.unsqueeze(-1) / self.scale

        out = torch.zeros(*x.shape, self.d_model, device=x.device, dtype=x.dtype)
        out[..., 0::2] = torch.sin(phase[..., 0::2])
        out[..., 1::2] = torch.cos(phase[..., 1::2])
        return out


@MODELS.register("crabnet_screener")
class CompositionScreener(nn.Module):
    """Simplex-aware CrabNet-style composition screener.

    Notes:
    - `src`: [B, T] atomic numbers, 0 is padding token.
    - `frac`: [B, T] stoichiometric fractions on the simplex.
    - Default behavior remains point prediction for backward compatibility.
    """

    def __init__(
        self,
        out_dims: int = 1,
        d_model: int = 512,
        N: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        simplex_transform: str = "escort",
        simplex_blend: float = 0.25,
        escort_power: float = 0.85,
        clr_temperature: float = 1.0,
        ilr_temperature: float = 1.0,
        uncertainty_head_mode: str = "none",
        uncertainty_min_std: float = 1e-6,
        return_distribution: bool = False,
        ensemble_size: int = 1,
        mc_dropout_samples: int = 0,
    ):
        super().__init__()
        self.out_dims = int(out_dims)
        self.d_model = int(d_model)
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model must be even for split fractional encoders, got {self.d_model}")
        self.return_distribution = bool(return_distribution)
        self.ensemble_size = int(max(ensemble_size, 1))
        self.mc_dropout_samples = int(max(mc_dropout_samples, 0))
        self._helmert_cache: dict[int, torch.Tensor] = {}

        transform_key = str(simplex_transform).strip().lower()
        if transform_key not in _SIMPLEX_TRANSFORMS:
            supported = ", ".join(sorted(_SIMPLEX_TRANSFORMS))
            raise ValueError(f"Unknown simplex_transform: {simplex_transform!r}. Supported: {supported}")
        self.simplex_transform = transform_key
        self.simplex_blend = float(min(max(simplex_blend, 0.0), 1.0))
        self.escort_power = float(max(escort_power, 1e-6))
        self.clr_temperature = float(max(clr_temperature, 1e-6))
        self.ilr_temperature = float(max(ilr_temperature, 1e-6))

        uq_key = str(uncertainty_head_mode).strip().lower()
        if uq_key not in _UQ_MODES:
            supported = ", ".join(sorted(_UQ_MODES))
            raise ValueError(f"Unknown uncertainty_head_mode: {uncertainty_head_mode!r}. Supported: {supported}")
        self.uncertainty_head_mode = uq_key
        self.uncertainty_min_std = float(max(uncertainty_min_std, 0.0))

        # Learnable element embedding. Padding index 0 stays fixed as zero vector.
        self.embedder = nn.Embedding(120, self.d_model, padding_idx=0)

        # Fraction encoders use continuous phases (math smoothness > quantized bins).
        self.pe = ContinuousFractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = ContinuousFractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.pos_scaler = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.pos_scaler_log = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model,
            nhead=heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)

        self.output_nns = nn.ModuleList(
            [ResidualNetwork(self.d_model, self.out_dims, [256, 128]) for _ in range(self.ensemble_size)]
        )
        self.uncertainty_nns = None
        if self.uncertainty_head_mode != "none":
            self.uncertainty_nns = nn.ModuleList(
                [ResidualNetwork(self.d_model, self.out_dims, [256, 128]) for _ in range(self.ensemble_size)]
            )

    @staticmethod
    def _renormalize_simplex(frac: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        frac = torch.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
        frac = torch.clamp(frac, min=0.0)
        frac = frac * mask.to(dtype=frac.dtype)

        mass = frac.sum(dim=1, keepdim=True)
        safe_mass = torch.clamp(mass, min=eps)
        frac_norm = frac / safe_mass

        # If a row has zero provided mass, fall back to uniform over active elements.
        count = torch.clamp(mask.sum(dim=1, keepdim=True), min=1)
        uniform = mask.to(dtype=frac.dtype) / count.to(dtype=frac.dtype)
        has_mass = mass > eps
        return torch.where(has_mass, frac_norm, uniform)

    def _helmert_matrix(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = self._helmert_cache.get(n)
        if cached is None:
            h = torch.zeros((n, n - 1), dtype=torch.float32)
            for j in range(1, n):
                inv = 1.0 / math.sqrt(j * (j + 1))
                h[:j, j - 1] = inv
                h[j, j - 1] = -j * inv
            self._helmert_cache[n] = h
            cached = h
        return cached.to(device=device, dtype=dtype)

    def _sanitize_inputs(self, src: torch.Tensor, frac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = src.to(dtype=torch.long)
        frac = frac.to(dtype=torch.float32)
        if src.shape != frac.shape:
            raise ValueError(f"src and frac must have identical shape, got {tuple(src.shape)} vs {tuple(frac.shape)}")
        valid_mask = src != 0
        frac = self._renormalize_simplex(frac, valid_mask)
        return src, frac, valid_mask

    def _ilr_softmax(self, frac: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        transformed = torch.zeros_like(frac)
        for row_idx in range(frac.shape[0]):
            active_idx = torch.nonzero(valid_mask[row_idx], as_tuple=False).squeeze(-1)
            active_count = int(active_idx.numel())
            if active_count == 0:
                continue
            if active_count == 1:
                transformed[row_idx, active_idx] = 1.0
                continue

            p = torch.clamp(frac[row_idx, active_idx], min=1e-12)
            logp = torch.log(p)
            clr = logp - logp.mean()
            # ILR coordinates on orthonormal Helmert basis.
            basis = self._helmert_matrix(active_count, p.device, p.dtype)
            ilr = clr @ basis
            ilr = ilr / self.ilr_temperature
            clr_scaled = ilr @ basis.transpose(0, 1)
            p_scaled = torch.softmax(clr_scaled, dim=0)
            transformed[row_idx, active_idx] = p_scaled
        return self._renormalize_simplex(transformed, valid_mask)

    def _apply_simplex_transform(self, frac: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if self.simplex_transform == "none" or self.simplex_blend <= 0.0:
            return frac

        transformed = frac
        if self.simplex_transform == "escort":
            # Aitchison power transform on the simplex:
            # p_i -> p_i^q / sum_j p_j^q
            transformed = torch.pow(torch.clamp(frac, min=1e-12), self.escort_power)
            transformed = self._renormalize_simplex(transformed, valid_mask)
        elif self.simplex_transform == "clr_softmax":
            # CLR-inspired mapping and softmax projection back to simplex.
            safe = torch.clamp(frac, min=1e-12)
            logp = torch.log(safe)
            m = valid_mask.to(dtype=frac.dtype)
            denom = torch.clamp(m.sum(dim=1, keepdim=True), min=1.0)
            clr_mean = (logp * m).sum(dim=1, keepdim=True) / denom
            clr = (logp - clr_mean).masked_fill(~valid_mask, float("-inf"))
            transformed = torch.softmax(clr / self.clr_temperature, dim=1)
            transformed = torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
            transformed = self._renormalize_simplex(transformed, valid_mask)
        elif self.simplex_transform == "ilr_softmax":
            transformed = self._ilr_softmax(frac, valid_mask)

        blend = (1.0 - self.simplex_blend) * frac + self.simplex_blend * transformed
        return self._renormalize_simplex(blend, valid_mask)

    @staticmethod
    def _decode_std(raw: torch.Tensor, mode: str, min_std: float) -> torch.Tensor:
        if mode == "log_std":
            std = torch.exp(raw)
        elif mode == "log_var":
            std = torch.exp(0.5 * raw)
        elif mode == "softplus_std":
            std = F.softplus(raw)
        else:
            raise ValueError(f"Unsupported uncertainty mode: {mode}")
        if min_std > 0.0:
            std = torch.clamp(std, min=min_std)
        return std

    @staticmethod
    def _weighted_pool(x: torch.Tensor, weights: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        active = valid_mask.to(dtype=weights.dtype)
        w = weights * active
        denom = torch.clamp(w.sum(dim=1, keepdim=True), min=1e-12)
        w = w / denom
        return torch.sum(x * w.unsqueeze(-1), dim=1)

    @staticmethod
    def _safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.sqrt(torch.clamp(x, min=eps))

    def _encode(self, src: torch.Tensor, frac: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, frac, valid_mask = self._sanitize_inputs(src, frac)
        frac = self._apply_simplex_transform(frac, valid_mask)
        x = self.embedder(src)
        two = torch.tensor(2.0, device=x.device, dtype=x.dtype)
        emb_scale = torch.pow(two, self.emb_scaler.to(device=x.device, dtype=x.dtype))
        x = x * emb_scale
        pe_feat = torch.zeros_like(x)
        ple_feat = torch.zeros_like(x)

        pe_scaler = torch.pow(two, torch.square(1.0 - self.pos_scaler.to(device=x.device, dtype=x.dtype)))
        ple_scaler = torch.pow(two, torch.square(1.0 - self.pos_scaler_log.to(device=x.device, dtype=x.dtype)))
        pe_feat[:, :, : self.d_model // 2] = self.pe(frac) * pe_scaler
        ple_feat[:, :, self.d_model // 2 :] = self.ple(frac) * ple_scaler

        x_src = x + pe_feat + ple_feat
        src_padding_mask = ~valid_mask
        x_enc = self.transformer_encoder(x_src, src_key_padding_mask=src_padding_mask)
        x_enc = x_enc.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
        pooled = self._weighted_pool(x_enc, frac, valid_mask)
        return pooled, frac

    def _member_predictions(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        member_means = torch.stack([head(pooled) for head in self.output_nns], dim=0)
        if self.uncertainty_head_mode == "none":
            return member_means, None

        decoded = []
        for head in self.uncertainty_nns:
            raw_std = head(pooled)
            decoded.append(self._decode_std(raw_std, self.uncertainty_head_mode, self.uncertainty_min_std))
        member_stds = torch.stack(decoded, dim=0)
        return member_means, member_stds

    def _aggregate_distribution(
        self,
        member_means: torch.Tensor,
        member_stds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mean = member_means.mean(dim=0)
        epistemic_var = torch.var(member_means, dim=0, unbiased=False)
        aleatoric_var = torch.mean(torch.square(member_stds), dim=0)
        total_var = epistemic_var + aleatoric_var
        return {
            "mean": mean,
            "std": self._safe_sqrt(total_var),
            "aleatoric_std": self._safe_sqrt(aleatoric_var),
            "epistemic_std": self._safe_sqrt(epistemic_var),
            "total_std": self._safe_sqrt(total_var),
            "ensemble_size": torch.tensor(float(member_means.shape[0]), device=mean.device, dtype=mean.dtype),
        }

    @staticmethod
    def gaussian_nll(
        target: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        reduction: str = "mean",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        var = torch.clamp(torch.square(std), min=eps)
        return F.gaussian_nll_loss(mean, target, var, reduction=reduction)

    def compute_training_loss(
        self,
        src: torch.Tensor,
        frac: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        pooled, _ = self._encode(src, frac)
        member_means, member_stds = self._member_predictions(pooled)
        mean = member_means.mean(dim=0)
        target = target.to(device=mean.device, dtype=mean.dtype)
        if target.shape != mean.shape:
            raise ValueError(f"target shape must match predictions, got {tuple(target.shape)} vs {tuple(mean.shape)}")

        if member_stds is None:
            return F.mse_loss(mean, target, reduction=reduction)

        aleatoric_std = self._safe_sqrt(torch.mean(torch.square(member_stds), dim=0), eps=self.uncertainty_min_std**2)
        return self.gaussian_nll(
            target,
            mean,
            aleatoric_std,
            reduction=reduction,
            eps=max(self.uncertainty_min_std**2, 1e-12),
        )

    @torch.no_grad()
    def _mc_dropout_epistemic_var(self, src: torch.Tensor, frac: torch.Tensor, samples: int) -> torch.Tensor | None:
        samples = int(max(samples, 0))
        if samples <= 0:
            return None

        prev_training = self.training
        self.train(True)
        mc_means = []
        for _ in range(samples):
            pooled, _ = self._encode(src, frac)
            member_means, _ = self._member_predictions(pooled)
            mc_means.append(member_means.mean(dim=0))
        self.train(prev_training)
        stacked = torch.stack(mc_means, dim=0)
        return torch.var(stacked, dim=0, unbiased=False)

    def forward(self, src: torch.Tensor, frac: torch.Tensor):
        pooled, transformed_frac = self._encode(src, frac)
        member_means, member_stds = self._member_predictions(pooled)
        mean = member_means.mean(dim=0)
        if self.uncertainty_head_mode == "none":
            return mean

        dist = self._aggregate_distribution(member_means, member_stds)
        dist["transformed_frac"] = transformed_frac
        if self.return_distribution:
            return dist
        return mean

    @torch.no_grad()
    def predict_distribution(
        self,
        src: torch.Tensor,
        frac: torch.Tensor,
        mc_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.uncertainty_head_mode == "none":
            raise RuntimeError("predict_distribution requires uncertainty_head_mode != 'none'.")
        pooled, transformed_frac = self._encode(src, frac)
        member_means, member_stds = self._member_predictions(pooled)
        out = self._aggregate_distribution(member_means, member_stds)
        out["transformed_frac"] = transformed_frac

        samples = self.mc_dropout_samples if mc_samples is None else int(max(mc_samples, 0))
        if samples > 0:
            mc_epistemic_var = self._mc_dropout_epistemic_var(src, frac, samples)
            if mc_epistemic_var is not None:
                base_epi_var = torch.square(out["epistemic_std"])
                epi_var = base_epi_var + mc_epistemic_var
                ale_var = torch.square(out["aleatoric_std"])
                total_var = ale_var + epi_var
                out["mc_epistemic_std"] = self._safe_sqrt(mc_epistemic_var)
                out["epistemic_std"] = self._safe_sqrt(epi_var)
                out["total_std"] = self._safe_sqrt(total_var)
                out["std"] = out["total_std"]
                out["mc_samples"] = torch.tensor(float(samples), device=out["mean"].device, dtype=out["mean"].dtype)
        return out
