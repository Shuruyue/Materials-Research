import logging

import torch.nn as nn

try:
    # Full native import from the Assimilated CrabNet source code
    from atlas.third_party.crabnet.crabnet_ import CrabNet as OriginalCrabNet
except ImportError as e:
    logging.warning(f"Could not import assimilated CrabNet: {e}")
    OriginalCrabNet = None

from atlas.utils.registry import MODELS

logger = logging.getLogger(__name__)

@MODELS.register("crabnet_native")
class NativeCrabnetScreener(nn.Module):
    """
    Full architecture assimilation of the CrabNet predictor.
    By natively launching the Original CrabNet module, we gain access to its intricate
    attention-map generation, robust fractional encoders, and all complex hyperparameters,
    while hiding it behind the clean ATLAS MODELS registry interface.
    """
    def __init__(self,
                 compute_device: str = "cpu",
                 d_model: int = 512,
                 heads: int = 4,
                 d_ffn: int = 2048,
                 N: int = 3,
                 pe_resolution: int = 5000,
                 f_prop: str = 'num_atoms',
                 **kwargs):
        super().__init__()

        if OriginalCrabNet is None:
            raise RuntimeError("Assimilated CrabNet source is missing from atlas/third_party/crabnet.")

        logger.info(f"Instantiating fully native CrabNet with d_model={d_model}, N={N} layers.")

        # Instantiate natively directly from the original structure
        self.engine = OriginalCrabNet(
            compute_device=compute_device,
            out_dims=3,             # e.g., predictions + uncertainties
            d_model=d_model,
            N=N,
            heads=heads,
            d_ffn=d_ffn,
            pe_resolution=pe_resolution,
            f_prop=f_prop,
            **kwargs
        )

    def forward(self, frac_idx, elem_idx):
        """
        Straight pass-through to Original CrabNet's forward method.
        """
        return self.engine(frac_idx, elem_idx)
