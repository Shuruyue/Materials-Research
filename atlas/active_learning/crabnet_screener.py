import torch
import torch.nn as nn
import numpy as np

class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network adapted from CrabNet.
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super().__init__()
        dims = [input_dim] + hidden_layer_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.res_fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                      if (dims[i] != dims[i+1]) else nn.Identity()
                                      for i in range(len(dims)-1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for _ in range(len(dims)-1)])
        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, fea):
        for fc, res_fc, act in zip(self.fcs, self.res_fcs, self.acts):
            fea = act(fc(fea)) + res_fc(fea)
        return self.fc_out(fea)

class FractionalEncoder(nn.Module):
    """
    Sinusoidal fractional encoder for element proportions (from CrabNet).
    """
    def __init__(self, d_model, resolution=100, log10=False):
        super().__init__()
        self.d_model = d_model // 2
        self.resolution = resolution
        self.log10 = log10

        x = torch.linspace(0, self.resolution - 1, self.resolution).view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1, self.d_model).view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x / torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * self.resolution).to(dtype=torch.long) - 1
        return self.pe[frac_idx]


class CompositionScreener(nn.Module):
    """
    CrabNet-inspired composition screening model. 
    Accepts elemental indices and fractional contents to predict properties.
    Greatly accelerates active learning by pre-filtering bad materials before MD.
    """
    def __init__(self, out_dims=1, d_model=512, N=3, heads=4):
        super().__init__()
        self.out_dims = out_dims
        self.d_model = d_model
        
        # Simple elemental embedding for demonstration (up to element 118)
        self.embedder = nn.Embedding(120, d_model)
        
        self.pe = FractionalEncoder(d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(d_model, resolution=5000, log10=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=heads, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)
        
        self.output_nn = ResidualNetwork(d_model, out_dims, [256, 128])

    def forward(self, src, frac):
        """
        src: [batch_size, max_elements] containing atomic numbers
        frac: [batch_size, max_elements] containing molar fractions
        """
        x = self.embedder(src)
        
        mask = frac.unsqueeze(dim=-1)
        mask = torch.matmul(mask, mask.transpose(-2, -1))
        mask[mask != 0] = 1
        src_mask = mask[:, 0] != 1

        pe_feat = torch.zeros_like(x)
        ple_feat = torch.zeros_like(x)
        
        pe_feat[:, :, :self.d_model//2] = self.pe(frac)
        ple_feat[:, :, self.d_model//2:] = self.ple(frac)

        # Combine element type and fraction
        x_src = x + pe_feat + ple_feat
        x_src = x_src.transpose(0, 1) # [seq, batch, embed]
        
        x_enc = self.transformer_encoder(x_src, src_key_padding_mask=src_mask)
        x_enc = x_enc.transpose(0, 1)

        # Pool active elements
        x_enc = x_enc * frac.unsqueeze(2).repeat(1, 1, self.d_model)
        
        # Aggregate
        agg_mask = (src == 0).unsqueeze(-1).repeat(1, 1, self.out_dims)
        output = self.output_nn(x_enc)
        output = output.masked_fill(agg_mask, 0)
        output = output.sum(dim=1) / (~agg_mask).sum(dim=1)
        
        return output
