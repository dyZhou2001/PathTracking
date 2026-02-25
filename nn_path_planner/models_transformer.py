from __future__ import annotations

import math

import torch
import torch.nn as nn

from .models_baseline import SimpleCNNBackbone


class SinCos2DPositionalEncoding(nn.Module):
    """2D sine/cosine positional encoding for (H,W) grids.

    Returns a tensor of shape (1, C, H, W).
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("positional dim must be divisible by 4")
        self.dim = int(dim)

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        dim = self.dim
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Half for y, half for x. Each uses sin+cos => dim/2 per axis.
        dim_each = dim // 2
        freqs = torch.arange(dim_each // 2, device=device, dtype=torch.float32)
        freqs = 1.0 / (10000 ** (freqs / (dim_each // 2)))

        # (H,W,dim_each/2)
        y_inp = yy[..., None] * freqs[None, None, :]
        x_inp = xx[..., None] * freqs[None, None, :]

        pe_y = torch.cat([torch.sin(y_inp), torch.cos(y_inp)], dim=-1)
        pe_x = torch.cat([torch.sin(x_inp), torch.cos(x_inp)], dim=-1)

        pe = torch.cat([pe_y, pe_x], dim=-1)  # (H,W,dim)
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # (1,dim,H,W)
        return pe


class TransformerPlannerNet(nn.Module):
    """CNN -> Transformer encoder/decoder with learned queries.

    Outputs:
      - points: (B, N, 2)
      - remaining_length_m: (B,)

    This is NOT for variable-length output; we keep N fixed (=15) but use
    transformer structure to model global context and ordered queries.
    """

    def __init__(
        self,
        *,
        num_points: int = 15,
        backbone_channels: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        state_dim: int = 0,
    ):
        super().__init__()
        self.num_points = int(num_points)
        self.state_dim = int(state_dim)

        self.backbone = SimpleCNNBackbone(base_channels=int(backbone_channels))
        c = self.backbone.out_channels

        self.input_proj = nn.Conv2d(c, int(d_model), kernel_size=1)
        self.pos2d = SinCos2DPositionalEncoding(int(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_encoder_layers))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(num_decoder_layers))

        # Learned queries, one per output point.
        self.query_embed = nn.Embedding(self.num_points, int(d_model))

        self.state_proj: nn.Module
        if self.state_dim > 0:
            self.state_proj = nn.Sequential(
                nn.Linear(self.state_dim, int(d_model)),
                nn.LayerNorm(int(d_model)),
                nn.GELU(),
                nn.Linear(int(d_model), int(d_model)),
            )
        else:
            self.state_proj = nn.Identity()

        self.points_mlp = nn.Sequential(
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), 2),
        )

        # Remaining length head uses pooled memory (+ optional state embedding).
        rem_in = int(d_model) * (2 if self.state_dim > 0 else 1)
        self.rem_head = nn.Sequential(
            nn.Linear(rem_in, int(d_model) // 2),
            nn.GELU(),
            nn.Linear(int(d_model) // 2, 1),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor | None = None) -> dict:
        feat = self.backbone(image)  # (B,C,h,w)
        mem = self.input_proj(feat)  # (B,D,h,w)
        B, D, H, W = mem.shape

        pos = self.pos2d(H, W, mem.device)  # (1,D,H,W)
        mem = mem + pos

        # Flatten to tokens
        mem_tokens = mem.flatten(2).transpose(1, 2)  # (B,HW,D)
        memory = self.encoder(mem_tokens)  # (B,HW,D)

        state_emb: torch.Tensor | None = None
        if self.state_dim > 0:
            if state is None:
                raise ValueError("state_dim>0 but state was not provided")
            if state.ndim != 2 or state.shape[0] != B or state.shape[1] != self.state_dim:
                raise ValueError(f"state must be (B,{self.state_dim}) but got {tuple(state.shape)}")
            state_emb = self.state_proj(state)  # (B,D)

        # Queries
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B,N,D)
        if state_emb is not None:
            q = q + state_emb.unsqueeze(1)
        hs = self.decoder(tgt=q, memory=memory)  # (B,N,D)

        points = self.points_mlp(hs)  # (B,N,2)

        pooled = memory.mean(dim=1)  # (B,D)
        if state_emb is not None:
            rem_in = torch.cat([pooled, state_emb], dim=-1)
        else:
            rem_in = pooled
        remaining = self.rem_head(rem_in).view(-1)  # (B,)

        return {"points": points, "remaining_length_m": remaining}
