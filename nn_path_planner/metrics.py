from __future__ import annotations

from typing import Dict

import torch


def masked_ade_fde(
    pred_points: torch.Tensor,  # (B,N,2)
    target_points: torch.Tensor,  # (B,N,2)
    target_mask: torch.Tensor,  # (B,N)
) -> Dict[str, torch.Tensor]:
    """Compute ADE and FDE with a per-point mask.

    - ADE: mean Euclidean error over valid points.
    - FDE: Euclidean error at the farthest valid point per sample.
    """

    eps = 1e-6
    B, N, _ = pred_points.shape

    mask = target_mask.float()
    diff = pred_points - target_points
    dist = torch.linalg.norm(diff, dim=-1)  # (B,N)

    denom = mask.sum(dim=1).clamp_min(1.0)
    ade = (dist * mask).sum(dim=1) / denom

    # farthest valid index (last one with mask=1). If none valid, use index 0.
    idx = (mask > 0.5).float() * torch.arange(N, device=mask.device).float()[None, :]
    last = idx.max(dim=1).values.long()

    batch_idx = torch.arange(B, device=mask.device)
    fde = dist[batch_idx, last]

    return {"ade": ade, "fde": fde}
