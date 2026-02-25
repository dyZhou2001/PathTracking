from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def masked_point_huber_loss(
    pred_points: torch.Tensor,  # (B,N,2)
    target_points: torch.Tensor,  # (B,N,2)
    target_mask: torch.Tensor,  # (B,N)
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    mask = target_mask.float().unsqueeze(-1)  # (B,N,1)
    per = F.smooth_l1_loss(pred_points, target_points, reduction="none", beta=float(delta))
    per = per.sum(dim=-1, keepdim=True)  # (B,N,1)
    num = (per * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den


def path_smoothness_loss(
    pred_points: torch.Tensor,  # (B,N,2)
    target_mask: torch.Tensor,  # (B,N)
) -> torch.Tensor:
    """Second-difference smoothness penalty (masked)."""
    B, N, _ = pred_points.shape
    if N < 3:
        return pred_points.new_tensor(0.0)

    p0 = pred_points[:, :-2, :]
    p1 = pred_points[:, 1:-1, :]
    p2 = pred_points[:, 2:, :]

    dd = p2 - 2.0 * p1 + p0  # (B,N-2,2)
    dd = torch.linalg.norm(dd, dim=-1)  # (B,N-2)

    m0 = target_mask[:, :-2]
    m1 = target_mask[:, 1:-1]
    m2 = target_mask[:, 2:]
    m = (m0 * m1 * m2).float()

    num = (dd * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def remaining_length_loss(pred_len: torch.Tensor, target_len: torch.Tensor, *, delta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(pred_len, target_len, reduction="mean", beta=float(delta))


def compute_losses(
    *,
    pred_points: torch.Tensor,
    pred_remaining_length_m: torch.Tensor,
    target_points: torch.Tensor,
    target_mask: torch.Tensor,
    target_remaining_length_m: torch.Tensor,
    w_smooth: float = 0.1,
    w_len: float = 0.2,
) -> Dict[str, torch.Tensor]:
    lp = masked_point_huber_loss(pred_points, target_points, target_mask)
    ls = path_smoothness_loss(pred_points, target_mask)
    ll = remaining_length_loss(pred_remaining_length_m, target_remaining_length_m)

    total = lp + float(w_smooth) * ls + float(w_len) * ll
    return {"total": total, "points": lp, "smooth": ls, "len": ll}
