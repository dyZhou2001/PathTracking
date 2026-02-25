from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math


@dataclass(frozen=True)
class ResampleResult:
    points_xy: List[Tuple[float, float]]  # length = num_samples
    mask: List[bool]  # length = num_samples
    available_length_m: float


def _pairwise_dist(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    return math.hypot(dx, dy)


def _clean_points(points_xy: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cleaned: List[Tuple[float, float]] = []
    for p in points_xy:
        try:
            x = float(p[0])
            y = float(p[1])
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        cleaned.append((x, y))

    if len(cleaned) <= 1:
        return cleaned

    # Remove consecutive duplicates (distance ~ 0) to avoid zero-length segments.
    deduped = [cleaned[0]]
    for p in cleaned[1:]:
        if _pairwise_dist(deduped[-1], p) > 1e-6:
            deduped.append(p)
    return deduped


def resample_polyline_by_arclength(
    points_xy: Sequence[Tuple[float, float]],
    *,
    spacing_m: float,
    num_samples: int,
    start_s_m: float = 1.0,
    pad_mode: str = "last",  # "last" or "zero"
) -> ResampleResult:
    """Resample a polyline at fixed arc-length positions.

    Samples points at s = start_s_m + k*spacing_m, for k=0..num_samples-1.

    - If the polyline is shorter than required, returns padded points with mask=False.
    - available_length_m is the total arc-length of the polyline.

    Args:
        points_xy: (x,y) points in vehicle frame.
        spacing_m: desired spacing along arc-length.
        num_samples: number of samples to output.
        start_s_m: first sample arc-length (meters). For your setting: 1.0.
        pad_mode: how to pad missing samples. "last" repeats last valid; "zero" uses (0,0).
    """

    if num_samples <= 0:
        return ResampleResult(points_xy=[], mask=[], available_length_m=0.0)

    spacing = max(float(spacing_m), 1e-6)
    start_s = float(start_s_m)

    cleaned = _clean_points(points_xy)
    if len(cleaned) < 2:
        pad_point = (0.0, 0.0)
        out_pts = [pad_point for _ in range(num_samples)]
        out_mask = [False for _ in range(num_samples)]
        return ResampleResult(points_xy=out_pts, mask=out_mask, available_length_m=0.0)

    # Build cumulative arc-length.
    cum_s: List[float] = [0.0]
    for i in range(len(cleaned) - 1):
        cum_s.append(cum_s[-1] + _pairwise_dist(cleaned[i], cleaned[i + 1]))
    total_len = float(cum_s[-1])

    def point_at_s(query_s: float) -> Optional[Tuple[float, float]]:
        if query_s < 0.0 or query_s > total_len:
            return None

        # Find segment containing query_s.
        # Linear scan is fine for typical short polylines.
        for i in range(len(cum_s) - 1):
            s0 = cum_s[i]
            s1 = cum_s[i + 1]
            if s1 <= s0:
                continue
            if query_s <= s1:
                t = (query_s - s0) / (s1 - s0)
                x0, y0 = cleaned[i]
                x1, y1 = cleaned[i + 1]
                return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

        return cleaned[-1]

    out_points: List[Tuple[float, float]] = []
    out_mask: List[bool] = []
    last_valid: Tuple[float, float] = cleaned[-1]

    for k in range(num_samples):
        query_s = start_s + k * spacing
        p = point_at_s(query_s)
        if p is None:
            out_mask.append(False)
            if pad_mode == "zero":
                out_points.append((0.0, 0.0))
            else:
                out_points.append(last_valid)
        else:
            out_mask.append(True)
            out_points.append(p)
            last_valid = p

    return ResampleResult(points_xy=out_points, mask=out_mask, available_length_m=total_len)


def lookahead_point_from_resampled(
    points_xy: Sequence[Tuple[float, float]],
    *,
    start_s_m: float,
    spacing_m: float,
    lookahead_m: float,
) -> Tuple[float, float]:
    """Pick the point corresponding to a given arc-length (e.g. 6m) from a resampled sequence.

    With start_s_m=1 and spacing_m=1, lookahead_m=6 corresponds to index 5.
    """
    idx = int(round((float(lookahead_m) - float(start_s_m)) / float(spacing_m)))
    idx = max(0, min(idx, len(points_xy) - 1))
    return (float(points_xy[idx][0]), float(points_xy[idx][1]))
