from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import numpy as np

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    Image = None  # type: ignore

import torch
from torch.utils.data import Dataset

from .geometry import ResampleResult, resample_polyline_by_arclength


@dataclass(frozen=True)
class PlannerSample:
    image: torch.Tensor  # (3,H,W)
    target_points: torch.Tensor  # (N,2)
    target_mask: torch.Tensor  # (N,)
    remaining_length_m: torch.Tensor  # (1,)
    state: Optional[torch.Tensor]  # (S,) optional extra inputs
    meta: Dict[str, Any]


class JsonlOffsetIndex:
    """Builds file offsets for a jsonl so we can random access lines efficiently."""

    def __init__(self, labels_path: Path):
        self.labels_path = labels_path
        self.offsets: List[int] = []

        with labels_path.open("rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(off)

    def __len__(self) -> int:
        return len(self.offsets)

    def read_obj(self, index: int) -> Dict[str, Any]:
        off = self.offsets[index]
        with self.labels_path.open("rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8")
        return json.loads(line)


def _load_image_tensor_rgb(path: Path, *, image_size: int = 224) -> torch.Tensor:
    if Image is None:
        raise RuntimeError("PIL not available. Please install pillow.")

    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((int(image_size), int(image_size)))

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    # to CHW
    arr = np.transpose(arr, (2, 0, 1))

    # ImageNet normalization (works well even if you don't use pretrained weights)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std

    return torch.from_numpy(arr)


class CarlaPathDataset(Dataset):
    """CARLA dataset: image -> local path (vehicle frame) resampled by arc-length."""

    def __init__(
        self,
        *,
        labels_jsonl: str,
        image_root: Optional[str] = None,
        image_size: int = 224,
        path_length_m: float = 15.0,
        spacing_m: float = 1.0,
        start_s_m: float = 1.0,
        pad_mode: str = "last",
        include_state: bool = False,
        state_xy_scale: float = 100.0,
        max_samples: Optional[int] = None,
    ):
        self.labels_path = Path(labels_jsonl)
        if not self.labels_path.exists():
            raise FileNotFoundError(str(self.labels_path))

        self.image_root = Path(image_root) if image_root else self.labels_path.parent
        self.image_size = int(image_size)
        self.path_length_m = float(path_length_m)
        self.spacing_m = float(spacing_m)
        self.start_s_m = float(start_s_m)
        self.pad_mode = str(pad_mode)

        self.include_state = bool(include_state)
        self.state_xy_scale = float(state_xy_scale)

        self.num_points = int(round(self.path_length_m / self.spacing_m))
        if self.num_points <= 0:
            raise ValueError("num_points must be > 0")

        self.index = JsonlOffsetIndex(self.labels_path)
        self._length = len(self.index)
        if max_samples is not None:
            self._length = min(self._length, int(max_samples))

    def __len__(self) -> int:
        return self._length

    def _extract_future_points(self, obj: Dict[str, Any]) -> List[Tuple[float, float]]:
        future = obj.get("future_route_vehicle", []) or []
        pts: List[Tuple[float, float]] = []
        for p in future:
            if not isinstance(p, dict):
                continue
            if "x" not in p or "y" not in p:
                continue
            pts.append((float(p["x"]), float(p["y"])))
        return pts

    def __getitem__(self, idx: int) -> PlannerSample:
        obj = self.index.read_obj(idx)

        rel_img = obj.get("image", None)
        if not rel_img:
            raise RuntimeError("Missing 'image' in labels")

        img_path = (self.image_root / str(rel_img)).resolve()
        image = _load_image_tensor_rgb(img_path, image_size=self.image_size)

        raw_pts = self._extract_future_points(obj)

        rr: ResampleResult = resample_polyline_by_arclength(
            raw_pts,
            spacing_m=self.spacing_m,
            num_samples=self.num_points,
            start_s_m=self.start_s_m,
            pad_mode=self.pad_mode,
        )

        target_points = torch.tensor(rr.points_xy, dtype=torch.float32)  # (N,2)
        target_mask = torch.tensor(rr.mask, dtype=torch.float32)  # float mask: 1/0
        remaining = torch.tensor([min(rr.available_length_m, self.path_length_m)], dtype=torch.float32)

        meta = {
            "frame": int(obj.get("frame", -1)),
            "episode_step": int(obj.get("episode_step", -1)),
            "sim_time": float(obj.get("sim_time", 0.0)),
            "image": str(rel_img),
        }

        state: Optional[torch.Tensor] = None
        if self.include_state:
            veh = obj.get("vehicle", {}) or {}
            try:
                x = float(veh.get("x", 0.0)) / max(self.state_xy_scale, 1e-6)
                y = float(veh.get("y", 0.0)) / max(self.state_xy_scale, 1e-6)
                yaw_deg = float(veh.get("yaw_deg", 0.0))
            except Exception:
                x, y, yaw_deg = 0.0, 0.0, 0.0

            # Use sin/cos to avoid angle wrap discontinuity.
            yaw_rad = float(np.deg2rad(yaw_deg))
            state = torch.tensor([x, y, float(np.sin(yaw_rad)), float(np.cos(yaw_rad))], dtype=torch.float32)

        return PlannerSample(
            image=image,
            target_points=target_points,
            target_mask=target_mask,
            remaining_length_m=remaining,
            state=state,
            meta=meta,
        )


def planner_collate(batch: List[PlannerSample]) -> Dict[str, Any]:
    images = torch.stack([b.image for b in batch], dim=0)
    points = torch.stack([b.target_points for b in batch], dim=0)
    masks = torch.stack([b.target_mask for b in batch], dim=0)
    lengths = torch.stack([b.remaining_length_m for b in batch], dim=0).squeeze(-1)
    meta = [b.meta for b in batch]

    has_state = all(b.state is not None for b in batch)
    state = torch.stack([b.state for b in batch], dim=0) if has_state else None

    return {
        "image": images,
        "state": state,
        "target_points": points,
        "target_mask": masks,
        "remaining_length_m": lengths,
        "meta": meta,
    }
