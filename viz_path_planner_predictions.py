"""Visualize trained transformer path planner predictions.

Creates a 3x3 grid. Each cell's main plot shows local path in vehicle frame:
- Green: GT (resampled at s=1..15m, masked)
- Red: prediction
- Markers at 6m lookahead point (index=5)

An RGB image inset is shown in the corner.

Example:
  python viz_path_planner_predictions.py \
    --labels dataset/run_20260125_163212/labels.jsonl \
    --checkpoint checkpoints_transformer/best.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch

from nn_path_planner.dataset import CarlaPathDataset
from nn_path_planner.models_transformer import TransformerPlannerNet


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(str(path), map_location=device)


def _build_model_from_ckpt(ckpt: Dict[str, Any]) -> TransformerPlannerNet:
    args = ckpt.get("args", {}) or {}
    use_state = bool(args.get("use_state", False))
    return TransformerPlannerNet(
        num_points=15,
        d_model=int(args.get("d_model", 256)),
        nhead=int(args.get("nhead", 8)),
        num_encoder_layers=int(args.get("enc_layers", 4)),
        num_decoder_layers=int(args.get("dec_layers", 4)),
        state_dim=(4 if use_state else 0),
    )


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="Path to labels.jsonl")
    p.add_argument("--image_root", default=None, help="Root folder for images (default: labels.jsonl parent)")
    p.add_argument("--checkpoint", default="checkpoints_transformer/best.pt", help="Path to model checkpoint")
    p.add_argument("--out", default="viz_predictions_9.png", help="Output PNG path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    p.add_argument("--max_samples", type=int, default=None, help="Optional cap on dataset size")

    p.add_argument(
        "--flip_pred_y",
        action="store_true",
        default=False,
        help="Debug: flip predicted y (right/left) sign before plotting.",
    )
    p.add_argument(
        "--xlim_min",
        type=float,
        default=-2.0,
        help="Min x-axis limit. Use <0 to reveal predictions behind the vehicle.",
    )
    p.add_argument(
        "--ylim",
        type=float,
        default=6.0,
        help="Abs y-axis limit in meters.",
    )

    args = p.parse_args()

    labels_path = Path(args.labels)
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)

    if not labels_path.exists():
        raise FileNotFoundError(str(labels_path))
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = _load_checkpoint(ckpt_path, device)
    ckpt_args = ckpt.get("args", {}) or {}
    use_state = bool(ckpt_args.get("use_state", False))
    state_xy_scale = float(ckpt_args.get("state_xy_scale", 100.0))

    model = _build_model_from_ckpt(ckpt).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Match training dataset settings.
    ds = CarlaPathDataset(
        labels_jsonl=str(labels_path),
        image_root=args.image_root,
        image_size=224,
        path_length_m=15.0,
        spacing_m=1.0,
        start_s_m=1.0,
        pad_mode="last",
        include_state=use_state,
        state_xy_scale=state_xy_scale,
        max_samples=args.max_samples,
    )

    if len(ds) < 9:
        raise RuntimeError(f"Dataset too small for 9 samples: {len(ds)}")

    rng = random.Random(int(args.seed))
    indices = rng.sample(range(len(ds)), 9)

    try:
        from PIL import Image
    except Exception:
        Image = None  # type: ignore

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=140)

    for ax, idx in zip(axes.flatten(), indices):
        sample = ds[idx]

        image_t = sample.image.unsqueeze(0).to(device)
        state_t = None
        if use_state:
            if sample.state is None:
                raise RuntimeError("Checkpoint expects state input, but dataset sample.state is None")
            state_t = sample.state.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(image_t, state=state_t) if use_state else model(image_t)

        pred_pts = _to_np(out["points"][0])  # (15,2)
        if bool(args.flip_pred_y):
            pred_pts = pred_pts.copy()
            pred_pts[:, 1] *= -1.0
        pred_len = float(out["remaining_length_m"][0].item())

        gt_pts = _to_np(sample.target_points)
        gt_mask = _to_np(sample.target_mask) > 0.5
        gt_len = float(sample.remaining_length_m.item())

        # Plot GT (masked)
        if gt_mask.any():
            last_valid = int(np.where(gt_mask)[0].max())
            gt_plot = gt_pts[: last_valid + 1]
            ax.plot(gt_plot[:, 0], gt_plot[:, 1], "-o", color="#00aa00", markersize=3, linewidth=1.5, label="GT")
        else:
            last_valid = -1

        # Plot prediction
        ax.plot(pred_pts[:, 0], pred_pts[:, 1], "-o", color="#cc0000", markersize=3, linewidth=1.5, label="Pred")

        # Mark 6m lookahead point: s=1..15 => 6m is index 5
        la_idx = 5
        ax.scatter([pred_pts[la_idx, 0]], [pred_pts[la_idx, 1]], s=60, c="#cc0000", marker="x")
        if last_valid >= la_idx:
            ax.scatter([gt_pts[la_idx, 0]], [gt_pts[la_idx, 1]], s=60, c="#00aa00", marker="x")

        ax.axhline(0.0, color="#cccccc", linewidth=1)
        ax.axvline(0.0, color="#cccccc", linewidth=1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.3)

        ax.set_xlim(float(args.xlim_min), 15.5)
        y_lim = float(abs(args.ylim))
        ax.set_ylim(-y_lim, y_lim)

        frame = sample.meta.get("frame", -1)
        # Show 6m point and whether it goes behind vehicle.
        la_idx = 5
        la_x = float(pred_pts[la_idx, 0])
        la_y = float(pred_pts[la_idx, 1])
        behind_flag = "BEHIND" if la_x < 0.0 else ""
        flip_flag = " flipY" if bool(args.flip_pred_y) else ""
        ax.set_title(
            f"frame={frame} | gt_len={gt_len:.1f} pred_len={pred_len:.1f} | la=({la_x:+.1f},{la_y:+.1f}) {behind_flag}{flip_flag}",
            fontsize=8,
        )

        # RGB inset
        if Image is not None:
            rel_img = sample.meta.get("image", None)
            if rel_img:
                img_path = (Path(args.image_root) if args.image_root else labels_path.parent) / str(rel_img)
                if img_path.exists():
                    try:
                        rgb = Image.open(img_path).convert("RGB")
                        rgb = rgb.resize((224, 224))
                        inset = ax.inset_axes([0.02, 0.62, 0.36, 0.36])
                        inset.imshow(rgb)
                        inset.set_xticks([])
                        inset.set_yticks([])
                        for spine in inset.spines.values():
                            spine.set_edgecolor("#ffffff")
                            spine.set_linewidth(1.0)
                    except Exception:
                        pass

    # single legend
    axes.flatten()[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Path planner predictions (vehicle frame) | main: path, inset: RGB", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()
