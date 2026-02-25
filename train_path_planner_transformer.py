"""Train a transformer-based CNN path planner.

Task:
  Input: single RGB image
  Output: local path in vehicle frame (x,y) at s=1..15m (1m spacing) + remaining_length_m.

Example:
  python train_path_planner_transformer.py --labels dataset/run_xxx/labels.jsonl --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, random_split

from nn_path_planner.dataset import CarlaPathDataset, planner_collate
from nn_path_planner.losses import compute_losses
from nn_path_planner.metrics import masked_ade_fde
from nn_path_planner.models_transformer import TransformerPlannerNet


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    loss_total = 0.0
    loss_pts = 0.0
    loss_len = 0.0
    loss_smooth = 0.0

    ade_all = []
    fde_all = []

    n_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        state = batch.get("state", None)
        state = state.to(device) if state is not None else None
        tgt_points = batch["target_points"].to(device)
        tgt_mask = batch["target_mask"].to(device)
        tgt_len = batch["remaining_length_m"].to(device)

        out = model(images, state=state) if state is not None else model(images)
        pred_points = out["points"]
        pred_len = out["remaining_length_m"]

        losses = compute_losses(
            pred_points=pred_points,
            pred_remaining_length_m=pred_len,
            target_points=tgt_points,
            target_mask=tgt_mask,
            target_remaining_length_m=tgt_len,
        )

        m = masked_ade_fde(pred_points, tgt_points, tgt_mask)

        loss_total += float(losses["total"].item())
        loss_pts += float(losses["points"].item())
        loss_len += float(losses["len"].item())
        loss_smooth += float(losses["smooth"].item())

        ade_all.append(m["ade"].detach().cpu())
        fde_all.append(m["fde"].detach().cpu())
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "ade": 0.0, "fde": 0.0}

    ade = torch.cat(ade_all).mean().item() if ade_all else 0.0
    fde = torch.cat(fde_all).mean().item() if fde_all else 0.0

    return {
        "loss": loss_total / n_batches,
        "loss_points": loss_pts / n_batches,
        "loss_len": loss_len / n_batches,
        "loss_smooth": loss_smooth / n_batches,
        "ade": float(ade),
        "fde": float(fde),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="Path to labels.jsonl")
    p.add_argument("--image_root", default=None, help="Root folder for images (default: labels.jsonl parent)")
    p.add_argument("--save_dir", default="checkpoints_transformer", help="Where to save checkpoints")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--device", default='cuda', help="cpu/cuda (default: auto)")

    # extra state input
    p.add_argument(
        "--use_state",
        action="store_true",
        default=False,
        help="Include extra state features (x,y,yaw) in addition to image.",
    )
    p.add_argument(
        "--state_xy_scale",
        type=float,
        default=100.0,
        help="Scale world x/y by this value before feeding the net (x/=scale, y/=scale).",
    )

    # model hyperparams
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=4)

    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(int(args.seed))

    ds = CarlaPathDataset(
        labels_jsonl=args.labels,
        image_root=args.image_root,
        image_size=224,
        path_length_m=15.0,
        spacing_m=1.0,
        start_s_m=1.0,
        pad_mode="last",
        include_state=bool(args.use_state),
        state_xy_scale=float(args.state_xy_scale),
        max_samples=args.max_samples,
    )

    n_val = max(1, int(len(ds) * float(args.val_ratio))) if len(ds) >= 2 else 0
    n_train = len(ds) - n_val

    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(int(args.seed)))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=planner_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=planner_collate,
        drop_last=False,
    )

    model = TransformerPlannerNet(
        num_points=15,
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_encoder_layers=int(args.enc_layers),
        num_decoder_layers=int(args.dec_layers),
        state_dim=(4 if bool(args.use_state) else 0),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, int(args.epochs)))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            state = batch.get("state", None)
            state = state.to(device) if state is not None else None
            tgt_points = batch["target_points"].to(device)
            tgt_mask = batch["target_mask"].to(device)
            tgt_len = batch["remaining_length_m"].to(device)

            out = model(images, state=state) if bool(args.use_state) else model(images)
            losses = compute_losses(
                pred_points=out["points"],
                pred_remaining_length_m=out["remaining_length_m"],
                target_points=tgt_points,
                target_mask=tgt_mask,
                target_remaining_length_m=tgt_len,
            )

            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            if step % 25 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={losses['total'].item():.4f} "
                    f"pts={losses['points'].item():.4f} len={losses['len'].item():.4f} smooth={losses['smooth'].item():.4f}"
                )

        scheduler.step()

        val_stats = run_eval(model, val_loader, device) if n_val > 0 else {"loss": 0.0, "ade": 0.0, "fde": 0.0}
        print(f"[val] epoch={epoch} stats={val_stats}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "args": vars(args),
            "val": val_stats,
        }
        torch.save(ckpt, save_dir / "last.pt")

        if n_val > 0 and float(val_stats["loss"]) < best_val:
            best_val = float(val_stats["loss"])
            torch.save(ckpt, save_dir / "best.pt")


if __name__ == "__main__":
    main()
