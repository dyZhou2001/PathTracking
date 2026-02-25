"""Closed-loop test: neural path planner -> Pure Pursuit controller -> CARLA.

Pipeline (per step):
  RGB camera image -> TransformerPlannerNet -> 15 points (vehicle frame)
  -> pick 6m point (index=5) as (target_x,target_y)
  -> AdaptiveController (PurePursuit + Speed PID) -> (throttle, brake, steer)
  -> CarlaEnv.step()

Defaults match the dataset collection setup in example.py:
  Town03, spawn_point_index=0, destination_index=1, goal_radius=3.0, target_speed=5.0.

Example:
  python test_nn_path_planner_control.py \
    --checkpoint checkpoints_transformer/best.pt \
    --device cuda
"""

from __future__ import annotations

import argparse
import queue
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import torch

try:
    import carla  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Failed to import CARLA PythonAPI (carla/libcarla). "
        "On Windows this is often caused by a Python version mismatch with the CARLA egg "
        "(e.g. CARLA 0.9.15 provides a py3.7 egg) or missing VC++ runtime/DLLs. "
        "Please run this script with a Python version that matches your CARLA egg, and ensure "
        "CARLA PythonAPI is on PYTHONPATH.\n\n"
        f"Original error: {e}"
    )

from carla_env import CarlaEnv
from pid_controller import AdaptiveController
from nn_path_planner.models_transformer import TransformerPlannerNet


def _imagenet_preprocess_pil_rgb(pil_img, *, image_size: int = 224) -> torch.Tensor:
    # Keep consistent with nn_path_planner.dataset._load_image_tensor_rgb
    img = pil_img.convert("RGB")
    img = img.resize((int(image_size), int(image_size)))

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std

    return torch.from_numpy(arr)


def _carla_image_to_pil_rgb(image: carla.Image):
    from PIL import Image

    w = int(image.width)
    h = int(image.height)
    # BGRA uint8
    data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((h, w, 4))
    bgr = data[:, :, :3]
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def _get_image_for_frame(
    image_queue: "queue.Queue[carla.Image]",
    *,
    target_frame: int,
    timeout_s: float,
) -> Optional[carla.Image]:
    # NOTE: This function is intentionally strict: it only returns the image whose
    # frame matches target_frame. Returning a "best effort" latest frame silently
    # introduces image/state misalignment, which can destabilize closed-loop control.
    deadline = datetime.now().timestamp() + float(timeout_s)
    while datetime.now().timestamp() < deadline:
        remaining = max(0.01, deadline - datetime.now().timestamp())
        try:
            img = image_queue.get(timeout=remaining)
        except queue.Empty:
            break

        f = int(getattr(img, "frame", -1))
        if f < int(target_frame):
            # Drop older frames to catch up.
            continue
        if f == int(target_frame):
            return img
        # f > target_frame: caller may have missed the exact frame (e.g., queue overflow).
        # Do NOT return a future frame here; the caller may cache it.
        return None

    return None


def _vehicle_xy_to_world(transform: carla.Transform, x: float, y: float) -> carla.Location:
    loc = transform.location
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    return carla.Location(
        x=float(loc.x + forward.x * x + right.x * y),
        y=float(loc.y + forward.y * x + right.y * y),
        z=float(loc.z + 0.2),
    )


def _draw_pred_path(
    world: carla.World,
    transform: carla.Transform,
    points_xy: np.ndarray,
    *,
    life_time: float,
) -> None:
    try:
        for i in range(len(points_xy)):
            p = points_xy[i]
            p_world = _vehicle_xy_to_world(transform, float(p[0]), float(p[1]))
            world.debug.draw_point(p_world, size=0.08, color=carla.Color(255, 0, 0), life_time=float(life_time))
            if i > 0:
                p_prev = points_xy[i - 1]
                p_prev_world = _vehicle_xy_to_world(transform, float(p_prev[0]), float(p_prev[1]))
                world.debug.draw_line(p_prev_world, p_world, thickness=0.05, color=carla.Color(255, 0, 0), life_time=float(life_time))
    except RuntimeError:
        return


def _build_model_from_ckpt(ckpt: dict) -> TransformerPlannerNet:
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


def _get_state_xy_yaw_sincos(env: CarlaEnv, *, xy_scale: float) -> torch.Tensor:
    """Build state features: [x/scale, y/scale, sin(yaw), cos(yaw)]."""
    try:
        tr = env.vehicle.get_transform()
        x = float(tr.location.x) / max(float(xy_scale), 1e-6)
        y = float(tr.location.y) / max(float(xy_scale), 1e-6)
        yaw_deg = float(tr.rotation.yaw)
    except Exception:
        x, y, yaw_deg = 0.0, 0.0, 0.0

    yaw_rad = float(np.deg2rad(yaw_deg))
    return torch.tensor([x, y, float(np.sin(yaw_rad)), float(np.cos(yaw_rad))], dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints_transformer/best.pt")
    p.add_argument("--device", default=None, help="cpu/cuda (default: auto)")

    # env defaults match example.py
    p.add_argument("--town", default="Town03")
    p.add_argument("--spawn_point_index", type=int, default=0)
    p.add_argument("--destination_index", type=int, default=1)
    p.add_argument("--goal_radius", type=float, default=3.0)

    p.add_argument("--target_speed", type=float, default=5.0)
    p.add_argument("--max_steps", type=int, default=5000)

    # camera defaults match dataset collection
    p.add_argument("--camera_width", type=int, default=800)
    p.add_argument("--camera_height", type=int, default=600)
    p.add_argument("--camera_fov", type=float, default=90.0)

    # Default True to match dataset collection; allow disabling via --no_synchronous_mode.
    p.add_argument("--synchronous_mode", action="store_true", default=True)
    p.add_argument("--no_synchronous_mode", action="store_false", dest="synchronous_mode")
    p.add_argument("--fixed_delta_seconds", type=float, default=0.05)

    p.add_argument("--image_timeout_s", type=float, default=1.0)
    p.add_argument("--lookahead_index", type=int, default=5, help="6m point with s=1..15 => index=5")

    p.add_argument(
        "--min_target_x",
        type=float,
        default=0.5,
        help="If predicted target_x is behind/too small, fallback to env target point.",
    )
    p.add_argument(
        "--flip_pred_y",
        action="store_true",
        default=False,
        help="Debug: flip predicted y (right/left) sign before control/drawing.",
    )
    p.add_argument(
        "--print_pred_every",
        type=int,
        default=50,
        help="Print predicted vs env target every N steps (0 disables).",
    )

    p.add_argument("--spectator_follow", action="store_true", default=True)
    p.add_argument("--debug_draw_pred", action="store_true", default=False)
    p.add_argument(
        "--debug_draw_every",
        type=int,
        default=1,
        help="Draw predicted path every N steps (default: 1).",
    )
    p.add_argument(
        "--debug_draw_life_time",
        type=float,
        default=None,
        help="Seconds that a debug-drawn path remains visible. Default: ~0.9*fixed_delta_seconds to avoid trails.",
    )

    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    use_state = bool(ckpt_args.get("use_state", False))
    state_xy_scale = float(ckpt_args.get("state_xy_scale", 100.0))

    model = _build_model_from_ckpt(ckpt).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    env = CarlaEnv(
        town=str(args.town),
        spectator_follow=bool(args.spectator_follow),
        max_episode_steps=int(args.max_steps),
        spawn_point_index=int(args.spawn_point_index),
        destination_index=int(args.destination_index),
        goal_radius=float(args.goal_radius),
        debug_draw=False,
        collect_dataset=False,
        synchronous_mode=bool(args.synchronous_mode),
        fixed_delta_seconds=float(args.fixed_delta_seconds),
    )

    controller = AdaptiveController(target_speed=float(args.target_speed), dt=float(args.fixed_delta_seconds))

    camera = None
    image_queue: "queue.Queue[carla.Image]" = queue.Queue(maxsize=32)

    # Cache a single future image if we ever observe frame > target_frame.
    cached_image: Optional[carla.Image] = None

    def _get_image_strict(target_frame: int, timeout_s: float) -> Optional[carla.Image]:
        nonlocal cached_image

        if cached_image is not None:
            f = int(getattr(cached_image, "frame", -1))
            if f == int(target_frame):
                img = cached_image
                cached_image = None
                return img
            if f < int(target_frame):
                cached_image = None

        # Wait for the exact target frame.
        deadline = datetime.now().timestamp() + float(timeout_s)
        while datetime.now().timestamp() < deadline:
            remaining = max(0.01, deadline - datetime.now().timestamp())
            try:
                img = image_queue.get(timeout=remaining)
            except queue.Empty:
                break

            f = int(getattr(img, "frame", -1))
            if f < int(target_frame):
                continue
            if f == int(target_frame):
                return img

            # f > target_frame: we missed the exact frame; stash this one for the next step.
            cached_image = img
            break

        return None

    def _on_image(image: carla.Image):
        try:
            image_queue.put_nowait(image)
        except queue.Full:
            try:
                _ = image_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                image_queue.put_nowait(image)
            except queue.Full:
                pass

    try:
        obs = env.reset()

        # Spawn RGB camera with the same settings as dataset collection.
        cam_bp = env.blueprint_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(int(args.camera_width)))
        cam_bp.set_attribute("image_size_y", str(int(args.camera_height)))
        cam_bp.set_attribute("fov", str(float(args.camera_fov)))
        # Try to align camera cadence with synchronous ticks.
        try:
            cam_bp.set_attribute("sensor_tick", str(float(args.fixed_delta_seconds)))
        except Exception:
            pass

        cam_transform = carla.Transform(
            carla.Location(x=1.6, z=1.4),
            carla.Rotation(pitch=-5.0),
        )
        camera = env.world.spawn_actor(cam_bp, cam_transform, attach_to=env.vehicle)
        camera.listen(_on_image)

        # Warm up a few ticks so the sensor starts producing frames.
        for _ in range(3):
            env.world.tick()

        for step in range(int(args.max_steps)):
            snapshot = env.world.get_snapshot()
            frame = int(snapshot.frame)

            image = _get_image_strict(frame, timeout_s=float(args.image_timeout_s))
            if image is None:
                print(f"[warn] image frame not available/aligned for frame={frame}; using env target for this step")
                # Fallback: use env's own target point (keeps control stable, avoids silent misalignment).
                action = controller.get_control(obs)
            else:
                pil_img = _carla_image_to_pil_rgb(image)
                image_t = _imagenet_preprocess_pil_rgb(pil_img, image_size=224).unsqueeze(0).to(device)

                state_t = None
                if use_state:
                    state_t = _get_state_xy_yaw_sincos(env, xy_scale=state_xy_scale).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(image_t, state=state_t) if use_state else model(image_t)
                pred_pts = out["points"][0].detach().cpu().numpy()  # (15,2)

                if bool(args.flip_pred_y):
                    pred_pts = pred_pts.copy()
                    pred_pts[:, 1] *= -1.0

                la_idx = int(np.clip(int(args.lookahead_index), 0, pred_pts.shape[0] - 1))
                target_x = float(pred_pts[la_idx, 0])
                target_y = float(pred_pts[la_idx, 1])

                # Compare with env's own lookahead target (10m by default) to debug sign issues.
                env_target_x = float(obs[7]) if len(obs) >= 9 else 0.0
                env_target_y = float(obs[8]) if len(obs) >= 9 else 0.0

                # If predicted point is behind / too close, fallback to env target to avoid circling.
                if not np.isfinite(target_x) or not np.isfinite(target_y) or float(target_x) < float(args.min_target_x):
                    target_x, target_y = env_target_x, env_target_y

                if int(args.print_pred_every) > 0 and (step % int(args.print_pred_every) == 0):
                    print(
                        f"pred_target=[{target_x:+.2f},{target_y:+.2f}] "
                        f"env_target=[{env_target_x:+.2f},{env_target_y:+.2f}] "
                        f"flip_y={bool(args.flip_pred_y)}"
                    )

                # Override obs target point so AdaptiveController uses PurePursuit.
                obs_for_ctl = np.array(obs, dtype=np.float32, copy=True)
                if obs_for_ctl.shape[0] < 9:
                    # shouldn't happen with current env, but keep safe
                    obs_for_ctl = np.pad(obs_for_ctl, (0, 9 - obs_for_ctl.shape[0]), mode="constant")
                obs_for_ctl[7] = target_x
                obs_for_ctl[8] = target_y

                action = controller.get_control(obs_for_ctl)

                if bool(args.debug_draw_pred) and (step % max(1, int(args.debug_draw_every)) == 0):
                    try:
                        transform = env.vehicle.get_transform()
                        life_time = (
                            float(args.debug_draw_life_time)
                            if args.debug_draw_life_time is not None
                            else max(0.001, float(args.fixed_delta_seconds) * 0.9)
                        )
                        _draw_pred_path(env.world, transform, pred_pts, life_time=life_time)

                        # Draw env target (cyan) and the actual control target (yellow)
                        try:
                            env_tgt_world = _vehicle_xy_to_world(transform, env_target_x, env_target_y)
                            env.world.debug.draw_point(
                                env_tgt_world,
                                size=0.10,
                                color=carla.Color(0, 255, 255),
                                life_time=life_time,
                            )
                            ctl_tgt_world = _vehicle_xy_to_world(transform, float(obs_for_ctl[7]), float(obs_for_ctl[8]))
                            env.world.debug.draw_point(
                                ctl_tgt_world,
                                size=0.10,
                                color=carla.Color(255, 255, 0),
                                life_time=life_time,
                            )
                        except Exception:
                            pass
                    except RuntimeError:
                        pass

            obs, reward, done, info = env.step(action)

            if step % 50 == 0:
                dist_goal = info.get("distance_to_goal", None)
                dist_goal_str = f"{dist_goal:.2f}m" if isinstance(dist_goal, (int, float)) else "N/A"
                speed = float(obs[6]) if len(obs) >= 7 else 0.0
                print(
                    f"step={step} frame={frame} speed={speed:.2f} "
                    f"action=[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f}] dist2goal={dist_goal_str}"
                )

            if done:
                print("Episode done")
                break

    finally:
        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
