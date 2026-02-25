"""Gym-style RL environment for fine-tuning the transformer path planner in CARLA.

Design choices aligned with the project's closed-loop test:
  - Observation: front RGB image, preprocessed exactly like test_nn_path_planner_control.py
  - Action: flattened local path points (15,2) in vehicle frame (x,y), produced by the transformer
  - Control: use AdaptiveController (PurePursuit + Speed PID) on a selected lookahead point

This env adds:
  - Lane invasion + collision sensors (for penalties / hard termination)
  - Soft lane-change penalty and lane-centering shaping
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import random

# Prefer gym (SB3 v1.x / old API), fallback to gymnasium only for import-time hints.
try:
    import gym  # type: ignore
except Exception:  # pragma: no cover
    gym = None  # type: ignore

try:
    import carla  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Failed to import CARLA PythonAPI (carla/libcarla). "
        "Please run with a Python version compatible with your CARLA egg/wheel.\n\n"
        f"Original error: {e}"
    )

from carla_env import CarlaEnv
from pid_controller import AdaptiveController


def _to_xyz_dict(v: Any) -> Optional[Dict[str, float]]:
    try:
        x = float(getattr(v, "x"))
        y = float(getattr(v, "y"))
        z = float(getattr(v, "z"))
        return {"x": x, "y": y, "z": z}
    except Exception:
        return None


def _safe_info_from_carla_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert CarlaEnv.step() info to deepcopy-safe Python types.

    SB3 DummyVecEnv deep-copies infos; carla.libcarla.* objects are not picklable.
    """

    out: Dict[str, Any] = {}

    out["frame"] = info.get("frame", None)
    out["sim_time"] = info.get("sim_time", None)
    out["distance_to_goal"] = info.get("distance_to_goal", None)

    # control is already a list of floats
    out["control"] = info.get("control", None)

    goal_loc = info.get("goal_location", None)
    goal_xyz = _to_xyz_dict(goal_loc)
    out["goal_location"] = goal_xyz

    # Optional: include current location/velocity if needed, but as plain dicts
    loc_xyz = _to_xyz_dict(info.get("location", None))
    if loc_xyz is not None:
        out["location"] = loc_xyz

    vel_xyz = _to_xyz_dict(info.get("velocity", None))
    if vel_xyz is not None:
        out["velocity"] = vel_xyz

    acc_xyz = _to_xyz_dict(info.get("acceleration", None))
    if acc_xyz is not None:
        out["acceleration"] = acc_xyz

    tgt_xyz = _to_xyz_dict(info.get("target_location", None))
    if tgt_xyz is not None:
        out["target_location"] = tgt_xyz

    return out


def _carla_image_to_pil_rgb(image: carla.Image):
    from PIL import Image

    w = int(image.width)
    h = int(image.height)
    data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((h, w, 4))
    bgr = data[:, :, :3]
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def _imagenet_preprocess_pil_rgb(pil_img, *, image_size: int = 224) -> np.ndarray:
    """Match test_nn_path_planner_control._imagenet_preprocess_pil_rgb exactly."""

    img = pil_img.convert("RGB")
    img = img.resize((int(image_size), int(image_size)))

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std

    return arr.astype(np.float32)


@dataclass
class RewardConfig:
    lane_center_w: float = 2.0
    lane_center_k: float = 3.0

    heading_w: float = 1.0

    speed_w: float = 0.3
    speed_k: float = 0.8

    steer_smooth_w: float = 0.2

    step_w: float = 0.0

    lane_change_penalty: float = -0.2

    success_bonus: float = 20.0
    collision_penalty: float = -20.0
    offroad_penalty: float = -20.0
    wrong_way_penalty: float = -20.0


class CarlaPathFollowingRLEnv(gym.Env):
    """Fine-tuning env.

    Old Gym API (obs, reward, done, info) for stable-baselines3 v1.x compatibility.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        town: str = "Town03",
        spawn_point_index: int = 0,
        destination_index: int = 1,
        goal_radius: float = 3.0,
        max_episode_steps: int = 1000,
        synchronous_mode: bool = True,
        fixed_delta_seconds: float = 0.05,
        target_speed: float = 5.0,
        lookahead_index: int = 5,
        min_target_x: float = 0.5,
        image_timeout_s: float = 1.0,
        camera_width: int = 800,
        camera_height: int = 600,
        camera_fov: float = 90.0,
        spectator_follow: bool = False,
        include_state: bool = False,
        state_xy_scale: float = 100.0,
        reward_cfg: Optional[RewardConfig] = None,
        debug: bool = False,
    ):
        if gym is None:
            raise RuntimeError("gym is required. Please install gym (or stable-baselines3 dependencies).")

        super().__init__()

        self.debug = bool(debug)

        self.target_speed = float(target_speed)
        self.lookahead_index = int(lookahead_index)
        self.min_target_x = float(min_target_x)
        self.image_timeout_s = float(image_timeout_s)
        self.include_state = bool(include_state)
        self.state_xy_scale = float(state_xy_scale)

        self.reward_cfg = reward_cfg or RewardConfig()
        self.spectator_follow = bool(spectator_follow)

        # Observation: Imagenet-normalized RGB tensor
        self._image_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(3, 224, 224),
            dtype=np.float32,
        )

        self._state_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space = (
            gym.spaces.Dict({"image": self._image_space, "state": self._state_space})
            if self.include_state
            else self._image_space
        )

        # Action: flattened 15x2 local path (vehicle frame, meters)
        self.action_space = gym.spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(30,),
            dtype=np.float32,
        )

        # Underlying CARLA env.
        self.env = CarlaEnv(
            town=str(town),
            spawn_point_index=int(spawn_point_index),
            destination_index=int(destination_index),
            goal_radius=float(goal_radius),
            max_episode_steps=int(max_episode_steps),
            synchronous_mode=bool(synchronous_mode),
            fixed_delta_seconds=float(fixed_delta_seconds),
            collect_dataset=False,
            enable_camera=True,
            enable_lane_invasion_sensor=True,
            enable_collision_sensor=True,
            camera_width=int(camera_width),
            camera_height=int(camera_height),
            camera_fov=float(camera_fov),
            reward_fn=lambda _env: 0.0,
            debug_draw=False,
            spectator_follow=bool(self.spectator_follow),
        )

        self.controller = AdaptiveController(target_speed=float(target_speed), dt=float(fixed_delta_seconds))

        self._prev_steer: float = 0.0
        self._prev_lane_id: Optional[int] = None
        self._ref_lane_sign: Optional[int] = None
        self._last_obs_image: Optional[np.ndarray] = None

        # Gym (old API) seeding
        self._np_random = np.random.RandomState()

    def seed(self, seed: Optional[int] = None):
        """OpenAI Gym-style seeding.

        shimmy calls env.seed(seed) when SB3 passes reset(seed=...).
        """
        if seed is None:
            seed = int(datetime.now().timestamp() * 1e6) % (2**31 - 1)
        seed = int(seed)

        try:
            self._np_random = np.random.RandomState(seed)
            np.random.seed(seed)
        except Exception:
            pass

        try:
            random.seed(seed)
        except Exception:
            pass

        return [seed]

    def _get_wp_any(self) -> Optional[carla.Waypoint]:
        if self.env.vehicle is None:
            return None
        try:
            loc = self.env.vehicle.get_location()
        except RuntimeError:
            return None

        # Try most explicit signature first.
        try:
            return self.env.map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any)
        except Exception:
            pass

        try:
            return self.env.map.get_waypoint(loc)
        except Exception:
            return None

    def _is_illegal_area(self, wp: Optional[carla.Waypoint]) -> Tuple[bool, str]:
        if wp is None:
            return True, "offroad"

        try:
            lane_type = wp.lane_type
        except Exception:
            lane_type = carla.LaneType.Any

        if lane_type != carla.LaneType.Driving:
            return True, f"lane_type={lane_type}"

        if self._ref_lane_sign is not None:
            try:
                cur_sign = int(np.sign(int(wp.lane_id)))
            except Exception:
                cur_sign = 0
            if cur_sign != int(self._ref_lane_sign):
                return True, "opposite_direction"

        return False, "ok"

    def _get_image_obs(self, *, target_frame: Optional[int]) -> np.ndarray:
        if target_frame is None:
            # Fallback to last known obs if available.
            if self._last_obs_image is not None:
                return self._last_obs_image
            return np.zeros(self._image_space.shape, dtype=np.float32)

        img = self.env.get_rgb_image_for_frame(target_frame=int(target_frame), timeout_s=float(self.image_timeout_s))
        if img is None:
            if self._last_obs_image is not None:
                return self._last_obs_image
            return np.zeros(self._image_space.shape, dtype=np.float32)

        pil_img = _carla_image_to_pil_rgb(img)
        obs = _imagenet_preprocess_pil_rgb(pil_img, image_size=224)
        self._last_obs_image = obs
        return obs

    def _get_state_obs(self) -> np.ndarray:
        """Match test_nn_path_planner_control._get_state_xy_yaw_sincos."""
        if self.env.vehicle is None:
            return np.zeros((4,), dtype=np.float32)
        try:
            tr = self.env.vehicle.get_transform()
            x = float(tr.location.x) / max(float(self.state_xy_scale), 1e-6)
            y = float(tr.location.y) / max(float(self.state_xy_scale), 1e-6)
            yaw_deg = float(tr.rotation.yaw)
        except Exception:
            return np.zeros((4,), dtype=np.float32)

        yaw_rad = float(np.deg2rad(yaw_deg))
        return np.array([x, y, float(np.sin(yaw_rad)), float(np.cos(yaw_rad))], dtype=np.float32)

    def _pack_obs(self, *, image_obs: np.ndarray) -> Any:
        if not self.include_state:
            return image_obs
        return {"image": image_obs, "state": self._get_state_obs()}

    def reset(self) -> Any:
        self.controller.reset()
        self._prev_steer = 0.0
        self._prev_lane_id = None
        self._last_obs_image = None

        _ = self.env.reset()

        wp0 = self._get_wp_any()
        try:
            self._ref_lane_sign = int(np.sign(int(wp0.lane_id))) if wp0 is not None else None
        except Exception:
            self._ref_lane_sign = None

        frame = None
        try:
            snap = self.env.world.get_snapshot()
            frame = int(snap.frame)
        except Exception:
            frame = None

        img_obs = self._get_image_obs(target_frame=frame)
        return self._pack_obs(image_obs=img_obs)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        pts = action.reshape(15, 2)
        idx = int(np.clip(self.lookahead_index, 0, 14))
        target_x, target_y = float(pts[idx, 0]), float(pts[idx, 1])

        # If target is invalid, fallback to route-based target from env observation.
        obs9_pre = self.env._get_observation()
        if target_x < float(self.min_target_x):
            target_x = float(obs9_pre[7])
            target_y = float(obs9_pre[8])

        obs9_for_control = np.array(obs9_pre, dtype=np.float32)
        if obs9_for_control.shape[0] >= 9:
            obs9_for_control[7] = float(target_x)
            obs9_for_control[8] = float(target_y)

        control = self.controller.get_control(obs9_for_control)

        _obs9, _unused_reward, env_done, carla_info = self.env.step(control)

        # Gather events
        lane_events = self.env.pop_lane_invasion_events()
        collision_events = self.env.pop_collision_events()

        frame = carla_info.get("frame", None)
        obs_img = self._get_image_obs(target_frame=frame)
        obs_out = self._pack_obs(image_obs=obs_img)

        # Termination checks
        done = bool(env_done)
        term_reason = None

        if collision_events:
            done = True
            term_reason = "collision"

        wp = self._get_wp_any()
        illegal, illegal_reason = self._is_illegal_area(wp)
        if illegal:
            done = True
            term_reason = illegal_reason

        # Reward shaping
        cfg = self.reward_cfg
        obs9 = self.env._get_observation()
        lateral_error = float(abs(obs9[4]))
        heading_error = float(obs9[5])
        speed = float(obs9[6])

        r_lane = float(np.exp(-cfg.lane_center_k * lateral_error)) * cfg.lane_center_w
        r_head = float(np.cos(heading_error)) * cfg.heading_w
        r_speed = float(np.exp(-cfg.speed_k * abs(speed - self.target_speed))) * cfg.speed_w

        steer = float(control[2])
        r_smooth = -abs(steer - float(self._prev_steer)) * cfg.steer_smooth_w
        self._prev_steer = steer

        r_step = float(cfg.step_w)

        # Lane-change (soft): penalize if lane_id changed OR lane invasion happened within driving area.
        lane_change = False
        if wp is not None:
            try:
                cur_lane_id = int(wp.lane_id)
            except Exception:
                cur_lane_id = None
            if cur_lane_id is not None and self._prev_lane_id is not None and cur_lane_id != int(self._prev_lane_id):
                lane_change = True
            if cur_lane_id is not None:
                self._prev_lane_id = int(cur_lane_id)

        if (not lane_change) and lane_events:
            lane_change = True

        r_lane_change = float(cfg.lane_change_penalty) if lane_change else 0.0

        reward = float(r_lane + r_head + r_speed + r_smooth + r_step + r_lane_change)

        # Terminal bonuses/penalties
        dist_to_goal = carla_info.get("distance_to_goal", None)
        if dist_to_goal is not None and dist_to_goal <= float(self.env.goal_radius):
            reward += float(cfg.success_bonus)

        if term_reason == "collision":
            reward += float(cfg.collision_penalty)
        elif term_reason == "offroad":
            reward += float(cfg.offroad_penalty)
        elif term_reason == "opposite_direction":
            reward += float(cfg.wrong_way_penalty)
        elif term_reason and term_reason.startswith("lane_type="):
            reward += float(cfg.offroad_penalty)

        info = _safe_info_from_carla_info(dict(carla_info))
        info.update(
            {
                "rl": {
                    "lane_events": int(len(lane_events)),
                    "collision_events": int(len(collision_events)),
                    "illegal_reason": illegal_reason,
                    "term_reason": term_reason,
                    "lateral_error": float(lateral_error),
                    "heading_error": float(heading_error),
                    "speed": float(speed),
                    "reward_terms": {
                        "lane": float(r_lane),
                        "heading": float(r_head),
                        "speed": float(r_speed),
                        "smooth": float(r_smooth),
                        "lane_change": float(r_lane_change),
                    },
                }
            }
        )

        return obs_out, float(reward), bool(done), info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

