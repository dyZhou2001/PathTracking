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
    lane_center_w: float = 1.8
    lane_center_k: float = 3.0
    lane_edge_penalty_w: float = 1.2
    lane_edge_soft_ratio: float = 0.45

    heading_w: float = 1.0

    speed_w: float = 0.3
    speed_k: float = 0.8

    steer_smooth_w: float = 0.4
    
    path_smoothness_w: float = 0.2
    path_smoothness_clip: float = 3.0

    progress_w: float = 5.0
    progress_clip: float = 1.5

    step_w: float = 0.0

    lane_change_penalty: float = -0.2
    broken_line_cross_penalty: float = -0.03
    solid_line_cross_penalty: float = -0.05

    success_bonus: float = 180.0
    collision_penalty: float = -120.0
    offroad_penalty: float = -120.0
    wrong_way_penalty: float = -120.0


@dataclass
class ViolationConfig:
    # Junction entrance grace (steps) to avoid false hard-kill when entering roundabout.
    junction_grace_steps: int = 20

    # Two-level state-machine thresholds
    soft_score_threshold: float = 1.0
    hard_score_threshold: float = 2.4
    soft_depth_threshold: float = 0.25
    hard_depth_threshold: float = 0.75

    soft_consecutive_steps: int = 4
    hard_consecutive_steps: int = 16
    clear_consecutive_steps: int = 6

    illegal_area_consecutive_steps: int = 8
    wrong_way_consecutive_steps: int = 8

    # Multi-signal weights (lane_type + lane_event + inner_depth)
    weight_lane_type: float = 1.0
    weight_lane_event: float = 0.3
    weight_inner_depth: float = 1.2

    # Reward shaping for suspected inner-shoulder behavior
    soft_penalty_scale: float = 0.35
    hard_penalty: float = -120.0

    # Ignore tiny steering intent near straight driving
    turn_sign_min_abs_target_y: float = 0.25


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
        violation_cfg: Optional[ViolationConfig] = None,
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
        self.violation_cfg = violation_cfg or ViolationConfig()
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
        self._prev_road_id: Optional[int] = None
        self._last_obs_image: Optional[np.ndarray] = None

        self._violation_soft_count: int = 0
        self._violation_hard_count: int = 0
        self._violation_clear_count: int = 0
        self._violation_state: str = "normal"

        self._prev_in_junction: bool = False
        self._junction_grace_remaining: int = 0

        self._illegal_area_count: int = 0
        self._wrong_way_count: int = 0
        self._prev_distance_to_goal: Optional[float] = None
        self._episode_reward_sum: float = 0.0
        self._episode_step_count: int = 0

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

        try:
            # project_to_road=True prevents wp=None while keeping lane type information.
            return self.env.map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Any,
            )
        except Exception:
            try:
                return self.env.map.get_waypoint(loc)
            except Exception:
                return None

    @staticmethod
    def _is_shoulder_like_lane(wp: Optional[carla.Waypoint]) -> bool:
        if wp is None:
            return False
        try:
            lane_type = wp.lane_type
        except Exception:
            return False
        return lane_type in [
            carla.LaneType.Shoulder,
            carla.LaneType.Sidewalk,
            carla.LaneType.Median,
        ]

    @staticmethod
    def _lane_event_score(lane_events: list) -> float:
        if not lane_events:
            return 0.0

        best = 0.1
        for ev in lane_events:
            markings = getattr(ev, "crossed_lane_markings", None) or []
            if not markings:
                best = max(best, 0.2)
                continue
            for mk in markings:
                mtype = str(getattr(mk, "type", ""))
                if ("Solid" in mtype) or ("Curb" in mtype):
                    best = max(best, 1.0)
                elif "Broken" in mtype:
                    best = max(best, 0.15)
                else:
                    best = max(best, 0.3)
        return float(best)

    def _lane_event_penalty(self, lane_events: list) -> Tuple[float, bool, bool]:
        """Return (penalty, has_broken_cross, has_solid_cross).

        Town03 roundabout has frequent dashed-line riding. We use a mild penalty for broken-line
        crossings and a stronger penalty for solid/curb crossings.
        """
        has_broken = False
        has_solid = False
        for ev in lane_events:
            markings = getattr(ev, "crossed_lane_markings", None) or []
            for mk in markings:
                mtype = str(getattr(mk, "type", ""))
                if ("Solid" in mtype) or ("Curb" in mtype):
                    has_solid = True
                elif "Broken" in mtype:
                    has_broken = True

        penalty = 0.0
        if has_broken:
            penalty += float(getattr(self.reward_cfg, "broken_line_cross_penalty", -0.03))
        if has_solid:
            penalty += float(self.reward_cfg.solid_line_cross_penalty)
        return penalty, bool(has_broken), bool(has_solid)

    def _update_junction_grace(self, wp: Optional[carla.Waypoint]) -> Tuple[bool, bool]:
        in_junction = False
        if wp is not None:
            try:
                in_junction = bool(wp.is_junction)
            except Exception:
                in_junction = False

        if in_junction and (not self._prev_in_junction):
            self._junction_grace_remaining = max(0, int(self.violation_cfg.junction_grace_steps))

        if not in_junction:
            self._junction_grace_remaining = 0

        in_grace = in_junction and (self._junction_grace_remaining > 0)
        if in_grace:
            self._junction_grace_remaining = max(0, int(self._junction_grace_remaining) - 1)

        self._prev_in_junction = in_junction
        return in_junction, in_grace

    def _estimate_inner_depth(self, *, lateral_error_signed: float, target_y: float, wp: Optional[carla.Waypoint]) -> float:
        cfg = self.violation_cfg

        if abs(float(target_y)) < float(cfg.turn_sign_min_abs_target_y):
            return 0.0

        turn_sign = 1.0 if float(target_y) > 0.0 else -1.0
        inner_offset = max(0.0, -turn_sign * float(lateral_error_signed))

        lane_half_width = 1.75
        if wp is not None:
            try:
                lane_half_width = max(0.5 * float(wp.lane_width), 0.5)
            except Exception:
                lane_half_width = 1.75

        return float(inner_offset / lane_half_width)

    def _update_inner_violation_state(
        self,
        *,
        wp: Optional[carla.Waypoint],
        lane_events: list,
        lateral_error_signed: float,
        target_y: float,
    ) -> Dict[str, Any]:
        cfg = self.violation_cfg

        in_junction, in_junction_grace = self._update_junction_grace(wp)

        lane_type_score = 1.0 if self._is_shoulder_like_lane(wp) else 0.0
        lane_event_score = self._lane_event_score(lane_events)
        inner_depth = self._estimate_inner_depth(
            lateral_error_signed=float(lateral_error_signed),
            target_y=float(target_y),
            wp=wp,
        )

        soft_depth = max(float(cfg.soft_depth_threshold), 1e-6)
        hard_depth = max(float(cfg.hard_depth_threshold), soft_depth)

        if inner_depth <= 0.0:
            depth_score = 0.0
        elif inner_depth >= hard_depth:
            depth_score = 1.0
        else:
            depth_score = float(np.clip(inner_depth / soft_depth, 0.0, 1.0))

        score = float(
            float(cfg.weight_lane_type) * lane_type_score
            + float(cfg.weight_lane_event) * lane_event_score
            + float(cfg.weight_inner_depth) * depth_score
        )

        soft_signal = (
            (score >= float(cfg.soft_score_threshold))
            or (inner_depth >= float(cfg.soft_depth_threshold))
            or (lane_type_score > 0.0 and lane_event_score > 0.0)
        )
        hard_signal = (score >= float(cfg.hard_score_threshold)) or (inner_depth >= float(cfg.hard_depth_threshold))

        if hard_signal:
            self._violation_hard_count += 1
        else:
            self._violation_hard_count = 0

        if soft_signal:
            self._violation_soft_count += 1
            self._violation_clear_count = 0
        else:
            self._violation_clear_count += 1
            if self._violation_clear_count >= int(cfg.clear_consecutive_steps):
                self._violation_soft_count = 0

        soft_confirmed = self._violation_soft_count >= int(cfg.soft_consecutive_steps)
        hard_confirmed = self._violation_hard_count >= int(cfg.hard_consecutive_steps)

        if hard_confirmed:
            self._violation_state = "violation"
        elif soft_confirmed:
            self._violation_state = "suspect"
        else:
            self._violation_state = "normal"

        soft_penalty = 0.0
        if soft_signal:
            soft_penalty = -float(cfg.soft_penalty_scale) * max(0.0, score)

        return {
            "score": float(score),
            "lane_type_score": float(lane_type_score),
            "lane_event_score": float(lane_event_score),
            "depth_score": float(depth_score),
            "inner_depth": float(inner_depth),
            "soft_signal": bool(soft_signal),
            "hard_signal": bool(hard_signal),
            "soft_confirmed": bool(soft_confirmed),
            "hard_confirmed": bool(hard_confirmed),
            "soft_penalty": float(soft_penalty),
            "in_junction": bool(in_junction),
            "in_junction_grace": bool(in_junction_grace),
            "state": str(self._violation_state),
            "soft_count": int(self._violation_soft_count),
            "hard_count": int(self._violation_hard_count),
            "clear_count": int(self._violation_clear_count),
        }

    def _is_illegal_area(self, wp: Optional[carla.Waypoint]) -> Tuple[bool, str]:
        """
        更鲁棒的可驾驶区域判断：
        - Junction 内放宽规则
        - 允许 Driving / Shoulder
        - 只在真正非法区域终止
        """

        if wp is None:
            return True, "offroad"

        try:
            lane_type = wp.lane_type
        except Exception:
            lane_type = carla.LaneType.Any

        # ===============================
        # 1️⃣ Junction 内放宽规则
        # ===============================
        try:
            if wp.is_junction:
                # 在环岛/路口区域，不因为 lane_type 直接终止
                return False, "junction_allowed"
        except Exception:
            pass

        # ===============================
        # 2️⃣ 明确非法区域
        # ===============================
        illegal_lane_types = [
            carla.LaneType.Sidewalk,
            carla.LaneType.Median,
            carla.LaneType.Bidirectional,
            carla.LaneType.Tram,
            carla.LaneType.Rail,
        ]

        if lane_type in illegal_lane_types:
            return True, f"lane_type={lane_type}"

        # ===============================
        # 3️⃣ 允许 Driving 和 Shoulder
        # ===============================
        if lane_type in [
            carla.LaneType.Driving,
            carla.LaneType.Shoulder,
            carla.LaneType.Parking
        ]:
            pass
        else:
            # 其他未知类型，保守处理
            return True, f"lane_type={lane_type}"

        # ===============================
        # 4️⃣ 反向行驶检测（放宽阈值）
        # ===============================
        if self.env.vehicle is not None:
            try:
                v_vec = self.env.vehicle.get_transform().get_forward_vector()
                wp_vec = wp.transform.get_forward_vector()
                dot = v_vec.x * wp_vec.x + v_vec.y * wp_vec.y

                # 只在严重反向才终止
                if dot < -0.8:
                    return True, "opposite_direction"
            except Exception:
                pass

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
        self._prev_road_id = None
        self._last_obs_image = None

        self._violation_soft_count = 0
        self._violation_hard_count = 0
        self._violation_clear_count = 0
        self._violation_state = "normal"
        self._prev_in_junction = False
        self._junction_grace_remaining = 0
        self._illegal_area_count = 0
        self._wrong_way_count = 0
        self._prev_distance_to_goal = None
        self._episode_reward_sum = 0.0
        self._episode_step_count = 0

        _ = self.env.reset()

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
        if illegal_reason == "opposite_direction":
            self._wrong_way_count = int(self._wrong_way_count) + 1 if illegal else 0
            self._illegal_area_count = 0
        elif illegal:
            self._illegal_area_count = int(self._illegal_area_count) + 1
            self._wrong_way_count = 0
        else:
            self._illegal_area_count = 0
            self._wrong_way_count = 0

        if (not done) and illegal and (illegal_reason == "opposite_direction"):
            if self._wrong_way_count >= int(self.violation_cfg.wrong_way_consecutive_steps):
                done = True
                term_reason = illegal_reason

        if (not done) and illegal and (illegal_reason != "opposite_direction"):
            if self._illegal_area_count >= int(self.violation_cfg.illegal_area_consecutive_steps):
                done = True
                term_reason = illegal_reason

        # Reward shaping
        cfg = self.reward_cfg
        obs9 = self.env._get_observation()
        lateral_error_signed = float(obs9[4])
        lateral_error = float(abs(lateral_error_signed))
        heading_error = float(obs9[5])
        speed = float(obs9[6])

        violation_diag = self._update_inner_violation_state(
            wp=wp,
            lane_events=lane_events,
            lateral_error_signed=float(lateral_error_signed),
            target_y=float(target_y),
        )

        if (not done) and violation_diag["hard_confirmed"] and (not violation_diag["in_junction"]) and (not violation_diag["in_junction_grace"]):
            done = True
            term_reason = "inner_shoulder_violation"

        r_lane = float(np.exp(-cfg.lane_center_k * lateral_error)) * cfg.lane_center_w

        lane_half_width = 1.75
        if wp is not None:
            try:
                lane_half_width = max(0.5 * float(wp.lane_width), 0.5)
            except Exception:
                lane_half_width = 1.75
        lane_ratio = float(np.clip(lateral_error / lane_half_width, 0.0, 1.5))
        edge_soft = float(np.clip(getattr(cfg, "lane_edge_soft_ratio", 0.45), 0.05, 0.95))
        if lane_ratio <= edge_soft:
            r_lane_edge = 0.0
        else:
            norm = float((lane_ratio - edge_soft) / max(1.0 - edge_soft, 1e-6))
            r_lane_edge = -float(getattr(cfg, "lane_edge_penalty_w", 1.2)) * float(np.clip(norm * norm, 0.0, 1.0))

        r_head = float(np.cos(heading_error)) * cfg.heading_w
        r_speed = float(np.exp(-cfg.speed_k * abs(speed - self.target_speed))) * cfg.speed_w

        steer = float(control[2])
        r_smooth = -abs(steer - float(self._prev_steer)) * cfg.steer_smooth_w
        self._prev_steer = steer

        # Path Smoothness penalty (use mean second-order difference + clip to avoid dominating total return)
        if pts.shape[0] >= 3:
            diff1 = pts[1:] - pts[:-1]
            diff2 = diff1[1:] - diff1[:-1]
            path_jerk = float(np.mean(np.linalg.norm(diff2, axis=1)))
            path_jerk = float(np.clip(path_jerk, 0.0, float(getattr(cfg, "path_smoothness_clip", 3.0))))
            r_path_smooth = -path_jerk * getattr(cfg, "path_smoothness_w", 0.5)
        else:
            r_path_smooth = 0.0

        r_step = float(cfg.step_w)

        cur_dist_to_goal = carla_info.get("distance_to_goal", None)
        r_progress = 0.0
        if (cur_dist_to_goal is not None) and (self._prev_distance_to_goal is not None):
            progress_m = float(self._prev_distance_to_goal) - float(cur_dist_to_goal)
            progress_m = float(np.clip(progress_m, -float(cfg.progress_clip), float(cfg.progress_clip)))
            r_progress = float(cfg.progress_w) * progress_m
        if cur_dist_to_goal is not None:
            self._prev_distance_to_goal = float(cur_dist_to_goal)

        # Lane-change (soft): penalize if lane_id changed OR lane invasion happened within driving area.
        lane_change = False
        if wp is not None:
            try:
                cur_lane_id = int(wp.lane_id)
                cur_road_id = int(wp.road_id)
            except Exception:
                cur_lane_id = None
                cur_road_id = None
                
            if cur_lane_id is not None and self._prev_lane_id is not None:
                # Only penalize lane id change if we are on the same road id 
                # (entering intersection/new road segment naturally changes lane IDs)
                if cur_road_id == self._prev_road_id and cur_lane_id != self._prev_lane_id:
                    lane_change = True
            
            if cur_lane_id is not None:
                self._prev_lane_id = cur_lane_id
            if cur_road_id is not None:
                self._prev_road_id = cur_road_id

        r_solid_cross, crossed_broken, crossed_solid = self._lane_event_penalty(lane_events)

        r_lane_change = float(cfg.lane_change_penalty) if lane_change else 0.0
        r_inner_soft = float(violation_diag["soft_penalty"])

        reward = float(r_lane + r_lane_edge + r_head + r_speed + r_smooth + r_path_smooth + r_step + r_progress + r_lane_change + r_solid_cross + r_inner_soft)
        self._episode_reward_sum += float(reward)
        self._episode_step_count += 1

        # Terminal bonuses/penalties
        dist_to_goal = carla_info.get("distance_to_goal", None)
        if dist_to_goal is not None and dist_to_goal <= float(self.env.goal_radius):
            if term_reason is None:
                term_reason = "success"
            reward += float(cfg.success_bonus)

        if term_reason == "collision":
            reward += float(cfg.collision_penalty)
        elif term_reason == "offroad":
            reward += float(cfg.offroad_penalty)
        elif term_reason == "opposite_direction":
            reward += float(cfg.wrong_way_penalty)
        elif term_reason and term_reason.startswith("lane_type="):
            reward += float(cfg.offroad_penalty)
        elif term_reason == "inner_shoulder_violation":
            reward += float(self.violation_cfg.hard_penalty)

        info = _safe_info_from_carla_info(dict(carla_info))
        info.update(
            {
                "rl": {
                    "lane_events": int(len(lane_events)),
                    "collision_events": int(len(collision_events)),
                    "illegal_reason": illegal_reason,
                    "term_reason": term_reason,
                    "lateral_error": float(lateral_error),
                    "lateral_error_signed": float(lateral_error_signed),
                    "heading_error": float(heading_error),
                    "speed": float(speed),
                    "inner_violation": {
                        "state": violation_diag["state"],
                        "score": float(violation_diag["score"]),
                        "inner_depth": float(violation_diag["inner_depth"]),
                        "soft_signal": bool(violation_diag["soft_signal"]),
                        "hard_signal": bool(violation_diag["hard_signal"]),
                        "soft_confirmed": bool(violation_diag["soft_confirmed"]),
                        "hard_confirmed": bool(violation_diag["hard_confirmed"]),
                        "in_junction": bool(violation_diag["in_junction"]),
                        "in_junction_grace": bool(violation_diag["in_junction_grace"]),
                        "soft_count": int(violation_diag["soft_count"]),
                        "hard_count": int(violation_diag["hard_count"]),
                        "clear_count": int(violation_diag["clear_count"]),
                        "score_terms": {
                            "lane_type": float(violation_diag["lane_type_score"]),
                            "lane_event": float(violation_diag["lane_event_score"]),
                            "depth": float(violation_diag["depth_score"]),
                        },
                    },
                    "reward_terms": {
                        "lane": float(r_lane),
                        "lane_edge": float(r_lane_edge),
                        "heading": float(r_head),
                        "speed": float(r_speed),
                        "smooth": float(r_smooth),
                        "path_smooth": float(r_path_smooth),
                        "progress": float(r_progress),
                        "lane_change": float(r_lane_change),
                        "solid_line_cross": float(r_solid_cross),
                        "inner_shoulder_soft": float(r_inner_soft),
                    },
                    "diagnostics": {
                        "crossed_broken_line": bool(crossed_broken),
                        "crossed_solid_line": bool(crossed_solid),
                        "illegal_area_count": int(self._illegal_area_count),
                        "wrong_way_count": int(self._wrong_way_count),
                    },
                }
            }
        )

        if done:
            print(
                "[EPISODE_REWARD] "
                f"steps={self._episode_step_count} "
                f"episode_reward={self._episode_reward_sum:.3f} "
                f"last_step_reward={float(reward):.3f} "
                f"term_reason={term_reason}"
            )
            print(
                "[EPISODE_REWARD_TERMS] "
                f"lane={r_lane:.3f} heading={r_head:.3f} speed={r_speed:.3f} "
                f"smooth={r_smooth:.3f} path_smooth={r_path_smooth:.3f} "
                f"progress={r_progress:.3f} lane_change={r_lane_change:.3f} "
                f"solid_cross={r_solid_cross:.3f} inner_soft={r_inner_soft:.3f}"
            )
            print(f"[DEBUG] Episode done! env_done={env_done}, term_reason={term_reason}, illegal_reason={illegal_reason}")
            if collision_events:
                print(f"[DEBUG] Collision events: {collision_events}")
            elif env_done:
                dist = carla_info.get("distance_to_goal", "unknown")
                print(f"[DEBUG] CarlaEnv done. Distance to goal: {dist}, episode_steps: {self.env.episode_step}/{self.env.max_episode_steps}")

        return obs_out, float(reward), bool(done), info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

