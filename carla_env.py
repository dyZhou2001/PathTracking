"""
Carla环境的Gym风格封装
支持自定义奖励函数，便于强化学习训练
"""
import carla
import numpy as np
import os
import sys
import json
import queue
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple, Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarlaEnv:
    """
    Gym风格的Carla环境封装
    
    使用方式:
        env = CarlaEnv(reward_fn=custom_reward_fn)
        obs = env.reset()
        for _ in range(1000):
            action = agent.act(obs)  # action: [throttle, brake, steer]
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        env.close()
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 2000,
        town: str = 'Town03',
        reward_fn: Callable = None,
        spawn_point_index: Optional[int] = 0,
        destination_index: Optional[int] = None,
        destination_location: Optional[carla.Location] = None,
        goal_radius: float = 3.0,
        route_resolution: float = 2.0,
        lookahead_distance: float = 10.0,
        debug_draw: bool = False,
        debug_draw_route: bool = True,
        debug_draw_trail: bool = True,
        debug_draw_target: bool = True,
        debug_draw_trail_persist: bool = False,
        debug_draw_interval: int = 5,
        debug_draw_life_time: float = 0.25,
        debug_draw_route_life_time: float = 30.0,

        # dataset collection
        collect_dataset: bool = False,
        # RL / online camera observation (no dataset saving)
        enable_camera: bool = False,
        enable_lane_invasion_sensor: bool = False,
        enable_collision_sensor: bool = False,
        dataset_dir: str = "dataset",
        dataset_run_name: Optional[str] = None,
        dataset_save_every: int = 5,
        dataset_future_points: int = 20,
        dataset_point_spacing: float = 2.0,
        camera_width: int = 800,
        camera_height: int = 600,
        camera_fov: float = 90.0,
        synchronous_mode: Optional[bool] = None,
        fixed_delta_seconds: float = 0.05,
        max_episode_steps: int = 1000,
        spectator_follow: bool = False,
        spectator_distance: float = 8.0,
        spectator_height: float = 3.0,
        spectator_pitch: float = -15.0,
    ):
        """
        初始化Carla环境
        
        Args:
            host: Carla服务器地址
            port: Carla服务器端口
            town: 地图名称 (Town01-Town13)
            reward_fn: 自定义奖励函数，签名应为 reward_fn(env) -> float
            spawn_point_index: 出生点索引。默认 0 表示每次固定使用第0个出生点；
                              设为 None 则每次从地图出生点中随机选择。
            destination_index: 终点出生点索引。默认 None 表示不设置终点/不做全局路线规划。
                               设为 int 时，会用该 spawn point 作为终点并生成全局路线。
            destination_location: 直接指定终点位置（优先级高于 destination_index）。
            goal_radius: 判定“到达终点”的半径（米）。
            route_resolution: 全局路线规划的采样分辨率（米），越小越精细但更慢。
            lookahead_distance: 前瞻距离（米），用于从 route_waypoints 中选择 target_point。
            debug_draw: 是否在 Carla 里可视化调试信息（路线/轨迹/前瞻点）。
            debug_draw_route: 是否绘制规划路线（蓝色）。
            debug_draw_trail: 是否绘制车辆走过的轨迹（红色）。
            debug_draw_target: 是否绘制前瞻目标点（绿色）。
            debug_draw_trail_persist: 是否让红色轨迹永久保留到仿真结束（life_time=0）。
            debug_draw_interval: 绘制频率（每N步绘制一次），用于降低开销。
            debug_draw_life_time: 单步绘制元素的存活时间（秒），适合轨迹/前瞻点。
            debug_draw_route_life_time: 路线绘制存活时间（秒）。

            collect_dataset: 是否采集训练数据（相机图像 + 参考轨迹标签）。
            dataset_dir: 数据集输出根目录。
            dataset_run_name: 本次采集的子目录名（默认按时间戳自动生成）。
            dataset_save_every: 每隔N步保存一次（步=env.step次数）。
            dataset_future_points: 每帧输出的未来轨迹点数量。
            dataset_point_spacing: 未来轨迹点间距（米，沿路线累计距离）。
            camera_width/camera_height/camera_fov: RGB相机参数。
            synchronous_mode: 是否强制设置 Carla 为同步模式（None 表示 collect_dataset=True 时自动启用）。
            fixed_delta_seconds: 同步模式下固定步长（秒）。
            spectator_follow: 是否让Carla窗口的观察者相机跟随车辆（可视化用）
            spectator_distance: 跟车相机后方距离（米）
            spectator_height: 跟车相机高度（米）
            spectator_pitch: 跟车相机俯仰角（度，负值向下看）
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.blueprint_lib = self.world.get_blueprint_library()
        
        self.vehicle = None
        self.sensors = {}
        self.spectator_follow = spectator_follow
        self.spectator_distance = spectator_distance
        self.spectator_height = spectator_height
        self.spectator_pitch = spectator_pitch
        self._spectator = self.world.get_spectator() if spectator_follow else None

        # 出生点配置
        self.spawn_point_index = spawn_point_index

        # 终点/路线配置
        self.destination_index = destination_index
        self.destination_location = destination_location
        self.goal_radius = float(goal_radius)
        self.route_resolution = float(route_resolution)
        self.lookahead_distance = float(lookahead_distance)

        # debug draw
        self.debug_draw = bool(debug_draw)
        self.debug_draw_route = bool(debug_draw_route)
        self.debug_draw_trail = bool(debug_draw_trail)
        self.debug_draw_target = bool(debug_draw_target)
        self.debug_draw_trail_persist = bool(debug_draw_trail_persist)
        self.debug_draw_interval = max(int(debug_draw_interval), 1)
        self.debug_draw_life_time = float(debug_draw_life_time)
        self.debug_draw_route_life_time = float(debug_draw_route_life_time)

        # dataset collection
        self.collect_dataset = bool(collect_dataset)
        self.enable_camera = bool(enable_camera)
        self.enable_lane_invasion_sensor = bool(enable_lane_invasion_sensor)
        self.enable_collision_sensor = bool(enable_collision_sensor)
        self.dataset_dir = str(dataset_dir)
        self.dataset_run_name = dataset_run_name
        self.dataset_save_every = max(int(dataset_save_every), 1)
        self.dataset_future_points = max(int(dataset_future_points), 0)
        self.dataset_point_spacing = float(dataset_point_spacing)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_fov = float(camera_fov)
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = float(fixed_delta_seconds)

        self._original_world_settings: Optional[carla.WorldSettings] = None
        self._image_queue: Optional["queue.Queue[carla.Image]"] = None
        self._cached_image: Optional[carla.Image] = None
        self._lane_invasion_queue: Optional["queue.Queue[Any]"] = None
        self._collision_queue: Optional["queue.Queue[Any]"] = None
        self._labels_fp = None
        self._dataset_run_dir: Optional[Path] = None
        self._dataset_images_dir: Optional[Path] = None
        
        # 奖励函数
        # 统一约定：reward_fn(env) -> float
        self.reward_fn = reward_fn if reward_fn is not None else self._default_reward
            
        # 状态记录
        self.prev_distance = 0
        self.episode_step = 0
        self.max_episode_steps = int(max_episode_steps)
        
        # 车道信息
        self.lane_id = None
        self.route_waypoints: List[carla.Waypoint] = []
        self.route_options: List[Any] = []
        self.goal_location: Optional[carla.Location] = None
        self._route_cursor_index: int = 0
        self._last_target_location: Optional[carla.Location] = None
        self._last_vehicle_location: Optional[carla.Location] = None
        self._last_action: Optional[Tuple[float, float, float]] = None
        
        logger.info(f"Carla环境初始化完成: {town}")
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始观测值 shape (7,): [vx, vy, yaw, yaw_rate, lateral_error, heading_error, speed]
        """
        # 清理旧车辆和传感器（幂等清理，避免重复destroy报错）
        self._safe_destroy_actors()

        # 根据需要应用世界设置（同步模式/固定步长）
        if self.collect_dataset:
            self._apply_world_settings_for_dataset()
        else:
            self._apply_world_settings(self.synchronous_mode)
        
        # 生成车辆出生点（默认固定；可设置为随机）
        spawn_points = self.map.get_spawn_points()
        spawn_point = self._select_spawn_point(spawn_points)
        
        # 创建车辆
        vehicle_bp = self.blueprint_lib.filter('vehicle.tesla.model3')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        if self.collect_dataset:
            self._setup_dataset_sensors()
        else:
            if self.enable_camera:
                self._setup_rgb_camera_sensor()

        if self.enable_lane_invasion_sensor:
            self._setup_lane_invasion_sensor()
        if self.enable_collision_sensor:
            self._setup_collision_sensor()
        
        # 设置初始速度
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        
        # 获取起点所在车道信息
        waypoint = self.map.get_waypoint(spawn_point.location)
        self.lane_id = waypoint.lane_id
        
        # 设置终点 + 生成路线（如果提供终点，则生成全局路线）
        self.goal_location = self._select_goal_location(spawn_points)
        self._generate_route(waypoint)
        self._route_cursor_index = 0
        self._last_target_location = None
        self._last_vehicle_location = None
        
        self.prev_distance = 0
        self.episode_step = 0
        
        # 让环境稳定
        for _ in range(10):
            self.world.tick()

        self._update_spectator()

        if self.debug_draw and self.debug_draw_route:
            self._debug_draw_route()
        
        obs = self._get_observation()
        logger.info("环境已重置")
        return obs

    @staticmethod
    def _validate_index(name: str, index: int, length: int) -> None:
        if not isinstance(index, int):
            raise TypeError(f"{name} 必须是 int 或 None")
        if index < 0 or index >= length:
            raise ValueError(f"{name} 超出范围: {index}, 有效范围: [0, {length - 1}]")

    def _select_spawn_point(self, spawn_points: list) -> carla.Transform:
        if not spawn_points:
            raise RuntimeError("地图未提供任何 spawn points")

        if self.spawn_point_index is None:
            return np.random.choice(spawn_points)

        self._validate_index("spawn_point_index", self.spawn_point_index, len(spawn_points))
        return spawn_points[self.spawn_point_index]

    def _select_goal_location(self, spawn_points: list) -> Optional[carla.Location]:
        # 优先使用明确给定的位置
        if self.destination_location is not None:
            return self.destination_location

        if self.destination_index is None:
            return None

        if not spawn_points:
            raise RuntimeError("地图未提供任何 spawn points")

        self._validate_index("destination_index", self.destination_index, len(spawn_points))
        return spawn_points[self.destination_index].location
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步环境交互
        
        Args:
            action: 动作向量 [throttle, brake, steer]
                   throttle: [0, 1] 油门
                   brake: [0, 1] 制动
                   steer: [-1, 1] 转向
        
        Returns:
            obs: 观测值
            reward: 奖励值
            done: 是否结束
            info: 额外信息字典
        """
        throttle, brake, steer = action
        throttle = np.clip(throttle, 0, 1)
        brake = np.clip(brake, 0, 1)
        steer = np.clip(steer, -1, 1)
        
        # 应用控制
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.steer = float(steer)
        self.vehicle.apply_control(control)
        self._last_action = (float(throttle), float(brake), float(steer))
        
        self.world.tick()

        self._update_spectator()
        
        obs = self._get_observation()
        reward = float(self.reward_fn(self))
        done = self._check_done()

        if self.collect_dataset and (self.episode_step % self.dataset_save_every == 0):
            self._dataset_save_sample(obs)

        if self.debug_draw and (self.episode_step % self.debug_draw_interval == 0):
            self._debug_draw_step()

        distance_to_goal = None
        if self.goal_location is not None:
            try:
                distance_to_goal = float(self.vehicle.get_location().distance(self.goal_location))
            except RuntimeError:
                distance_to_goal = None
        
        frame = None
        sim_time = None
        try:
            snapshot = self.world.get_snapshot()
            frame = int(snapshot.frame)
            sim_time = float(snapshot.timestamp.elapsed_seconds)
        except RuntimeError:
            frame = None
            sim_time = None

        info = {
            'frame': frame,
            'sim_time': sim_time,
            'location': self.vehicle.get_location(),
            'velocity': self.vehicle.get_velocity(),
            'acceleration': self.vehicle.get_acceleration(),
            'control': [throttle, brake, steer],
            'goal_location': self.goal_location,
            'distance_to_goal': distance_to_goal,
            'target_location': self._last_target_location,
        }
        
        self.episode_step += 1
        
        return obs, reward, done, info

    def _apply_world_settings_for_dataset(self) -> None:
        """采集数据时建议开启同步模式，便于帧对齐。"""
        desired_sync = True if self.synchronous_mode is None else bool(self.synchronous_mode)
        self._apply_world_settings(desired_sync)

    def _apply_world_settings(self, synchronous_mode: Optional[bool]) -> None:
        """按需设置 Carla 世界同步模式与固定步长。

        - synchronous_mode=None: 不改动当前世界设置（保持外部启动参数/已有设置）
        - synchronous_mode=True: 开启同步模式并设置 fixed_delta_seconds
        - synchronous_mode=False: 关闭同步模式并清空 fixed_delta_seconds
        """
        if synchronous_mode is None:
            return

        try:
            if self._original_world_settings is None:
                self._original_world_settings = self.world.get_settings()
        except RuntimeError:
            return

        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = bool(synchronous_mode)
            if settings.synchronous_mode:
                settings.fixed_delta_seconds = float(self.fixed_delta_seconds)
            else:
                settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except RuntimeError:
            return

    def _setup_dataset_sensors(self) -> None:
        """创建 RGB 相机并准备输出目录/label 文件。"""
        if self.vehicle is None:
            return

        run_name = self.dataset_run_name
        if not run_name:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            # 让默认 run_name 在多次 reset 之间保持一致，便于同一次运行的数据落到同一个目录
            self.dataset_run_name = run_name

        run_dir = Path(self.dataset_dir) / run_name
        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        self._dataset_run_dir = run_dir
        self._dataset_images_dir = images_dir

        # labels.jsonl 追加写
        self._labels_fp = open(run_dir / "labels.jsonl", "a", encoding="utf-8")

        self._image_queue = queue.Queue(maxsize=32)
        self._cached_image = None

        cam_bp = self.blueprint_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.camera_width))
        cam_bp.set_attribute('image_size_y', str(self.camera_height))
        cam_bp.set_attribute('fov', str(self.camera_fov))

        # 一个比较通用的前视相机位姿（可按需要调整）
        cam_transform = carla.Transform(
            carla.Location(x=1.6, z=1.4),
            carla.Rotation(pitch=-5.0),
        )
        camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.sensors['rgb_camera'] = camera

        def _on_image(image: carla.Image):
            if self._image_queue is None:
                return
            try:
                self._image_queue.put_nowait(image)
            except queue.Full:
                try:
                    _ = self._image_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._image_queue.put_nowait(image)
                except queue.Full:
                    pass

        camera.listen(_on_image)

    def _setup_rgb_camera_sensor(self) -> None:
        """创建 RGB 相机用于在线观测（不保存数据集）。"""
        if self.vehicle is None:
            return

        self._image_queue = queue.Queue(maxsize=32)
        self._cached_image = None

        cam_bp = self.blueprint_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.camera_width))
        cam_bp.set_attribute('image_size_y', str(self.camera_height))
        cam_bp.set_attribute('fov', str(self.camera_fov))

        cam_transform = carla.Transform(
            carla.Location(x=1.6, z=1.4),
            carla.Rotation(pitch=-5.0),
        )
        camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.sensors['rgb_camera'] = camera

        def _on_image(image: carla.Image):
            if self._image_queue is None:
                return
            try:
                self._image_queue.put_nowait(image)
            except queue.Full:
                try:
                    _ = self._image_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._image_queue.put_nowait(image)
                except queue.Full:
                    pass

        camera.listen(_on_image)

    def _setup_lane_invasion_sensor(self) -> None:
        if self.vehicle is None:
            return

        self._lane_invasion_queue = queue.Queue(maxsize=64)
        bp = self.blueprint_lib.find('sensor.other.lane_invasion')
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['lane_invasion'] = sensor

        def _on_event(event: Any) -> None:
            if self._lane_invasion_queue is None:
                return
            try:
                self._lane_invasion_queue.put_nowait(event)
            except queue.Full:
                try:
                    _ = self._lane_invasion_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._lane_invasion_queue.put_nowait(event)
                except queue.Full:
                    pass

        sensor.listen(_on_event)

    def _setup_collision_sensor(self) -> None:
        if self.vehicle is None:
            return

        self._collision_queue = queue.Queue(maxsize=64)
        bp = self.blueprint_lib.find('sensor.other.collision')
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors['collision'] = sensor

        def _on_event(event: Any) -> None:
            if self._collision_queue is None:
                return
            try:
                self._collision_queue.put_nowait(event)
            except queue.Full:
                try:
                    _ = self._collision_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._collision_queue.put_nowait(event)
                except queue.Full:
                    pass

        sensor.listen(_on_event)

    def pop_lane_invasion_events(self, *, max_events: int = 128) -> List[Any]:
        q = self._lane_invasion_queue
        if q is None:
            return []
        events: List[Any] = []
        for _ in range(int(max_events)):
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                break
        return events

    def pop_collision_events(self, *, max_events: int = 128) -> List[Any]:
        q = self._collision_queue
        if q is None:
            return []
        events: List[Any] = []
        for _ in range(int(max_events)):
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                break
        return events

    def get_rgb_image_for_frame(self, *, target_frame: int, timeout_s: float = 1.0) -> Optional[carla.Image]:
        """严格按 frame 获取对应的相机图像。

        返回 None 表示未拿到精确帧（例如队列溢出或超时）。
        """
        if self._image_queue is None:
            return None

        if self._cached_image is not None:
            try:
                if int(getattr(self._cached_image, 'frame', -1)) == int(target_frame):
                    img = self._cached_image
                    self._cached_image = None
                    return img
            except Exception:
                self._cached_image = None

        deadline = datetime.now().timestamp() + float(timeout_s)
        while datetime.now().timestamp() < deadline:
            remaining = max(0.01, deadline - datetime.now().timestamp())
            try:
                img = self._image_queue.get(timeout=remaining)
            except queue.Empty:
                break

            f = int(getattr(img, 'frame', -1))
            if f < int(target_frame):
                continue
            if f == int(target_frame):
                return img

            # f > target_frame: 先缓存，等待下一次调用
            self._cached_image = img
            return None

        return None

    def _dataset_save_sample(self, obs: np.ndarray) -> None:
        if self.vehicle is None or self._image_queue is None or self._labels_fp is None:
            return
        if self._dataset_images_dir is None:
            return

        try:
            snapshot = self.world.get_snapshot()
            frame = int(snapshot.frame)
            sim_time = float(snapshot.timestamp.elapsed_seconds)
        except RuntimeError:
            return

        image = self._get_image_for_frame(frame, timeout_s=2.0)
        if image is None:
            return

        # 保存图像
        image_path = self._dataset_images_dir / f"{frame:08d}.png"
        try:
            image.save_to_disk(str(image_path), carla.ColorConverter.Raw)
        except RuntimeError:
            return

        # 计算未来轨迹点（车辆坐标系 + 世界坐标）
        transform = self.vehicle.get_transform()
        future_world = self._get_future_route_locations(
            start_index=self._route_cursor_index,
            num_points=self.dataset_future_points,
            spacing_m=self.dataset_point_spacing,
        )
        future_vehicle = [self._world_to_vehicle_xy(transform, loc) for loc in future_world]

        # 当前 target point（车辆坐标系）
        target_x = float(obs[7]) if len(obs) >= 9 else 0.0
        target_y = float(obs[8]) if len(obs) >= 9 else 0.0

        label = {
            "frame": frame,
            "episode_step": int(self.episode_step),
            "sim_time": sim_time,
            "image": str(Path("images") / image_path.name),
            "vehicle": {
                "x": float(transform.location.x),
                "y": float(transform.location.y),
                "z": float(transform.location.z),
                "yaw_deg": float(transform.rotation.yaw),
            },
            "action": None if self._last_action is None else {
                "throttle": float(self._last_action[0]),
                "brake": float(self._last_action[1]),
                "steer": float(self._last_action[2]),
            },
            "speed": float(obs[6]) if len(obs) >= 7 else None,
            "goal": None if self.goal_location is None else {
                "x": float(self.goal_location.x),
                "y": float(self.goal_location.y),
                "z": float(self.goal_location.z),
            },
            "target_point_vehicle": {"x": target_x, "y": target_y},
            "route_cursor_index": int(self._route_cursor_index),
            "future_route_vehicle": [{"x": float(x), "y": float(y)} for x, y in future_vehicle],
            "future_route_world": [{"x": float(l.x), "y": float(l.y), "z": float(l.z)} for l in future_world],
        }

        self._labels_fp.write(json.dumps(label, ensure_ascii=False) + "\n")
        self._labels_fp.flush()

    def _get_image_for_frame(self, target_frame: int, timeout_s: float = 2.0) -> Optional[carla.Image]:
        if self._image_queue is None:
            return None

        deadline = datetime.now().timestamp() + float(timeout_s)
        best = None
        while datetime.now().timestamp() < deadline:
            remaining = max(0.01, deadline - datetime.now().timestamp())
            try:
                img = self._image_queue.get(timeout=remaining)
            except queue.Empty:
                break

            if int(getattr(img, 'frame', -1)) == int(target_frame):
                return img

            # 保留最新的，避免队列堆积
            best = img

        return best

    def _get_future_route_locations(self, start_index: int, num_points: int, spacing_m: float) -> List[carla.Location]:
        if not self.route_waypoints or num_points <= 0:
            return []

        n = len(self.route_waypoints)
        i = int(np.clip(start_index, 0, n - 1))

        points: List[carla.Location] = []
        accum = 0.0
        spacing = max(float(spacing_m), 0.01)

        # 从当前位置附近的 route index 向前取点
        prev = self.route_waypoints[i].transform.location
        while i < n and len(points) < num_points:
            cur = self.route_waypoints[i].transform.location
            try:
                seg = float(prev.distance(cur))
            except RuntimeError:
                seg = 0.0
            accum += seg
            if accum >= spacing or len(points) == 0:
                points.append(cur)
                accum = 0.0
            prev = cur
            i += 1

        return points

    @staticmethod
    def _world_to_vehicle_xy(vehicle_transform: carla.Transform, world_location: carla.Location) -> Tuple[float, float]:
        vehicle_location = vehicle_transform.location
        forward = vehicle_transform.get_forward_vector()
        right = vehicle_transform.get_right_vector()
        d = world_location - vehicle_location
        x = float(d.x * forward.x + d.y * forward.y + d.z * forward.z)
        y = float(d.x * right.x + d.y * right.y + d.z * right.z)
        return x, y

    def _update_spectator(self) -> None:
        """让Carla窗口中的观察者相机跟随车辆，便于可视化。"""
        if not self.spectator_follow:
            return
        if self._spectator is None or self.vehicle is None:
            return
        try:
            if not self.vehicle.is_alive:
                return
        except RuntimeError:
            return

        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        location = transform.location

        cam_location = carla.Location(
            x=location.x - forward.x * float(self.spectator_distance),
            y=location.y - forward.y * float(self.spectator_distance),
            z=location.z + float(self.spectator_height),
        )
        cam_rotation = carla.Rotation(
            pitch=float(self.spectator_pitch),
            yaw=transform.rotation.yaw,
            roll=0.0,
        )
        try:
            self._spectator.set_transform(carla.Transform(cam_location, cam_rotation))
        except RuntimeError:
            # Carla关闭/窗口切换等情况下可能失败，忽略即可
            pass
    
    def _get_observation(self) -> np.ndarray:
        """
        获取观测值
        
        Returns:
                        观测向量:
                            - 前7维保持兼容: [vx, vy, yaw, yaw_rate, distance_to_lane_center, heading_error, speed]
                            - 追加2维 target_point(车辆坐标系): [target_x, target_y]

                        其中 target_x/target_y 定义为：
                            - target_x: 目标点在车辆前方的投影（沿车辆 forward，单位米）
                            - target_y: 目标点在车辆右侧的投影（沿车辆 right，单位米）
        """
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        
        # 提取速度分量
        vx = velocity.x
        vy = velocity.y
        vz = velocity.z
        
        # 计算速度大小
        speed = np.sqrt(vx**2 + vy**2)
        
        # 提取偏航角
        yaw = np.radians(transform.rotation.yaw)
        
        # 计算角速度（利用转向角估计）
        yaw_rate = 0.0  # 可以通过传感器扩展获取
        
        # 获取与车道中心的距离(带符号)和航向误差(带符号)
        waypoint = self.map.get_waypoint(transform.location)
        
        # 计算横向误差（带符号）
        # 约定：+ 表示车道中心在车辆右侧；- 表示车道中心在车辆左侧
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        to_waypoint = waypoint.transform.location - transform.location
        lateral_error = float(np.dot([to_waypoint.x, to_waypoint.y], [right.x, right.y]))
        
        # 计算航向误差（带符号）
        # 约定：+ 表示车辆航向相对车道方向偏左；- 表示偏右
        # 使用 atan2(cross, dot) 获取带符号角度，避免 arccos 的无符号与数值越界问题
        waypoint_forward = waypoint.transform.get_forward_vector()
        fwd = np.array([forward.x, forward.y], dtype=np.float64)
        wp_fwd = np.array([waypoint_forward.x, waypoint_forward.y], dtype=np.float64)

        fwd_norm = np.linalg.norm(fwd)
        wp_fwd_norm = np.linalg.norm(wp_fwd)
        denom = max(float(fwd_norm * wp_fwd_norm), 1e-12)

        dot = float(np.dot(wp_fwd, fwd) / denom)
        dot = float(np.clip(dot, -1.0, 1.0))
        cross = float((wp_fwd[0] * fwd[1] - wp_fwd[1] * fwd[0]) / denom)
        heading_error = float(np.arctan2(cross, dot))
        
        target_x, target_y = self._get_target_point_vehicle_frame(transform)

        obs = np.array(
            [vx, vy, yaw, yaw_rate, lateral_error, heading_error, speed, target_x, target_y],
            dtype=np.float32,
        )
        
        return obs

    def _get_target_point_vehicle_frame(self, vehicle_transform: carla.Transform) -> Tuple[float, float]:
        """从路线中选取前瞻目标点，并转换到车辆坐标系（forward/right）。"""
        try:
            vehicle_location = vehicle_transform.location
            forward = vehicle_transform.get_forward_vector()
            right = vehicle_transform.get_right_vector()
        except RuntimeError:
            return 0.0, 0.0

        target_location = self._get_lookahead_target_location(vehicle_location)
        self._last_target_location = target_location
        if target_location is None:
            return 0.0, 0.0

        to_target = target_location - vehicle_location
        target_x = float(to_target.x * forward.x + to_target.y * forward.y + to_target.z * forward.z)
        target_y = float(to_target.x * right.x + to_target.y * right.y + to_target.z * right.z)
        return target_x, target_y

    def _get_lookahead_target_location(self, vehicle_location: carla.Location) -> Optional[carla.Location]:
        """在 route_waypoints 上找一个“前瞻距离”处的目标点（世界坐标）。

        若 route_waypoints 为空，则退化为当前车道 waypoint 前方 lookahead_distance 的点。
        """
        if not self.route_waypoints:
            try:
                wp = self.map.get_waypoint(vehicle_location)
                next_wps = wp.next(float(self.lookahead_distance))
                if next_wps:
                    return next_wps[0].transform.location
            except RuntimeError:
                return None
            return None

        # 在路线附近窗口内寻找最近点，避免每次全量扫描
        n = len(self.route_waypoints)
        cursor = int(np.clip(self._route_cursor_index, 0, n - 1))
        search_back = 10
        search_ahead = 80
        start = max(0, cursor - search_back)
        end = min(n - 1, cursor + search_ahead)

        best_i = cursor
        best_d = float("inf")
        for i in range(start, end + 1):
            try:
                d = float(vehicle_location.distance(self.route_waypoints[i].transform.location))
            except RuntimeError:
                continue
            if d < best_d:
                best_d = d
                best_i = i

        self._route_cursor_index = best_i

        # 从 best_i 开始沿路线累计距离，找到前瞻点
        target_i = best_i
        accum = 0.0
        while target_i + 1 < n and accum < float(self.lookahead_distance):
            p0 = self.route_waypoints[target_i].transform.location
            p1 = self.route_waypoints[target_i + 1].transform.location
            try:
                seg = float(p0.distance(p1))
            except RuntimeError:
                seg = 0.0
            accum += seg
            target_i += 1

        return self.route_waypoints[target_i].transform.location

    def _debug_draw_route(self) -> None:
        if not self.route_waypoints:
            return
        if self.world is None:
            return

        try:
            dbg = self.world.debug
        except RuntimeError:
            return

        color = carla.Color(0, 128, 255)  # blue
        life = float(self.debug_draw_route_life_time)

        # Draw polyline along the route
        prev = None
        for wp in self.route_waypoints[::2]:
            loc = wp.transform.location
            loc = carla.Location(loc.x, loc.y, loc.z + 0.3)
            dbg.draw_point(loc, size=0.08, color=color, life_time=life)
            if prev is not None:
                dbg.draw_line(prev, loc, thickness=0.05, color=color, life_time=life)
            prev = loc

        # Mark goal
        if self.goal_location is not None:
            goal = carla.Location(self.goal_location.x, self.goal_location.y, self.goal_location.z + 0.6)
            dbg.draw_point(goal, size=0.15, color=carla.Color(255, 255, 0), life_time=life)

    def _debug_draw_step(self) -> None:
        if self.vehicle is None:
            return
        try:
            dbg = self.world.debug
            vloc = self.vehicle.get_location()
        except RuntimeError:
            return

        life = float(self.debug_draw_life_time)
        trail_life = 0.0 if self.debug_draw_trail_persist else life

        # Trail (red)
        if self.debug_draw_trail:
            cur = carla.Location(vloc.x, vloc.y, vloc.z + 0.2)
            if self._last_vehicle_location is not None:
                prev = carla.Location(
                    self._last_vehicle_location.x,
                    self._last_vehicle_location.y,
                    self._last_vehicle_location.z + 0.2,
                )
                dbg.draw_line(prev, cur, thickness=0.07, color=carla.Color(255, 0, 0), life_time=trail_life)
            self._last_vehicle_location = vloc

        # Target point (green)
        if self.debug_draw_target and self._last_target_location is not None:
            tgt = carla.Location(
                self._last_target_location.x,
                self._last_target_location.y,
                self._last_target_location.z + 0.6,
            )
            dbg.draw_point(tgt, size=0.12, color=carla.Color(0, 255, 0), life_time=life)
            try:
                cur = carla.Location(vloc.x, vloc.y, vloc.z + 0.6)
                dbg.draw_line(cur, tgt, thickness=0.04, color=carla.Color(0, 255, 0), life_time=life)
            except RuntimeError:
                pass
    
    def _check_done(self) -> bool:
        """检查是否完成一个episode"""
        if self.episode_step >= self.max_episode_steps:
            return True

        # 到达终点
        if self.goal_location is not None and self.vehicle is not None:
            try:
                if self.vehicle.get_location().distance(self.goal_location) <= float(self.goal_radius):
                    return True
            except RuntimeError:
                pass
        
        # 检查是否发生碰撞（可扩展）
        # 检查是否超出地图边界
        location = self.vehicle.get_location()
        if abs(location.x) > 5000 or abs(location.y) > 5000:
            return True
        
        return False
    
    @staticmethod
    def _default_reward(env: "CarlaEnv") -> float:
        """
        默认奖励函数
        
        奖励设置:
        - 保持在车道中心: +1.0
        - 偏离车道中心: 根据距离惩罚
        - 保持目标速度: +0.5
        """
        obs = env._get_observation()
        distance_to_center = float(abs(obs[4]))
        heading_error = obs[5]
        speed = obs[6]
        
        # 车道保持奖励
        lane_reward = np.exp(-distance_to_center * 10) * 1.0
        
        # 航向对齐奖励
        heading_reward = np.cos(heading_error) * 0.5
        
        # 速度保持奖励（目标速度10 m/s）
        target_speed = 10.0
        speed_reward = np.exp(-np.abs(speed - target_speed) * 0.1) * 0.5
        
        reward = lane_reward + heading_reward + speed_reward
        
        return float(reward)

    def _safe_destroy_actors(self) -> None:
        """安全销毁车辆和传感器，允许重复调用（幂等）。"""
        if self._labels_fp is not None:
            try:
                self._labels_fp.close()
            except Exception:
                pass
            finally:
                self._labels_fp = None

        self._image_queue = None
        self._cached_image = None
        self._lane_invasion_queue = None
        self._collision_queue = None

        if self.vehicle is not None:
            try:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
            except RuntimeError:
                pass
            finally:
                self.vehicle = None

        for key, sensor in list(self.sensors.items()):
            try:
                if sensor is not None and sensor.is_alive:
                    try:
                        sensor.stop()
                    except Exception:
                        pass
                    sensor.destroy()
            except RuntimeError:
                pass
            finally:
                self.sensors.pop(key, None)
    
    def _generate_route(self, start_waypoint: carla.Waypoint):
        """生成路线。

        - 若设置了 goal_location，则使用 CARLA 全局路线规划器生成路线（推荐，用于路口/环岛出入口选择）。
        - 若未设置 goal_location，则退化为“沿当前车道向前”的局部路线（车道保持）。
        """
        self.route_waypoints = []
        self.route_options = []

        if self.goal_location is None:
            current = start_waypoint
            for _ in range(100):
                self.route_waypoints.append(current)
                next_waypoints = current.next(2.0)
                if not next_waypoints:
                    break
                current = next_waypoints[0]
            return

        self._ensure_carla_agents_on_path()
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except Exception as e:
            raise RuntimeError(
                "无法导入 CARLA 的 GlobalRoutePlanner。\n"
                "常见原因：你只安装了 carla 的 egg/wheel（仅含 carla 模块），但未把 CARLA 源码中的 PythonAPI/carla 加入 PYTHONPATH。\n"
                "解决方式：\n"
                "1) 设置环境变量 CARLA_ROOT 指向 CARLA 根目录（包含 PythonAPI 文件夹），然后重试；或\n"
                "2) 直接把 <CARLA_ROOT>/PythonAPI/carla 加入 sys.path/PYTHONPATH。\n"
            ) from e

        # 不同 CARLA 版本的 agents 实现略有差异：
        # - 有些版本提供 GlobalRoutePlannerDAO + GlobalRoutePlanner(dao) + setup()
        # - 有些版本只有 GlobalRoutePlanner(wmap, sampling_resolution)（例如你当前安装包）
        try:
            from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO  # type: ignore

            dao = GlobalRoutePlannerDAO(self.map, float(self.route_resolution))
            grp = GlobalRoutePlanner(dao)
            if hasattr(grp, "setup"):
                grp.setup()
        except Exception:
            grp = GlobalRoutePlanner(self.map, float(self.route_resolution))

        start_loc = start_waypoint.transform.location
        route = grp.trace_route(start_loc, self.goal_location)
        if not route:
            logger.warning("全局路线规划失败：未生成任何 route points")
            return

        for wp, road_option in route:
            self.route_waypoints.append(wp)
            self.route_options.append(road_option)

    @staticmethod
    def _try_add_sys_path(path: Path) -> bool:
        try:
            p = str(path)
        except Exception:
            return False

        if not p:
            return False
        if p in sys.path:
            return True
        if path.exists() and path.is_dir():
            sys.path.insert(0, p)
            return True
        return False

    def _ensure_carla_agents_on_path(self) -> None:
        """确保可导入 agents.navigation.*。

        CARLA 的 agents 模块通常位于 <CARLA_ROOT>/PythonAPI/carla/agents。
        但很多安装方式只把 carla egg/wheel 装进了 site-packages，不包含 agents 文件夹。
        """
        # 已经能导入则直接返回
        try:
            import agents  # type: ignore

            _ = agents
            return
        except Exception:
            pass

        candidates: List[Path] = []

        # 1) 从环境变量推断
        carla_root = os.environ.get("CARLA_ROOT") or os.environ.get("CARLA_HOME")
        if carla_root:
            root = Path(carla_root)
            candidates.append(root / "PythonAPI" / "carla")

        # 2) 从 carla 模块所在位置推断（常见：.../PythonAPI/carla/dist/carla-*.egg/...）
        try:
            carla_file = Path(getattr(carla, "__file__", "")).resolve()
        except Exception:
            carla_file = None

        if carla_file and carla_file.exists():
            for parent in carla_file.parents:
                # 命中 .../PythonAPI/carla/dist/...
                if parent.name == "dist" and (parent.parent / "agents").exists():
                    candidates.append(parent.parent)
                    break
                # 命中 .../PythonAPI/carla/...
                if parent.name == "carla" and (parent / "agents").exists():
                    candidates.append(parent)
                    break

        # 去重并尝试加入 sys.path
        for c in candidates:
            self._try_add_sys_path(c)

        # 再次尝试导入，失败则保持原样让上层抛更友好错误
        try:
            import agents  # type: ignore

            _ = agents
        except Exception:
            return
    
    def close(self):
        """关闭环境，销毁所有演员"""
        self._safe_destroy_actors()

        # 恢复世界设置
        if self._original_world_settings is not None:
            try:
                self.world.apply_settings(self._original_world_settings)
            except RuntimeError:
                pass
            finally:
                self._original_world_settings = None
        logger.info("环境已关闭")
    
    def set_reward_fn(self, reward_fn: Callable):
        """设置新的奖励函数"""
        self.reward_fn = reward_fn
        logger.info("奖励函数已更新")
    
    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except Exception:
            # 析构阶段不应抛异常影响进程退出
            pass
