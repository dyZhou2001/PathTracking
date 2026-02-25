"""example.py

示例脚本：
- PID 控制演示（原有）
- 键盘转向 + 自动定速 的数据采集（新增）

关于“窗口焦点”：
- `pynput`：全局键盘监听，通常不依赖焦点（Carla 窗口/终端都行）。
- `msvcrt`：从终端读取按键，焦点必须在终端窗口。
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from carla_env import CarlaEnv
from pid_controller import AdaptiveController, SpeedController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
# 运行配置（可按需修改）
# ==============================
# 固定出生点索引（0 表示第0个 spawn point；设为 None 则随机）
SPAWN_POINT_INDEX = 0
# 终点索引：设为 int 会启用“全局路线规划”（路口左/右/直由规划决定）
# 设为 None 则不设置终点，只做车道保持（可能在环岛一直绕圈）
DESTINATION_INDEX = 1
# 到达终点判定半径（米）
GOAL_RADIUS = 3.0

# Carla 可视化调试（在 CarlaUE4 窗口里画线/点）
DEBUG_DRAW = False

# 同步步长（建议与采集/控制一致）
FIXED_DELTA_SECONDS = 0.05

# ==============================
# 数据集采集（图像 + 参考轨迹标签）
# ==============================
COLLECT_DATASET = True
DATASET_DIR = "dataset"
DATASET_RUN_NAME = None  # 例如 "town03_spawn0_to1"；None 则自动生成并在多次 reset 间复用
DATASET_SAVE_EVERY = 5
DATASET_FUTURE_POINTS = 20
DATASET_POINT_SPACING = 2.0
DATASET_SYNC = True
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90.0

# ==============================
# 键盘控制（用于采集纠偏数据）
# ==============================
# 推荐优先使用 pynput（全局监听，不依赖焦点）；没有安装会自动回退到 msvcrt（需要终端焦点）
KEYBOARD_BACKEND = "pynput"  # "pynput" or "msvcrt"

# 控制手感
KEY_STEER_RATE = 1.8     # steer/s，按住方向键时的变化速度
KEY_STEER_RETURN = 2.5   # steer/s，松开后回正速度
KEY_STEER_DEADBAND = 0.02

# 键盘采集时的目标速度（仍由 SpeedController 自动控制油门/刹车）
KEY_TARGET_SPEED = 0.5

# 是否将键盘采集模式设为默认入口
RUN_KEYBOARD_COLLECT = True


def custom_reward_fn(env):
    """
    自定义奖励函数示例
    
    可以根据具体需求定制奖励函数的逻辑
    """
    obs = env._get_observation()
    distance_to_center = float(abs(obs[4]))
    heading_error = obs[5]
    speed = obs[6]
    
    # 车道保持奖励（重权重）
    lane_reward = np.exp(-distance_to_center * 20) * 2.0
    
    # 航向控制奖励
    heading_reward = np.cos(heading_error) * 1.0
    
    # 速度控制奖励（目标速度12 m/s）
    target_speed = 12.0
    speed_reward = np.exp(-np.abs(speed - target_speed) * 0.1) * 1.0
    
    # 偏离车道过远的惩罚
    if distance_to_center > 2.0:
        penalty = -5.0
    else:
        penalty = 0.0
    
    total_reward = lane_reward + heading_reward + speed_reward + penalty
    
    return float(total_reward)


def example_pid_control():
    """
    示例1: 使用PID控制器进行车道保持和速度控制
    """
    logger.info("=" * 50)
    logger.info("示例1: PID控制器演示")
    logger.info("=" * 50)
    
    # 创建环境（使用默认奖励函数）
    # spectator_follow=True 时，会让 CarlaUE4 窗口的观察者相机跟随车辆，便于可视化
    env = CarlaEnv(
        town='Town03',
        spectator_follow=True,
        max_episode_steps=10000,
        spawn_point_index=SPAWN_POINT_INDEX,
        destination_index=DESTINATION_INDEX,
        goal_radius=GOAL_RADIUS,
        debug_draw=DEBUG_DRAW,
        debug_draw_trail_persist=DEBUG_DRAW,

        # dataset
        collect_dataset=COLLECT_DATASET,
        dataset_dir=DATASET_DIR,
        dataset_run_name=DATASET_RUN_NAME,
        dataset_save_every=DATASET_SAVE_EVERY,
        dataset_future_points=DATASET_FUTURE_POINTS,
        dataset_point_spacing=DATASET_POINT_SPACING,
        synchronous_mode=DATASET_SYNC,
        fixed_delta_seconds=FIXED_DELTA_SECONDS,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_fov=CAMERA_FOV,
    )
    
    # 创建自适应控制器
    controller = AdaptiveController(target_speed=5.0)
    
    try:
        obs = env.reset()
        
        for step in range(10000):
            # 使用PID控制器生成动作
            action = controller.get_control(obs)
            
            obs, reward, done, info = env.step(action)
            
            if step % 50 == 0:
                speed = obs[6]
                distance = obs[4]
                heading = obs[5]
                dist_goal = info.get('distance_to_goal', None)
                dist_goal_str = f"{dist_goal:.2f}m" if isinstance(dist_goal, (int, float)) else "N/A"
                logger.info(
                    f"Step {step}: Speed={speed:.2f}m/s, Distance={distance:.2f}m, "
                    f"Heading={heading:.2f}rad, Reward={reward:.2f}, Dist2Goal={dist_goal_str}"
                )
            
            if done:
                logger.info("Episode完成")
                break
        
    finally:
        env.close()


@dataclass
class _KeyboardState:
    left: bool = False
    right: bool = False
    quit: bool = False
    brake: bool = False


class KeyboardSteering:
    """键盘转向输入：输出 steer in [-1,1]。

    - 左/右: A/D 或 ←/→
    - 空格: 直接刹车（覆盖 SpeedController 输出）
    - Q 或 ESC: 退出循环
    """

    def __init__(
        self,
        *,
        dt: float,
        backend: str = "pynput",
        steer_rate: float = 1.8,
        return_rate: float = 2.5,
        deadband: float = 0.02,
    ):
        self.dt = float(dt)
        self.backend = str(backend).lower()
        self.steer_rate = float(steer_rate)
        self.return_rate = float(return_rate)
        self.deadband = float(deadband)

        self.state = _KeyboardState()
        self.steer = 0.0

        self._listener = None
        self._msvcrt = None

    def start(self) -> None:
        if self.backend == "pynput":
            try:
                from pynput import keyboard as pynput_keyboard  # type: ignore
            except Exception as e:
                logger.warning(
                    "pynput 不可用，回退到 msvcrt（需要终端窗口焦点）。"
                    "如需全局监听，请运行: pip install pynput\n"
                    f"原始错误: {e}"
                )
                self.backend = "msvcrt"
            else:
                self._start_pynput(pynput_keyboard)

        if self.backend == "msvcrt":
            try:
                import msvcrt
            except Exception as e:
                raise RuntimeError("msvcrt 不可用（非 Windows 环境？）") from e
            self._msvcrt = msvcrt

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _start_pynput(self, pynput_keyboard) -> None:
        Key = pynput_keyboard.Key

        def _is_left(k) -> bool:
            if k == Key.left:
                return True
            try:
                return getattr(k, "char", None) in ("a", "A")
            except Exception:
                return False

        def _is_right(k) -> bool:
            if k == Key.right:
                return True
            try:
                return getattr(k, "char", None) in ("d", "D")
            except Exception:
                return False

        def _is_quit(k) -> bool:
            if k in (Key.esc,):
                return True
            try:
                return getattr(k, "char", None) in ("q", "Q")
            except Exception:
                return False

        def _is_brake(k) -> bool:
            return k == Key.space

        def on_press(key):
            if _is_left(key):
                self.state.left = True
            if _is_right(key):
                self.state.right = True
            if _is_brake(key):
                self.state.brake = True
            if _is_quit(key):
                self.state.quit = True

        def on_release(key):
            if _is_left(key):
                self.state.left = False
            if _is_right(key):
                self.state.right = False
            if _is_brake(key):
                self.state.brake = False

        self._listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def _poll_msvcrt(self) -> None:
        """msvcrt：非阻塞读取按键事件（需要终端窗口焦点）。

        说明：msvcrt 没有“按住/松开”的精确信号，我们用“最近一次按键”来更新 state，
        并在无输入时自动回正。
        """
        msvcrt = self._msvcrt
        if msvcrt is None:
            return

        self.state.left = False
        self.state.right = False
        self.state.brake = False

        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            # Arrow keys come as '\x00' or '\xe0' prefix then a code.
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                if code == "K":  # left
                    self.state.left = True
                elif code == "M":  # right
                    self.state.right = True
                continue

            if ch in ("a", "A"):
                self.state.left = True
            elif ch in ("d", "D"):
                self.state.right = True
            elif ch in ("q", "Q"):
                self.state.quit = True
            elif ch == " ":
                self.state.brake = True

    def update(self) -> float:
        if self.backend == "msvcrt":
            self._poll_msvcrt()

        if self.state.left and not self.state.right:
            self.steer -= self.steer_rate * self.dt
        elif self.state.right and not self.state.left:
            self.steer += self.steer_rate * self.dt
        else:
            # 回正
            if abs(self.steer) <= self.deadband:
                self.steer = 0.0
            else:
                step = self.return_rate * self.dt
                self.steer -= np.sign(self.steer) * min(abs(self.steer), step)

        self.steer = float(np.clip(self.steer, -1.0, 1.0))
        return self.steer


def example_keyboard_collect():
    """示例: 键盘转向采集数据。

    推荐设置：
    - COLLECT_DATASET = True
    - DATASET_SYNC = True
    """
    logger.info("=" * 50)
    logger.info("示例: 键盘转向 + 自动定速 采集数据")
    logger.info("控制: A/D 或 ←/→ 转向；空格刹车；Q/ESC 退出")
    logger.info("=" * 50)

    env = CarlaEnv(
        town="Town03",
        spectator_follow=True,
        max_episode_steps=100000,
        spawn_point_index=SPAWN_POINT_INDEX,
        destination_index=DESTINATION_INDEX,
        goal_radius=GOAL_RADIUS,
        debug_draw=DEBUG_DRAW,
        debug_draw_trail_persist=DEBUG_DRAW,

        # dataset
        collect_dataset=COLLECT_DATASET,
        dataset_dir=DATASET_DIR,
        dataset_run_name=DATASET_RUN_NAME,
        dataset_save_every=DATASET_SAVE_EVERY,
        dataset_future_points=DATASET_FUTURE_POINTS,
        dataset_point_spacing=DATASET_POINT_SPACING,
        synchronous_mode=DATASET_SYNC,
        fixed_delta_seconds=FIXED_DELTA_SECONDS,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_fov=CAMERA_FOV,
    )

    speed_ctl = SpeedController(target_speed=float(KEY_TARGET_SPEED), dt=float(FIXED_DELTA_SECONDS))
    kb = KeyboardSteering(
        dt=float(FIXED_DELTA_SECONDS),
        backend=str(KEYBOARD_BACKEND),
        steer_rate=float(KEY_STEER_RATE),
        return_rate=float(KEY_STEER_RETURN),
        deadband=float(KEY_STEER_DEADBAND),
    )

    try:
        obs = env.reset()
        kb.start()

        for step in range(100000):
            steer = kb.update()

            speed = float(obs[6]) if len(obs) >= 7 else 0.0
            throttle, brake = speed_ctl.get_control(speed)

            if kb.state.brake:
                throttle, brake = 0.0, 1.0

            action = np.array([throttle, brake, steer], dtype=np.float32)
            obs, reward, done, info = env.step(action)

            if kb.state.quit:
                logger.info("退出键触发，停止采集")
                break

            if step % 50 == 0:
                dist_goal = info.get("distance_to_goal", None)
                dist_goal_str = f"{dist_goal:.2f}m" if isinstance(dist_goal, (int, float)) else "N/A"
                lateral = float(obs[4]) if len(obs) >= 5 else 0.0
                heading = float(obs[5]) if len(obs) >= 6 else 0.0
                logger.info(
                    f"Step {step}: Speed={speed:.2f}m/s steer={steer:+.2f} "
                    f"lat={lateral:+.2f} head={heading:+.2f} Dist2Goal={dist_goal_str}"
                )

            if done:
                logger.info("Episode完成")
                break

            # 给异步输入一点时间片；同步仿真下通常不需要，但能降低 CPU 占用
            time.sleep(0.0)

    finally:
        kb.stop()
        env.close()


def example_rl_training():
    """
    示例2: 强化学习训练框架示例
    """
    logger.info("=" * 50)
    logger.info("示例2: 强化学习训练框架")
    logger.info("=" * 50)
    
    # 创建环境并设置自定义奖励函数
    env = CarlaEnv(town='Town03', reward_fn=custom_reward_fn)
    
    try:
        num_episodes = 3
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(200):
                # 这里应该接入你的RL Agent
                # 现在我们使用随机动作作为演示
                action = np.random.uniform([-1, 0, -1], [1, 1, 1])  # [throttle, brake, steer]
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                       f"Total Reward: {episode_reward:.2f}, "
                       f"Steps: {step + 1}")
    
    finally:
        env.close()


def example_variable_speed():
    """
    示例3: 演示变速控制
    """
    logger.info("=" * 50)
    logger.info("示例3: 变速控制演示")
    logger.info("=" * 50)
    
    env = CarlaEnv(town='Town03')
    controller = AdaptiveController(target_speed=8.0)
    
    try:
        obs = env.reset()
        
        for step in range(600):
            # 每200步改变目标速度
            if step == 200:
                controller.set_target_speed(12.0)
                logger.info("目标速度变更为 12.0 m/s")
            elif step == 400:
                controller.set_target_speed(6.0)
                logger.info("目标速度变更为 6.0 m/s")
            
            action = controller.get_control(obs)
            obs, reward, done, info = env.step(action)
            
            if step % 100 == 0:
                speed = obs[6]
                logger.info(f"Step {step}: Current Speed={speed:.2f}m/s")
            
            if done:
                break
        
    finally:
        env.close()


def test_pid_controller():
    """
    单独测试PID控制器
    """
    logger.info("=" * 50)
    logger.info("测试: PID控制器")
    logger.info("=" * 50)
    
    from pid_controller import PIDController
    
    # 创建一个简单的PID控制器
    pid = PIDController(kp=1.0, ki=0.1, kd=0.3, dt=0.05)
    
    # 模拟误差信号，观察PID输出
    logger.info("模拟跟踪阶跃输入（目标值=1）")
    for i in range(100):
        # 模拟一个误差信号：目标-当前
        error = 1.0 - (0.9 ** (i / 20.0))  # 指数趋近1
        
        output = pid.update(error)
        
        if i % 10 == 0:
            logger.info(f"Step {i}: Error={error:.3f}, Output={output:.3f}")


if __name__ == "__main__":
    # 根据需要运行不同的示例
    # 注意：运行这些示例需要Carla服务器运行在后台
    
    # test_pid_controller()  # 取消注释运行PID测试
    
    if RUN_KEYBOARD_COLLECT:
        example_keyboard_collect()
    else:
        example_pid_control()  # 运行PID控制演示
    
    # example_rl_training()  # 取消注释运行RL训练框架
    
    # example_variable_speed()  # 取消注释运行变速控制演示
    
    logger.info("所有示例运行完成！")
