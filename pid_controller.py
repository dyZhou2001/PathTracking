"""
PID控制器实现
用于车道保持控制和车速保持功能
"""
import numpy as np
from typing import Tuple


class PIDController:
    """
    标准PID控制器
    
    使用方式:
        pid = PIDController(kp=1.0, ki=0.01, kd=0.1, dt=0.05)
        output = pid.update(error)
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, 
                 dt: float = 0.05, output_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        初始化PID控制器
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
            dt: 时间步长（秒）
            output_range: 输出范围 (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_min, self.output_max = output_range
        
        # 内部状态
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
    
    def update(self, error: float) -> float:
        """
        更新PID控制器并计算输出
        
        Args:
            error: 当前误差值
        
        Returns:
            控制输出值
        """
        # 比例项
        p_output = self.kp * error
        
        # 积分项（累积误差）
        self.integral += error * self.dt
        i_output = self.ki * self.integral
        
        # 微分项（误差变化率）
        derivative = (error - self.prev_error) / self.dt
        d_output = self.kd * derivative
        
        # 计算总输出
        output = p_output + i_output + d_output
        
        # 限制输出范围
        output = np.clip(output, self.output_min, self.output_max)
        
        # 更新历史值
        self.prev_error = error
        self.prev_output = output
        
        return float(output)
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
    
    def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
        """动态调整PID参数"""
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd


class LaneKeepingController:
    """
    车道保持控制器
    
    通过PID控制器调节转向角，使车辆保持在车道中心
    """
    
    def __init__(self, dt: float = 0.05):
        """
        初始化车道保持控制器
        
        Args:
            dt: 时间步长
        """
        # 转向PID控制器
        # 根据与车道中心的距离调节转向
        self.steering_controller = PIDController(
            kp=0.5,    # 比例增益
            ki=0.02,    # 积分增益
            kd=0.05,    # 微分增益
            dt=dt,
            output_range=(-1.0, 1.0)  # 转向范围
        )
        
        self.dt = dt
    
    def get_control(self, distance_to_center: float, heading_error: float) -> float:
        """
        计算转向控制命令
        
        Args:
            distance_to_center: 与车道中心的横向距离（米）
            heading_error: 航向误差（弧度）
        
        Returns:
            转向命令 [-1, 1]
        """
        # 综合横向误差和航向误差作为误差信号
        # 约定：distance_to_center(带符号) + 表示车道中心在右侧；heading_error(带符号) + 表示车辆偏左
        # 这种约定下，steer>0(向右) 有助于减小正误差
        error = float(distance_to_center) + float(heading_error) * 0.1
        
        steering = self.steering_controller.update(error)
        return steering
    
    def reset(self):
        """重置控制器"""
        self.steering_controller.reset()


class PurePursuitController:
    """Pure Pursuit 路径跟踪转向控制器。

    输入为车辆坐标系下的目标点 (target_x, target_y)：
      - target_x: 前方距离（沿车辆 forward，米）
      - target_y: 右侧偏移（沿车辆 right，米）

    输出为 CARLA VehicleControl 的 steer 范围 [-1, 1]。
    """

    def __init__(
        self,
        wheelbase: float = 2.7,
        max_steer_angle_rad: float = np.deg2rad(40.0),
        min_lookahead: float = 1.0,
    ):
        self.wheelbase = float(wheelbase)
        self.max_steer_angle_rad = float(max_steer_angle_rad)
        self.min_lookahead = float(min_lookahead)

    def get_control(self, target_x: float, target_y: float) -> float:
        # 目标点距离
        ld = float(np.hypot(target_x, target_y))
        ld = max(ld, self.min_lookahead)

        # Pure Pursuit 几何转角（弧度）
        # delta = atan2(2L * y, Ld^2)
        delta = float(np.arctan2(2.0 * self.wheelbase * float(target_y), ld * ld))

        # 映射到 [-1, 1] 的 steer
        steer = float(delta / self.max_steer_angle_rad)
        steer = float(np.clip(steer, -1.0, 1.0))
        return steer


class SpeedController:
    """
    车速控制器
    
    通过油门和制动调节车速，使车辆保持目标速度
    """
    
    def __init__(self, target_speed: float = 10.0, dt: float = 0.05):
        """
        初始化车速控制器
        
        Args:
            target_speed: 目标速度（m/s）
            dt: 时间步长
        """
        # 油门/制动PID控制器
        self.speed_controller = PIDController(
            kp=1.0,     # 比例增益
            ki=0.2,     # 积分增益
            kd=0.3,     # 微分增益
            dt=dt,
            output_range=(-1.0, 1.0)  # 输出范围（-1为满制动，+1为满油门）
        )
        
        self.target_speed = target_speed
        self.dt = dt
        self.max_accel = 5.0  # 最大加速度 m/s^2
        self.max_decel = 8.0  # 最大减速度 m/s^2
    
    def get_control(self, current_speed: float) -> Tuple[float, float]:
        """
        计算油门和制动命令
        
        Args:
            current_speed: 当前速度（m/s）
        
        Returns:
            (throttle, brake) 范围都在 [0, 1]
        """
        # 计算速度误差
        speed_error = self.target_speed - current_speed
        
        # PID计算
        control_output = self.speed_controller.update(speed_error)
        
        # 分离油门和制动
        if control_output >= 0:
            # 需要加速
            throttle = np.clip(control_output, 0, 1)
            brake = 0.0
        else:
            # 需要减速
            throttle = 0.0
            brake = np.clip(-control_output, 0, 1)
        
        return float(throttle), float(brake)
    
    def set_target_speed(self, target_speed: float):
        """设置目标速度"""
        self.target_speed = target_speed
    
    def reset(self):
        """重置控制器"""
        self.speed_controller.reset()


class AdaptiveController:
    """
    自适应控制器
    
    结合车道保持和车速控制，提供完整的自动驾驶控制
    """
    
    def __init__(self, target_speed: float = 10.0, dt: float = 0.05):
        """
        初始化自适应控制器
        
        Args:
            target_speed: 目标速度（m/s）
            dt: 时间步长
        """
        self.lane_controller = LaneKeepingController(dt=dt)
        self.path_controller = PurePursuitController()
        self.speed_controller = SpeedController(target_speed=target_speed, dt=dt)
        self.dt = dt
    
    def get_control(self, observation: np.ndarray) -> np.ndarray:
        """
        根据观测值计算完整的控制命令
        
        Args:
            observation: 观测向量 [vx, vy, yaw, yaw_rate, distance_to_center, 
                                 heading_error, speed]
        
        Returns:
            控制向量 [throttle, brake, steer]
        """
        # 兼容两种观测：
        # - 旧版: 7维 [vx, vy, yaw, yaw_rate, distance_to_center, heading_error, speed]
        # - 新版: 9维，末尾追加 target_x, target_y
        vx, vy, yaw, yaw_rate, distance_to_center, heading_error, speed = observation[:7]

        if len(observation) >= 9:
            target_x, target_y = float(observation[7]), float(observation[8])
            steering = self.path_controller.get_control(target_x, target_y)
        else:
            steering = self.lane_controller.get_control(distance_to_center, heading_error)
        
        # 计算油门和制动
        throttle, brake = self.speed_controller.get_control(speed)
        
        # 当航向误差过大时，减速以保证安全
        # 注意：若车速已接近 0，再强制最小刹车可能导致“停住后再也起不来”。
        min_speed_for_forced_brake = 0.5  # m/s
        if abs(heading_error) > np.pi / 2:  # 30度
            throttle *= 0.5
            if speed > min_speed_for_forced_brake:
                brake = max(brake, 0.3)
        
        # 当偏离车道较远时，减速
        if distance_to_center > 3:
            throttle *= 0.5
            if speed > min_speed_for_forced_brake:
                brake = max(brake, 0.2)
        
        return np.array([throttle, brake, steering], dtype=np.float32)
    
    def set_target_speed(self, target_speed: float):
        """设置目标速度"""
        self.speed_controller.set_target_speed(target_speed)
    
    def reset(self):
        """重置所有控制器"""
        self.lane_controller.reset()
        self.speed_controller.reset()
