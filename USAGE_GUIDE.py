"""
Carla Gym风格环境 - 快速使用指南
"""

# ============================================================================
# 1. 基本使用 - Gym风格调用
# ============================================================================

from carla_env import CarlaEnv
import numpy as np

# 创建环境
env = CarlaEnv(
    host='localhost',
    port=2000,
    town='Town03'
)

# 主训练循环
obs = env.reset()
for step in range(1000):
    # 生成动作 [throttle, brake, steer]
    action = np.array([0.3, 0.0, 0.1])  # 加速，转向
    
    # 与环境交互
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()

env.close()


# ============================================================================
# 2. 自定义奖励函数
# ============================================================================

def my_custom_reward(env):
    """
    自定义奖励函数
    
    env.observation 或 env._get_observation() 可获取：
    - [0] vx: 车体X方向速度
    - [1] vy: 车体Y方向速度
    - [2] yaw: 偏航角
    - [3] yaw_rate: 偏航角速度
    - [4] distance_to_center: 到车道中心的距离
    - [5] heading_error: 航向误差
    - [6] speed: 速度大小
    """
    obs = env._get_observation()
    distance = obs[4]  # 到车道中心距离
    speed = obs[6]     # 当前速度
    
    # 简单奖励：保持在车道内
    reward = 1.0 if distance < 1.0 else -1.0
    
    # 可以添加速度约束
    if speed > 15:
        reward -= 0.5
    
    return reward


# 使用自定义奖励函数创建环境
env = CarlaEnv(town='Town03', reward_fn=my_custom_reward)


# ============================================================================
# 3. 使用PID控制器
# ============================================================================

from pid_controller import AdaptiveController

# 创建自适应控制器
controller = AdaptiveController(target_speed=10.0)

env = CarlaEnv(town='Town03')
obs = env.reset()

for step in range(1000):
    # PID控制器自动生成动作
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    
    if done:
        controller.reset()
        obs = env.reset()

env.close()


# ============================================================================
# 4. 单独使用各个控制器
# ============================================================================

from pid_controller import LaneKeepingController, SpeedController, PIDController

# 车道保持控制器
lane_controller = LaneKeepingController(dt=0.05)
steering = lane_controller.get_control(
    distance_to_center=0.5,  # 到车道中心0.5米
    heading_error=0.1        # 航向误差0.1弧度
)

# 车速控制器
speed_controller = SpeedController(target_speed=10.0)
throttle, brake = speed_controller.get_control(current_speed=8.5)

# 通用PID控制器
pid = PIDController(kp=1.0, ki=0.1, kd=0.3)
output = pid.update(error=0.5)
pid.reset()


# ============================================================================
# 5. 动态调整参数
# ============================================================================

env = CarlaEnv(town='Town03')
controller = AdaptiveController(target_speed=10.0)

obs = env.reset()
for step in range(1000):
    # 每500步改变目标速度
    if step == 500:
        controller.set_target_speed(15.0)
    
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()

env.close()


# ============================================================================
# 6. 获取详细信息
# ============================================================================

env = CarlaEnv(town='Town03')
obs = env.reset()

for step in range(100):
    action = np.array([0.1, 0.0, 0.0])
    obs, reward, done, info = env.step(action)
    
    # info 字典包含：
    # - location: 车辆位置 (carla.Location)
    # - velocity: 车辆速度 (carla.Vector3D)
    # - acceleration: 车辆加速度 (carla.Vector3D)
    # - control: 应用的控制 [throttle, brake, steer]
    
    location = info['location']
    velocity = info['velocity']
    print(f"位置: ({location.x:.1f}, {location.y:.1f}), "
          f"速度: ({velocity.x:.1f}, {velocity.y:.1f})")

env.close()


# ============================================================================
# 7. 强化学习集成示例
# ============================================================================

from carla_env import CarlaEnv
import numpy as np

# 定义RL特定的奖励函数
def rl_reward_fn(env):
    obs = env._get_observation()
    distance = obs[4]
    heading = obs[5]
    speed = obs[6]
    
    # 多目标奖励
    rewards = {
        'lane_keeping': np.exp(-distance * 10),
        'heading_control': np.cos(heading),
        'speed_control': 1.0 if 8 <= speed <= 12 else 0.5,
    }
    
    total = sum(rewards.values()) / len(rewards)
    
    # 碰撞惩罚等
    if distance > 2.0:
        total -= 2.0
    
    return total


env = CarlaEnv(town='Town03', reward_fn=rl_reward_fn)

# 这里可以接入你的RL算法 (DQN, PPO, A3C等)
# agent = YourRLAgent(env)
# agent.train(episodes=1000)


# ============================================================================
# 8. 环境参数说明
# ============================================================================

"""
CarlaEnv 初始化参数:
    host (str): Carla服务器主机地址，默认 'localhost'
    port (int): Carla服务器端口，默认 2000
    town (str): 加载的地图名称
               可选: 'Town01', 'Town02', 'Town03', 'Town04', 'Town05',
                    'Town06', 'Town07', 'Town10HD', 'Town12', 'Town13'
    reward_fn (callable): 奖励函数，签名为 reward_fn(env) -> float
    render (bool): 是否渲染（当前保留用于未来扩展）

观测值 (obs) 向量:
    [0] vx: 车体X方向速度 (m/s)
    [1] vy: 车体Y方向速度 (m/s)
    [2] yaw: 偏航角 (弧度)
    [3] yaw_rate: 偏航角速度 (rad/s)
    [4] distance_to_center: 到车道中心的横向距离 (m)
    [5] heading_error: 车辆航向与车道方向的误差 (弧度)
    [6] speed: 车辆速度大小 (m/s)

动作 (action) 向量:
    [0] throttle: 油门 [0, 1]
    [1] brake: 制动 [0, 1]
    [2] steer: 转向 [-1, 1]
"""
