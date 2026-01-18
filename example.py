"""
示例脚本：展示如何使用Carla环境和PID控制器
"""
import numpy as np
from carla_env import CarlaEnv
from pid_controller import AdaptiveController
import logging

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
    
    example_pid_control()  # 运行PID控制演示
    
    # example_rl_training()  # 取消注释运行RL训练框架
    
    # example_variable_speed()  # 取消注释运行变速控制演示
    
    logger.info("所有示例运行完成！")
