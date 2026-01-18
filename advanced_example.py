"""
高级示例：强化学习框架集成
展示如何将Carla环境与主流RL框架集成
"""
import numpy as np
from carla_env import CarlaEnv
from pid_controller import AdaptiveController
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 示例1: 自定义RL代理 + 自定义奖励
# ============================================================================

class SimpleQLearningAgent:
    """
    简单的Q学习代理（用于演示）
    实际项目中应使用DQN、PPO等更高级算法
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # 简单的线性权重（实际应使用神经网络）
        self.weights = np.random.randn(state_dim, action_dim) * 0.1
    
    def select_action(self, obs, epsilon=0.1):
        """ε-greedy策略"""
        if np.random.random() < epsilon:
            # 随机动作
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # 贪心动作
            q_values = obs @ self.weights
            action = q_values / (np.linalg.norm(q_values) + 1e-8)
        
        return np.clip(action, -1, 1)
    
    def update(self, obs, action, reward, next_obs):
        """更新权重"""
        target = reward + 0.99 * np.max(next_obs @ self.weights)
        prediction = np.dot(obs, self.weights[:, 0])
        error = target - prediction
        
        self.weights += self.learning_rate * error * obs[:, np.newaxis]


def train_with_rl_agent():
    """
    使用自定义RL代理进行训练
    """
    logger.info("=" * 60)
    logger.info("示例1: 强化学习代理训练")
    logger.info("=" * 60)
    
    # 自定义奖励函数
    def rl_reward_fn(env):
        obs = env._get_observation()
        distance = obs[4]
        heading = obs[5]
        speed = obs[6]
        
        # 多目标优化
        lane_reward = np.exp(-distance * 15) * 2.0      # 车道保持
        heading_reward = np.cos(heading) * 1.0          # 航向对齐
        speed_reward = 1.0 if 8 <= speed <= 12 else 0.5 # 速度控制
        
        reward = lane_reward + heading_reward + speed_reward
        
        # 风险惩罚
        if distance > 2.0:
            reward -= 5.0
        if speed > 15:
            reward -= 1.0
        
        return reward
    
    env = CarlaEnv(town='Town03', reward_fn=rl_reward_fn)
    agent = SimpleQLearningAgent(state_dim=7, action_dim=3)
    
    try:
        for episode in range(5):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(200):
                # 代理选择动作
                action = agent.select_action(obs, epsilon=0.1 * (5 - episode) / 5)
                
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # 更新代理
                agent.update(obs, action, reward, next_obs)
                
                obs = next_obs
                
                if done:
                    break
            
            logger.info(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}, "
                       f"Steps = {step + 1}")
    
    finally:
        env.close()


# ============================================================================
# 示例2: 多个目标的加权奖励
# ============================================================================

class WeightedReward:
    """
    加权奖励系统，可动态调整各项权重
    """
    
    def __init__(self):
        self.weights = {
            'lane_keeping': 2.0,
            'heading_control': 1.0,
            'speed_control': 1.0,
            'smooth_steering': 0.5,
            'collision_avoidance': 5.0,
        }
    
    def compute(self, env):
        """计算加权奖励"""
        obs = env._get_observation()
        vx, vy, yaw, yaw_rate, distance, heading, speed = obs
        
        rewards = {}
        
        # 车道保持奖励
        rewards['lane_keeping'] = np.exp(-distance * 10)
        
        # 航向控制奖励
        rewards['heading_control'] = np.cos(heading)
        
        # 速度控制奖励
        target_speed = 10.0
        rewards['speed_control'] = np.exp(-np.abs(speed - target_speed) * 0.1)
        
        # 平滑转向奖励
        rewards['smooth_steering'] = 1.0 - np.abs(yaw_rate) * 0.1
        
        # 碰撞避免奖励
        if distance > 1.5:
            rewards['collision_avoidance'] = 0.0
        else:
            rewards['collision_avoidance'] = 1.0
        
        # 计算加权和
        total_reward = sum(rewards[k] * self.weights[k] for k in rewards)
        
        return total_reward / sum(self.weights.values())
    
    def set_weight(self, key, value):
        """调整权重"""
        if key in self.weights:
            self.weights[key] = value


def train_with_weighted_reward():
    """
    使用加权奖励系统进行训练
    """
    logger.info("=" * 60)
    logger.info("示例2: 加权奖励系统")
    logger.info("=" * 60)
    
    reward_system = WeightedReward()
    
    env = CarlaEnv(town='Town03', reward_fn=reward_system.compute)
    controller = AdaptiveController(target_speed=10.0)
    
    try:
        obs = env.reset()
        total_reward = 0
        
        for step in range(500):
            # 每100步调整权重
            if step == 100:
                logger.info("调整奖励权重：增加速度控制权重")
                reward_system.set_weight('speed_control', 2.0)
            elif step == 200:
                logger.info("调整奖励权重：增加车道保持权重")
                reward_system.set_weight('lane_keeping', 3.0)
            
            action = controller.get_control(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if step % 100 == 0:
                logger.info(f"Step {step}: Cumulative Reward = {total_reward:.2f}")
            
            if done:
                break
        
        logger.info(f"Total Episode Reward: {total_reward:.2f}")
    
    finally:
        env.close()


# ============================================================================
# 示例3: 课程学习 (Curriculum Learning)
# ============================================================================

def curriculum_learning():
    """
    课程学习示例：逐步增加任务难度
    """
    logger.info("=" * 60)
    logger.info("示例3: 课程学习")
    logger.info("=" * 60)
    
    def make_reward_fn(difficulty):
        """根据难度级别生成奖励函数"""
        def reward_fn(env):
            obs = env._get_observation()
            distance = obs[4]
            heading = obs[5]
            speed = obs[6]
            
            # 难度越高，要求越严格
            if difficulty == 1:
                # 简单：只要求保持在车道内
                return 1.0 if distance < 2.0 else -1.0
            elif difficulty == 2:
                # 中等：要求车道保持 + 基本速度控制
                lane = np.exp(-distance * 10)
                speed_ctrl = 1.0 if 8 <= speed <= 12 else 0.5
                return lane + speed_ctrl
            else:
                # 困难：所有要求都很严格
                lane = np.exp(-distance * 20)
                heading_ctrl = np.cos(heading)
                speed_ctrl = np.exp(-np.abs(speed - 10) * 0.2)
                if distance > 1.5:
                    return -5.0
                return lane + heading_ctrl + speed_ctrl
        
        return reward_fn
    
    difficulties = [1, 2, 3]
    
    for difficulty in difficulties:
        logger.info(f"\n--- 难度等级 {difficulty} ---")
        
        env = CarlaEnv(town='Town03', reward_fn=make_reward_fn(difficulty))
        controller = AdaptiveController(target_speed=10.0)
        
        try:
            obs = env.reset()
            episode_reward = 0
            
            for step in range(300):
                action = controller.get_control(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            logger.info(f"难度 {difficulty} 完成: 总奖励 = {episode_reward:.2f}")
        
        finally:
            env.close()


# ============================================================================
# 示例4: 多车道环境测试
# ============================================================================

def test_different_scenarios():
    """
    测试不同的场景（不同的地图和难度）
    """
    logger.info("=" * 60)
    logger.info("示例4: 不同场景测试")
    logger.info("=" * 60)
    
    towns = ['Town01', 'Town03', 'Town05']
    target_speeds = [8.0, 10.0, 12.0]
    
    for town in towns:
        logger.info(f"\n--- 地图: {town} ---")
        
        env = CarlaEnv(town=town)
        controller = AdaptiveController(target_speed=10.0)
        
        try:
            obs = env.reset()
            avg_speed = 0
            steps = 0
            
            for step in range(200):
                action = controller.get_control(obs)
                obs, reward, done, info = env.step(action)
                
                avg_speed += obs[6]
                steps += 1
                
                if done:
                    break
            
            logger.info(f"地图 {town}: 平均速度 = {avg_speed / steps:.2f} m/s, "
                       f"运行步数 = {steps}")
        
        finally:
            env.close()


# ============================================================================
# 示例5: 动态参数调整演示
# ============================================================================

def dynamic_parameter_tuning():
    """
    演示如何动态调整PID参数以适应不同场景
    """
    logger.info("=" * 60)
    logger.info("示例5: 动态参数调整")
    logger.info("=" * 60)
    
    env = CarlaEnv(town='Town03')
    controller = AdaptiveController(target_speed=10.0)
    
    try:
        obs = env.reset()
        
        for step in range(600):
            # 根据当前状态调整PID参数
            distance_to_center = obs[4]
            
            if distance_to_center > 1.0:
                # 偏离较远，增加转向反应
                logger.info("距离过大，增加转向反应性")
                controller.lane_controller.steering_controller.set_gains(kp=3.0, kd=0.8)
            elif distance_to_center < 0.3:
                # 回到中心，减小反应以避免过冲
                logger.info("接近中心，降低转向反应性")
                controller.lane_controller.steering_controller.set_gains(kp=1.5, kd=0.3)
            
            action = controller.get_control(obs)
            obs, reward, done, info = env.step(action)
            
            if step % 100 == 0:
                logger.info(f"Step {step}: Distance={obs[4]:.3f}, Speed={obs[6]:.2f}")
            
            if done:
                break
        
    finally:
        env.close()


# ============================================================================
# 示例6: 性能评估和指标收集
# ============================================================================

class PerformanceMetrics:
    """
    性能指标收集和分析
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_reward = 0
        self.distances = []
        self.speeds = []
        self.headings = []
        self.steps = 0
    
    def update(self, obs, reward):
        self.total_reward += reward
        self.distances.append(obs[4])
        self.speeds.append(obs[6])
        self.headings.append(obs[5])
        self.steps += 1
    
    def get_summary(self):
        """获取性能总结"""
        return {
            'total_reward': self.total_reward,
            'avg_distance': np.mean(self.distances),
            'std_distance': np.std(self.distances),
            'avg_speed': np.mean(self.speeds),
            'std_speed': np.std(self.speeds),
            'avg_heading': np.mean(self.headings),
            'max_distance': np.max(self.distances),
            'steps': self.steps,
        }


def evaluate_performance():
    """
    评估控制器性能
    """
    logger.info("=" * 60)
    logger.info("示例6: 性能评估")
    logger.info("=" * 60)
    
    env = CarlaEnv(town='Town03')
    controller = AdaptiveController(target_speed=10.0)
    metrics = PerformanceMetrics()
    
    try:
        obs = env.reset()
        
        for step in range(500):
            action = controller.get_control(obs)
            obs, reward, done, info = env.step(action)
            
            metrics.update(obs, reward)
            
            if done:
                break
        
        summary = metrics.get_summary()
        logger.info("\n性能指标:")
        logger.info(f"  总奖励: {summary['total_reward']:.2f}")
        logger.info(f"  平均距离车道中心: {summary['avg_distance']:.3f}m "
                   f"(±{summary['std_distance']:.3f}m)")
        logger.info(f"  平均速度: {summary['avg_speed']:.2f}m/s "
                   f"(±{summary['std_speed']:.2f}m/s)")
        logger.info(f"  最大偏离距离: {summary['max_distance']:.3f}m")
        logger.info(f"  运行步数: {summary['steps']}")
    
    finally:
        env.close()


if __name__ == "__main__":
    # 运行示例（选择其中之一或全部）
    
    # train_with_rl_agent()           # 示例1
    # train_with_weighted_reward()    # 示例2
    # curriculum_learning()           # 示例3
    # test_different_scenarios()      # 示例4
    # dynamic_parameter_tuning()      # 示例5
    evaluate_performance()            # 示例6
    
    logger.info("\n所有示例运行完成！")
