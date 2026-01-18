# 快速参考卡 (Quick Reference)

## 🎯 10秒快速开始

```python
from carla_env import CarlaEnv
from pid_controller import AdaptiveController

env = CarlaEnv(town='Town03')
controller = AdaptiveController(target_speed=10.0)
obs = env.reset()

for step in range(1000):
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    if done: break

env.close()
```

---

## 📦 模块导入速查

| 功能 | 导入 | 使用 |
|------|------|------|
| 主环境 | `from carla_env import CarlaEnv` | `env = CarlaEnv()` |
| 综合控制 | `from pid_controller import AdaptiveController` | `controller = AdaptiveController()` |
| 转向控制 | `from pid_controller import LaneKeepingController` | `controller = LaneKeepingController()` |
| 速度控制 | `from pid_controller import SpeedController` | `controller = SpeedController()` |
| 基础PID | `from pid_controller import PIDController` | `pid = PIDController()` |
| 配置预设 | `from config import get_pid_config` | `conf = get_pid_config()` |

---

## 🎮 环境使用

### 创建环境
```python
# 基础
env = CarlaEnv()

# 自定义地图和奖励
env = CarlaEnv(
    town='Town03',                    # 地图
    reward_fn=my_custom_reward_fn,   # 奖励函数
    host='localhost',                 # 服务器地址
    port=2000                        # 服务器端口
)
```

### 主循环
```python
obs = env.reset()                    # 获取初始观测

for _ in range(episodes):
    obs = env.reset()
    
    for step in range(max_steps):
        action = agent.act(obs)      # 你的智能体
        
        # obs: [vx,vy,yaw,yaw_rate,dist,heading,speed]
        # action: [throttle, brake, steer]
        obs, reward, done, info = env.step(action)
        
        if done:
            break

env.close()
```

---

## 🎛️ PID控制器使用

### 快速使用 (推荐)
```python
from pid_controller import AdaptiveController

controller = AdaptiveController(target_speed=10.0)
action = controller.get_control(obs)  # 返回 [throttle, brake, steer]
```

### 分别控制
```python
from pid_controller import LaneKeepingController, SpeedController

lane_controller = LaneKeepingController()
speed_controller = SpeedController(target_speed=10.0)

steering = lane_controller.get_control(distance, heading_error)
throttle, brake = speed_controller.get_control(current_speed)
```

### 参数调整
```python
controller.lane_controller.steering_controller.set_gains(kp=2.0, ki=0.1, kd=0.5)
controller.set_target_speed(12.0)
controller.reset()
```

---

## 🏆 奖励函数

### 默认奖励
```python
env = CarlaEnv(town='Town03')  # 使用内置默认奖励
```

### 自定义奖励
```python
def my_reward(env):
    obs = env._get_observation()
    distance = obs[4]      # 距车道中心距离
    heading = obs[5]       # 航向误差
    speed = obs[6]         # 车速
    
    reward = (
        np.exp(-distance * 10) * 1.0 +      # 车道保持
        np.cos(heading) * 0.5 +              # 航向对齐
        (1.0 if 8 <= speed <= 12 else 0.5)   # 速度控制
    )
    
    if distance > 2.0:
        reward -= 5.0  # 偏离车道惩罚
    
    return reward

env = CarlaEnv(town='Town03', reward_fn=my_reward)
```

### 预设奖励
```python
from config import get_reward_config

# lane_keeping_focused, balanced, speed_focused, comfort_optimized
config = get_reward_config('comfort_optimized')
```

---

## 🗺️ 地图选择

```python
towns = [
    'Town01', 'Town02', 'Town03', 'Town04', 'Town05',
    'Town06', 'Town07', 'Town10HD', 'Town12', 'Town13'
]

env = CarlaEnv(town='Town03')  # 推荐用于测试
```

---

## 🔍 观测空间详解

```python
obs = env.reset()  # 返回7维向量

# 索引	名称              单位    解释
# 0     vx               m/s     车体X方向速度
# 1     vy               m/s     车体Y方向速度
# 2     yaw              rad     偏航角
# 3     yaw_rate         rad/s   角速度
# 4     distance_center  m       到车道中心距离
# 5     heading_error    rad     航向误差
# 6     speed            m/s     速度大小

distance_to_center = obs[4]
heading_error = obs[5]
current_speed = obs[6]
```

---

## 🎬 动作空间详解

```python
action = np.array([throttle, brake, steer])

# 元素       范围        单位    解释
# [0]        [0, 1]      -       油门
# [1]        [0, 1]      -       制动
# [2]        [-1, 1]     -       转向

# 示例
action = np.array([0.5, 0.0, 0.1])  # 50%油门，转向0.1
```

---

## ⚙️ 常用配置

### 保守型控制
```python
from config import get_pid_config

config = get_pid_config('conservative')
# 结果：反应慢但稳定
```

### 激进型控制
```python
config = get_pid_config('aggressive')
# 结果：反应快但容易振荡
```

### 环境难度
```python
from config import get_env_config

# easy, medium, hard, expert
config = get_env_config('medium')
env = CarlaEnv(
    town=config['town'],
    target_speed=config['target_speed']
)
```

---

## 📊 性能评估

```python
class Metrics:
    def __init__(self):
        self.rewards = []
        self.distances = []
        self.speeds = []
    
    def update(self, reward, obs):
        self.rewards.append(reward)
        self.distances.append(obs[4])
        self.speeds.append(obs[6])
    
    def print_summary(self):
        print(f"Avg Reward: {np.mean(self.rewards):.2f}")
        print(f"Avg Distance: {np.mean(self.distances):.3f}m")
        print(f"Avg Speed: {np.mean(self.speeds):.2f}m/s")

metrics = Metrics()
obs = env.reset()
for _ in range(1000):
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    metrics.update(reward, obs)
    if done: break

metrics.print_summary()
```

---

## 🐛 调试技巧

### 打印状态
```python
obs = env.reset()
print(f"Speed: {obs[6]:.2f} m/s")
print(f"Distance to center: {obs[4]:.3f} m")
print(f"Heading error: {obs[5]:.3f} rad")
```

### 保存轨迹
```python
trajectory = []
obs = env.reset()
for _ in range(1000):
    trajectory.append({
        'obs': obs.copy(),
        'reward': reward,
    })
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    if done: break
```

### 单步调试
```python
obs = env.reset()
action = np.array([0.1, 0.0, 0.1])
obs, reward, done, info = env.step(action)

print(f"Info: {info}")
print(f"Location: {info['location']}")
print(f"Velocity: {info['velocity']}")
```

---

## 🚀 与RL框架集成

### Stable-Baselines3
```python
from stable_baselines3 import PPO
from carla_env import CarlaEnv

env = CarlaEnv()
model = PPO('MlpPolicy', env)
model.learn(total_timesteps=10000)
```

### PyTorch
```python
import torch

obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
action = policy_network(obs_tensor)
```

### NumPy (简单算法)
```python
# 线性策略
action = np.dot(obs, weights)
action = np.clip(action, [-1, 0, -1], [1, 1, 1])
```

---

## ⚠️ 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| 连接超时 | Carla未启动 | `./CarlaUE4.sh` |
| NaN奖励 | 观测值无效 | 检查 `reward_fn` |
| 车辆不动 | 油门过小 | 增加 `throttle` |
| 车辆振荡 | Kp过大 | 降低PID Kp |
| 速度过快 | 目标速度过高 | 降低 `target_speed` |

---

## 📚 文件导航

| 文件 | 用途 |
|------|------|
| `carla_env.py` | 主环境类 |
| `pid_controller.py` | 控制器实现 |
| `config.py` | 参数预设 |
| `example.py` | 基础示例 |
| `advanced_example.py` | 高级用法 |
| `USAGE_GUIDE.py` | 详细代码示例 |
| `README.md` | 完整文档 |
| `PROJECT_SUMMARY.md` | 项目总结 |

---

## 🔗 快速链接

- 完整文档: [README.md](README.md)
- 详细指南: [USAGE_GUIDE.py](USAGE_GUIDE.py)
- 基础示例: [example.py](example.py)
- 高级用法: [advanced_example.py](advanced_example.py)
- 参数配置: [config.py](config.py)
- 项目总结: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## 💡 最佳实践

1. ✅ 从`AdaptiveController`开始
2. ✅ 使用配置预设快速获得参考性能
3. ✅ 逐步调整参数而不是大幅改动
4. ✅ 定期评估性能指标
5. ✅ 使用不同的地图测试泛化能力
6. ✅ 记录所有实验配置和结果

---

**快速参考卡 v1.0** | 最后更新: 2026年1月
