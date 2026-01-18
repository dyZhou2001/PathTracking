# Carla Gym风格环境封装

一个功能完整的Carla强化学习环境封装，提供Gym风格的API和PID控制器实现。

## 功能特性

✅ **Gym风格API** - 熟悉的 `reset()`、`step()` 接口
✅ **自定义奖励函数** - 灵活定制你的奖励逻辑
✅ **PID控制器** - 开箱即用的车道保持和速度控制
✅ **完整的观测空间** - 多维度的车辆状态信息
✅ **易于集成** - 与任何RL框架完美配合

## 项目结构

```
.
├── carla_env.py           # 主环境封装类
├── pid_controller.py      # PID控制器实现
├── example.py            # 使用示例
├── USAGE_GUIDE.py        # 详细使用指南
└── README.md             # 本文件
```

## 快速开始

### 安装依赖

```bash
# 确保已安装 Carla Python API
pip install carla numpy

# 或者如果从CARLA源码编译
export PYTHONPATH=$PYTHONPATH:/home/user/carla/PythonAPI/carla/dist/carla-X.X.X-py3.x-linux-x86_64.egg
```

### 启动CARLA服务器

```bash
# 方式1: 带UI运行
./CarlaUE4.sh

# 方式2: 无UI运行（更快）
./CarlaUE4.sh -RenderOffScreen
```

### 基本使用

```python
from carla_env import CarlaEnv
from pid_controller import AdaptiveController

# 创建环境
env = CarlaEnv(town='Town03')

# 创建控制器
controller = AdaptiveController(target_speed=10.0)

# 运行一个episode
obs = env.reset()
for step in range(1000):
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
```

## 核心组件

### 1. CarlaEnv - 环境类

**主要方法:**

- `reset()` → obs - 重置环境并返回初始观测
- `step(action)` → (obs, reward, done, info) - 执行一步
- `set_reward_fn(fn)` - 设置自定义奖励函数
- `close()` - 关闭环境

**初始化参数:**

```python
env = CarlaEnv(
    host='localhost',      # 服务器地址
    port=2000,            # 服务器端口
    town='Town03',        # 地图名称
    reward_fn=None,       # 自定义奖励函数
    render=False          # 是否渲染
)
```

**观测值 (7维):**

| 索引 | 名称 | 单位 | 范围 | 说明 |
|-----|------|------|------|------|
| 0 | vx | m/s | - | 车体X方向速度 |
| 1 | vy | m/s | - | 车体Y方向速度 |
| 2 | yaw | rad | [-π, π] | 偏航角 |
| 3 | yaw_rate | rad/s | - | 偏航角速度 |
| 4 | distance_to_center | m | ≥0 | 到车道中心距离 |
| 5 | heading_error | rad | [0, π] | 航向误差 |
| 6 | speed | m/s | ≥0 | 车速大小 |

**动作值 (3维):**

| 索引 | 名称 | 范围 | 说明 |
|-----|------|------|------|
| 0 | throttle | [0, 1] | 油门 |
| 1 | brake | [0, 1] | 制动 |
| 2 | steer | [-1, 1] | 转向 |

### 2. PID控制器

#### PIDController - 基础PID

```python
from pid_controller import PIDController

pid = PIDController(
    kp=1.0,                              # 比例增益
    ki=0.1,                              # 积分增益
    kd=0.3,                              # 微分增益
    dt=0.05,                             # 时间步长
    output_range=(-1.0, 1.0)             # 输出范围
)

output = pid.update(error=0.5)
pid.reset()
```

#### LaneKeepingController - 车道保持

```python
from pid_controller import LaneKeepingController

controller = LaneKeepingController(dt=0.05)

steering = controller.get_control(
    distance_to_center=0.5,              # 到车道中心距离
    heading_error=0.1                    # 航向误差
)
```

#### SpeedController - 速度控制

```python
from pid_controller import SpeedController

controller = SpeedController(target_speed=10.0, dt=0.05)

throttle, brake = controller.get_control(current_speed=8.5)
controller.set_target_speed(12.0)        # 改变目标速度
```

#### AdaptiveController - 综合控制

```python
from pid_controller import AdaptiveController

controller = AdaptiveController(target_speed=10.0)

action = controller.get_control(obs)      # 返回 [throttle, brake, steer]
controller.set_target_speed(12.0)
controller.reset()
```

### 3. 自定义奖励函数

```python
def my_reward_fn(env):
    """
    自定义奖励函数
    
    参数:
        env: CarlaEnv 实例
    
    返回:
        reward: 浮点数
    """
    obs = env._get_observation()
    
    distance_to_center = obs[4]
    heading_error = obs[5]
    speed = obs[6]
    
    # 定义奖励逻辑
    lane_reward = np.exp(-distance_to_center * 10) * 1.0
    heading_reward = np.cos(heading_error) * 0.5
    speed_reward = 1.0 if 8 <= speed <= 12 else -0.5
    
    total_reward = lane_reward + heading_reward + speed_reward
    
    return total_reward

# 使用自定义奖励函数
env = CarlaEnv(town='Town03', reward_fn=my_reward_fn)
```

## 使用示例

### 示例1: PID控制演示

```python
from carla_env import CarlaEnv
from pid_controller import AdaptiveController

env = CarlaEnv(town='Town03')
controller = AdaptiveController(target_speed=10.0)

obs = env.reset()
for step in range(1000):
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    print(f"Step {step}: Reward={reward:.3f}")
    if done:
        break

env.close()
```

### 示例2: 强化学习集成

```python
from carla_env import CarlaEnv
import numpy as np

def rl_reward_fn(env):
    obs = env._get_observation()
    # 你的RL特定奖励
    return np.random.random()

env = CarlaEnv(town='Town03', reward_fn=rl_reward_fn)

# 连接你的RL算法
# agent = DQNAgent(env)
# agent.train(episodes=1000)

env.close()
```

### 示例3: 变速控制

```python
from carla_env import CarlaEnv
from pid_controller import AdaptiveController

env = CarlaEnv(town='Town03')
controller = AdaptiveController(target_speed=8.0)

obs = env.reset()
for step in range(1000):
    if step == 250:
        controller.set_target_speed(12.0)
    elif step == 500:
        controller.set_target_speed(6.0)
    
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

## 与RL框架集成

### 与 Stable-Baselines3 集成

```python
from stable_baselines3 import PPO
from carla_env import CarlaEnv

env = CarlaEnv(town='Town03')

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

### 与 PyTorch 集成

```python
import torch
from carla_env import CarlaEnv

env = CarlaEnv(town='Town03')

# 你的神经网络模型
model = YourPolicyNetwork()

obs = env.reset()
for step in range(10000):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action = model(obs_tensor).squeeze().numpy()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

## 可用的地图

- Town01 - 简单城镇
- Town02 - 多车道城镇
- Town03 - 有小岛的城镇
- Town04 - 高速公路
- Town05 - 城市环岛
- Town06 - 长直道
- Town07 - 城市十字路口
- Town10HD - 城市高清版
- Town12 - 大城市
- Town13 - 城市环绕道

## 性能优化建议

1. **使用无UI模式运行CARLA** - 提高约50%速度
   ```bash
   ./CarlaUE4.sh -RenderOffScreen
   ```

2. **减少传感器数量** - 目前实现不使用传感器（基于变换），已经很快

3. **使用批处理** - 并行运行多个环境实例

4. **调整时间步** - 在 `AdaptiveController` 中调整 `dt` 参数

## 常见问题

**Q: 如何自定义观测空间？**
A: 修改 `CarlaEnv._get_observation()` 方法，添加或移除观测维度。

**Q: 如何添加更多的动作维度？**
A: 修改 `CarlaEnv.step()` 中的动作处理部分。

**Q: 如何实现其他控制算法？**
A: 继承 `PIDController` 或直接创建新的控制器类。

**Q: 怎样记录运行数据？**
A: 在 `step()` 方法后记录 `obs` 和 `info`，或在 `reward_fn` 中添加日志。

## 扩展功能

### 添加传感器数据

```python
def add_camera_sensor(self):
    # 添加摄像头
    camera_bp = self.blueprint_lib.find('sensor.camera.rgb')
    camera = self.world.spawn_actor(camera_bp, carla.Transform(...))
    camera.listen(lambda image: self.process_image(image))
    self.sensors['camera'] = camera

def add_lidar_sensor(self):
    # 添加激光雷达
    lidar_bp = self.blueprint_lib.find('sensor.lidar.ray_cast')
    lidar = self.world.spawn_actor(lidar_bp, carla.Transform(...))
    lidar.listen(lambda data: self.process_lidar(data))
    self.sensors['lidar'] = lidar
```

### 添加碰撞检测

```python
def add_collision_sensor(self):
    collision_bp = self.blueprint_lib.find('sensor.other.collision')
    collision = self.world.spawn_actor(collision_bp, carla.Transform())
    collision.listen(lambda event: self.on_collision(event))
    self.sensors['collision'] = collision
    self.collision_flag = False

def on_collision(self, event):
    self.collision_flag = True
```

## 参考资源

- [CARLA官方文档](https://carla.readthedocs.io/)
- [OpenAI Gym文档](https://gym.openai.com/)
- [Stable-Baselines3文档](https://stable-baselines3.readthedocs.io/)

## 许可证

MIT License

## 作者

Created for autonomous driving research and RL training.
