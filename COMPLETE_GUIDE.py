"""
项目完整指南和文件说明

这是一个为CARLA模拟器设计的强化学习环境框架
支持Gym风格API、自定义奖励函数和PID控制器
"""

# ============================================================================
# 📋 项目文件说明
# ============================================================================

PROJECT_FILES = {
    '核心文件': {
        'carla_env.py': {
            'description': 'CARLA环境的Gym风格封装',
            'key_classes': ['CarlaEnv'],
            'key_methods': ['reset()', 'step(action)', 'set_reward_fn()'],
            'usage': '# 创建环境\nenv = CarlaEnv(town="Town03")\nobs = env.reset()',
        },
        'pid_controller.py': {
            'description': 'PID控制器和自适应控制器实现',
            'key_classes': ['PIDController', 'LaneKeepingController', 
                          'SpeedController', 'AdaptiveController'],
            'key_methods': ['update(error)', 'get_control(obs)', 'reset()'],
            'usage': '# 创建控制器\ncontroller = AdaptiveController(target_speed=10.0)\naction = controller.get_control(obs)',
        },
    },
    
    '配置和示例': {
        'config.py': {
            'description': '配置预设和参数优化建议',
            'contents': ['LANE_KEEPING_CONFIG', 'SPEED_CONTROLLER_CONFIG', 
                        'REWARD_CONFIGS', 'ENVIRONMENT_DIFFICULTIES'],
            'usage': 'from config import get_pid_config, get_reward_config',
        },
        'example.py': {
            'description': '基础使用示例',
            'examples': ['PID控制演示', 'RL训练框架', '变速控制演示', 'PID测试'],
            'usage': 'python example.py',
        },
        'advanced_example.py': {
            'description': '高级用法和集成示例',
            'examples': ['RL代理训练', '加权奖励', '课程学习', '多场景测试',
                        '动态参数调整', '性能评估'],
            'usage': 'python advanced_example.py',
        },
    },
    
    '文档': {
        'README.md': {
            'description': '完整的项目文档和API说明',
            'sections': ['功能特性', '安装依赖', '快速开始', '核心组件',
                        '使用示例', '性能优化', '常见问题', '扩展功能'],
            'usage': '读取了解详细API和使用方法',
        },
        'USAGE_GUIDE.py': {
            'description': '详细的代码使用示例',
            'sections': ['基本使用', '自定义奖励', '使用PID', '单独使用控制器',
                        '动态调整参数', '获取信息', 'RL集成'],
            'usage': '查看具体代码示例',
        },
        'PROJECT_SUMMARY.md': {
            'description': '项目总结和快速导航',
            'sections': ['项目概览', '文件结构', '核心特性', '与RL框架集成'],
            'usage': '快速了解项目概况',
        },
        'QUICK_REFERENCE.md': {
            'description': '快速参考卡（最常用）',
            'sections': ['10秒快速开始', '模块导入速查', '环境使用', 
                        '控制器使用', '奖励函数', '调试技巧'],
            'usage': '快速查询API和常用代码片段',
        },
    },
}

# ============================================================================
# 🎯 典型工作流程
# ============================================================================

WORKFLOWS = {
    '快速上手': """
    1. 启动CARLA: ./CarlaUE4.sh -RenderOffScreen
    2. 查看快速参考: 阅读 QUICK_REFERENCE.md
    3. 运行基础示例: python example.py
    4. 修改参数测试不同配置
    """,
    
    'RL训练': """
    1. 定义奖励函数
    2. 创建环境: env = CarlaEnv(reward_fn=my_reward)
    3. 创建智能体
    4. 使用环境API进行训练
    5. 在不同地图上测试
    """,
    
    '自定义控制': """
    1. 理解观测空间 (7维向量)
    2. 创建控制器类
    3. 集成到环境中
    4. 测试和调优参数
    5. 评估性能指标
    """,
    
    '参数优化': """
    1. 从预设配置开始
    2. 评估当前性能
    3. 根据问题调整单个参数
    4. 记录实验结果
    5. 迭代优化
    """,
}

# ============================================================================
# 📊 API 速查表
# ============================================================================

API_REFERENCE = {
    'CarlaEnv': {
        'init': 'CarlaEnv(host="localhost", port=2000, town="Town03", reward_fn=None)',
        'methods': {
            'reset()': '重置环境，返回初始观测 (7,)',
            'step(action)': '执行动作，返回 (obs, reward, done, info)',
            'set_reward_fn(fn)': '设置自定义奖励函数',
            'close()': '关闭环境',
        },
        'properties': {
            'vehicle': 'CARLA车辆对象',
            'world': 'CARLA世界对象',
            'sensors': '传感器字典',
        },
    },
    
    'AdaptiveController': {
        'init': 'AdaptiveController(target_speed=10.0, dt=0.05)',
        'methods': {
            'get_control(obs)': '根据观测返回控制向量 [throttle, brake, steer]',
            'set_target_speed(speed)': '设置目标速度',
            'reset()': '重置控制器',
        },
        'attributes': {
            'lane_controller': 'LaneKeepingController',
            'speed_controller': 'SpeedController',
        },
    },
    
    'PIDController': {
        'init': 'PIDController(kp=1.0, ki=0.1, kd=0.3, dt=0.05)',
        'methods': {
            'update(error)': '更新并返回控制输出',
            'reset()': '重置内部状态',
            'set_gains(kp, ki, kd)': '调整PID参数',
        },
    },
}

# ============================================================================
# 🚀 快速启动脚本
# ============================================================================

QUICK_START_SCRIPTS = {
    '最简单的示例': """
from carla_env import CarlaEnv
from pid_controller import AdaptiveController

env = CarlaEnv()
controller = AdaptiveController()
obs = env.reset()

for _ in range(1000):
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
""",
    
    '自定义奖励示例': """
import numpy as np
from carla_env import CarlaEnv

def my_reward(env):
    obs = env._get_observation()
    dist = obs[4]
    speed = obs[6]
    return np.exp(-dist * 10) + (1 if 8 <= speed <= 12 else 0)

env = CarlaEnv(reward_fn=my_reward)
# ... 继续使用
""",
    
    'RL集成示例': """
from stable_baselines3 import PPO
from carla_env import CarlaEnv

env = CarlaEnv()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
""",
}

# ============================================================================
# 🔧 常用代码片段
# ============================================================================

CODE_SNIPPETS = {
    '获取观测信息': """
obs = env.reset()
vx, vy, yaw, yaw_rate, distance, heading, speed = obs
print(f"Speed: {speed:.2f} m/s, Distance: {distance:.3f} m")
""",
    
    '调整控制器参数': """
# 调整转向增益
controller.lane_controller.steering_controller.set_gains(kp=2.5, kd=0.6)

# 调整速度控制
controller.set_target_speed(12.0)

# 重置控制器
controller.reset()
""",
    
    '记录训练数据': """
trajectory = []
obs = env.reset()

for step in range(1000):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    
    trajectory.append({
        'step': step,
        'obs': obs.copy(),
        'action': action.copy(),
        'reward': reward,
        'done': done,
    })
    
    if done:
        break

# 保存为numpy文件
import numpy as np
np.save('trajectory.npy', trajectory)
""",
    
    '计算性能指标': """
import numpy as np

rewards = []
distances = []

obs = env.reset()
for _ in range(1000):
    action = controller.get_control(obs)
    obs, reward, done, info = env.step(action)
    
    rewards.append(reward)
    distances.append(obs[4])
    
    if done:
        break

print(f"Total Reward: {sum(rewards):.2f}")
print(f"Avg Distance: {np.mean(distances):.3f} m")
print(f"Std Distance: {np.std(distances):.3f} m")
""",
}

# ============================================================================
# 🎓 学习路径
# ============================================================================

LEARNING_PATH = """
初级 (1-2天):
  1. 阅读 QUICK_REFERENCE.md
  2. 运行 example.py 中的基础示例
  3. 尝试修改目标速度和地图
  4. 理解观测和动作空间

中级 (3-5天):
  1. 学习自定义奖励函数
  2. 实现自己的控制器
  3. 运行 advanced_example.py
  4. 理解PID参数调优

高级 (1-2周):
  1. 与RL框架集成 (Stable-Baselines3)
  2. 实现课程学习
  3. 多智能体系统
  4. 实际数据集生成

专家 (持续):
  1. 传感器集成
  2. 复杂场景设计
  3. 算法创新
  4. 性能优化
"""

# ============================================================================
# ✅ 检查清单
# ============================================================================

CHECKLIST = {
    '环境检查': [
        '✓ CARLA已安装',
        '✓ Python API可用 (import carla)',
        '✓ CARLA服务器可启动',
        '✓ 网络连接正常',
    ],
    
    '代码检查': [
        '✓ 所有导入正确',
        '✓ 环境创建成功',
        '✓ 重置返回观测',
        '✓ 步进返回正确格式',
    ],
    
    '性能检查': [
        '✓ 无异常错误',
        '✓ 奖励值合理',
        '✓ 车辆响应正常',
        '✓ 性能指标可接受',
    ],
    
    '优化检查': [
        '✓ 参数已调优',
        '✓ 多地图测试',
        '✓ 泛化能力良好',
        '✓ 文档已完善',
    ],
}

# ============================================================================
# 📞 常见问题快速解答
# ============================================================================

FAQ_QUICK = {
    '我如何开始?': '→ 读 QUICK_REFERENCE.md，运行 example.py',
    '如何自定义奖励?': '→ 查看 USAGE_GUIDE.py 中"自定义奖励函数"部分',
    'PID参数怎么调?': '→ 看 config.py 中的 TUNING_GUIDE',
    '如何与RL框架集成?': '→ 参考 advanced_example.py 和 README.md',
    '支持多智能体吗?': '→ 创建多个独立的环境实例即可',
    '性能太差怎么办?': '→ 查看 config.py 中的 TROUBLESHOOTING',
}

# ============================================================================
# 🎬 运行说明
# ============================================================================

EXECUTION_GUIDE = """
第一步: 启动CARLA服务器
    ./CarlaUE4.sh -RenderOffScreen
    或
    ./CarlaUE4.sh (带UI调试)

第二步: 打开新的终端窗口

第三步: 激活环境（如果需要）
    conda activate carla
    或
    source /path/to/venv/bin/activate

第四步: 运行脚本
    # 基础示例
    python example.py
    
    # 高级示例
    python advanced_example.py
    
    # 自己的脚本
    python your_script.py

第五步: 查看输出
    检查日志信息和性能指标
    
第六步: 关闭CARLA
    按Ctrl+C停止服务器
"""

# ============================================================================
# 📈 进度追踪
# ============================================================================

PROGRESS_TRACKER = """
Week 1:
  ☐ 安装和配置环境
  ☐ 理解API和观测空间
  ☐ 运行基础示例
  ☐ 尝试不同的地图

Week 2:
  ☐ 实现自定义奖励
  ☐ 调整PID参数
  ☐ 性能评估
  ☐ 文档学习

Week 3:
  ☐ RL框架集成
  ☐ 训练简单模型
  ☐ 多地图测试
  ☐ 结果分析

Week 4:
  ☐ 高级功能开发
  ☐ 性能优化
  ☐ 论文复现
  ☐ 项目总结
"""

# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CARLA强化学习环境框架 - 完整指南")
    print("=" * 70)
    
    print("\n📋 项目文件:")
    for category, files in PROJECT_FILES.items():
        print(f"\n{category}:")
        for fname, info in files.items():
            print(f"  • {fname}: {info['description']}")
    
    print("\n🚀 快速开始:")
    print(EXECUTION_GUIDE)
    
    print("\n📚 学习路径:")
    print(LEARNING_PATH)
    
    print("\n✅ 检查清单:")
    for category, items in CHECKLIST.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")
    
    print("\n❓ 常见问题:")
    for question, answer in FAQ_QUICK.items():
        print(f"  Q: {question}")
        print(f"  A: {answer}\n")
    
    print("=" * 70)
    print("准备好开始了吗? 运行: python example.py")
    print("=" * 70)
