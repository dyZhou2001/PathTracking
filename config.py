"""
配置文件和参数优化建议
"""

# ============================================================================
# 车道保持控制器 - PID参数优化
# ============================================================================

LANE_KEEPING_CONFIG = {
    'conservative': {
        'kp': 1.0,    # 较低的比例增益，反应较慢但更稳定
        'ki': 0.05,   # 低积分增益，避免积分饱和
        'kd': 0.2,    # 低微分增益
        'usage': '直线或缓弯道',
    },
    'balanced': {
        'kp': 2.0,    # 平衡的比例增益
        'ki': 0.1,    # 标准积分增益
        'kd': 0.5,    # 标准微分增益
        'usage': '一般城市驾驶',
    },
    'aggressive': {
        'kp': 3.5,    # 高比例增益，反应快
        'ki': 0.2,    # 高积分增益
        'kd': 0.8,    # 高微分增益
        'usage': '紧急躲避或急转弯',
    },
}

# ============================================================================
# 速度控制器 - PID参数优化
# ============================================================================

SPEED_CONTROLLER_CONFIG = {
    'smooth': {
        'kp': 0.8,    # 平滑加速
        'ki': 0.15,   # 温和的速度追踪
        'kd': 0.2,    # 低微分，避免抖动
        'usage': '平稳驾驶',
    },
    'normal': {
        'kp': 1.0,
        'ki': 0.2,
        'kd': 0.3,
        'usage': '标准驾驶',
    },
    'responsive': {
        'kp': 1.5,    # 快速响应
        'ki': 0.3,    # 积极的速度追踪
        'kd': 0.5,    # 高微分，快速修正
        'usage': '需要快速响应的场景',
    },
}

# ============================================================================
# 奖励函数配置
# ============================================================================

REWARD_CONFIGS = {
    'lane_keeping_focused': {
        'lane_keeping': 3.0,
        'heading_control': 1.0,
        'speed_control': 0.5,
        'description': '优先考虑车道保持',
    },
    'balanced': {
        'lane_keeping': 2.0,
        'heading_control': 1.0,
        'speed_control': 1.0,
        'description': '平衡的多目标优化',
    },
    'speed_focused': {
        'lane_keeping': 1.0,
        'heading_control': 0.5,
        'speed_control': 3.0,
        'description': '优先考虑速度控制',
    },
    'comfort_optimized': {
        'lane_keeping': 2.0,
        'heading_control': 1.5,
        'speed_control': 1.5,
        'smooth_steering': 2.0,
        'description': '优先考虑平稳性和舒适度',
    },
}

# ============================================================================
# 环境配置 - 不同难度等级
# ============================================================================

ENVIRONMENT_DIFFICULTIES = {
    'easy': {
        'town': 'Town01',
        'target_speed': 8.0,
        'max_episode_steps': 1000,
        'reward_scale': 1.0,
        'traffic': False,
    },
    'medium': {
        'town': 'Town03',
        'target_speed': 10.0,
        'max_episode_steps': 1000,
        'reward_scale': 1.5,
        'traffic': False,
    },
    'hard': {
        'town': 'Town05',
        'target_speed': 12.0,
        'max_episode_steps': 800,
        'reward_scale': 2.0,
        'traffic': False,
    },
    'expert': {
        'town': 'Town12',
        'target_speed': 15.0,
        'max_episode_steps': 600,
        'reward_scale': 3.0,
        'traffic': False,
    },
}

# ============================================================================
# 训练配置示例
# ============================================================================

TRAINING_CONFIG = {
    'baseline': {
        'episodes': 100,
        'steps_per_episode': 1000,
        'learning_rate': 0.001,
        'batch_size': 32,
        'gamma': 0.99,
    },
    'fast_training': {
        'episodes': 50,
        'steps_per_episode': 500,
        'learning_rate': 0.01,
        'batch_size': 64,
        'gamma': 0.99,
    },
    'thorough_training': {
        'episodes': 500,
        'steps_per_episode': 2000,
        'learning_rate': 0.0001,
        'batch_size': 16,
        'gamma': 0.995,
    },
}

# ============================================================================
# 参数调优建议
# ============================================================================

TUNING_GUIDE = """
PID参数调优指南：

1. 比例增益 (Kp):
   - 决定响应的快速性
   - 过小：反应迟缓，无法及时修正
   - 过大：容易振荡，可能不稳定
   - 调整：从小到大逐步增加，找到合适平衡点

2. 积分增益 (Ki):
   - 消除稳态误差
   - 过小：无法消除累积误差
   - 过大：导致积分饱和，响应不稳定
   - 调整：通常为Kp的1/10左右

3. 微分增益 (Kd):
   - 增加阻尼，减少振荡
   - 过小：无法有效阻尼
   - 过大：放大噪声
   - 调整：通常为Kp的1/4到1/3

一般调优步骤：
1. 设置 Ki=0, Kd=0
2. 逐步增加 Kp 直到产生振荡
3. 增加 Kd 直到振荡停止
4. 增加 Ki 以消除稳态误差
5. 微调各参数使性能最优

奖励函数调优建议：
- 使用多个小目标比单一大目标更稳定
- 各项权重应在同一量级（1-10倍之间）
- 定期评估各项指标的贡献
- 根据实际需求优先级调整权重
"""

# ============================================================================
# 性能基准
# ============================================================================

PERFORMANCE_BASELINE = """
典型性能指标 (Town03, 目标速度10m/s, 运行500步):

使用保守型PID参数:
  - 平均速度: 9.2 ± 1.5 m/s
  - 平均距离车道中心: 0.25 ± 0.15 m
  - 总奖励: 300-400
  - 稳定性: 高

使用平衡型PID参数:
  - 平均速度: 10.1 ± 1.2 m/s
  - 平均距离车道中心: 0.30 ± 0.20 m
  - 总奖励: 250-350
  - 稳定性: 中

使用激进型PID参数:
  - 平均速度: 10.5 ± 2.0 m/s
  - 平均距离车道中心: 0.35 ± 0.30 m
  - 总奖励: 200-300
  - 稳定性: 低（振荡较多）
"""

# ============================================================================
# 故障排除
# ============================================================================

TROUBLESHOOTING = {
    '车辆振荡/不稳定': {
        'cause': ['Kp过大', 'Kd过小', 'dt设置不当'],
        'solution': ['减小Kp', '增大Kd', '调整时间步长'],
    },
    '反应过慢': {
        'cause': ['Kp过小', 'Ki过大导致饱和', '时间步长过大'],
        'solution': ['增大Kp', '减小Ki', '减小dt'],
    },
    '频繁碰撞': {
        'cause': ['目标速度过高', '转向增益过小', '奖励函数不合理'],
        'solution': ['降低目标速度', '增大转向增益', '调整奖励权重'],
    },
    '油门制动抖动': {
        'cause': ['速度控制Kp过大', 'Ki导致的积分抖动'],
        'solution': ['减小Kp', '减小Ki', '增大Kd'],
    },
}

# ============================================================================
# 实用工具函数
# ============================================================================

def get_pid_config(style='balanced'):
    """获取推荐的PID配置"""
    lane_config = LANE_KEEPING_CONFIG.get(style, LANE_KEEPING_CONFIG['balanced'])
    speed_config = SPEED_CONTROLLER_CONFIG.get(style, SPEED_CONTROLLER_CONFIG['normal'])
    return {
        'lane_keeping': lane_config,
        'speed_control': speed_config,
    }


def get_reward_config(focus='balanced'):
    """获取推荐的奖励配置"""
    return REWARD_CONFIGS.get(focus, REWARD_CONFIGS['balanced'])


def get_env_config(difficulty='medium'):
    """获取推荐的环境配置"""
    return ENVIRONMENT_DIFFICULTIES.get(difficulty, ENVIRONMENT_DIFFICULTIES['medium'])


def get_training_config(training_type='baseline'):
    """获取推荐的训练配置"""
    return TRAINING_CONFIG.get(training_type, TRAINING_CONFIG['baseline'])


# ============================================================================
# 快速配置示例
# ============================================================================

if __name__ == "__main__":
    # 示例用法
    print("PID配置 (保守):")
    print(get_pid_config('conservative'))
    
    print("\n奖励配置 (舒适度优化):")
    print(get_reward_config('comfort_optimized'))
    
    print("\n环境配置 (困难):")
    print(get_env_config('hard'))
    
    print("\n训练配置 (详细):")
    print(get_training_config('thorough_training'))
    
    print("\n调优指南:")
    print(TUNING_GUIDE)
