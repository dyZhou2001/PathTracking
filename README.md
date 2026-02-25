# CARLA PathTracking（数据采集 → 图像到局部路径 → 闭环控制 → RL 微调）

这个项目包含两条主线：

1) **经典控制 / Gym 风格 CARLA 环境**：`carla_env.py` + `pid_controller.py`，支持 `reset()/step()`、自定义奖励、全局路线规划、调试绘制等。
2) **神经路径规划（image → local path）**：采集 `labels.jsonl` + RGB 图像，监督训练 Baseline/Transformer，闭环把网络输出喂给 Pure Pursuit + Speed PID，再可选用 PPO 在 CARLA 里做在线微调。

推荐从 [START_HERE.md](START_HERE.md) 开始（3 分钟跑通）。

---

## 目录结构（最新版）

```
PathTracking/
├── carla_env.py                      # Gym 风格 CARLA 环境（支持采集/相机/碰撞/压线）
├── pid_controller.py                 # Pure Pursuit + 速度 PID 等控制器
├── example.py                        # 采集数据 + 基础运行示例（键盘转向/自动定速）
├── test_nn_path_planner_control.py   # 闭环测试：图像→Transformer→控制器→CARLA
├── train_path_planner_baseline.py    # 监督学习：CNN baseline
├── train_path_planner_transformer.py # 监督学习：Transformer
├── viz_path_planner_predictions.py   # 预测可视化（9宫格 + 图像 inset）
├── train_path_planner_rl_ppo.py      # PPO 微调：Actor=Transformer（预训练），Critic=独立 CNN
├── rl_carla_path_env.py              # PPO 训练用 Gym 环境（图像观测 + 路径动作）
├── rl_transformer_policy.py          # SB3 自定义策略（actor/critic + 高斯探索）
├── nn_path_planner/                  # 数据集/几何/网络/损失/指标
├── dataset/                          # 采集输出（run_xxx/labels.jsonl + images/）
├── checkpoints_transformer/          # Transformer 权重（SL + RL）
└── checkpoints_baseline/             # Baseline 权重
```

---

## 快速开始（Windows）

### 0) 环境前置

- **CARLA Server**（例如 0.9.15）
- **Python 与 CARLA PythonAPI 版本必须匹配**（Windows 上常见问题）

建议先确认你当前 conda 环境已激活（你现在的工作区看起来是 `conda activate carla`）。

### 1) 启动 CARLA Server

在另一个终端里：

```powershell
cd D:\CARLA_0.9.15\WindowsNoEditor
./CarlaUE4.exe -RenderOffScreen
```

### 2) 跑通采集/控制示例

```powershell
cd d:\ZDY_Drift\PathTracking
python example.py
```

`example.py` 默认会启用数据采集（见文件顶部配置项，如 `COLLECT_DATASET=True`、`DATASET_DIR=dataset`）。

---

## 端到端工作流（推荐顺序）

### A. 采集数据（RGB + 参考轨迹标签）

入口：`example.py`

采集输出目录形如：

```
dataset/
  run_YYYYMMDD_HHMMSS/
    labels.jsonl
    images/
      00001234.png
      ...
```

`labels.jsonl` 每行是一条样本，至少需要字段：

- `image`: 相对路径（例如 `images/00001234.png`）
- `future_route_vehicle`: 未来路径点列表（车辆坐标系）`[{"x":..,"y":..}, ...]`

如果你用 Transformer 的 `--use_state`，还会读取：

- `vehicle.x`, `vehicle.y`, `vehicle.yaw_deg`（用于构造 `[x/scale, y/scale, sin(yaw), cos(yaw)]`）

### B. 监督训练（Baseline / Transformer）

#### Baseline（CNN）

```powershell
python train_path_planner_baseline.py --labels dataset\run_xxx\labels.jsonl --epochs 20
```

输出：

- `checkpoints_baseline/last.pt`
- `checkpoints_baseline/best.pt`（有验证集时）

提示：该脚本默认 `--device cuda`。如果你没有 GPU 或想用 CPU 训练：

```powershell
python train_path_planner_transformer.py --labels dataset\run_xxx\labels.jsonl --epochs 20 --device cpu
```
#### Transformer

```powershell
python train_path_planner_transformer.py --labels dataset\run_xxx\labels.jsonl --epochs 20
```

可选：加入额外状态输入（位置/朝向）：

```powershell
python train_path_planner_transformer.py --labels dataset\run_xxx\labels.jsonl --epochs 20 --use_state
```

输出：

- `checkpoints_transformer/last.pt`
- `checkpoints_transformer/best.pt`（有验证集时）

训练指标（验证集）：

- `loss`（点回归 + 平滑正则 + remaining_length 辅助损失）
- `ade` / `fde`（masked）

### C. 预测可视化（离线）

```powershell
python viz_path_planner_predictions.py --labels dataset\run_xxx\labels.jsonl --checkpoint checkpoints_transformer\best.pt --out viz_predictions_9.png
```

说明：

- GT（绿色）与预测（红色）都在**车辆坐标系**下绘制
- 默认标注 `6m` 前瞻点（`s=1..15m`，因此 `6m -> index=5`）

### D. 闭环测试（在线）

入口：`test_nn_path_planner_control.py`

Pipeline：

`RGB image -> TransformerPlannerNet -> 15 个局部点 -> 取 index=5 的 6m 点 -> AdaptiveController -> (throttle, brake, steer) -> CarlaEnv.step()`

运行：

```powershell
python test_nn_path_planner_control.py --checkpoint checkpoints_transformer\best.pt --device cuda
```

常用参数：

- `--lookahead_index 5`：默认 6m 点
- `--flip_pred_y`：调试左右符号（只用于排查坐标系约定问题）
- `--spectator_follow`：跟车视角

### E. PPO 微调（可选）

入口：`train_path_planner_rl_ppo.py`

项目约束（脚本内已固化）：

- Actor 必须是 **现有 TransformerPlannerNet**（从监督学习 checkpoint 初始化，输入/输出不变）
- Critic 是**独立的小网络**（不共享 actor 权重）
- 观测预处理必须与闭环脚本一致（Imagenet normalize）

训练：

```powershell
python train_path_planner_rl_ppo.py --sl_checkpoint checkpoints_transformer\best.pt --total_timesteps 200000 --device cuda
```

断点续训：

```powershell
python train_path_planner_rl_ppo.py --resume_zip checkpoints_transformer\ppo_rl_last.zip --total_timesteps 200000 --device cuda
```

输出（不会覆盖监督学习权重）：

- `checkpoints_transformer/best_rl.pt`
- `checkpoints_transformer/last_rl.pt`
- `checkpoints_transformer/ppo_rl_last.zip`（SB3 训练状态）

微调后闭环测试：

```powershell
python test_nn_path_planner_control.py --checkpoint checkpoints_transformer\best_rl.pt --device cuda
```

---

## CarlaEnv / 控制器速览

### 环境 API

```python
from carla_env import CarlaEnv

env = CarlaEnv(town="Town03", spawn_point_index=0, destination_index=1, goal_radius=3.0)
obs = env.reset()
obs, reward, done, info = env.step([0.3, 0.0, 0.0])
env.close()
```

观测 `obs`（7维）：

`[vx, vy, yaw, yaw_rate, distance_to_center, heading_error, speed]`

动作 `action`（3维）：

`[throttle ∈ [0,1], brake ∈ [0,1], steer ∈ [-1,1]]`

### 控制器（推荐）

```python
from pid_controller import AdaptiveController

controller = AdaptiveController(target_speed=5.0)
action = controller.get_control(obs)
```

---

## 依赖建议

最小训练/可视化依赖（监督学习 + 可视化）：

- `torch`
- `pillow`
- `matplotlib`
- `numpy`

在当前环境中安装（示例，按需调整为你的 CUDA 版本对应的 PyTorch）：

```powershell
pip install -U numpy pillow matplotlib
pip install -U torch
```

PPO 微调依赖（可选）：

- `stable-baselines3`（以及其依赖）
- `gym`（SB3 v1.x 使用旧 Gym API；脚本里做了兼容提示）

```powershell
pip install -U stable-baselines3 gym
```

CARLA 依赖：

- `carla` PythonAPI（egg/wheel 与 Python 版本匹配）

---

## 文档导航

- 3 分钟跑通：`START_HERE.md`
- 快速查表：`QUICK_REFERENCE.md`
- 训练脚本说明：`TRAIN_PATH_PLANNER.md`
- 项目索引：`INDEX.md`

---

## 常见问题（高频）

### 1) Windows 导入 carla 失败

通常是 **Python 版本与 CARLA egg 不匹配** 或缺少运行库/DLL。请使用与 CARLA PythonAPI 对应的 Python 版本，并确保 Python 能找到 carla egg/wheel。

### 2) 闭环抖动 / 车辆走偏

- 检查 `lookahead_index`（默认 `5` 对应 6m）
- 用 `--flip_pred_y` 快速排查左右符号约定
- 确保相机预处理与训练一致（本项目已统一为 Imagenet normalize）

### 3) CARLA 很慢

优先用 `-RenderOffScreen` 启动 CARLA，并尽量避免打开额外渲染/调试绘制。

