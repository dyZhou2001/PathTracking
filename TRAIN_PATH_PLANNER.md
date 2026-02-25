# 图像到局部路径：训练脚本

这套训练脚本基于你采集的 `labels.jsonl`：
- 输入：单目 RGB 图像
- 输出：车辆坐标系下的未来路径点 `(x,y)`，按弧长重采样到 **15 个点**，对应 `s=1..15m`（间隔 1m）
- 额外输出：`remaining_length_m`（辅助标量监督）

## 依赖

- Python 3.8+
- `torch`
- `pillow`

（可选）GPU：CUDA 对应版本的 PyTorch。

## 数据要求

你的数据目录形如：

- `dataset/run_xxx/labels.jsonl`
- `dataset/run_xxx/images/00001234.png`

其中 `labels.jsonl` 每行必须包含：
- `image`: 相对路径（例如 `images/00001234.png`）
- `future_route_vehicle`: `[{"x":..,"y":..}, ...]`

## Baseline 训练

```bash
python train_path_planner_baseline.py --labels dataset/run_xxx/labels.jsonl --epochs 20
```

输出权重：
- `checkpoints_baseline/last.pt`
- `checkpoints_baseline/best.pt`（有验证集时）

## Transformer 训练

```bash
python train_path_planner_transformer.py --labels dataset/run_xxx/labels.jsonl --epochs 20
```

提示：该脚本默认 `--device cuda`。如果没有 GPU，可加 `--device cpu`：

```bash
python train_path_planner_transformer.py --labels dataset/run_xxx/labels.jsonl --epochs 20 --device cpu
```

输出权重：
- `checkpoints_transformer/last.pt`
- `checkpoints_transformer/best.pt`（有验证集时）

## 训练输出指标

训练过程中会打印验证集统计：
- `loss`：总损失（点回归 + 平滑正则 + remaining_length 辅助损失）
- `ade`：平均点误差（masked）
- `fde`：最远点误差（masked）

## 与控制器对接（留接口）

模型输出 15 个点，代表 `s=1..15m`。
你打算用 `6m` 处的点作为控制器 lookahead：
- 对应索引：`6m -> index=5`（因为从 1m 开始）

具体如何把点喂给你的控制器（Pure Pursuit/MPC）你后续可以自行调整。
