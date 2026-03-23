# Ultrasound Anomaly MVP

目标：仅用健康样本训练 Deep SVDD，验证是否能检测脱粘异常。

## 1. 数据准备指导

本项目输入是原始一维时域信号，当前支持两种文件格式：

- `.mat`：必须包含变量 `x`（时间轴）和 `y`（信号矩阵）
  - `y` 会被整理为 `(n_points, signal_length)`
- `.npy`：支持 1D 或 2D
  - 1D 会自动转成 `(1, signal_length)`

推荐目录（示例）：

```text
data/
  raw/
    healthy_train/
      h1.npy
      h2.npy
    healthy_test/
      h3.npy
    damaged_test/
      d1.npy
      d2.npy
```

在配置文件里按“路径划分”填写：

- `data.train_healthy_paths`：只放健康训练路径（文件或目录）
- `data.test_healthy_paths`：测试健康路径
- `data.test_damaged_paths`：测试损伤路径

注意：

- 训练集必须仅包含健康数据。
- 训练/测试必须按路径拆分，不能随机打散后划分。
- `window_size` 不应大于有效信号长度；若信号较短，代码会自动补齐窗口。

## 2. 训练参数配置指导

编辑文件：`configs/config.yaml`

最小可用配置示例：

```yaml
project_name: ultrasound-anomaly-mvp
seed: 42

data:
  raw_dir: data/raw
  processed_dir: data/processed
  train_healthy_paths:
    - data/raw/healthy_train
  test_healthy_paths:
    - data/raw/healthy_test
  test_damaged_paths:
    - data/raw/damaged_test
  window_size: 256
  stride: 128
  normalization: zscore
  split_by: path

model:
  embedding_dim: 128
  input_channels: 1
  hidden_channels: [16, 32, 64]
  svdd_objective: one-class

train:
  batch_size: 64
  epochs: 50
  lr: 0.001
  weight_decay: 1.0e-6
  device: cpu

eval:
  aggregate: mean
  plot_score_distribution: true
  plot_roc: true
  plot_pca: true

output:
  logs_dir: logs
  ckpt_dir: checkpoints
  results_dir: results
```

## 3. 训练参数解释

### data 段

- `train_healthy_paths`：健康训练数据路径列表（文件/目录都可）
- `test_healthy_paths`：健康测试路径列表
- `test_damaged_paths`：损伤测试路径列表
- `window_size`：滑动窗口长度（每个样本的时序长度）
- `stride`：窗口步长
- `normalization`：窗口归一化方式，支持 `zscore`、`minmax`
- `split_by`：固定为 `path`

### model 段

- `embedding_dim`：编码器输出特征维度
- `input_channels`：输入通道数，MVP 固定为 `1`
- `hidden_channels`：1D CNN 隐藏通道配置
- `svdd_objective`：当前使用 `one-class`

### train 段

- `batch_size`：训练批大小
- `epochs`：训练轮数
- `lr`：学习率
- `weight_decay`：权重衰减
- `device`：`cpu` 或 `cuda`

### eval 段

- `aggregate`：路径级分数聚合方式，`mean`/`max`/`p95`
- `plot_score_distribution`：是否保存分数分布图
- `plot_roc`：是否保存 ROC 曲线图
- `plot_pca`：是否保存 PCA 可视化图

### output 段

- `logs_dir`：训练日志目录
- `ckpt_dir`：模型权重目录
- `results_dir`：评估图表与指标目录

## 4. 如何执行推理（评估）

本项目把“推理”定义为：加载训练好的 Deep SVDD 模型，对测试窗口输出 anomaly score，并生成评估结果。

执行命令：

```bash
python evaluate.py --config configs/config.yaml
```

推理产物默认保存到 `results/`：

- `score_distribution.png`
- `roc_curve.png`
- `pca.png`
- `metrics.json`

其中 `metrics.json` 主要包含：

- `window_level`：窗口级 AUC、FPR/TPR、分数统计
- `path_level`：路径级 AUC、FPR/TPR、聚合方式与统计

## 5. 运行命令（整条 pipeline / 单独步骤）

### 5.1 运行整个 pipeline（推荐顺序）

```bash
python train.py --config configs/config.yaml
python evaluate.py --config configs/config.yaml
```

### 5.2 仅训练

```bash
python train.py --config configs/config.yaml
```

训练后会生成：

- `checkpoints/deep_svdd.pt`
- `logs/train_history.json`

### 5.3 仅推理/评估

```bash
python evaluate.py --config configs/config.yaml
```

前提：`checkpoints/deep_svdd.pt` 已存在，且测试路径已在配置中填写。
