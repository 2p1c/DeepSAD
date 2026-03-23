# Pipeline Agent Guide

## 目标
统一调度训练与评估，形成可复现实验闭环。

## 功能要求
- 加载 `configs/config.yaml`。
- 编排训练/测试流程。
- 记录日志并保存 checkpoint/result。

## 必须支持
- `python train.py`
- `python evaluate.py`

## 输出目录
- `logs/`
- `checkpoints/`
- `results/`
