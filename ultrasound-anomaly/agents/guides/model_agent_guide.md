# Model Agent Guide

## 目标
学习健康信号的特征分布，并输出异常分数。

## 模型结构
- 1D CNN Encoder。
- 输出 embedding `z`，默认维度 `128`。

## 训练目标
最小化 `||z - c||^2`（Deep SVDD）。

## 必须步骤
1. 用健康训练数据前向一次初始化中心 `c`。
2. 训练 encoder 收敛到健康分布。
3. 推理时输出窗口级 anomaly score（到 `c` 的距离）。

## 注意事项
- 不使用损伤数据参与训练。
- 监控 collapse（例如 embedding 方差过低）。
