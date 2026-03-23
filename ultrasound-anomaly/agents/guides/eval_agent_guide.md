# Evaluation Agent Guide

## 目标
验证模型是否能区分健康/损伤。

## 必须完成
1. anomaly score 分布图（健康 vs 损伤）。
2. ROC 曲线和 AUC。
3. PCA 或 t-SNE 可视化（embedding 空间）。

## 关键验证
- 损伤样本分数分布整体高于健康样本。
- 类间分离可视化可观察到明显趋势。

## 进阶方向
- 对比不同缺陷尺寸/深度下的检测能力。
