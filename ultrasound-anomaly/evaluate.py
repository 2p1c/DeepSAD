"""Entry point for model evaluation and visualization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agents.data_agent import UltrasoundDataAgent
from agents.eval_agent import EvaluationAgent
from agents.model_agent import DeepSVDDAgent
from agents.pipeline_agent import PROJECT_ROOT, PipelineConfig


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _expand_signal_paths(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured data path does not exist: {path}")

        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(list(path.rglob("*.mat")) + list(path.rglob("*.npy")))

        for candidate in candidates:
            key = str(candidate.resolve())
            if key not in seen:
                expanded.append(candidate.resolve())
                seen.add(key)
    return expanded


def _apply_feature_norm(feat_vec: np.ndarray, feature_norm: dict | None) -> np.ndarray:
    feat = np.asarray(feat_vec, dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {feat.shape}.")
    if not feature_norm:
        return feat

    method = str(feature_norm.get("method", "zscore"))
    if method == "robust_iqr":
        center = np.asarray(feature_norm["center"], dtype=np.float32)
        scale = np.asarray(feature_norm["scale"], dtype=np.float32)
    else:
        # Backward compatibility with older checkpoints.
        center = np.asarray(feature_norm["mean"], dtype=np.float32)
        scale = np.asarray(feature_norm["std"], dtype=np.float32)

    if center.ndim != 1 or scale.ndim != 1:
        raise ValueError("feature_norm center/scale must be 1D vectors.")
    if feat.shape[1] != center.shape[0] or feat.shape[1] != scale.shape[0]:
        raise ValueError(
            "feature_norm dimension mismatch: "
            f"feature dim={feat.shape[1]}, center dim={center.shape[0]}, scale dim={scale.shape[0]}"
        )
    return (feat - center) / scale


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deep SVDD model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)

    ckpt_dir = Path(cfg.output.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (PROJECT_ROOT / ckpt_dir).resolve()
    checkpoint_path = ckpt_dir / "deep_svdd.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_norm = checkpoint.get("feature_norm")

    data_agent = UltrasoundDataAgent(
        window_size=cfg.data.window_size,
        stride=cfg.data.stride,
        normalization=cfg.data.normalization,
    )
    splits = data_agent.make_splits(
        train_healthy_paths=[],
        test_healthy_paths=_expand_signal_paths(cfg.data.test_healthy_paths),
        test_damaged_paths=_expand_signal_paths(cfg.data.test_damaged_paths),
    )
    test_samples = splits["test_samples"]
    if not test_samples:
        raise ValueError(
            "No test samples found. Please provide both data.test_healthy_paths and data.test_damaged_paths."
        )

    test_batch = data_agent.to_batch(test_samples)
    x_test = np.asarray(test_batch["x"], dtype=np.float32)
    feat_test_raw = np.asarray(test_batch["feat_vec"], dtype=np.float32)
    y_test = np.asarray(test_batch["y"], dtype=np.int64)
    path_ids = test_batch["path_ids"]

    fusion_enabled = cfg.fusion.enabled and cfg.fusion.mode == "late"
    if fusion_enabled and feature_norm:
        if "center" in feature_norm:
            feat_dim = int(len(feature_norm["center"]))
        else:
            feat_dim = int(len(feature_norm["mean"]))
    else:
        feat_dim = int(feat_test_raw.shape[1])
    feat_test = feat_test_raw
    if fusion_enabled and feature_norm is not None:
        feat_test = _apply_feature_norm(feat_test_raw, feature_norm)

    model = DeepSVDDAgent(
        window_size=cfg.data.window_size,
        embedding_dim=cfg.model.embedding_dim,
        hidden_channels=cfg.model.hidden_channels,
        fusion_enabled=fusion_enabled,
        fusion_feat_dim=feat_dim,
        fusion_feat_hidden_dim=cfg.fusion.feat_hidden_dim,
        fusion_dropout=cfg.fusion.dropout,
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    if "center_c" in checkpoint:
        center = np.asarray(checkpoint["center_c"], dtype=np.float32).reshape(-1)
    else:
        center = model.center_c.detach().cpu().numpy().reshape(-1)

    device = torch.device(cfg.train.device)
    model = model.to(device)
    x_tensor = torch.from_numpy(x_test).to(device)
    feat_tensor = torch.from_numpy(feat_test).to(device)

    with torch.no_grad():
        embeddings = model.forward(x_tensor, feat_vec=feat_tensor).detach().cpu().numpy()
    scores = model.score_samples(x_tensor, feat_vec=feat_tensor)

    evaluator = EvaluationAgent()
    window_metrics = evaluator.evaluate_window_level(y_test, scores)
    path_metrics = evaluator.evaluate_path_level(
        y_test, scores, path_ids, aggregate=cfg.eval.aggregate
    )
    pca_embeddings = evaluator.build_embedding_pca(embeddings, y_test, n_components=2)

    results_dir = Path(cfg.output.results_dir)
    if not results_dir.is_absolute():
        results_dir = (PROJECT_ROOT / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    score_distribution_path = results_dir / "score_distribution.png"
    roc_curve_path = results_dir / "roc_curve.png"
    pca_path = results_dir / "pca.png"

    if cfg.eval.plot_score_distribution:
        evaluator.plot_score_distribution(y_test, scores, score_distribution_path)
    if cfg.eval.plot_roc:
        evaluator.plot_roc(window_metrics["fpr"], window_metrics["tpr"], window_metrics["auc"], roc_curve_path)
    if cfg.eval.plot_pca:
        evaluator.plot_pca(pca_embeddings, y_test, pca_path)

    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "results_dir": str(results_dir),
        "data_source": "filesystem",
        "config": asdict(cfg),
        "window_level": {
            "auc": float(window_metrics["auc"]),
            "fpr": window_metrics["fpr"],
            "tpr": window_metrics["tpr"],
            "thresholds": window_metrics["thresholds"],
            "num_samples": int(len(y_test)),
            "healthy_count": int((np.asarray(y_test) == 0).sum()),
            "damaged_count": int((np.asarray(y_test) == 1).sum()),
            "healthy_score_mean": float(np.asarray(scores)[np.asarray(y_test) == 0].mean()),
            "damaged_score_mean": float(np.asarray(scores)[np.asarray(y_test) == 1].mean()),
        },
        "path_level": {
            "auc": float(path_metrics["auc"]),
            "fpr": path_metrics["fpr"],
            "tpr": path_metrics["tpr"],
            "thresholds": path_metrics["thresholds"],
            "aggregate": cfg.eval.aggregate,
            "num_paths": int(len(path_metrics["path_scores"])),
            "healthy_count": int((np.asarray(path_metrics["path_labels"]) == 0).sum()),
            "damaged_count": int((np.asarray(path_metrics["path_labels"]) == 1).sum()),
        },
    }

    metrics_path = results_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics), f, indent=2, ensure_ascii=False)

    print(f"Saved evaluation results to: {results_dir}")


if __name__ == "__main__":
    main()
