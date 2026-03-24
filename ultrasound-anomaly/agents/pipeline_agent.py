"""Pipeline Agent: config loading, orchestration, logging, checkpointing."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from agents.data_agent import UltrasoundDataAgent
from agents.eval_agent import EvaluationAgent
from agents.model_agent import DeepSVDDAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NORM_EPS = 1e-8


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_healthy_paths: list[str] = field(default_factory=list)
    test_healthy_paths: list[str] = field(default_factory=list)
    test_damaged_paths: list[str] = field(default_factory=list)
    window_size: int = 256
    stride: int = 128
    normalization: str = "zscore"
    split_by: str = "path"


@dataclass
class ModelConfig:
    embedding_dim: int = 128
    input_channels: int = 1
    hidden_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    svdd_objective: str = "one-class"


@dataclass
class FusionConfig:
    enabled: bool = False
    mode: str = "late"
    sources: list[str] = field(default_factory=lambda: ["time"])
    feat_dim: int = 7
    feat_hidden_dim: int = 32
    dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-6
    device: str = "cpu"


@dataclass
class EvalConfig:
    aggregate: str = "mean"
    plot_score_distribution: bool = True
    plot_roc: bool = True
    plot_pca: bool = True


@dataclass
class OutputConfig:
    logs_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    results_dir: str = "results"


@dataclass
class PipelineConfig:
    project_name: str = "ultrasound-anomaly-mvp"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PipelineConfig":
        path = Path(config_path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        data = DataConfig(**(payload.get("data") or {}))
        model = ModelConfig(**(payload.get("model") or {}))
        fusion = FusionConfig(**(payload.get("fusion") or {}))
        train = TrainConfig(**(payload.get("train") or {}))
        eval_cfg = EvalConfig(**(payload.get("eval") or {}))
        output = OutputConfig(**(payload.get("output") or {}))

        cfg = cls(
            project_name=str(payload.get("project_name", "ultrasound-anomaly-mvp")),
            seed=int(payload.get("seed", 42)),
            data=data,
            model=model,
            fusion=fusion,
            train=train,
            eval=eval_cfg,
            output=output,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.data.split_by != "path":
            raise ValueError("Only split_by='path' is supported.")
        if self.model.input_channels != 1:
            raise ValueError("MVP currently supports input_channels=1 only.")
        if self.data.window_size <= 0 or self.data.stride <= 0:
            raise ValueError("data.window_size and data.stride must be positive.")
        if self.train.batch_size <= 0 or self.train.epochs <= 0:
            raise ValueError("train.batch_size and train.epochs must be positive.")
        if self.fusion.mode not in {"late", "mid", "gated", "early"}:
            raise ValueError("fusion.mode must be one of {'late', 'mid', 'gated', 'early'}.")
        if self.fusion.feat_dim <= 0 or self.fusion.feat_hidden_dim <= 0:
            raise ValueError("fusion.feat_dim and fusion.feat_hidden_dim must be positive.")
        if self.fusion.dropout < 0.0 or self.fusion.dropout >= 1.0:
            raise ValueError("fusion.dropout must be in [0, 1).")


class PipelineAgent:
    @classmethod
    def run_train(cls, config_path: str | Path) -> dict[str, Any]:
        return cls()._run_train(config_path)

    def _run_train(self, config_path: str | Path) -> dict[str, Any]:
        cfg = PipelineConfig.from_yaml(config_path)
        self._set_seed(cfg.seed)

        device = torch.device(cfg.train.device)
        if device.type.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")

        data_agent = UltrasoundDataAgent(
            window_size=cfg.data.window_size,
            stride=cfg.data.stride,
            normalization=cfg.data.normalization,
        )

        train_healthy_paths = self._expand_signal_paths(cfg.data.train_healthy_paths)
        test_healthy_paths = self._expand_signal_paths(cfg.data.test_healthy_paths)
        test_damaged_paths = self._expand_signal_paths(cfg.data.test_damaged_paths)

        splits = data_agent.make_splits(
            train_healthy_paths=train_healthy_paths,
            test_healthy_paths=test_healthy_paths,
            test_damaged_paths=test_damaged_paths,
        )
        train_samples = splits["train_samples"]
        test_samples = splits["test_samples"]

        if not train_samples:
            raise ValueError(
                "No training samples were built. Please provide data.train_healthy_paths in configs/config.yaml."
            )

        train_batch = data_agent.to_batch(train_samples)
        x_train = np.asarray(train_batch["x"], dtype=np.float32)
        feat_train_raw = np.asarray(train_batch["feat_vec"], dtype=np.float32)

        fusion_enabled = cfg.fusion.enabled and cfg.fusion.mode == "late"
        feature_norm = None
        feat_train = feat_train_raw
        if fusion_enabled:
            feature_norm = self._build_feature_norm(feat_train_raw)
            feat_train = self._apply_feature_norm(feat_train_raw, feature_norm)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(feat_train)),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=False,
        )

        fusion_feat_dim = int(feat_train.shape[1]) if fusion_enabled else int(cfg.fusion.feat_dim)
        model = DeepSVDDAgent(
            window_size=cfg.data.window_size,
            embedding_dim=cfg.model.embedding_dim,
            hidden_channels=cfg.model.hidden_channels,
            fusion_enabled=fusion_enabled,
            fusion_feat_dim=fusion_feat_dim,
            fusion_feat_hidden_dim=cfg.fusion.feat_hidden_dim,
            fusion_dropout=cfg.fusion.dropout,
        )
        history = model.fit(
            train_loader=train_loader,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            device=device,
            log_interval=max(1, cfg.train.epochs // 10),
        )

        ckpt_dir = self._resolve_output_dir(cfg.output.ckpt_dir)
        logs_dir = self._resolve_output_dir(cfg.output.logs_dir)
        results_dir = self._resolve_output_dir(cfg.output.results_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = ckpt_dir / "deep_svdd.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "center_c": model.center_c.detach().cpu(),
            "config": asdict(cfg),
            "feature_dim": int(feat_train.shape[1]),
            "feature_norm": feature_norm,
        }
        torch.save(checkpoint, checkpoint_path)

        train_log: dict[str, Any] = {
            "project_name": cfg.project_name,
            "seed": cfg.seed,
            "config": asdict(cfg),
            "data": {
                "num_train_samples": int(x_train.shape[0]),
                "num_test_samples": int(len(test_samples)),
                "window_size": int(cfg.data.window_size),
                "stride": int(cfg.data.stride),
                "normalization": cfg.data.normalization,
                "feature_dim": int(feat_train.shape[1]),
                "feature_norm": feature_norm,
            },
            "history": history,
            "summary": {
                "final_loss": float(history[-1]["loss"]),
                "best_loss": float(min(item["loss"] for item in history)),
                "center_norm": float(model.center_c.norm().item()),
                "checkpoint": str(checkpoint_path),
            },
        }

        if test_samples:
            test_batch = data_agent.to_batch(test_samples)
            x_test = np.asarray(test_batch["x"], dtype=np.float32)
            feat_test_raw = np.asarray(test_batch["feat_vec"], dtype=np.float32)
            feat_test = feat_test_raw
            if fusion_enabled and feature_norm is not None:
                feat_test = self._apply_feature_norm(feat_test_raw, feature_norm)
            y_test = np.asarray(test_batch["y"], dtype=np.int64)
            test_path_ids = test_batch["path_ids"]
            test_scores = model.score_samples(
                torch.from_numpy(x_test),
                feat_vec=torch.from_numpy(feat_test),
            )
            evaluator = EvaluationAgent()
            window_metrics = evaluator.evaluate_window_level(y_test, test_scores)
            path_metrics = evaluator.evaluate_path_level(
                y_test, test_scores, test_path_ids, aggregate=cfg.eval.aggregate
            )
            train_log["evaluation"] = {
                "window_auc": float(window_metrics["auc"]),
                "path_auc": float(path_metrics["auc"]),
            }

        history_path = logs_dir / "train_history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=2, ensure_ascii=False)

        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Saved training log: {history_path}")
        return train_log

    @staticmethod
    def _resolve_output_dir(path_like: str) -> Path:
        path = Path(path_like)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    @staticmethod
    def _expand_signal_paths(paths: list[str]) -> list[Path]:
        expanded: list[Path] = []
        seen: set[str] = set()
        for item in paths:
            path = Path(item)
            if not path.is_absolute():
                path = (PROJECT_ROOT / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Configured data path does not exist: {path}")

            candidates: list[Path]
            if path.is_file():
                candidates = [path]
            else:
                mat_files = list(path.rglob("*.mat"))
                npy_files = list(path.rglob("*.npy"))
                candidates = sorted(mat_files + npy_files)

            for candidate in candidates:
                key = str(candidate.resolve())
                if key not in seen:
                    expanded.append(candidate.resolve())
                    seen.add(key)
        return expanded

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _build_feature_norm(feat_vec: np.ndarray) -> dict[str, list[float] | float]:
        feat = np.asarray(feat_vec, dtype=np.float32)
        if feat.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {feat.shape}.")
        if feat.size == 0:
            raise ValueError("Cannot build feature normalization stats from an empty feature matrix.")

        mean = feat.mean(axis=0)
        std = feat.std(axis=0)
        std = np.where(std < FEATURE_NORM_EPS, 1.0, std)
        return {
            "method": "zscore",
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "eps": float(FEATURE_NORM_EPS),
        }

    @staticmethod
    def _apply_feature_norm(feat_vec: np.ndarray, feature_norm: dict[str, Any]) -> np.ndarray:
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


__all__ = ["PipelineConfig", "PipelineAgent"]
