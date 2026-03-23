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
        train = TrainConfig(**(payload.get("train") or {}))
        eval_cfg = EvalConfig(**(payload.get("eval") or {}))
        output = OutputConfig(**(payload.get("output") or {}))

        cfg = cls(
            project_name=str(payload.get("project_name", "ultrasound-anomaly-mvp")),
            seed=int(payload.get("seed", 42)),
            data=data,
            model=model,
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

        x_train, _, _ = data_agent.to_arrays(train_samples)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train)),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=False,
        )

        model = DeepSVDDAgent(
            window_size=cfg.data.window_size,
            embedding_dim=cfg.model.embedding_dim,
            hidden_channels=cfg.model.hidden_channels,
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
            x_test, y_test, test_path_ids = data_agent.to_arrays(test_samples)
            test_scores = model.score_samples(torch.from_numpy(x_test))
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


__all__ = ["PipelineConfig", "PipelineAgent"]
