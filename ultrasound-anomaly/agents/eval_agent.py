"""Evaluation Agent: score distribution, ROC/AUC, PCA visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.metrics import aggregate_scores_by_path, auc_from_roc, roc_curve_binary


class EvaluationAgent:
    """Evaluation helpers for window-level and path-level anomaly analysis."""

    def __init__(self, model=None, center=None, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        if center is None:
            self.center = None
        else:
            center_arr = np.asarray(center, dtype=np.float32).reshape(-1)
            self.center = center_arr

    def evaluate_window_level(self, y_true, scores) -> dict:
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        fpr, tpr, thresholds = roc_curve_binary(y_true, scores)
        auc = auc_from_roc(fpr, tpr)
        return {
            "auc": float(auc),
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }

    def evaluate_path_level(self, y_true, scores, path_ids, aggregate: str = "mean") -> dict:
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        path_ids = np.asarray(path_ids).ravel()

        if not (len(y_true) == len(scores) == len(path_ids)):
            raise ValueError("y_true, scores and path_ids must have the same length")

        unique_paths = []
        path_labels = []
        seen = {}
        for label, path_id in zip(y_true, path_ids):
            key = path_id.item() if hasattr(path_id, "item") else path_id
            if key not in seen:
                seen[key] = len(unique_paths)
                unique_paths.append(key)
                path_labels.append(int(label))
            elif path_labels[seen[key]] != int(label):
                raise ValueError("path_id maps to multiple labels")

        aggregated_scores = aggregate_scores_by_path(scores, path_ids, method=aggregate)
        fpr, tpr, thresholds = roc_curve_binary(path_labels, aggregated_scores)
        auc = auc_from_roc(fpr, tpr)
        return {
            "auc": float(auc),
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "path_scores": aggregated_scores,
            "path_labels": np.asarray(path_labels, dtype=np.int64),
            "path_ids": np.asarray(unique_paths),
            "aggregate": aggregate,
        }

    def build_embedding_pca(self, embeddings, y_true, n_components: int = 2) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[0] == 0:
            return np.zeros((0, n_components), dtype=np.float64)

        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        if centered.shape[1] == 0:
            return np.zeros((centered.shape[0], n_components), dtype=np.float64)

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[: min(n_components, vt.shape[0])].T
        transformed = centered @ components

        if transformed.shape[1] < n_components:
            padding = np.zeros((transformed.shape[0], n_components - transformed.shape[1]), dtype=transformed.dtype)
            transformed = np.hstack([transformed, padding])
        return transformed[:, :n_components]

    def plot_score_distribution(self, y_true, scores, save_path) -> None:
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        scores = np.asarray(scores, dtype=np.float64).ravel()
        healthy = scores[y_true == 0]
        damaged = scores[y_true != 0]

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        bins = max(12, min(40, int(np.sqrt(max(len(scores), 1)))))
        ax.hist(healthy, bins=bins, alpha=0.7, label=f"Healthy (n={healthy.size})", color="#2a9d8f")
        ax.hist(damaged, bins=bins, alpha=0.7, label=f"Damaged (n={damaged.size})", color="#e76f51")
        ax.set_title("Anomaly Score Distribution")
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def plot_roc(self, fpr, tpr, auc, save_path) -> None:
        fpr = np.asarray(fpr, dtype=np.float64).ravel()
        tpr = np.asarray(tpr, dtype=np.float64).ravel()

        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
        ax.plot(fpr, tpr, color="#1d3557", linewidth=2, label=f"ROC AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#8d99ae", linewidth=1)
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="lower right")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def plot_pca(self, pca_embeddings, y_true, save_path) -> None:
        pca_embeddings = np.asarray(pca_embeddings, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        if pca_embeddings.ndim != 2 or pca_embeddings.shape[1] < 2:
            raise ValueError("pca_embeddings must be a 2D array with at least 2 columns")

        healthy = pca_embeddings[y_true == 0]
        damaged = pca_embeddings[y_true != 0]

        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
        ax.scatter(
            healthy[:, 0],
            healthy[:, 1],
            s=18,
            alpha=0.75,
            label=f"Healthy (n={healthy.shape[0]})",
            color="#2a9d8f",
            edgecolors="none",
        )
        ax.scatter(
            damaged[:, 0],
            damaged[:, 1],
            s=18,
            alpha=0.75,
            label=f"Damaged (n={damaged.shape[0]})",
            color="#e76f51",
            edgecolors="none",
        )
        ax.set_title("Embedding PCA Projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def score_embeddings(self, embeddings) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if self.center is None:
            return np.sum(np.square(embeddings), axis=1)
        center = np.asarray(self.center, dtype=np.float32).reshape(1, -1)
        if center.shape[1] != embeddings.shape[1]:
            raise ValueError("center and embeddings must have the same dimension")
        return np.sum(np.square(embeddings - center), axis=1)

    def infer_embeddings_and_scores(self, windows: np.ndarray, batch_size: int = 256):
        if self.model is None:
            raise ValueError("model is required for inference")

        self.model.eval()
        outputs = []
        scores = []
        windows = np.asarray(windows, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, windows.shape[0], batch_size):
                batch = torch.from_numpy(windows[start : start + batch_size]).to(self.device)
                batch = batch.unsqueeze(1) if batch.ndim == 2 else batch
                emb = self.model(batch).detach().cpu().numpy()
                outputs.append(emb)
                scores.append(self.score_embeddings(emb))

        embeddings = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)
        scores = np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)
        return embeddings, scores
