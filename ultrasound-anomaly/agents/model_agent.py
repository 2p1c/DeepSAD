"""Model Agent: 1D encoder + optional feature late fusion + Deep SVDD/DeepSAD."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from models.encoder import Conv1dEncoder


class DeepSVDDAgent(nn.Module):
    """Deep SVDD / DeepSAD agent built on top of a lightweight 1D CNN encoder."""

    def __init__(
        self,
        window_size: int,
        embedding_dim: int = 128,
        hidden_channels: int | Iterable[int] = 32,
        fusion_enabled: bool = False,
        fusion_feat_dim: int = 7,
        fusion_feat_hidden_dim: int = 32,
        fusion_dropout: float = 0.0,
        objective: str = "deep_svdd",
        eta: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_channels = hidden_channels if isinstance(hidden_channels, int) else tuple(hidden_channels)
        self.fusion_enabled = bool(fusion_enabled)
        self.fusion_feat_dim = int(fusion_feat_dim)
        self.fusion_feat_hidden_dim = int(fusion_feat_hidden_dim)
        self.encoder = Conv1dEncoder(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim,
            hidden_channels=hidden_channels,
        )
        self.objective = self._normalize_objective(objective)
        self.eta = float(eta)
        if self.fusion_enabled:
            if self.fusion_feat_dim <= 0 or self.fusion_feat_hidden_dim <= 0:
                raise ValueError("fusion_feat_dim and fusion_feat_hidden_dim must be positive when fusion is enabled")
            self.feature_mlp = nn.Sequential(
                nn.Linear(self.fusion_feat_dim, self.fusion_feat_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.fusion_feat_hidden_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
            )
            self.fusion_projection = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.Dropout(p=float(fusion_dropout)),
            )
            self.fusion_norm = nn.LayerNorm(self.embedding_dim)
            self.fusion_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer("center_c", torch.zeros(self.embedding_dim))

    @staticmethod
    def _extract_inputs(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x = None
        aux = None
        y = None
        if torch.is_tensor(batch):
            x = batch
        elif isinstance(batch, dict):
            x_keys = ("x", "input", "inputs", "signal", "window", "series", "data")
            aux_keys = ("feat_vec", "aux", "features")
            label_keys = ("y", "label", "labels", "target", "targets", "anomaly_label", "is_anomaly")
            for key in x_keys:
                if key in batch:
                    x = batch[key]
                    break
            for key in aux_keys:
                if key in batch:
                    aux = batch[key]
                    break
            for key in label_keys:
                if key in batch:
                    y = batch[key]
                    break
            if x is None:
                x = next(iter(batch.values()))
        elif isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("empty batch received from dataloader")
            x = batch[0]
            aux = batch[1] if len(batch) > 1 else None
            y = batch[2] if len(batch) > 2 else None
        else:
            x = batch

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if aux is not None and not torch.is_tensor(aux):
            aux = torch.as_tensor(aux)
        if y is not None and not torch.is_tensor(y):
            y = torch.as_tensor(y)
        return x, aux, y

    @staticmethod
    def _normalize_objective(objective: str) -> str:
        normalized = str(objective).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "deep_svdd": "deep_svdd",
            "one_class": "deep_svdd",
            "oneclass": "deep_svdd",
            "deepsad": "deepsad",
            "semi_supervised": "deepsad",
            "semisupervised": "deepsad",
        }
        if normalized not in aliases:
            raise ValueError(
                "objective must be one of: deep_svdd, deepsad, one-class, semi-supervised"
            )
        return aliases[normalized]

    @staticmethod
    def _format_input(x: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = x.to(device=device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[1] != 1 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"expected input shape [B, 1, W], got {tuple(x.shape)}")
        return x

    @staticmethod
    def _format_feat_vec(
        feat_vec: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if feat_vec is None:
            return None
        feat_vec = feat_vec.to(device=device, dtype=torch.float32)
        if feat_vec.ndim != 2:
            raise ValueError(f"expected feature vector shape [B, F], got {tuple(feat_vec.shape)}")
        if feat_vec.shape[0] != batch_size:
            raise ValueError(
                f"feature batch size must match input batch size: {feat_vec.shape[0]} != {batch_size}"
            )
        return feat_vec

    @staticmethod
    def _format_labels(
        labels: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if labels is None:
            return None
        labels = labels.to(device=device)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        elif labels.ndim != 1:
            raise ValueError(f"expected label shape [B] or [B, 1], got {tuple(labels.shape)}")
        if labels.shape[0] != batch_size:
            raise ValueError(f"label batch size must match input batch size: {labels.shape[0]} != {batch_size}")
        return labels.to(dtype=torch.float32)

    def _compute_objective_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.center_c.to(device=z.device, dtype=z.dtype)
        sq_dist = torch.sum((z - c) ** 2, dim=1)
        svdd_loss = torch.mean(sq_dist)
        anomaly_penalty = torch.zeros((), device=z.device, dtype=z.dtype)

        if self.objective == "deepsad" and labels is not None:
            labels = self._format_labels(labels, batch_size=z.shape[0], device=z.device)
            if labels is not None:
                normal_mask = labels <= 0
                anomaly_mask = labels > 0
                eps = torch.tensor(1e-6, device=z.device, dtype=z.dtype)
                dist = torch.sqrt(sq_dist + eps)

                normal_term = sq_dist[normal_mask] if torch.any(normal_mask) else None
                anomaly_term = dist[anomaly_mask] if torch.any(anomaly_mask) else None

                loss_parts: list[torch.Tensor] = []
                if normal_term is not None:
                    loss_parts.append(torch.mean(normal_term))
                if anomaly_term is not None:
                    anomaly_penalty = torch.mean(self.eta / (anomaly_term + eps))
                    loss_parts.append(anomaly_penalty)

                if loss_parts:
                    if len(loss_parts) == 1:
                        objective_loss = loss_parts[0]
                    else:
                        objective_loss = torch.mean(torch.stack(loss_parts))
                    return objective_loss, svdd_loss, anomaly_penalty

        return svdd_loss, svdd_loss, anomaly_penalty

    def forward(self, x: torch.Tensor, feat_vec: torch.Tensor | None = None) -> torch.Tensor:
        x = self._format_input(x, next(self.parameters()).device)
        z_cnn = self.encoder(x)
        if not self.fusion_enabled:
            return z_cnn

        feat_vec = self._format_feat_vec(feat_vec, batch_size=x.shape[0], device=x.device)
        if feat_vec is None:
            raise ValueError("fusion is enabled but feat_vec was not provided")
        if feat_vec.shape[1] != self.fusion_feat_dim:
            raise ValueError(
                f"feature dimension mismatch: expected {self.fusion_feat_dim}, got {feat_vec.shape[1]}"
            )

        z_feat = self.feature_mlp(feat_vec)
        z_cat = torch.cat([z_cnn, z_feat], dim=1)
        fused_delta = self.fusion_projection(z_cat)
        return self.fusion_norm(z_cnn + self.fusion_alpha * fused_delta)

    def init_center_c(self, dataloader: Any, device: torch.device) -> torch.Tensor:
        """Initialize the hypersphere center from healthy training data."""

        device = torch.device(device)
        self.to(device)

        was_training = self.training
        self.eval()

        c = torch.zeros(self.embedding_dim, device=device)
        fallback_c = torch.zeros(self.embedding_dim, device=device)
        n_samples = 0
        fallback_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                x, feat_vec, labels = self._extract_inputs(batch)
                x = self._format_input(x, device)
                feat_vec = self._format_feat_vec(feat_vec, batch_size=x.shape[0], device=device)
                labels = self._format_labels(labels, batch_size=x.shape[0], device=device)
                z = self.forward(x, feat_vec=feat_vec)
                fallback_c += z.sum(dim=0)
                fallback_samples += z.shape[0]
                if self.objective == "deepsad" and labels is not None:
                    normal_mask = labels <= 0
                    if torch.any(normal_mask):
                        z = z[normal_mask]
                batch_size = z.shape[0]
                if batch_size == 0:
                    continue
                c += z.sum(dim=0)
                n_samples += batch_size

        if n_samples == 0:
            if fallback_samples == 0:
                raise ValueError("dataloader produced no samples; cannot initialize center c")
            c = fallback_c
            n_samples = fallback_samples

        c /= float(n_samples)

        # Avoid collapse by keeping any near-zero component away from exact zero.
        eps = torch.tensor(1e-6, device=device, dtype=c.dtype)
        near_zero = c.abs() < eps
        c = torch.where(near_zero, torch.where(c >= 0, eps, -eps), c)

        self.center_c.data.copy_(c)
        self.train(was_training)
        return self.center_c.detach().clone()

    def loss(
        self,
        x: torch.Tensor,
        feat_vec: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.forward(x, feat_vec=feat_vec)
        objective_loss, _, _ = self._compute_objective_loss(z, labels=y)
        return objective_loss

    def score_samples(self, x: torch.Tensor, feat_vec: torch.Tensor | None = None) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            z = self.forward(x, feat_vec=feat_vec)
            c = self.center_c.to(device=z.device, dtype=z.dtype)
            scores = torch.sum((z - c) ** 2, dim=1)
        return scores.detach().cpu().numpy()

    def fit(
        self,
        train_loader: Any,
        epochs: int,
        lr: float,
        weight_decay: float,
        device: torch.device,
        log_interval: int = 10,
        var_reg_weight: float = 0.2,
        target_embedding_std: float = 0.01,
        var_reg_warmup_epochs: int = 5,
        collapse_std_threshold: float = 0.001,
        collapse_patience: int = 5,
        min_epochs: int = 10,
    ) -> list[dict[str, float | int]]:
        device = torch.device(device)
        self.to(device)
        self.init_center_c(train_loader, device)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        history: list[dict[str, float | int]] = []
        self.train()
        collapse_counter = 0
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_svdd_loss = 0.0
            total_anomaly_penalty = 0.0
            total_var_penalty = 0.0
            total_samples = 0
            total_embedding_std = 0.0
            num_batches = 0

            for batch in train_loader:
                x, feat_vec, labels = self._extract_inputs(batch)
                x = self._format_input(x, device)
                feat_vec = self._format_feat_vec(feat_vec, batch_size=x.shape[0], device=device)
                labels = self._format_labels(labels, batch_size=x.shape[0], device=device)
                optimizer.zero_grad(set_to_none=True)
                z = self.forward(x, feat_vec=feat_vec)
                objective_loss, svdd_loss, anomaly_penalty = self._compute_objective_loss(z, labels=labels)
                batch_embedding_std = z.std(dim=0, unbiased=False).mean()
                std_deficit = torch.relu(
                    torch.tensor(float(target_embedding_std), device=z.device, dtype=z.dtype)
                    - batch_embedding_std
                )
                var_penalty = std_deficit * std_deficit
                warmup_scale = 1.0
                if var_reg_warmup_epochs > 0:
                    warmup_scale = min(1.0, float(epoch) / float(var_reg_warmup_epochs))
                loss = objective_loss + float(var_reg_weight) * warmup_scale * var_penalty
                loss.backward()
                optimizer.step()

                batch_size = x.shape[0]
                total_loss += loss.item() * batch_size
                total_svdd_loss += svdd_loss.item() * batch_size
                total_anomaly_penalty += anomaly_penalty.item() * batch_size
                total_var_penalty += var_penalty.item() * batch_size
                total_samples += batch_size
                total_embedding_std += batch_embedding_std.detach().item()
                num_batches += 1

            mean_loss = total_loss / max(total_samples, 1)
            mean_svdd_loss = total_svdd_loss / max(total_samples, 1)
            mean_anomaly_penalty = total_anomaly_penalty / max(total_samples, 1)
            mean_var_penalty = total_var_penalty / max(total_samples, 1)
            mean_embedding_std = total_embedding_std / max(num_batches, 1)
            record = {
                "epoch": epoch,
                "loss": float(mean_loss),
                "svdd_loss": float(mean_svdd_loss),
                "anomaly_penalty": float(mean_anomaly_penalty),
                "var_penalty": float(mean_var_penalty),
                "embedding_std": float(mean_embedding_std),
                "center_norm": float(self.center_c.norm().item()),
                "lr": float(lr),
            }
            history.append(record)

            if log_interval and (epoch % log_interval == 0 or epoch == epochs):
                print(
                    f"[DeepSVDDAgent] epoch {epoch}/{epochs} "
                    f"loss={mean_loss:.6f} svdd={mean_svdd_loss:.6f} "
                    f"anom_pen={mean_anomaly_penalty:.6f} "
                    f"var_pen={mean_var_penalty:.6f} embedding_std={mean_embedding_std:.6f}"
                )

            if epoch >= int(min_epochs):
                if mean_embedding_std < float(collapse_std_threshold):
                    collapse_counter += 1
                else:
                    collapse_counter = 0
                if collapse_counter >= int(collapse_patience):
                    print(
                        "[DeepSVDDAgent] Early stop triggered by anti-collapse guard: "
                        f"embedding_std<{collapse_std_threshold} for {collapse_counter} epochs."
                    )
                    break

        self.eval()
        return history
