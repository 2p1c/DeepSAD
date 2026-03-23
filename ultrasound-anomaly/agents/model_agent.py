"""Model Agent: 1D encoder + Deep SVDD training and inference score."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from models.encoder import Conv1dEncoder


class DeepSVDDAgent(nn.Module):
    """Deep SVDD agent built on top of a lightweight 1D CNN encoder."""

    def __init__(
        self,
        window_size: int,
        embedding_dim: int = 128,
        hidden_channels: int | Iterable[int] = 32,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_channels = hidden_channels if isinstance(hidden_channels, int) else tuple(hidden_channels)
        self.encoder = Conv1dEncoder(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim,
            hidden_channels=hidden_channels,
        )
        self.register_buffer("center_c", torch.zeros(self.embedding_dim))

    @staticmethod
    def _extract_inputs(batch: Any) -> torch.Tensor:
        if torch.is_tensor(batch):
            x = batch
        elif isinstance(batch, dict):
            preferred_keys = ("x", "input", "inputs", "signal", "window", "series", "data")
            x = None
            for key in preferred_keys:
                if key in batch:
                    x = batch[key]
                    break
            if x is None:
                x = next(iter(batch.values()))
        elif isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("empty batch received from dataloader")
            x = batch[0]
        else:
            x = batch

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        return x

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._format_input(x, next(self.parameters()).device)
        return self.encoder(x)

    def init_center_c(self, dataloader: Any, device: torch.device) -> torch.Tensor:
        """Initialize the hypersphere center from healthy training data."""

        device = torch.device(device)
        self.to(device)

        was_training = self.training
        self.eval()

        c = torch.zeros(self.embedding_dim, device=device)
        n_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                x = self._format_input(self._extract_inputs(batch), device)
                z = self.encoder(x)
                batch_size = z.shape[0]
                c += z.sum(dim=0)
                n_samples += batch_size

        if n_samples == 0:
            raise ValueError("dataloader produced no samples; cannot initialize center c")

        c /= float(n_samples)

        # Avoid collapse by keeping any near-zero component away from exact zero.
        eps = torch.tensor(0.1, device=device, dtype=c.dtype)
        near_zero = c.abs() < eps
        c = torch.where(near_zero, torch.where(c >= 0, eps, -eps), c)

        self.center_c.data.copy_(c)
        self.train(was_training)
        return self.center_c.detach().clone()

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward(x)
        c = self.center_c.to(device=z.device, dtype=z.dtype)
        return torch.mean(torch.sum((z - c) ** 2, dim=1))

    def score_samples(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            z = self.forward(x)
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
    ) -> list[dict[str, float | int]]:
        device = torch.device(device)
        self.to(device)
        self.init_center_c(train_loader, device)

        optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        history: list[dict[str, float | int]] = []
        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_samples = 0
            total_embedding_std = 0.0
            num_batches = 0

            for batch in train_loader:
                x = self._format_input(self._extract_inputs(batch), device)
                optimizer.zero_grad(set_to_none=True)
                z = self.encoder(x)
                c = self.center_c.to(device=z.device, dtype=z.dtype)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                batch_size = x.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                total_embedding_std += z.detach().std(dim=0, unbiased=False).mean().item()
                num_batches += 1

            mean_loss = total_loss / max(total_samples, 1)
            mean_embedding_std = total_embedding_std / max(num_batches, 1)
            record = {
                "epoch": epoch,
                "loss": float(mean_loss),
                "embedding_std": float(mean_embedding_std),
                "center_norm": float(self.center_c.norm().item()),
                "lr": float(lr),
            }
            history.append(record)

            if log_interval and (epoch % log_interval == 0 or epoch == epochs):
                print(
                    f"[DeepSVDDAgent] epoch {epoch}/{epochs} "
                    f"loss={mean_loss:.6f} embedding_std={mean_embedding_std:.6f}"
                )

        self.eval()
        return history
