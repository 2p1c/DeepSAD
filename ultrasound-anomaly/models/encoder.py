"""1D encoder backbone for ultrasound windows."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class Conv1dEncoder(nn.Module):
    """Lightweight 1D CNN encoder for Deep SVDD."""

    def __init__(
        self,
        window_size: int,
        embedding_dim: int,
        hidden_channels: int | Sequence[int],
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if isinstance(hidden_channels, int):
            if hidden_channels <= 0:
                raise ValueError("hidden_channels must be positive")
            channels: list[int] = [
                hidden_channels,
                hidden_channels * 2,
                hidden_channels * 4,
            ]
        else:
            channels = [int(ch) for ch in hidden_channels]
            if not channels:
                raise ValueError("hidden_channels must not be empty")
            if any(ch <= 0 for ch in channels):
                raise ValueError("hidden_channels values must be positive")

        layers: list[nn.Module] = []
        in_channels = 1
        for idx, out_channels in enumerate(channels):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=5,
                    padding=2,
                    bias=True,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            if idx < len(channels) - 1:
                layers.append(nn.MaxPool1d(kernel_size=2, ceil_mode=True))
            in_channels = out_channels

        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.hidden_channels = tuple(channels)
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(in_channels, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input shape [B, 1, W], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"expected channel dimension 1, got {x.shape[1]}")

        x = x.float()
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.projection(x)
