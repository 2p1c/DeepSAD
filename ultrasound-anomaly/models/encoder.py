"""1D encoder backbone for ultrasound windows."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class Conv1dEncoder(nn.Module):
    """1D CNN encoder focused on damage-sensitive guided-wave representation."""

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

        if len(channels) not in {3, 4}:
            raise ValueError("hidden_channels length must be 3 (MVP) or 4 (full recommended).")

        if len(channels) == 4:
            kernel_sizes = [7, 5, 5, 3]
            strides = [1, 2, 2, 2]
        else:
            # MVP: 3 blocks with unified kernel size 5
            kernel_sizes = [5, 5, 5]
            strides = [1, 2, 2]

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, k, s in zip(channels, kernel_sizes, strides):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    stride=s,
                    padding=k // 2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.hidden_channels = tuple(channels)
        # Minimum per-sample dynamic range before the conv stack.
        # Signals below this range are adaptively up-scaled for stability.
        self.min_input_absmax = 1e-3
        self.input_scale_eps = 1e-12
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(in_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input shape [B, 1, W], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"expected channel dimension 1, got {x.shape[1]}")

        x = x.float()
        # Guardrail for tiny-amplitude inputs (e.g. pm-level) to avoid
        # near-zero activations throughout the backbone.
        absmax = x.detach().abs().amax(dim=2, keepdim=True)
        safe_absmax = absmax.clamp_min(self.input_scale_eps)
        scale = torch.where(
            absmax < self.min_input_absmax,
            self.min_input_absmax / safe_absmax,
            torch.ones_like(safe_absmax),
        )
        x = x * scale
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        return self.projection(x)
