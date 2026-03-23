"""Signal preprocessing helpers: slicing and normalization."""

from __future__ import annotations

from typing import Literal

import numpy as np

NormalizationMode = Literal["zscore", "minmax", None]

_EPS = 1e-8


def _validate_1d_signal(signal: np.ndarray) -> np.ndarray:
    """Return a validated 1D signal as a NumPy array."""
    arr = np.asarray(signal)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got array with shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Signal must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError("Signal contains NaN or infinite values.")
    return arr.astype(np.float32, copy=False)


def sliding_window_1d(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Slice a 1D signal into overlapping windows.

    Returns an array with shape (n_windows, window_size). If the signal is
    shorter than the requested window size, an empty array is returned.
    """
    arr = _validate_1d_signal(signal)

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, got {window_size!r}.")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride!r}.")
    if window_size > arr.size:
        return np.empty((0, window_size), dtype=np.float32)

    starts = range(0, arr.size - window_size + 1, stride)
    windows = [arr[start : start + window_size] for start in starts]
    if not windows:
        return np.empty((0, window_size), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32, copy=False)


def normalize_windows(windows: np.ndarray, mode: NormalizationMode) -> np.ndarray:
    """
    Normalize each window independently.

    Supported modes:
    - ``zscore``: zero mean, unit variance
    - ``minmax``: rescale to [0, 1]
    - ``None``: no normalization
    """
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of windows, got shape {arr.shape}.")
    if arr.size == 0:
        return arr.copy()
    if not np.isfinite(arr).all():
        raise ValueError("Windows contain NaN or infinite values.")

    if mode is None:
        return arr.copy()
    if mode not in {"zscore", "minmax"}:
        raise ValueError("normalization mode must be one of {'zscore', 'minmax', None}.")

    work = arr.astype(np.float64, copy=False)
    if mode == "zscore":
        mean = work.mean(axis=1, keepdims=True)
        std = work.std(axis=1, keepdims=True)
        std = np.where(std < _EPS, 1.0, std)
        normalized = (work - mean) / std
    else:
        min_val = work.min(axis=1, keepdims=True)
        max_val = work.max(axis=1, keepdims=True)
        scale = max_val - min_val
        scale = np.where(scale < _EPS, 1.0, scale)
        normalized = (work - min_val) / scale

    return normalized.astype(np.float32, copy=False)


def extract_windows_from_signal(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    normalization: NormalizationMode,
) -> np.ndarray:
    """
    Extract windows from a 1D or 2D signal and optionally normalize them.

    For 2D input, each row is treated as an independent 1D signal and the
    resulting windows are concatenated.
    """
    arr = np.asarray(signal)
    if arr.ndim == 1:
        windows = sliding_window_1d(arr, window_size=window_size, stride=stride)
    elif arr.ndim == 2:
        pieces = [
            sliding_window_1d(row, window_size=window_size, stride=stride)
            for row in arr
        ]
        non_empty = [piece for piece in pieces if piece.size > 0]
        windows = (
            np.concatenate(non_empty, axis=0)
            if non_empty
            else np.empty((0, window_size), dtype=np.float32)
        )
    else:
        raise ValueError(f"Expected a 1D or 2D signal, got array with shape {arr.shape}.")

    return normalize_windows(windows, mode=normalization)
