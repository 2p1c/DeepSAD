"""Signal preprocessing helpers: slicing and normalization."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import hilbert

NormalizationMode = Literal["zscore", "minmax", None]

_EPS = 1e-8
_BAND_RATIO_LO = 0.15
_BAND_RATIO_HI = 0.45
FEATURE_VECTOR_NAMES = (
    "time_energy",
    "envelope_energy",
    "band_energy_ratio",
    "phase_var",
    "mean_abs",
    "std",
    "crest_factor",
)
FEATURE_VECTOR_DIM = len(FEATURE_VECTOR_NAMES)


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


def extract_window_feature_vector(window: np.ndarray) -> np.ndarray:
    """
    Extract a compact feature vector from a single raw window.

    The vector is intentionally small and stable so it can be attached to the
    sample without changing the existing window tensor flow.
    """
    arr = _validate_1d_signal(window).astype(np.float64, copy=False)
    centered = arr - arr.mean()

    time_energy = float(np.mean(arr**2))
    envelope = np.abs(hilbert(centered))
    envelope_energy = float(np.mean(envelope**2))

    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    total_spectrum_energy = float(np.sum(spectrum))
    freqs = np.fft.rfftfreq(centered.size, d=1.0)
    band_mask = (freqs >= _BAND_RATIO_LO) & (freqs <= _BAND_RATIO_HI)
    band_energy = float(np.sum(spectrum[band_mask]))
    band_energy_ratio = band_energy / max(total_spectrum_energy, _EPS)

    phase = np.unwrap(np.angle(hilbert(centered)))
    phase_var = float(np.var(phase))

    mean_abs = float(np.mean(np.abs(arr)))
    std = float(np.std(arr))
    rms = float(np.sqrt(max(time_energy, 0.0)))
    crest_factor = float(np.max(np.abs(arr)) / max(rms, _EPS))

    return np.array(
        [
            time_energy,
            envelope_energy,
            band_energy_ratio,
            phase_var,
            mean_abs,
            std,
            crest_factor,
        ],
        dtype=np.float32,
    )


def extract_feature_vectors_from_windows(windows: np.ndarray) -> np.ndarray:
    """Vectorized wrapper for extracting features from a 2D window tensor."""
    arr = np.asarray(windows)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of windows, got shape {arr.shape}.")
    if arr.size == 0:
        return np.empty((0, FEATURE_VECTOR_DIM), dtype=np.float32)

    features = [extract_window_feature_vector(row) for row in arr]
    return np.stack(features, axis=0).astype(np.float32, copy=False)


def extract_windows_from_signal(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    normalization: NormalizationMode,
    return_features: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Extract windows from a 1D or 2D signal and optionally normalize them.

    For 2D input, each row is treated as an independent 1D signal and the
    resulting windows are concatenated.
    """
    arr = np.asarray(signal)
    if arr.ndim == 1:
        raw_windows = sliding_window_1d(arr, window_size=window_size, stride=stride)
    elif arr.ndim == 2:
        pieces = [
            sliding_window_1d(row, window_size=window_size, stride=stride)
            for row in arr
        ]
        non_empty = [piece for piece in pieces if piece.size > 0]
        raw_windows = (
            np.concatenate(non_empty, axis=0)
            if non_empty
            else np.empty((0, window_size), dtype=np.float32)
        )
    else:
        raise ValueError(f"Expected a 1D or 2D signal, got array with shape {arr.shape}.")

    feature_vectors = None
    if return_features:
        feature_vectors = extract_feature_vectors_from_windows(raw_windows)

    windows = normalize_windows(raw_windows, mode=normalization)
    if return_features:
        return windows, feature_vectors
    return windows
