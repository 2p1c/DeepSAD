"""Data Agent: load raw 1D ultrasound signals, windowing, normalization, and leakage-safe splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.io import loadmat

from utils.preprocessing import extract_windows_from_signal


@dataclass(slots=True)
class WindowSample:
    """A single normalized window sample with its metadata."""

    window: np.ndarray
    label: int
    path_id: str
    source: str
    index: int


class UltrasoundDataAgent:
    """Build Deep SVDD-ready samples from raw ultrasound signals."""

    def __init__(self, window_size: int, stride: int, normalization: str | None):
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {window_size!r}.")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(f"stride must be a positive integer, got {stride!r}.")
        if normalization not in {"zscore", "minmax", None}:
            raise ValueError("normalization must be one of {'zscore', 'minmax', None}.")

        self.window_size = window_size
        self.stride = stride
        self.normalization = normalization

    @staticmethod
    def _validate_2d_signal(signal: np.ndarray, path: Path) -> np.ndarray:
        arr = np.asarray(signal)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"Expected loaded signal from {path} to be 1D or 2D, got shape {arr.shape}."
            )
        if arr.size == 0:
            raise ValueError(f"Signal loaded from {path} is empty.")
        if not np.isfinite(arr).all():
            raise ValueError(f"Signal loaded from {path} contains NaN or infinite values.")
        return arr

    @staticmethod
    def _path_id(path: Path) -> str:
        return str(path.expanduser().resolve())

    @staticmethod
    def _source_from_path(path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in {"mat", "npy"}:
            raise ValueError(f"Unsupported file extension for {path}: expected .mat or .npy.")
        return suffix

    def load_signal_file(self, path: str | Path) -> np.ndarray:
        """
        Load a raw ultrasound signal file.

        Supported formats:
        - .mat: requires variables ``x`` and ``y``
        - .npy: direct 1D or 2D NumPy array
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Signal file not found: {path_obj}")

        suffix = path_obj.suffix.lower()
        if suffix == ".mat":
            data = loadmat(path_obj)
            if "x" not in data or "y" not in data:
                raise KeyError(f"{path_obj} must contain MATLAB variables 'x' and 'y'.")

            x = np.asarray(data["x"]).squeeze()
            y = np.asarray(data["y"]).squeeze()
            if x.ndim != 1 or x.size == 0:
                raise ValueError(f"Variable 'x' in {path_obj} must be a non-empty 1D array.")
            if y.ndim == 0:
                raise ValueError(f"Variable 'y' in {path_obj} must not be a scalar.")
            if y.ndim > 2:
                raise ValueError(
                    f"Variable 'y' in {path_obj} must be 1D or 2D, got shape {y.shape}."
                )

            if y.ndim == 1:
                signal = y.reshape(1, -1)
            else:
                if y.shape[0] == x.size and y.shape[1] != x.size:
                    signal = y.T
                elif y.shape[1] == x.size:
                    signal = y
                elif y.shape[0] == x.size and y.shape[1] == x.size:
                    signal = y
                else:
                    raise ValueError(
                        f"Cannot align y shape {y.shape} with x length {x.size} in {path_obj}. "
                        "Expected one axis of y to match len(x)."
                    )
        elif suffix == ".npy":
            signal = np.load(path_obj, allow_pickle=False)
            signal = self._validate_2d_signal(signal, path_obj)
        else:
            raise ValueError(f"Unsupported file extension for {path_obj}: expected .mat or .npy.")

        return self._validate_2d_signal(signal, path_obj).astype(np.float32, copy=False)

    def build_samples(self, paths: Sequence[str | Path], label: int) -> List[WindowSample]:
        """Generate one WindowSample per extracted window for each input path."""
        if not isinstance(label, int):
            raise ValueError(f"label must be an integer, got {label!r}.")

        samples: List[WindowSample] = []
        for path in paths:
            path_obj = Path(path)
            signal = self.load_signal_file(path_obj)
            windows = extract_windows_from_signal(
                signal,
                window_size=self.window_size,
                stride=self.stride,
                normalization=self.normalization,
            )
            path_id = self._path_id(path_obj)
            source = self._source_from_path(path_obj)
            for index, window in enumerate(windows):
                samples.append(
                    WindowSample(
                        window=np.asarray(window, dtype=np.float32),
                        label=int(label),
                        path_id=path_id,
                        source=source,
                        index=index,
                    )
                )
        return samples

    def make_splits(
        self,
        train_healthy_paths: Sequence[str | Path],
        test_healthy_paths: Sequence[str | Path],
        test_damaged_paths: Sequence[str | Path],
    ) -> Dict[str, List[WindowSample]]:
        """Build leakage-safe train/test splits for Deep SVDD."""
        train_samples = self.build_samples(train_healthy_paths, label=0)
        test_samples = self.build_samples(test_healthy_paths, label=0) + self.build_samples(
            test_damaged_paths, label=1
        )
        return {
            "train_samples": train_samples,
            "test_samples": test_samples,
        }

    @staticmethod
    def to_arrays(samples: Sequence[WindowSample]) -> tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert samples to model-ready arrays.

        Returns:
        - X: float32 array with shape (N, 1, W)
        - y: int64 array with shape (N,)
        - path_ids: list of source file identifiers in sample order
        """
        if len(samples) == 0:
            return (
                np.empty((0, 1, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                [],
            )

        windows = []
        labels = []
        path_ids: List[str] = []
        expected_shape = None
        for sample in samples:
            window = np.asarray(sample.window, dtype=np.float32)
            if window.ndim != 1:
                raise ValueError(
                    f"Each sample.window must be 1D, got shape {window.shape} for {sample.path_id}."
                )
            if expected_shape is None:
                expected_shape = window.shape[0]
            elif window.shape[0] != expected_shape:
                raise ValueError(
                    "All windows must have the same length to build a batch. "
                    f"Expected {expected_shape}, got {window.shape[0]} for {sample.path_id}."
                )
            windows.append(window)
            labels.append(int(sample.label))
            path_ids.append(sample.path_id)

        x = np.stack(windows, axis=0).astype(np.float32, copy=False)[:, None, :]
        y = np.asarray(labels, dtype=np.int64)
        return x, y, path_ids
