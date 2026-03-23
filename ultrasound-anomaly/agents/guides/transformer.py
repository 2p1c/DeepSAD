"""
Data Transformer for Experimental .mat Files

Converts experimental ultrasonic signal data from .mat format to
the format required by train.py (numpy arrays in data/ directory).

Features:
- Loads noisy (21x21) and clean (41x41) signals from .mat files
- Interpolates noisy signals from 21x21 to 41x41 grid
- Performs data augmentation to increase dataset size
- Splits into training and validation sets
- Saves in format compatible with data_utils.py file mode

Usage:
    uv run python transformer.py --noisy path/to/noisy.mat --clean path/to/clean.mat
    uv run python transformer.py --noisy noisy.mat --clean clean.mat --augment_factor 5
"""

import argparse
import numpy as np
from scipy import io as sio
from scipy import interpolate
from pathlib import Path
from typing import Tuple


def load_mat_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load .mat file and extract x (time) and y (signal) data.

    Args:
        filepath: Path to .mat file

    Returns:
        Tuple of (time_vector, signal_data)
        - time_vector: shape (signal_length,)
        - signal_data: shape (n_points, signal_length)
    """
    print(f"[INFO] Loading {filepath}...")
    mat_data = sio.loadmat(filepath)

    # Extract x (time) and y (signal) variables
    x = mat_data["x"].flatten()  # Time vector
    y = mat_data["y"]  # Signal data (n_points, signal_length)

    # Ensure y is 2D with shape (n_points, signal_length)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    # Handle case where dimensions might be swapped
    if y.shape[0] > y.shape[1]:
        # Likely (signal_length, n_points), transpose it
        if y.shape[0] == len(x):
            y = y.T

    print(f"  Time vector shape: {x.shape}")
    print(f"  Signal data shape: {y.shape}")
    print(f"  Signal length: {len(x)}")
    print(f"  Number of points: {y.shape[0]}")

    return x, y


def reshape_to_grid(signal_data: np.ndarray, n_cols: int, n_rows: int) -> np.ndarray:
    """
    Reshape 1D array of signals to 3D grid (x, y, t).

    Assumes snake-scan order (standard for scanning systems).

    Args:
        signal_data: shape (n_points, signal_length)
        n_cols: Number of columns (x direction)
        n_rows: Number of rows (y direction)

    Returns:
        3D array of shape (n_cols, n_rows, signal_length)
    """
    signal_length = signal_data.shape[1]
    grid = np.zeros((n_cols, n_rows, signal_length), dtype=np.float64)

    for col in range(n_cols):
        start_idx = col * n_rows
        end_idx = (col + 1) * n_rows
        col_data = signal_data[start_idx:end_idx, :]

        # Snake scan correction: flip even columns (uncomment if needed)
        # if col % 2 == 1:
        #     col_data = np.flipud(col_data)

        grid[col, :, :] = col_data

    return grid


def flatten_grid(grid: np.ndarray) -> np.ndarray:
    """
    Flatten 3D grid back to 2D array of signals.

    Args:
        grid: shape (n_cols, n_rows, signal_length)

    Returns:
        2D array of shape (n_cols * n_rows, signal_length)
    """
    n_cols, n_rows, signal_length = grid.shape
    signals = np.zeros((n_cols * n_rows, signal_length), dtype=np.float64)

    for col in range(n_cols):
        start_idx = col * n_rows
        end_idx = (col + 1) * n_rows
        signals[start_idx:end_idx, :] = grid[col, :, :]

    return signals


def interpolate_spatial(
    grid_small: np.ndarray, target_cols: int, target_rows: int, method: str = "cubic"
) -> np.ndarray:
    """
    Spatially interpolate signal grid from smaller to larger size.

    Interpolates each time point independently in the spatial domain.

    IMPORTANT: Cubic interpolation has numerical precision issues with very small
    amplitude signals (< ~1e-10). This function automatically detects such signals
    and uses linear interpolation instead to preserve signal integrity.

    Args:
        grid_small: Input grid of shape (n_cols_small, n_rows_small, signal_length)
        target_cols: Target number of columns
        target_rows: Target number of rows
        method: Interpolation method ('linear', 'cubic'). Note: will automatically
                downgrade to 'linear' if signal amplitude is too small for cubic.

    Returns:
        Interpolated grid of shape (target_cols, target_rows, signal_length)
    """
    n_cols_small, n_rows_small, signal_length = grid_small.shape

    # Fast path: no spatial resize needed.
    # When source and target grids are identical, interpolation is a no-op.
    # Bypassing the interpolator avoids potential numerical artifacts on
    # ultra-small amplitude signals and preserves values exactly.
    if n_cols_small == target_cols and n_rows_small == target_rows:
        print(
            f"[INFO] Interpolating from {n_cols_small}x{n_rows_small} to {target_cols}x{target_rows} using '{method}' method..."
        )
        print("[INFO] Source and target grid sizes are identical; skipping interpolation.")
        grid_large = grid_small.astype(np.float64, copy=True)
        print(f"  Interpolation complete. Output shape: {grid_large.shape}")
        return grid_large

    # Check signal amplitude to determine best interpolation method
    signal_amplitude = grid_small.max() - grid_small.min()
    small_signal_threshold = 1e-9  # Signals below this should use linear interpolation

    # Automatically choose interpolation method based on signal amplitude
    original_method = method
    if signal_amplitude < small_signal_threshold and method == "cubic":
        method = "linear"
        print(
            f"[INFO] Signal amplitude ({signal_amplitude:.2e}) < threshold ({small_signal_threshold:.2e})"
        )
        print(
            f"[INFO] Automatically using 'linear' interpolation to preserve small signals"
        )
        print(
            f"       (cubic interpolation has numerical precision issues with signals < 1e-10)"
        )

    print(
        f"[INFO] Interpolating from {n_cols_small}x{n_rows_small} to {target_cols}x{target_rows} using '{method}' method..."
    )

    # Create coordinate grids
    x_small = np.linspace(0, 1, n_cols_small)
    y_small = np.linspace(0, 1, n_rows_small)
    x_large = np.linspace(0, 1, target_cols)
    y_large = np.linspace(0, 1, target_rows)

    # Output grid
    grid_large = np.zeros((target_cols, target_rows, signal_length), dtype=np.float64)

    # Interpolate each time point
    for t in range(signal_length):
        # Get spatial slice at time t
        slice_small = grid_small[:, :, t]

        # Create interpolation function
        interp_func = interpolate.RegularGridInterpolator(
            (x_small, y_small),
            slice_small,
            method=method,  # Use potentially adjusted method
            bounds_error=False,
            fill_value=None,
        )

        # Create target points
        xx_large, yy_large = np.meshgrid(x_large, y_large, indexing="ij")
        points = np.stack([xx_large.flatten(), yy_large.flatten()], axis=-1)

        # Interpolate
        slice_large = interp_func(points).reshape(target_cols, target_rows)
        grid_large[:, :, t] = slice_large

    print(f"  Interpolation complete. Output shape: {grid_large.shape}")

    # Verify that signal was preserved (not zeroed out)
    output_amplitude = grid_large.max() - grid_large.min()
    if (
        signal_amplitude > small_signal_threshold * 0.1
        and output_amplitude < signal_amplitude * 0.01
    ):
        print(
            f"  ⚠️  WARNING: Signal amplitude dropped significantly during interpolation!"
        )
        print(f"      Input amplitude: {signal_amplitude:.2e}")
        print(f"      Output amplitude: {output_amplitude:.2e}")
        print(f"      This may indicate a numerical precision issue.")

    return grid_large


def truncate_signals(
    time_vector: np.ndarray, signal_data: np.ndarray, target_length: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncate signals to target length.

    If signals are longer than target_length, keep the first target_length points.
    If signals are already at or shorter than target_length, return as-is.

    Args:
        time_vector: Time vector of shape (signal_length,)
        signal_data: Signal data of shape (n_points, signal_length)
        target_length: Target number of time points (default: 1000)

    Returns:
        Tuple of (truncated_time, truncated_signals)
    """
    current_length = signal_data.shape[1]

    if current_length <= target_length:
        print(
            f"[INFO] Signal length ({current_length}) <= target ({target_length}), no truncation needed."
        )
        return time_vector, signal_data

    print(f"[INFO] Truncating signals: {current_length} -> {target_length} points")
    truncated_time = time_vector[:target_length]
    truncated_signals = signal_data[:, :target_length]

    return truncated_time, truncated_signals


def normalize_signal(signal: np.ndarray, min_threshold: float = None) -> np.ndarray:
    """
    Normalize signal to [-1, 1] range with improved handling of small-amplitude signals.

    This function now properly handles signals with very small amplitudes (e.g., ~1e-10)
    by using a dynamic threshold based on floating-point precision instead of a fixed
    threshold. For signals with amplitude range below the threshold, the function
    preserves the signal's center value instead of zeroing it out.

    Args:
        signal: Input signal array
        min_threshold: Minimum threshold for amplitude range. If None, uses a very small
                      threshold based on machine epsilon (~1e-16 for float64), allowing
                      processing of extremely small but valid signals.

    Returns:
        Normalized signal in [-1, 1] range (dtype: float32)

    Examples:
        >>> # Normal signal
        >>> sig = np.array([0.0, 0.5, 1.0])
        >>> normalize_signal(sig)
        array([-1.,  0.,  1.], dtype=float32)

        >>> # Very small amplitude signal (previously would be zeroed)
        >>> small_sig = np.array([1e-12, 2e-12, 3e-12])
        >>> result = normalize_signal(small_sig)
        >>> # Now preserves the signal structure instead of returning zeros
    """
    min_val = signal.min()
    max_val = signal.max()
    amplitude_range = max_val - min_val

    # Use dynamic threshold based on floating-point precision
    if min_threshold is None:
        # Use smaller multiplier for better sensitivity to tiny signals
        # For float64: ~2.22e-15, threshold = ~2.22e-16 (100x smaller than before)
        # For float32: ~1.19e-7, threshold = ~1.19e-7
        # This allows handling of signals down to ~1e-15 range
        eps = np.finfo(signal.dtype).eps
        min_threshold = eps if signal.dtype == np.float32 else eps * 0.1

    if amplitude_range > min_threshold:
        # Normal normalization: map [min_val, max_val] to [-1, 1]
        normalized = 2 * (signal - min_val) / amplitude_range - 1
    else:
        # Signal is nearly flat (amplitude range < threshold)
        # Instead of zeroing, preserve the signal by centering it around 0
        # This handles very small but valid signals (e.g., ~1e-10 to ~1e-15 amplitude)
        center = (min_val + max_val) / 2

        # For extremely small signals, preserve their structure
        if amplitude_range > 0:
            # Signal has some variation, even if tiny - preserve it
            # Normalize to full [-1, 1] range to maximize signal visibility
            normalized = 2 * (signal - min_val) / amplitude_range - 1
        elif np.abs(center) > min_threshold:
            # Flat signal with DC offset
            # Center it and scale to small value
            normalized = (signal - center) / (np.abs(center) + min_threshold)
            normalized = np.clip(normalized, -1, 1)
        else:
            # Signal is essentially zero (flat and centered at zero)
            normalized = signal - center

    return normalized.astype(np.float32)


def augment_signal_pair(
    noisy: np.ndarray, clean: np.ndarray, augment_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation to a signal pair.

    Args:
        noisy: Noisy signal (signal_length,)
        clean: Clean signal (signal_length,)
        augment_type: Type of augmentation

    Returns:
        Tuple of (augmented_noisy, augmented_clean)
    """
    if augment_type == "time_shift":
        # Random circular shift
        shift = np.random.randint(-100, 100)
        noisy_aug = np.roll(noisy, shift)
        clean_aug = np.roll(clean, shift)

    elif augment_type == "amplitude_scale":
        # Random amplitude scaling
        scale = np.random.uniform(0.7, 1.3)
        noisy_aug = noisy * scale
        clean_aug = clean * scale

    elif augment_type == "flip":
        # Time reversal
        noisy_aug = np.flip(noisy).copy()
        clean_aug = np.flip(clean).copy()

    elif augment_type == "add_noise":
        # Add small synthetic noise (simulates measurement variation)
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.randn(len(noisy)) * noise_level
        noisy_aug = noisy + noise
        clean_aug = clean.copy()  # Keep clean signal unchanged

    elif augment_type == "time_stretch":
        # Slight time stretching/compression
        stretch_factor = np.random.uniform(0.95, 1.05)
        length = len(noisy)
        new_length = int(length * stretch_factor)

        # Resample
        x_old = np.linspace(0, 1, new_length)
        x_new = np.linspace(0, 1, length)

        noisy_stretched = np.interp(
            np.linspace(0, 1, new_length), np.linspace(0, 1, length), noisy
        )
        clean_stretched = np.interp(
            np.linspace(0, 1, new_length), np.linspace(0, 1, length), clean
        )

        # Resample back to original length
        noisy_aug = np.interp(x_new, x_old, noisy_stretched)
        clean_aug = np.interp(x_new, x_old, clean_stretched)

    elif augment_type == "window_crop":
        # Random window crop and resize
        crop_size = np.random.randint(800, 950)
        start = np.random.randint(0, len(noisy) - crop_size)

        noisy_crop = noisy[start : start + crop_size]
        clean_crop = clean[start : start + crop_size]

        # Resize back to original length
        noisy_aug = np.interp(
            np.linspace(0, 1, len(noisy)), np.linspace(0, 1, crop_size), noisy_crop
        )
        clean_aug = np.interp(
            np.linspace(0, 1, len(clean)), np.linspace(0, 1, crop_size), clean_crop
        )
    else:
        # No augmentation
        noisy_aug = noisy.copy()
        clean_aug = clean.copy()

    return noisy_aug.astype(np.float64), clean_aug.astype(np.float64)


def create_augmented_dataset(
    noisy_signals: np.ndarray, clean_signals: np.ndarray, augment_factor: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create augmented dataset from original signals.

    Args:
        noisy_signals: shape (n_points, signal_length)
        clean_signals: shape (n_points, signal_length)
        augment_factor: How many times to augment (1 = no augmentation)

    Returns:
        Tuple of (augmented_noisy, augmented_clean)
    """
    n_points, signal_length = noisy_signals.shape

    augment_types = [
        "time_shift",
        "amplitude_scale",
        "flip",
        "add_noise",
        "time_stretch",
        "window_crop",
    ]

    all_noisy = [noisy_signals.copy()]  # Include original
    all_clean = [clean_signals.copy()]

    print(f"[INFO] Augmenting data (factor={augment_factor})...")
    print(f"  Original pairs: {n_points}")

    for aug_round in range(augment_factor - 1):
        noisy_aug = np.zeros_like(noisy_signals)
        clean_aug = np.zeros_like(clean_signals)

        for i in range(n_points):
            # Randomly select augmentation type
            aug_type = np.random.choice(augment_types)
            noisy_aug[i], clean_aug[i] = augment_signal_pair(
                noisy_signals[i], clean_signals[i], aug_type
            )

        all_noisy.append(noisy_aug)
        all_clean.append(clean_aug)

    # Concatenate all
    final_noisy = np.concatenate(all_noisy, axis=0)
    final_clean = np.concatenate(all_clean, axis=0)

    print(f"  Augmented pairs: {final_noisy.shape[0]}")

    return final_noisy, final_clean


def save_dataset(
    noisy_signals: np.ndarray,
    clean_signals: np.ndarray,
    output_dir: str,
    train_ratio: float = 0.8,
    normalize: bool = True,
) -> None:
    """
    Save dataset in format compatible with data_utils.py file mode.

    Creates directory structure:
        output_dir/
            train/
                clean/
                    0000.npy, 0001.npy, ...
                noisy/
                    0000.npy, 0001.npy, ...
            val/
                clean/
                    0000.npy, 0001.npy, ...
                noisy/
                    0000.npy, 0001.npy, ...

    Args:
        noisy_signals: shape (n_samples, signal_length)
        clean_signals: shape (n_samples, signal_length)
        output_dir: Output directory path
        train_ratio: Ratio of training samples
        normalize: Whether to normalize signals to [-1, 1]
    """
    output_path = Path(output_dir)

    # Create directory structure
    dirs = [
        output_path / "train" / "clean",
        output_path / "train" / "noisy",
        output_path / "val" / "clean",
        output_path / "val" / "noisy",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    n_samples = noisy_signals.shape[0]

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    # Split indices
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    print(f"[INFO] Saving dataset to {output_dir}")
    print(f"  Total samples: {n_samples}")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")

    # Save training data
    for i, idx in enumerate(train_indices):
        noisy = noisy_signals[idx]
        clean = clean_signals[idx]

        if normalize:
            # Normalize each signal individually
            noisy = normalize_signal(noisy)
            clean = normalize_signal(clean)

        np.save(output_path / "train" / "noisy" / f"{i:04d}.npy", noisy)
        np.save(output_path / "train" / "clean" / f"{i:04d}.npy", clean)

    # Save validation data
    for i, idx in enumerate(val_indices):
        noisy = noisy_signals[idx]
        clean = clean_signals[idx]

        if normalize:
            noisy = normalize_signal(noisy)
            clean = normalize_signal(clean)

        np.save(output_path / "val" / "noisy" / f"{i:04d}.npy", noisy)
        np.save(output_path / "val" / "clean" / f"{i:04d}.npy", clean)

    print(f"  Dataset saved successfully!")

    # Save metadata
    metadata = {
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "signal_length": noisy_signals.shape[1],
        "normalized": normalize,
    }
    np.save(output_path / "metadata.npy", metadata)
    print(f"  Metadata saved to {output_path / 'metadata.npy'}")


def transform_data(
    noisy_path: str,
    clean_path: str,
    output_dir: str = "data",
    noisy_grid_size: Tuple[int, int] = (21, 21),
    clean_grid_size: Tuple[int, int] = (41, 41),
    augment_factor: int = 5,
    train_ratio: float = 0.8,
    interpolation_method: str = "cubic",
    normalize: bool = True,
    seed: int = 42,
    target_signal_length: int = 1000,
) -> None:
    """
    Main transformation pipeline.

    Args:
        noisy_path: Path to noisy signal .mat file
        clean_path: Path to clean signal .mat file
        output_dir: Output directory
        noisy_grid_size: (cols, rows) of noisy data grid
        clean_grid_size: (cols, rows) of clean data grid
        augment_factor: Data augmentation factor
        train_ratio: Training/validation split ratio
        interpolation_method: Spatial interpolation method
        normalize: Whether to normalize signals
        seed: Random seed
        target_signal_length: Target signal length, truncate if longer
    """
    np.random.seed(seed)

    print("=" * 60)
    print("Data Transformation Pipeline")
    print("=" * 60)

    # Load data
    print("\n[Step 1] Loading .mat files...")
    noisy_time, noisy_data = load_mat_file(noisy_path)
    clean_time, clean_data = load_mat_file(clean_path)
    print(
        f"[DEBUG] After load_mat_file: noisy min={noisy_data.min():.4g}, max={noisy_data.max():.4g}, mean={noisy_data.mean():.4g}"
    )

    # Truncate signals to target length if needed
    print(f"\n[Step 1.5] Checking signal length (target: {target_signal_length})...")
    noisy_time, noisy_data = truncate_signals(
        noisy_time, noisy_data, target_signal_length
    )
    clean_time, clean_data = truncate_signals(
        clean_time, clean_data, target_signal_length
    )
    print(
        f"[DEBUG] After truncate_signals: noisy min={noisy_data.min():.4g}, max={noisy_data.max():.4g}, mean={noisy_data.mean():.4g}"
    )

    # Verify dimensions
    noisy_n_cols, noisy_n_rows = noisy_grid_size
    clean_n_cols, clean_n_rows = clean_grid_size

    expected_noisy_points = noisy_n_cols * noisy_n_rows
    expected_clean_points = clean_n_cols * clean_n_rows

    assert noisy_data.shape[0] == expected_noisy_points, (
        f"Noisy data has {noisy_data.shape[0]} points, expected {expected_noisy_points}"
    )
    assert clean_data.shape[0] == expected_clean_points, (
        f"Clean data has {clean_data.shape[0]} points, expected {expected_clean_points}"
    )

    signal_length = noisy_data.shape[1]
    assert clean_data.shape[1] == signal_length, (
        f"Signal length mismatch: noisy={signal_length}, clean={clean_data.shape[1]}"
    )

    print(f"\n[Step 2] Reshaping to grids...")
    print(f"  Noisy: {noisy_n_cols}x{noisy_n_rows}x{signal_length}")
    print(f"  Clean: {clean_n_cols}x{clean_n_rows}x{signal_length}")

    # Reshape to 3D grids
    noisy_grid = reshape_to_grid(noisy_data, noisy_n_cols, noisy_n_rows)
    clean_grid = reshape_to_grid(clean_data, clean_n_cols, clean_n_rows)
    print(
        f"[DEBUG] After reshape_to_grid: noisy min={noisy_grid.min():.4g}, max={noisy_grid.max():.4g}, mean={noisy_grid.mean():.4g}"
    )

    print(f"\n[Step 3] Spatial interpolation...")
    # Interpolate noisy from 21x21 to 41x41
    noisy_grid_interp = interpolate_spatial(
        noisy_grid, clean_n_cols, clean_n_rows, method=interpolation_method
    )
    print(
        f"[DEBUG] After interpolate_spatial: noisy min={noisy_grid_interp.min():.4g}, max={noisy_grid_interp.max():.4g}, mean={noisy_grid_interp.mean():.4g}"
    )

    print(f"\n[Step 4] Flattening grids...")
    # Flatten back to 2D
    noisy_signals = flatten_grid(noisy_grid_interp)
    clean_signals = flatten_grid(clean_grid)
    print(
        f"[DEBUG] After flatten_grid: noisy min={noisy_signals.min():.4g}, max={noisy_signals.max():.4g}, mean={noisy_signals.mean():.4g}"
    )

    print(f"  Noisy signals shape: {noisy_signals.shape}")
    print(f"  Clean signals shape: {clean_signals.shape}")

    print(f"\n[Step 5] Data augmentation...")
    # Augment data
    noisy_aug, clean_aug = create_augmented_dataset(
        noisy_signals, clean_signals, augment_factor=augment_factor
    )
    print(
        f"[DEBUG] After create_augmented_dataset: noisy min={noisy_aug.min():.4g}, max={noisy_aug.max():.4g}, mean={noisy_aug.mean():.4g}"
    )

    print(f"\n[Step 6] Saving dataset...")
    # Save to file
    save_dataset(
        noisy_aug, clean_aug, output_dir, train_ratio=train_ratio, normalize=normalize
    )

    print("\n" + "=" * 60)
    print("Transformation Complete!")
    print("=" * 60)
    print(f"\nDataset saved to: {output_dir}/")
    print(f"  - Training:   {output_dir}/train/  (clean/ & noisy/)")
    print(f"  - Validation: {output_dir}/val/    (clean/ & noisy/)")


def main():
    parser = argparse.ArgumentParser(
        description="Transform experimental .mat data for neural network training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--noisy",
        type=str,
        required=True,
        help="Path to noisy signal .mat file (21x21 grid)",
    )
    parser.add_argument(
        "--clean",
        type=str,
        required=True,
        help="Path to clean signal .mat file (41x41 grid)",
    )

    # Optional arguments
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--noisy_cols", type=int, default=21, help="Number of columns in noisy grid"
    )
    parser.add_argument(
        "--noisy_rows", type=int, default=21, help="Number of rows in noisy grid"
    )
    parser.add_argument(
        "--clean_cols", type=int, default=41, help="Number of columns in clean grid"
    )
    parser.add_argument(
        "--clean_rows", type=int, default=41, help="Number of rows in clean grid"
    )
    parser.add_argument(
        "--augment_factor",
        type=int,
        default=5,
        help="Data augmentation factor (1=no augmentation)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--interp_method",
        type=str,
        default="cubic",
        choices=["linear", "cubic"],
        help="Spatial interpolation method",
    )
    parser.add_argument(
        "--no_normalize", action="store_true", help="Disable signal normalization"
    )
    parser.add_argument(
        "--signal_length",
        type=int,
        default=1000,
        help="Target signal length (truncate if longer)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    transform_data(
        noisy_path=args.noisy,
        clean_path=args.clean,
        output_dir=args.output,
        noisy_grid_size=(args.noisy_cols, args.noisy_rows),
        clean_grid_size=(args.clean_cols, args.clean_rows),
        augment_factor=args.augment_factor,
        train_ratio=args.train_ratio,
        interpolation_method=args.interp_method,
        normalize=not args.no_normalize,
        seed=args.seed,
        target_signal_length=args.signal_length,
    )


if __name__ == "__main__":
    main()
