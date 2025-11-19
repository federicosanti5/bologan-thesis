"""
Utility functions for monitoring data analysis.

This module provides helper functions for:
- RAPL counter overflow handling (32-bit counters)
- stdout.log parsing (Gloss, Dloss, training metrics)
- Temporal resampling to uniform grid
- Safe mathematical operations with NaN handling
"""

import re
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


def handle_rapl_overflow(energy_series: pd.Series,
                         max_value: int = None,
                         lookback_window: int = 5) -> pd.Series:
    """
    Handle RAPL counter overflow by estimating lost energy from previous power trend.

    When RAPL counters overflow/wrap, we lose data. This function detects wraps and
    estimates the energy consumed during the wrap interval based on the average power
    consumption observed before the wrap.

    Detection: E[i+1] < E[i] indicates overflow (simple consecutive comparison)
    Correction: Estimate delta from average power of previous samples

    Args:
        energy_series: pandas Series with raw RAPL energy values (microJoules)
        max_value: Unused (kept for API compatibility)
        lookback_window: Number of previous samples to use for power estimation (default: 5)

    Returns:
        pandas Series with corrected cumulative energy values

    Example:
        >>> # Normal samples at ~100 uJ/sample, then wrap occurs
        >>> raw_energy = pd.Series([100, 200, 300, 400, 50])  # 50 < 400 = overflow
        >>> corrected = handle_rapl_overflow(raw_energy)
        >>> # Delta estimated from avg power: (200-100 + 300-200 + 400-300)/3 = 100 uJ/sample
        >>> # So delta at wrap ≈ 100 uJ

    Notes:
        - Overflow detection based on consecutive value comparison only
        - Estimates lost energy from power trend before wrap
        - Handles multiple overflow events in sequence
        - Returns cumulative energy starting from 0 (always increasing)
    """
    if len(energy_series) == 0:
        return energy_series

    # Convert to float array for processing
    values = np.array(energy_series, dtype=np.float64)
    corrected = np.zeros(len(values), dtype=np.float64)

    # Track cumulative energy starting from 0
    cumulative = 0.0

    for i in range(len(values)):
        if pd.isna(values[i]):
            corrected[i] = np.nan
            continue

        if i == 0:
            # First value: initialize cumulative at 0
            corrected[i] = 0.0
        else:
            # Simple overflow detection: current < previous means wrap occurred
            if values[i] < values[i-1]:
                # Overflow/wrap detected: counter reset, data lost
                # Estimate energy from average power of previous samples

                # Calculate how many previous samples we can use
                start_idx = max(1, i - lookback_window)

                # Compute deltas (energy per interval) from previous samples
                prev_deltas = []
                for j in range(start_idx, i):
                    if not pd.isna(values[j]) and not pd.isna(values[j-1]):
                        # Only use non-wrap deltas
                        if values[j] >= values[j-1]:
                            prev_deltas.append(values[j] - values[j-1])

                # Estimate delta from average of previous deltas
                if prev_deltas:
                    delta = np.mean(prev_deltas)
                else:
                    # Fallback: if no previous valid deltas, use current value as rough estimate
                    delta = values[i]
            else:
                # Normal case: simple difference
                delta = values[i] - values[i-1]

            cumulative += delta
            corrected[i] = cumulative

    return pd.Series(corrected, index=energy_series.index)


def parse_stdout_losses(stdout_path: str, stderr_path: str = None) -> Dict[str, Union[List[float], List[int]]]:
    """
    Parse training losses and metrics from stdout.log and/or stderr.log file.

    Extracts from lines like:
    "Iter: 1000; Dloss: -0.0991; Gloss: -0.2144; TotalTime: 599.04; ..."

    Args:
        stdout_path: Path to train_stdout.log file
        stderr_path: Path to train_stderr.log file (optional, will try to infer if not provided)

    Returns:
        Dictionary with:
            - 'iterations': List of iteration numbers
            - 'dloss': List of discriminator losses
            - 'gloss': List of generator losses
            - 'total_time': List of cumulative times (seconds)
            - 'get_next_time': List of data loading times
            - 'train_loop_time': List of training loop times
            - 'training_size': Tuple (num_samples, num_features) or None

    Example:
        >>> metrics = parse_stdout_losses('logs/train_stdout.log', 'logs/train_stderr.log')
        >>> print(f"Iterations: {len(metrics['iterations'])}")
        >>> print(f"Final Gloss: {metrics['gloss'][-1]:.4f}")

    Raises:
        FileNotFoundError: If stdout_path doesn't exist
        ValueError: If no valid training data found in file

    Notes:
        - Training size is typically in stdout
        - Loss values may be in stdout or stderr (depends on logging configuration)
        - If stderr_path is not provided, will try stdout_path.replace('stdout', 'stderr')
    """
    iterations = []
    dloss_values = []
    gloss_values = []
    total_times = []
    get_next_times = []
    train_loop_times = []
    training_size = None

    # Regex patterns
    iter_pattern = re.compile(
        r'Iter:\s*(\d+);\s*'
        r'Dloss:\s*([-+]?\d*\.?\d+);\s*'
        r'Gloss:\s*([-+]?\d*\.?\d+);\s*'
        r'TotalTime:\s*([-+]?\d*\.?\d+);?\s*'
        r'(?:GetNext:\s*([-+]?\d*\.?\d+),?)?\s*'
        r'(?:.*TrainLoop:\s*([-+]?\d*\.?\d+))?'
    )

    # Multiple patterns for training size (different formats)
    size_pattern1 = re.compile(r'Training size X:\s*\((\d+),\s*(\d+)\)')
    size_pattern2 = re.compile(r'Training size[^\(]*\((\d+),\s*(\d+)\)')  # More flexible

    # If stderr_path not provided, infer from stdout_path
    if stderr_path is None:
        stderr_path = stdout_path.replace('stdout.log', 'stderr.log')

    # Parse stdout first (for training_size and potential loss values)
    try:
        with open(stdout_path, 'r') as f:
            for line in f:
                # Parse training size (try multiple patterns)
                if training_size is None:
                    size_match = size_pattern1.search(line)
                    if not size_match:
                        size_match = size_pattern2.search(line)
                    if size_match:
                        training_size = (int(size_match.group(1)), int(size_match.group(2)))

                # Parse iteration metrics
                iter_match = iter_pattern.search(line)
                if iter_match:
                    iterations.append(int(iter_match.group(1)))
                    dloss_values.append(float(iter_match.group(2)))
                    gloss_values.append(float(iter_match.group(3)))
                    total_times.append(float(iter_match.group(4)))

                    # Optional fields (may be None)
                    get_next = iter_match.group(5)
                    train_loop = iter_match.group(6)

                    get_next_times.append(float(get_next) if get_next else 0.0)
                    train_loop_times.append(float(train_loop) if train_loop else 0.0)

    except FileNotFoundError:
        raise FileNotFoundError(f"stdout.log not found: {stdout_path}")

    # If no iterations found in stdout, try stderr (loss values may be there)
    if len(iterations) == 0:
        from pathlib import Path
        if Path(stderr_path).exists():
            try:
                with open(stderr_path, 'r') as f:
                    for line in f:
                        # Parse training size from stderr too (may be duplicated there)
                        if training_size is None:
                            size_match = size_pattern1.search(line)
                            if not size_match:
                                size_match = size_pattern2.search(line)
                            if size_match:
                                training_size = (int(size_match.group(1)), int(size_match.group(2)))

                        # Parse iteration metrics
                        iter_match = iter_pattern.search(line)
                        if iter_match:
                            iterations.append(int(iter_match.group(1)))
                            dloss_values.append(float(iter_match.group(2)))
                            gloss_values.append(float(iter_match.group(3)))
                            total_times.append(float(iter_match.group(4)))

                            # Optional fields (may be None)
                            get_next = iter_match.group(5)
                            train_loop = iter_match.group(6)

                            get_next_times.append(float(get_next) if get_next else 0.0)
                            train_loop_times.append(float(train_loop) if train_loop else 0.0)

            except (FileNotFoundError, IOError):
                pass  # stderr not found or not readable, continue with what we have

    # Allow parsing even without iteration data if we found training_size
    if len(iterations) == 0 and training_size is None:
        raise ValueError(f"No training iteration data or training size found in {stdout_path} or {stderr_path}")
    elif len(iterations) == 0:
        # No iterations but we have training_size - still useful
        warnings.warn(f"No training iteration data found in {stdout_path} or {stderr_path}, but training_size parsed")

    return {
        'iterations': iterations,
        'dloss': dloss_values,
        'gloss': gloss_values,
        'total_time': total_times,
        'get_next_time': get_next_times,
        'train_loop_time': train_loop_times,
        'training_size': training_size,
    }


def resample_to_grid(df: pd.DataFrame,
                     timestamp_col: str = 'timestamp',
                     freq: str = '1s',
                     aggregation: str = 'mean') -> pd.DataFrame:
    """
    Resample DataFrame to uniform temporal grid.

    Aligns all monitoring data to a regular time grid (e.g., 1 second intervals)
    to enable direct comparison and correlation analysis across different data sources.

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column (default: 'timestamp')
        freq: Resampling frequency:
            - '1s': 1 second (1 Hz) - recommended
            - '0.5s' or '500ms': 0.5 seconds (2 Hz)
            - '100ms': 0.1 seconds (10 Hz)
        aggregation: Aggregation method for resampling:
            - 'mean': Average values in each bin (default, best for most metrics)
            - 'sum': Sum values (use for cumulative counters like energy deltas)
            - 'max': Maximum value in bin
            - 'min': Minimum value in bin
            - 'median': Median value

    Returns:
        Resampled DataFrame with uniform timestamps

    Example:
        >>> # Resample RAPL data (sampled at 2Hz) to 1Hz grid
        >>> rapl_df = pd.DataFrame({
        ...     'timestamp': [1.0, 1.5, 2.0, 2.5, 3.0],
        ...     'power_w': [150, 155, 152, 148, 151]
        ... })
        >>> resampled = resample_to_grid(rapl_df, freq='1s', aggregation='mean')
        >>> # Result: timestamps [1.0, 2.0, 3.0] with averaged power

    Notes:
        - Timestamp column is converted to datetime64[ns]
        - Original timestamps are lost (replaced by grid points)
        - NaN values are forward-filled after resampling
        - For energy data, consider aggregation='sum' for delta-energy columns
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")

    # Create a copy to avoid modifying original
    df = df.copy()

    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        # Try Unix timestamp first, fallback to ISO 8601
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
        except (ValueError, TypeError):
            # Fallback: parse as ISO 8601 or other formats
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Set timestamp as index for resampling
    df = df.set_index(timestamp_col)

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Resample numeric columns with specified aggregation
    if len(numeric_cols) > 0:
        df_numeric = df[numeric_cols]

        if aggregation == 'mean':
            df_numeric_resampled = df_numeric.resample(freq).mean()
        elif aggregation == 'sum':
            df_numeric_resampled = df_numeric.resample(freq).sum()
        elif aggregation == 'max':
            df_numeric_resampled = df_numeric.resample(freq).max()
        elif aggregation == 'min':
            df_numeric_resampled = df_numeric.resample(freq).min()
        elif aggregation == 'median':
            df_numeric_resampled = df_numeric.resample(freq).median()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Forward-fill NaN values
        df_numeric_resampled = df_numeric_resampled.ffill()
    else:
        df_numeric_resampled = pd.DataFrame()

    # Resample non-numeric columns (take first value in each bin)
    if len(non_numeric_cols) > 0:
        df_non_numeric = df[non_numeric_cols]
        df_non_numeric_resampled = df_non_numeric.resample(freq).first()

        # Forward-fill NaN values
        df_non_numeric_resampled = df_non_numeric_resampled.ffill()
    else:
        df_non_numeric_resampled = pd.DataFrame()

    # Combine numeric and non-numeric columns
    if len(numeric_cols) > 0 and len(non_numeric_cols) > 0:
        df_resampled = pd.concat([df_numeric_resampled, df_non_numeric_resampled], axis=1)
    elif len(numeric_cols) > 0:
        df_resampled = df_numeric_resampled
    else:
        df_resampled = df_non_numeric_resampled

    # Reset index to have timestamp as column again
    df_resampled = df_resampled.reset_index()

    return df_resampled


def safe_divide(numerator: Union[float, np.ndarray, pd.Series],
                denominator: Union[float, np.ndarray, pd.Series],
                default: float = np.nan) -> Union[float, np.ndarray, pd.Series]:
    """
    Safe division with handling of division by zero and NaN values.

    Args:
        numerator: Dividend (scalar, array, or Series)
        denominator: Divisor (scalar, array, or Series)
        default: Value to return when division is undefined (default: NaN)

    Returns:
        Result of division, with default value for undefined cases

    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        nan
        >>> safe_divide(10, 0, default=0.0)
        0.0
        >>> safe_divide(np.array([10, 20]), np.array([2, 0]))
        array([ 5., nan])

    Notes:
        - Handles zero denominators
        - Handles NaN in numerator or denominator
        - Preserves array/Series type if inputs are arrays/Series
    """
    # Handle scalar case
    if np.isscalar(numerator) and np.isscalar(denominator):
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator

    # Handle array/Series case
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    # Perform division
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator

    # Replace inf and nan with default
    result = np.where(np.isfinite(result), result, default)

    return result


def compute_power_from_energy(energy_uj: pd.Series,
                               timestamp_s: pd.Series) -> pd.Series:
    """
    Compute instantaneous power from cumulative energy measurements.

    Power = ΔEnergy / Δtime

    Args:
        energy_uj: Cumulative energy in microJoules (already overflow-corrected)
        timestamp_s: Timestamps in seconds (Unix epoch with decimals) or datetime64

    Returns:
        Instantaneous power in Watts

    Example:
        >>> energy = pd.Series([0, 100000, 200000])  # microJoules
        >>> time = pd.Series([0.0, 1.0, 2.0])  # seconds
        >>> power = compute_power_from_energy(energy, time)
        >>> print(power)  # [NaN, 0.1, 0.1] Watts (100 mW average)

    Notes:
        - First value is NaN (no previous point for delta)
        - Converts microJoules to Joules (÷ 1e6)
        - Handles irregular sampling intervals
        - Handles both numeric seconds and datetime64 timestamps
    """
    # Compute deltas
    delta_energy_uj = energy_uj.diff()  # microJoules
    delta_time = timestamp_s.diff()

    # Convert delta_time to seconds if it's timedelta64
    if pd.api.types.is_timedelta64_dtype(delta_time):
        delta_time_s = delta_time.dt.total_seconds()
    else:
        delta_time_s = delta_time  # already in seconds

    # Convert microJoules to Joules
    delta_energy_j = delta_energy_uj / 1e6

    # Power = Energy / Time (Watts = Joules / seconds)
    power_w = safe_divide(delta_energy_j, delta_time_s, default=np.nan)

    return power_w


def align_timestamps(df1: pd.DataFrame,
                     df2: pd.DataFrame,
                     timestamp_col: str = 'timestamp',
                     tolerance: str = '1s') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two DataFrames by timestamp using nearest-neighbor matching.

    Useful for correlating data from different monitoring sources that may not
    have exactly synchronized timestamps.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        timestamp_col: Name of timestamp column in both DataFrames
        tolerance: Maximum time difference for matching (e.g., '1s', '500ms')

    Returns:
        Tuple of (aligned_df1, aligned_df2) with matching timestamps

    Example:
        >>> df_rapl = pd.DataFrame({'timestamp': [1.0, 2.0, 3.0], 'power': [100, 110, 105]})
        >>> df_freq = pd.DataFrame({'timestamp': [1.1, 2.1, 2.9], 'freq': [2000, 2100, 2050]})
        >>> rapl_aligned, freq_aligned = align_timestamps(df_rapl, df_freq, tolerance='200ms')
        >>> # Now df1 and df2 have same length and aligned timestamps

    Notes:
        - Uses pandas merge_asof for efficient nearest-neighbor join
        - Timestamps outside tolerance are dropped
        - Original DataFrames are not modified (returns copies)
    """
    df1 = df1.copy()
    df2 = df2.copy()

    # Ensure timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(df1[timestamp_col]):
        df1[timestamp_col] = pd.to_datetime(df1[timestamp_col], unit='s')
    if not pd.api.types.is_datetime64_any_dtype(df2[timestamp_col]):
        df2[timestamp_col] = pd.to_datetime(df2[timestamp_col], unit='s')

    # Sort by timestamp (required for merge_asof)
    df1 = df1.sort_values(timestamp_col)
    df2 = df2.sort_values(timestamp_col)

    # Merge using nearest timestamp matching
    merged = pd.merge_asof(df1, df2, on=timestamp_col,
                           tolerance=pd.Timedelta(tolerance),
                           direction='nearest')

    return merged


# Module-level constants
RAPL_MAX_VALUE = 2**32  # microJoules
RAPL_OVERFLOW_THRESHOLD_SECONDS = 262  # ~262s at 100W

__all__ = [
    'handle_rapl_overflow',
    'parse_stdout_losses',
    'resample_to_grid',
    'safe_divide',
    'compute_power_from_energy',
    'align_timestamps',
    'RAPL_MAX_VALUE',
    'RAPL_OVERFLOW_THRESHOLD_SECONDS',
]
