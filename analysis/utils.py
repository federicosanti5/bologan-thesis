"""
Utility functions for monitoring data analysis.

This module provides helper functions for:
- RAPL counter overflow handling (32-bit counters)
- stdout.log parsing (Gloss, Dloss, training metrics)
- Temporal resampling to uniform grid
- Safe mathematical operations with NaN handling
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


def handle_rapl_overflow(energy_series: pd.Series,
                         max_value: int = 2**32) -> pd.Series:
    """
    Handle RAPL counter overflow for 32-bit energy counters.

    RAPL (Running Average Power Limit) energy counters are 32-bit unsigned integers
    that overflow approximately every 262 seconds at 100W power consumption.

    Detection: E[i+1] < E[i] indicates overflow
    Correction: cumulative_energy += (MAX - E[i]) + E[i+1]

    Args:
        energy_series: pandas Series with raw RAPL energy values (microJoules)
        max_value: Maximum counter value before overflow (default: 2^32)

    Returns:
        pandas Series with corrected cumulative energy values

    Example:
        >>> raw_energy = pd.Series([100, 200, 50, 150])  # 50 < 200 = overflow
        >>> corrected = handle_rapl_overflow(raw_energy, max_value=256)
        >>> print(corrected)
        0    100
        1    200
        2    306  # (256-200) + 50
        3    406  # 306 + (150-50)

    Notes:
        - Assumes monotonically increasing energy consumption
        - Handles multiple overflow events in sequence
        - Returns cumulative energy (always increasing)
    """
    if len(energy_series) == 0:
        return energy_series

    corrected = np.zeros(len(energy_series), dtype=np.float64)
    cumulative = 0.0
    prev_value = 0.0

    for i, value in enumerate(energy_series):
        if pd.isna(value):
            corrected[i] = np.nan
            continue

        if i == 0:
            # First value: initialize
            cumulative = float(value)
            prev_value = float(value)
        else:
            if value < prev_value:
                # Overflow detected
                delta = (max_value - prev_value) + value
                cumulative += delta
            else:
                # Normal increment
                delta = value - prev_value
                cumulative += delta
            prev_value = float(value)

        corrected[i] = cumulative

    return pd.Series(corrected, index=energy_series.index)


def parse_stdout_losses(stdout_path: str) -> Dict[str, Union[List[float], List[int]]]:
    """
    Parse training losses and metrics from stdout.log file.

    Extracts from lines like:
    "Iter: 1000; Dloss: -0.0991; Gloss: -0.2144; TotalTime: 599.04; ..."

    Args:
        stdout_path: Path to train_stdout.log file

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
        >>> metrics = parse_stdout_losses('logs/train_stdout.log')
        >>> print(f"Iterations: {len(metrics['iterations'])}")
        >>> print(f"Final Gloss: {metrics['gloss'][-1]:.4f}")

    Raises:
        FileNotFoundError: If stdout_path doesn't exist
        ValueError: If no valid training data found in file
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

    size_pattern = re.compile(r'Training size X: \((\d+),\s*(\d+)\)')

    try:
        with open(stdout_path, 'r') as f:
            for line in f:
                # Parse training size
                size_match = size_pattern.search(line)
                if size_match and training_size is None:
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

    if len(iterations) == 0:
        raise ValueError(f"No training iteration data found in {stdout_path}")

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

    # Resample based on aggregation method
    if aggregation == 'mean':
        df_resampled = df.resample(freq).mean()
    elif aggregation == 'sum':
        df_resampled = df.resample(freq).sum()
    elif aggregation == 'max':
        df_resampled = df.resample(freq).max()
    elif aggregation == 'min':
        df_resampled = df.resample(freq).min()
    elif aggregation == 'median':
        df_resampled = df.resample(freq).median()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Forward-fill NaN values (assume previous value persists)
    df_resampled = df_resampled.ffill()

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
        timestamp_s: Timestamps in seconds (Unix epoch with decimals)

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
    """
    # Compute deltas
    delta_energy_uj = energy_uj.diff()  # microJoules
    delta_time_s = timestamp_s.diff()   # seconds

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
