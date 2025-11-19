"""
Monitoring Data Loader - Load and preprocess monitoring CSV files

This module handles loading all CSV files from experiment directories,
applying necessary preprocessing (RAPL overflow, resampling, validation).
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import warnings

try:
    from .utils import (
        handle_rapl_overflow,
        resample_to_grid,
        parse_stdout_losses,
    )
except ImportError:
    from utils import (
        handle_rapl_overflow,
        resample_to_grid,
        parse_stdout_losses,
    )


class MonitoringDataLoader:
    """
    Load and preprocess monitoring data from experiment directories.

    This class handles:
    - Loading all 12+ CSV files from system/process monitoring
    - RAPL overflow correction
    - Timestamp standardization and resampling
    - Data validation and integrity checks
    """

    # Expected CSV files mapping
    CSV_FILES = {
        # System monitoring
        'rapl': 'system_monitoring/train_system_energy_rapl.csv',
        'cpu_freq': 'system_monitoring/train_system_cpu_freq.csv',
        'thermal': 'system_monitoring/train_system_thermal.csv',
        'vmstat': 'system_monitoring/train_system_vmstat.csv',
        'iostat_cpu': 'system_monitoring/train_system_iostat_cpu.csv',
        'iostat_dev': 'system_monitoring/train_system_iostat_dev.csv',
        'free_mem': 'system_monitoring/train_system_free_mem.csv',
        'free_swap': 'system_monitoring/train_system_free_swap.csv',
        'pidstat_system': 'system_monitoring/train_system_pidstat.csv',
        'monitoring_overhead': 'system_monitoring/train_system_monitoring_overhead.csv',

        # Process monitoring
        'pidstat_process': 'process_monitoring/train_process_pidstat.csv',
        'process_io': 'process_monitoring/train_process_io.csv',
        'perf': 'process_monitoring/train_process_perf.csv',
    }

    # Metadata files
    METADATA_FILES = {
        'experiment': 'metadata/experiment_metadata.json',
        'summary': 'metadata/execution_summary.json',
    }

    # Log files
    LOG_FILES = {
        'stdout': 'logs/train_stdout.log',
        'stderr': 'logs/train_stderr.log',
    }

    def __init__(self, verbose: bool = True):
        """
        Initialize the data loader.

        Args:
            verbose: Print loading progress and warnings
        """
        self.verbose = verbose
        self.experiment_dir = None
        self.loaded_files = {}
        self.missing_files = []

    def load_experiment(self,
                       exp_dir: str,
                       resample_freq: Optional[str] = '1s',
                       validate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all monitoring data from an experiment directory.

        Args:
            exp_dir: Path to experiment directory (e.g., results/monitoring/exp_*)
            resample_freq: Resampling frequency ('1s', '0.5s', None to skip)
            validate: Run data validation checks

        Returns:
            Dictionary of DataFrames:
                {
                    'rapl': DataFrame,
                    'cpu_freq': DataFrame,
                    'vmstat': DataFrame,
                    ...
                    'metadata': dict,
                    'training_metrics': dict (from stdout parsing)
                }

        Raises:
            FileNotFoundError: If experiment directory doesn't exist
            ValueError: If critical files are missing

        Example:
            >>> loader = MonitoringDataLoader()
            >>> data = loader.load_experiment('results/monitoring/exp_20251117_203747')
            >>> print(f"Loaded {len(data)} datasets")
            >>> print(f"RAPL data: {len(data['rapl'])} rows")
        """
        self.experiment_dir = Path(exp_dir)

        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        if self.verbose:
            print(f"Loading experiment: {self.experiment_dir.name}")
            print("="*60)

        dfs = {}

        # Load CSV files
        for key, rel_path in self.CSV_FILES.items():
            csv_path = self.experiment_dir / rel_path
            if csv_path.exists():
                if self.verbose:
                    print(f"  Loading {key:20s} ... ", end='', flush=True)

                try:
                    # Special handling for specific files
                    if key == 'rapl':
                        df = self._load_rapl_with_overflow(csv_path)
                    elif key == 'perf':
                        df = self._load_perf_csv(csv_path)
                    else:
                        df = self._load_csv_generic(csv_path)

                    dfs[key] = df
                    self.loaded_files[key] = csv_path

                    if self.verbose:
                        print(f"✓ ({len(df):6d} rows)")

                except Exception as e:
                    if self.verbose:
                        print(f"✗ ERROR: {e}")
                    warnings.warn(f"Failed to load {key}: {e}")
            else:
                self.missing_files.append(key)
                if self.verbose:
                    print(f"  {key:20s} ... ✗ NOT FOUND")

        # Load metadata
        dfs['metadata'] = self._load_metadata()

        # Parse training metrics from stdout
        stdout_path = self.experiment_dir / self.LOG_FILES['stdout']
        if stdout_path.exists():
            try:
                dfs['training_metrics'] = parse_stdout_losses(str(stdout_path))
                if self.verbose:
                    n_iter = len(dfs['training_metrics']['iterations'])
                    print(f"  Training metrics     ... ✓ ({n_iter:6d} iterations)")
            except Exception as e:
                if self.verbose:
                    print(f"  Training metrics     ... ✗ ERROR: {e}")
                warnings.warn(f"Failed to parse stdout: {e}")
        else:
            if self.verbose:
                print(f"  Training metrics     ... ✗ NOT FOUND")

        # Resample to uniform grid
        if resample_freq is not None:
            if self.verbose:
                print(f"\nResampling to {resample_freq} grid...")
            dfs = self._resample_all_to_grid(dfs, freq=resample_freq)

        # Validate data
        if validate:
            if self.verbose:
                print("\nValidating data...")
            validation_results = self.validate_data(dfs)
            if self.verbose:
                self._print_validation_results(validation_results)

        if self.verbose:
            print("="*60)
            print(f"✅ Loaded {len([k for k in dfs if isinstance(dfs[k], pd.DataFrame)])} datasets")
            if self.missing_files:
                print(f"⚠️  Missing {len(self.missing_files)} files: {', '.join(self.missing_files)}")

        return dfs

    def _load_csv_generic(self, csv_path: Path) -> pd.DataFrame:
        """Load CSV file with standard settings"""
        return pd.read_csv(csv_path)

    def _load_rapl_with_overflow(self, rapl_path: Path) -> pd.DataFrame:
        """
        Load RAPL energy CSV and correct for 32-bit overflow.

        Columns:
            timestamp, intel-rapl:0_package_0_uj, intel-rapl:0:0_dram_uj,
            intel-rapl:1_package_1_uj, intel-rapl:1:0_dram_uj
        """
        df = pd.read_csv(rapl_path)

        # Normalize timestamp column name (old format: timestamp_ms, new format: timestamp)
        if 'timestamp_ms' in df.columns and 'timestamp' not in df.columns:
            df['timestamp'] = df['timestamp_ms'] / 1000.0  # Convert ms to seconds
            df = df.drop(columns=['timestamp_ms'])
        elif 'timestamp_ms' in df.columns and 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp_ms'])  # Keep timestamp, drop timestamp_ms

        # Identify energy columns (ending with _uj)
        energy_cols = [col for col in df.columns if col.endswith('_uj')]

        # Apply overflow correction to each energy column
        for col in energy_cols:
            df[col] = handle_rapl_overflow(df[col])

        return df

    def _load_perf_csv(self, perf_path: Path) -> pd.DataFrame:
        """
        Load perf CSV file (new standard format with header).

        Columns:
            timestamp, value, unit, event, time_enabled, time_running,
            metric_value, metric_unit
        """
        df = pd.read_csv(perf_path)

        # Handle empty perf file (no events captured)
        if len(df) == 0:
            warnings.warn("Perf CSV is empty (no events captured)")

        return df

    def _load_metadata(self) -> dict:
        """Load metadata JSON files"""
        import json

        metadata = {}

        for key, rel_path in self.METADATA_FILES.items():
            json_path = self.experiment_dir / rel_path
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        metadata[key] = json.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to load {key} metadata: {e}")

        return metadata

    def _resample_all_to_grid(self,
                             dfs: Dict[str, pd.DataFrame],
                             freq: str = '1s') -> Dict[str, pd.DataFrame]:
        """
        Resample all time-series DataFrames to uniform temporal grid.

        Args:
            dfs: Dictionary of DataFrames
            freq: Resampling frequency

        Returns:
            Dictionary with resampled DataFrames
        """
        resampled = {}

        for key, df in dfs.items():
            # Skip non-DataFrame entries
            if not isinstance(df, pd.DataFrame):
                resampled[key] = df
                continue

            # Skip DataFrames without timestamp column
            if 'timestamp' not in df.columns:
                resampled[key] = df
                continue

            # Skip perf data - it has multiple events per timestamp
            # Resampling would lose event distinction
            if key == 'perf':
                resampled[key] = df
                continue

            try:
                # Resample with appropriate aggregation
                if key in ['rapl', 'process_io']:
                    # Energy deltas: sum within each bin
                    # Actually, RAPL is already cumulative, so mean is better
                    df_resampled = resample_to_grid(df, freq=freq, aggregation='mean')
                else:
                    # Other metrics: average within each bin
                    df_resampled = resample_to_grid(df, freq=freq, aggregation='mean')

                resampled[key] = df_resampled

            except Exception as e:
                warnings.warn(f"Failed to resample {key}: {e}")
                resampled[key] = df

        return resampled

    def validate_data(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """
        Validate loaded data for integrity and consistency.

        Checks:
        - Missing values percentage
        - Timestamp consistency
        - Value ranges (detect outliers)
        - Data completeness

        Args:
            dfs: Dictionary of DataFrames

        Returns:
            Dictionary of validation results per dataset
        """
        validation = {}

        for key, df in dfs.items():
            if not isinstance(df, pd.DataFrame):
                continue

            result = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_percentage': 0.0,
                'timestamp_issues': [],
                'value_warnings': [],
            }

            # Check missing values
            if len(df) > 0:
                missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
                result['missing_percentage'] = missing_pct

            # Check timestamp consistency
            if 'timestamp' in df.columns:
                # Check for duplicates
                if df['timestamp'].duplicated().any():
                    result['timestamp_issues'].append("Duplicate timestamps found")

                # Check for NaN timestamps
                if df['timestamp'].isna().any():
                    result['timestamp_issues'].append("NaN timestamps found")

                # Check for monotonicity (should be increasing)
                if len(df) > 1:
                    timestamps = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                    if not timestamps.is_monotonic_increasing:
                        result['timestamp_issues'].append("Timestamps not monotonic")

            # Value range checks (specific to dataset type)
            if key == 'rapl':
                # Power should be positive and reasonable (0-500W for dual-socket)
                energy_cols = [col for col in df.columns if col.endswith('_uj')]
                for col in energy_cols:
                    if (df[col] < 0).any():
                        result['value_warnings'].append(f"{col}: Negative values detected")

            elif key == 'vmstat':
                # CPU percentages should sum to ~100
                if 'us' in df.columns and 'sy' in df.columns and 'id' in df.columns:
                    cpu_sum = df['us'] + df['sy'] + df['id']
                    if ((cpu_sum < 95) | (cpu_sum > 105)).any():
                        result['value_warnings'].append("CPU percentages don't sum to 100")

            validation[key] = result

        return validation

    def _print_validation_results(self, validation: Dict[str, dict]):
        """Print validation results in readable format"""
        for key, result in validation.items():
            if result['missing_percentage'] > 5.0:
                print(f"  ⚠️  {key:20s}: {result['missing_percentage']:.1f}% missing values")

            if result['timestamp_issues']:
                print(f"  ⚠️  {key:20s}: {', '.join(result['timestamp_issues'])}")

            if result['value_warnings']:
                for warning in result['value_warnings']:
                    print(f"  ⚠️  {key:20s}: {warning}")

    def get_time_range(self, dfs: Dict[str, pd.DataFrame]) -> tuple:
        """
        Get overall time range across all datasets.

        Args:
            dfs: Dictionary of DataFrames

        Returns:
            Tuple of (start_time, end_time) as datetime objects
        """
        min_time = None
        max_time = None

        for df in dfs.values():
            if not isinstance(df, pd.DataFrame):
                continue

            if 'timestamp' not in df.columns:
                continue

            timestamps = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            timestamps = timestamps.dropna()

            if len(timestamps) == 0:
                continue

            if min_time is None or timestamps.min() < min_time:
                min_time = timestamps.min()

            if max_time is None or timestamps.max() > max_time:
                max_time = timestamps.max()

        return min_time, max_time

    def get_data_summary(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary table of loaded datasets.

        Returns:
            DataFrame with columns: [dataset, rows, columns, time_span_s, missing_%]
        """
        summary_data = []

        for key, df in dfs.items():
            if not isinstance(df, pd.DataFrame):
                continue

            row = {
                'dataset': key,
                'rows': len(df),
                'columns': len(df.columns),
                'time_span_s': None,
                'missing_pct': 0.0,
            }

            # Calculate time span
            if 'timestamp' in df.columns and len(df) > 1:
                try:
                    timestamps = pd.to_datetime(df['timestamp'], unit='s')
                    time_span = (timestamps.max() - timestamps.min()).total_seconds()
                    row['time_span_s'] = time_span
                except:
                    pass

            # Calculate missing percentage
            if len(df) > 0:
                missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
                row['missing_pct'] = missing_pct

            summary_data.append(row)

        return pd.DataFrame(summary_data)


# Convenience function
def load_experiment(exp_dir: str,
                   resample_freq: str = '1s',
                   verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load an experiment.

    Args:
        exp_dir: Path to experiment directory
        resample_freq: Resampling frequency ('1s', '0.5s', None)
        verbose: Print loading progress

    Returns:
        Dictionary of DataFrames

    Example:
        >>> data = load_experiment('results/monitoring/exp_20251117_203747')
        >>> rapl_df = data['rapl']
    """
    loader = MonitoringDataLoader(verbose=verbose)
    return loader.load_experiment(exp_dir, resample_freq=resample_freq)


__all__ = ['MonitoringDataLoader', 'load_experiment']
