#!/usr/bin/env python3
"""Debug timestamp issue in RAPL data"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring_data_loader import MonitoringDataLoader
from metrics_calculator import MetricsCalculator

# Load cluster data
exp_dir = Path(__file__).parent.parent / 'results' / 'oph' / 'monitoring' / 'exp_20250918_182245'
loader = MonitoringDataLoader()

print("Loading data...")
dfs = loader.load_experiment(str(exp_dir), resample_freq='1s', validate=False)

rapl_df = dfs['rapl']

print(f"\nRAPL DataFrame info:")
print(f"  Rows: {len(rapl_df)}")
print(f"  Columns: {rapl_df.columns.tolist()}")
print(f"\nFirst 5 timestamps:")
print(rapl_df['timestamp'].head())
print(f"\nTimestamp dtype: {rapl_df['timestamp'].dtype}")
print(f"\nFirst timestamp: {rapl_df['timestamp'].iloc[0]}")
print(f"  Type: {type(rapl_df['timestamp'].iloc[0])}")
print(f"\nLast timestamp: {rapl_df['timestamp'].iloc[-1]}")
print(f"  Type: {type(rapl_df['timestamp'].iloc[-1])}")

print(f"\nTimestamp difference:")
ts_diff = rapl_df['timestamp'].iloc[-1] - rapl_df['timestamp'].iloc[0]
print(f"  Raw diff: {ts_diff}")
print(f"  Type: {type(ts_diff)}")
print(f"  Has total_seconds: {hasattr(ts_diff, 'total_seconds')}")

if hasattr(ts_diff, 'total_seconds'):
    duration_s = ts_diff.total_seconds()
    print(f"  Duration (s): {duration_s}")
else:
    print(f"  Duration (s): {float(ts_diff)}")

print(f"\nEnergy columns:")
energy_cols = [col for col in rapl_df.columns if col.endswith('_uj')]
for col in energy_cols:
    print(f"  {col}: {rapl_df[col].iloc[0]:.2f} → {rapl_df[col].iloc[-1]:.2f}")
