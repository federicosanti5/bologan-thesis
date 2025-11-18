"""
Test MonitoringDataLoader on real experiment data
"""

import sys
from pathlib import Path

# Add analysis to path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring_data_loader import load_experiment


def test_load_experiment():
    """Test loading real experiment data"""

    # Use most recent experiment
    exp_dir = '/home/saint/Documents/UNIBO/tesi/results/monitoring/exp_20251117_203747'

    print("Testing MonitoringDataLoader")
    print("="*70)
    print(f"Experiment: {exp_dir}\n")

    # Load experiment
    data = load_experiment(exp_dir, resample_freq='1s', verbose=True)

    # Print summary
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)

    for key in sorted(data.keys()):
        if key in ['metadata', 'training_metrics']:
            print(f"{key:25s}: {type(data[key]).__name__}")
        else:
            df = data[key]
            print(f"{key:25s}: {len(df):6d} rows × {len(df.columns):3d} cols")

    # Check RAPL data
    if 'rapl' in data:
        print("\n" + "="*70)
        print("RAPL DATA CHECK")
        print("="*70)
        rapl = data['rapl']
        print(f"Columns: {list(rapl.columns)}")
        print(f"First 3 rows:")
        print(rapl.head(3))
        print(f"\nEnergy range (package_0):")
        col = [c for c in rapl.columns if 'package_0' in c][0]
        print(f"  Min: {rapl[col].min():.2e} µJ")
        print(f"  Max: {rapl[col].max():.2e} µJ")

    # Check training metrics
    if 'training_metrics' in data:
        print("\n" + "="*70)
        print("TRAINING METRICS CHECK")
        print("="*70)
        tm = data['training_metrics']
        print(f"Iterations: {len(tm['iterations'])}")
        if len(tm['iterations']) > 0:
            print(f"First iteration: {tm['iterations'][0]}")
            print(f"Last iteration:  {tm['iterations'][-1]}")
            print(f"Final Gloss: {tm['gloss'][-1]:.4f}")
            print(f"Final Dloss: {tm['dloss'][-1]:.4f}")
        if tm['training_size']:
            print(f"Training size: {tm['training_size']}")

    print("\n" + "="*70)
    print("✅ TEST PASSED - Data loaded successfully!")
    print("="*70)


if __name__ == "__main__":
    test_load_experiment()
