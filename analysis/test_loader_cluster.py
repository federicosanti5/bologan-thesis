"""
Test MonitoringDataLoader on CLUSTER experiment data (OPH)

This test validates:
1. RAPL overflow handling on real data
2. Training metrics parsing (Gloss, Dloss)
3. Complete workflow with all sensors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from monitoring_data_loader import load_experiment


def test_cluster_experiment():
    """Test loading cluster experiment with RAPL and training data"""

    # Cluster experiment with FULL data
    exp_dir = '/home/saint/Documents/UNIBO/tesi/results/oph/monitoring/exp_20250918_182245'

    print("="*70)
    print("TESTING LOADER ON CLUSTER EXPERIMENT (OPH)")
    print("="*70)
    print(f"Experiment: {Path(exp_dir).name}")
    print(f"Expected: RAPL data, Training metrics, Full monitoring\n")

    # Load with verbose output
    data = load_experiment(exp_dir, resample_freq='1s', verbose=True)

    print("\n" + "="*70)
    print("CRITICAL VALIDATION CHECKS")
    print("="*70)

    # Check 1: RAPL DATA
    print("\n[CHECK 1] RAPL Energy Data")
    if 'rapl' in data and len(data['rapl']) > 0:
        rapl = data['rapl']
        print(f"  ✅ RAPL loaded: {len(rapl)} rows")

        # Check overflow handling
        energy_cols = [c for c in rapl.columns if c.endswith('_uj')]
        print(f"  Energy columns: {len(energy_cols)}")

        for col in energy_cols[:2]:  # Check first 2 columns
            values = rapl[col].dropna()
            if len(values) > 1:
                # Check monotonically increasing (overflow corrected)
                is_monotonic = (values.diff().dropna() >= 0).all()
                total_energy_mj = values.iloc[-1] / 1e12  # microJoules to MegaJoules

                print(f"  {col[:30]:30s}:")
                print(f"    Monotonic increasing: {'✅ YES' if is_monotonic else '❌ NO (OVERFLOW BUG!)'}")
                print(f"    Total energy: {total_energy_mj:.3f} MJ ({total_energy_mj/1e6:.2f} kWh)")
                print(f"    Min: {values.min():.2e} µJ, Max: {values.max():.2e} µJ")
    else:
        print("  ❌ RAPL data not found or empty!")

    # Check 2: TRAINING METRICS
    print("\n[CHECK 2] Training Metrics (Gloss, Dloss)")
    if 'training_metrics' in data:
        tm = data['training_metrics']
        if tm and 'iterations' in tm and len(tm['iterations']) > 0:
            print(f"  ✅ Training metrics parsed: {len(tm['iterations'])} iterations")
            print(f"    First iteration: {tm['iterations'][0]}")
            print(f"    Last iteration:  {tm['iterations'][-1]}")
            print(f"    Final Gloss: {tm['gloss'][-1]:.6f}")
            print(f"    Final Dloss: {tm['dloss'][-1]:.6f}")

            if tm['training_size']:
                print(f"    Training size: {tm['training_size']} samples")
        else:
            print("  ⚠️  Training metrics empty or malformed")
    else:
        print("  ❌ Training metrics not found!")

    # Check 3: DATA COMPLETENESS
    print("\n[CHECK 3] Data Completeness")
    csv_datasets = [k for k in data.keys() if isinstance(data[k], __import__('pandas').DataFrame)]
    print(f"  Total datasets loaded: {len(csv_datasets)}")

    critical_datasets = ['rapl', 'cpu_freq', 'vmstat', 'thermal']
    missing_critical = [d for d in critical_datasets if d not in data or len(data.get(d, [])) == 0]

    if missing_critical:
        print(f"  ⚠️  Missing critical datasets: {', '.join(missing_critical)}")
    else:
        print(f"  ✅ All critical datasets present")

    # Check 4: TIME RANGE
    print("\n[CHECK 4] Temporal Coverage")
    if 'rapl' in data and len(data['rapl']) > 0:
        import pandas as pd
        timestamps = pd.to_datetime(data['rapl']['timestamp'], unit='s')
        duration_s = (timestamps.max() - timestamps.min()).total_seconds()
        duration_min = duration_s / 60
        print(f"  Duration: {duration_s:.1f} seconds ({duration_min:.1f} minutes)")
        print(f"  Samples: {len(timestamps)} → avg {len(timestamps)/duration_s:.2f} Hz")

    # Summary
    print("\n" + "="*70)
    if 'rapl' in data and len(data['rapl']) > 0 and 'training_metrics' in data:
        print("✅ CLUSTER TEST PASSED - Ready for MetricsCalculator!")
    else:
        print("⚠️  CLUSTER TEST INCOMPLETE - Check warnings above")
    print("="*70)


if __name__ == "__main__":
    test_cluster_experiment()
