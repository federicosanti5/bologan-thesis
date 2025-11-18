#!/usr/bin/env python3
"""
Unit tests for MetricsCalculator using synthetic data

This test suite validates:
1. Energy metrics calculation with controlled RAPL data
2. Performance metrics calculation with synthetic monitoring data
3. Efficiency metrics calculation with known inputs
4. Edge cases: NaN, empty DataFrames, missing columns, zero division

Strategy: Synthetic data with predictable outputs for precise assertions
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from metrics_calculator import MetricsCalculator
except ImportError:
    from analysis.metrics_calculator import MetricsCalculator


def create_synthetic_rapl_data():
    """
    Create synthetic RAPL data with known energy values.

    Setup:
    - Duration: 4 seconds (timestamps 0-3)
    - Package 0: 1 MJ/s = 1 J/s = 1 W
    - Package 1: 2 MJ/s = 2 J/s = 2 W
    - DRAM 0: 0.5 MJ/s = 0.5 J/s = 0.5 W
    - DRAM 1: 0.5 MJ/s = 0.5 J/s = 0.5 W

    Expected totals:
    - Total energy: 12 J (4s * 3W)
    - CPU energy: 9 J (3 W * 3s)
    - DRAM energy: 3 J (1 W * 3s)
    - CPU %: 75%, DRAM %: 25%
    - Average power: 4 W
    """
    return pd.DataFrame({
        'timestamp': [0.0, 1.0, 2.0, 3.0],
        'intel-rapl:0_package_0_uj': [0, 1e6, 2e6, 3e6],      # 1 J/s
        'intel-rapl:0_package_1_uj': [0, 2e6, 4e6, 6e6],      # 2 J/s
        'intel-rapl:0_dram_0_uj': [0, 0.5e6, 1e6, 1.5e6],     # 0.5 J/s
        'intel-rapl:0_dram_1_uj': [0, 0.5e6, 1e6, 1.5e6],     # 0.5 J/s
    })


def create_synthetic_vmstat_data():
    """
    Create synthetic vmstat data.

    Setup:
    - CPU us (user): 60%
    - CPU sy (system): 20%
    - Total CPU: 80%
    - I/O wait: 5%
    """
    return pd.DataFrame({
        'timestamp': [0.0, 1.0, 2.0, 3.0],
        'us': [60, 60, 60, 60],
        'sy': [20, 20, 20, 20],
        'wa': [5, 5, 5, 5],
        'id': [15, 15, 15, 15],
    })


def create_synthetic_pidstat_data():
    """
    Create synthetic pidstat process data.

    Setup:
    - Memory RSS: grows from 1GB to 2GB
    - Peak: 2048 MB
    - Average: ~1536 MB
    """
    return pd.DataFrame({
        'timestamp': [0.0, 1.0, 2.0, 3.0],
        'PID': [12345, 12345, 12345, 12345],
        'RSS': [1024*1024, 1536*1024, 1792*1024, 2048*1024],  # kB
        '%CPU': [80.0, 80.0, 80.0, 80.0],
    })


def create_synthetic_perf_data():
    """
    Create synthetic perf data.

    Setup:
    - IPC = 1.5 (1500 instructions / 1000 cycles)
    - Cache hit rate = 90% (900 hits / 1000 refs)
    """
    return pd.DataFrame({
        'timestamp': [0.0, 0.0, 0.0, 0.0],
        'value': [1500, 1000, 1000, 100],
        'unit': ['', '', '', ''],
        'event': ['instructions', 'cycles', 'cache-references', 'cache-misses'],
    })


def create_synthetic_iostat_data():
    """
    Create synthetic iostat device data.

    Setup:
    - Read: 1000 kB/s
    - Write: 500 kB/s
    """
    return pd.DataFrame({
        'timestamp': [0.0, 1.0, 2.0, 3.0],
        'Device': ['sda', 'sda', 'sda', 'sda'],
        'rkB/s': [1000, 1000, 1000, 1000],
        'wkB/s': [500, 500, 500, 500],
    })


def create_synthetic_training_metrics():
    """
    Create synthetic training metrics.

    Setup:
    - 100 iterations
    - 10000 training samples
    - Duration: 3 seconds
    - Throughput: 3333 samples/s
    """
    return {
        'iterations': list(range(100)),
        'gloss': [1.5 - i*0.01 for i in range(100)],  # Decreasing
        'dloss': [0.5 + i*0.001 for i in range(100)],  # Increasing
        'total_time': [0.03 * i for i in range(100)],  # 0.03s per iter
        'training_size': (10000, 128),  # (samples, channels)
    }


def test_energy_metrics():
    """Test compute_energy_metrics() with synthetic RAPL data"""
    print("\n" + "="*70)
    print("TEST 1: Energy Metrics Calculation")
    print("="*70)

    calc = MetricsCalculator(verbose=False)
    rapl_df = create_synthetic_rapl_data()
    training_metrics = create_synthetic_training_metrics()

    result = calc.compute_energy_metrics(rapl_df, training_metrics)

    # Validate structure
    assert 'total_energy_j' in result
    assert 'average_power_w' in result
    assert 'energy_breakdown' in result
    assert 'per_component' in result

    # Validate values
    # Total: package_0(3J) + package_1(6J) + dram_0(1.5J) + dram_1(1.5J) = 12J
    assert abs(result['total_energy_j'] - 12.0) < 0.01, \
        f"Expected 12J, got {result['total_energy_j']}"

    # Average power: 12J / 3s = 4W
    assert abs(result['average_power_w'] - 4.0) < 0.01, \
        f"Expected 4W, got {result['average_power_w']}"

    # CPU: 3 + 6 = 9J (75%)
    assert abs(result['energy_breakdown']['cpu_j'] - 9.0) < 0.01
    assert abs(result['energy_breakdown']['cpu_percent'] - 75.0) < 0.1

    # DRAM: 1.5 + 1.5 = 3J (25%)
    assert abs(result['energy_breakdown']['dram_j'] - 3.0) < 0.01
    assert abs(result['energy_breakdown']['dram_percent'] - 25.0) < 0.1

    # Energy per 1000 events: 12J / (10000/1000) = 1.2J
    assert result['energy_per_1000_events_j'] is not None
    assert abs(result['energy_per_1000_events_j'] - 1.2) < 0.01

    print("✅ Energy metrics: PASSED")
    print(f"   Total energy: {result['total_energy_j']:.2f} J (expected: 12 J)")
    print(f"   Average power: {result['average_power_w']:.2f} W (expected: 4 W)")
    print(f"   CPU breakdown: {result['energy_breakdown']['cpu_percent']:.1f}% (expected: 75%)")
    print(f"   Energy/1000-events: {result['energy_per_1000_events_j']:.2f} J (expected: 1.2 J)")

    return True


def test_performance_metrics():
    """Test compute_performance_metrics() with synthetic data"""
    print("\n" + "="*70)
    print("TEST 2: Performance Metrics Calculation")
    print("="*70)

    calc = MetricsCalculator(verbose=False)
    vmstat_df = create_synthetic_vmstat_data()
    pidstat_df = create_synthetic_pidstat_data()
    perf_df = create_synthetic_perf_data()
    iostat_df = create_synthetic_iostat_data()
    training_metrics = create_synthetic_training_metrics()

    result = calc.compute_performance_metrics(
        pidstat_df=pidstat_df,
        vmstat_df=vmstat_df,
        perf_df=perf_df,
        iostat_df=iostat_df,
        training_metrics=training_metrics,
        duration_s=3.0
    )

    # Validate structure
    assert 'training_time_s' in result
    assert 'throughput_samples_per_s' in result
    assert 'cpu_utilization_avg_percent' in result
    assert 'memory_peak_mb' in result
    assert 'ipc' in result
    assert 'cache_hit_rate_percent' in result

    # Validate values
    # Training time: last total_time = 0.03 * 99 = 2.97s
    assert abs(result['training_time_s'] - 2.97) < 0.01

    # Throughput: 10000 samples / 2.97s = ~3367 samples/s
    assert result['throughput_samples_per_s'] is not None
    assert 3300 < result['throughput_samples_per_s'] < 3400

    # CPU utilization: 60 + 20 = 80%
    assert abs(result['cpu_utilization_avg_percent'] - 80.0) < 0.1

    # I/O wait: 5%
    assert abs(result['iowait_avg_percent'] - 5.0) < 0.1

    # Memory peak: 2048 MB
    assert abs(result['memory_peak_mb'] - 2048.0) < 0.1

    # IPC: 1500 / 1000 = 1.5
    assert abs(result['ipc'] - 1.5) < 0.01

    # Cache hit rate: (1000 - 100) / 1000 = 90%
    assert abs(result['cache_hit_rate_percent'] - 90.0) < 0.1

    # I/O throughput: 1000 + 500 = 1500 kB/s
    assert abs(result['io_total_kbps'] - 1500.0) < 0.1

    print("✅ Performance metrics: PASSED")
    print(f"   Throughput: {result['throughput_samples_per_s']:.1f} samples/s")
    print(f"   CPU utilization: {result['cpu_utilization_avg_percent']:.1f}% (expected: 80%)")
    print(f"   Memory peak: {result['memory_peak_mb']:.1f} MB (expected: 2048 MB)")
    print(f"   IPC: {result['ipc']:.2f} (expected: 1.5)")
    print(f"   Cache hit rate: {result['cache_hit_rate_percent']:.1f}% (expected: 90%)")

    return True


def test_efficiency_metrics():
    """Test compute_efficiency_metrics() with known inputs"""
    print("\n" + "="*70)
    print("TEST 3: Efficiency Metrics Calculation")
    print("="*70)

    calc = MetricsCalculator(verbose=False)

    # Known inputs
    energy_metrics = {
        'total_energy_j': 12.0,
        'average_power_w': 4.0,
    }

    performance_metrics = {
        'throughput_samples_per_s': 3333.0,
        'training_time_s': 3.0,
    }

    result = calc.compute_efficiency_metrics(energy_metrics, performance_metrics)

    # Validate structure
    assert 'performance_per_watt' in result
    assert 'samples_per_joule' in result
    assert 'energy_delay_product' in result

    # Validate values
    # Performance-per-Watt: 3333 samples/s / 4W = 833.25 samples/s/W
    assert abs(result['performance_per_watt'] - 833.25) < 1.0

    # Samples per Joule: (3333 * 3) / 12 = 833.25 samples/J
    assert abs(result['samples_per_joule'] - 833.25) < 1.0

    # Energy-Delay Product: 12 * 3^2 = 108
    assert abs(result['energy_delay_product'] - 108.0) < 0.1

    print("✅ Efficiency metrics: PASSED")
    print(f"   Performance-per-Watt: {result['performance_per_watt']:.2f} samples/s/W")
    print(f"   Samples/Joule: {result['samples_per_joule']:.2f}")
    print(f"   Energy-Delay Product: {result['energy_delay_product']:.2f}")

    return True


def test_edge_cases():
    """Test edge cases: empty data, NaN, missing columns"""
    print("\n" + "="*70)
    print("TEST 4: Edge Cases")
    print("="*70)

    calc = MetricsCalculator(verbose=False)

    # Test 4.1: Empty RAPL DataFrame
    print("\n[4.1] Empty RAPL DataFrame")
    try:
        rapl_empty = pd.DataFrame()
        result = calc.compute_energy_metrics(rapl_empty)
        print("❌ Should have raised ValueError for empty DataFrame")
        return False
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")

    # Test 4.2: Missing energy columns
    print("\n[4.2] Missing energy columns")
    try:
        rapl_no_energy = pd.DataFrame({'timestamp': [0, 1, 2]})
        result = calc.compute_energy_metrics(rapl_no_energy)
        print("❌ Should have raised ValueError for missing energy columns")
        return False
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")

    # Test 4.3: None DataFrames for optional parameters
    print("\n[4.3] None DataFrames (should return N/A values)")
    result = calc.compute_performance_metrics(
        pidstat_df=None,
        vmstat_df=None,
        perf_df=None,
        iostat_df=None,
        training_metrics=None,
        duration_s=None
    )
    assert result['cpu_utilization_avg_percent'] is None
    assert result['ipc'] is None
    print("✅ Correctly handled None DataFrames")

    # Test 4.4: Zero division safety
    print("\n[4.4] Zero division safety")
    energy_zero = {'total_energy_j': 0.0, 'average_power_w': 0.0}
    perf_zero = {'throughput_samples_per_s': 0.0, 'training_time_s': 0.0}
    result = calc.compute_efficiency_metrics(energy_zero, perf_zero)
    assert result['performance_per_watt'] is None  # safe_divide should return None
    print("✅ Safely handled zero division")

    # Test 4.5: Missing timestamp column
    print("\n[4.5] Missing timestamp in RAPL")
    rapl_no_ts = pd.DataFrame({
        'intel-rapl:0_package_0_uj': [0, 1e6, 2e6],
    })
    result = calc.compute_energy_metrics(rapl_no_ts)
    assert result['total_energy_j'] == 2.0  # Should still calculate energy
    assert result['average_power_w'] == 0.0  # But power defaults to 0
    print("✅ Handled missing timestamp (energy OK, power=0)")

    print("\n✅ All edge cases: PASSED")
    return True


def test_monitoring_overhead():
    """Test compute_monitoring_overhead() with synthetic data"""
    print("\n" + "="*70)
    print("TEST 5: Monitoring Overhead Calculation")
    print("="*70)

    calc = MetricsCalculator(verbose=False)

    # Create synthetic overhead data
    overhead_df = pd.DataFrame({
        'timestamp': [0, 0, 0, 1, 1, 1],
        'Command': ['vmstat', 'iostat', 'perf', 'vmstat', 'iostat', 'perf'],
        '%CPU': [0.5, 0.8, 1.2, 0.5, 0.8, 1.2],
        'RSS': [1024, 2048, 4096, 1024, 2048, 4096],  # kB
    })

    result = calc.compute_monitoring_overhead(overhead_df, vmstat_df=None)

    # Validate structure
    assert 'cpu_overhead_percent' in result
    assert 'memory_overhead_mb' in result
    assert 'tools' in result

    # Validate values
    # CPU overhead: avg(0.5, 0.5) + avg(0.8, 0.8) + avg(1.2, 1.2) = 2.5%
    assert abs(result['cpu_overhead_percent'] - 2.5) < 0.01

    # Memory: (1024 + 2048 + 4096) / 1024 = 7 MB
    assert abs(result['memory_overhead_mb'] - 7.0) < 0.01

    # Tools count
    assert len(result['tools']) == 3
    assert 'vmstat' in result['tools']
    assert 'iostat' in result['tools']
    assert 'perf' in result['tools']

    print("✅ Monitoring overhead: PASSED")
    print(f"   CPU overhead: {result['cpu_overhead_percent']:.2f}% (expected: 2.5%)")
    print(f"   Memory overhead: {result['memory_overhead_mb']:.2f} MB (expected: 7 MB)")
    print(f"   Tools: {list(result['tools'].keys())}")

    return True


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "="*70)
    print("METRICS CALCULATOR UNIT TESTS")
    print("="*70)
    print("Strategy: Synthetic data with controlled values")
    print()

    tests = [
        ("Energy Metrics", test_energy_metrics),
        ("Performance Metrics", test_performance_metrics),
        ("Efficiency Metrics", test_efficiency_metrics),
        ("Edge Cases", test_edge_cases),
        ("Monitoring Overhead", test_monitoring_overhead),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except AssertionError as e:
            print(f"\n❌ {name}: FAILED")
            print(f"   Assertion error: {e}")
            results.append((name, False))
        except Exception as e:
            print(f"\n❌ {name}: ERROR")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} - {name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
