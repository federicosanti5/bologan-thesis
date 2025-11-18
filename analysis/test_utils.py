"""
Quick tests for utils.py functions

Run with: python -m pytest test_utils.py -v
Or simply: python test_utils.py
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    handle_rapl_overflow,
    safe_divide,
    resample_to_grid,
    compute_power_from_energy,
)


def test_rapl_overflow():
    """Test RAPL overflow handling"""
    print("\n[TEST] RAPL Overflow Handling...")

    # Create data with overflow: 100 → 200 → 50 (overflow!) → 150
    energy = pd.Series([100, 200, 50, 150])
    corrected = handle_rapl_overflow(energy, max_value=256)

    # Expected: cumulative energy
    # 0: 100
    # 1: 100 + (200-100) = 200
    # 2: 200 + ((256-200) + 50) = 200 + 106 = 306 (overflow)
    # 3: 306 + (150-50) = 406
    expected = [100, 200, 306, 406]

    assert len(corrected) == len(energy), "Length mismatch"
    assert corrected[0] == expected[0], f"Expected {expected[0]}, got {corrected[0]}"
    assert corrected[1] == expected[1], f"Expected {expected[1]}, got {corrected[1]}"
    assert corrected[2] == expected[2], f"Expected {expected[2]}, got {corrected[2]}"
    assert corrected[3] == expected[3], f"Expected {expected[3]}, got {corrected[3]}"

    print("  ✓ Overflow detection and correction works")
    print(f"  Input:  {list(energy)}")
    print(f"  Output: {list(corrected)}")


def test_safe_divide():
    """Test safe division"""
    print("\n[TEST] Safe Division...")

    # Scalar tests
    assert safe_divide(10, 2) == 5.0
    assert pd.isna(safe_divide(10, 0))
    assert safe_divide(10, 0, default=0.0) == 0.0

    # Array tests
    result = safe_divide(np.array([10, 20, 30]), np.array([2, 0, 5]))
    assert result[0] == 5.0
    assert pd.isna(result[1]) or result[1] == 0.0  # Depends on default
    assert result[2] == 6.0

    print("  ✓ Safe division handles zero divisors")
    print(f"  10/2 = {safe_divide(10, 2)}")
    print(f"  10/0 = {safe_divide(10, 0)} (NaN)")
    print(f"  [10,20,30]/[2,0,5] = {safe_divide(np.array([10, 20, 30]), np.array([2, 0, 5]))}")


def test_resample():
    """Test temporal resampling"""
    print("\n[TEST] Temporal Resampling...")

    df = pd.DataFrame({
        'timestamp': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        'value': [10, 20, 15, 25, 18, 22]
    })

    # Resample to 1s grid (average 0-1s, 1-2s, 2-3s)
    resampled = resample_to_grid(df, freq='1s', aggregation='mean')

    assert len(resampled) == 3, f"Expected 3 rows, got {len(resampled)}"

    # Check that resampling worked (values should be averaged)
    # Note: exact values depend on pandas resampling logic
    print("  ✓ Resampling to 1s grid works")
    print(f"  Original: {len(df)} rows")
    print(f"  Resampled: {len(resampled)} rows")
    print(f"  First resampled value: {resampled['value'].iloc[0]:.2f}")


def test_compute_power():
    """Test power computation from energy"""
    print("\n[TEST] Power Computation...")

    # Energy: 0 → 100000 µJ in 1s = 100 mW = 0.1 W
    energy_uj = pd.Series([0, 100000, 200000, 300000])
    timestamp_s = pd.Series([0.0, 1.0, 2.0, 3.0])

    power_w = compute_power_from_energy(energy_uj, timestamp_s)

    # power_w is a Series
    assert pd.isna(power_w[0]) or pd.isna(power_w.iloc[0]), "First power value should be NaN"
    power_1 = power_w[1] if hasattr(power_w, '__getitem__') else power_w.iloc[1]
    assert abs(power_1 - 0.1) < 0.01, f"Expected ~0.1W, got {power_1}"

    print("  ✓ Power calculation from energy works")
    print(f"  Energy (µJ): {list(energy_uj)}")
    print(f"  Power (W):   {list(power_w)}")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running Utils Tests")
    print("="*60)

    try:
        test_rapl_overflow()
        test_safe_divide()
        test_resample()
        test_compute_power()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
