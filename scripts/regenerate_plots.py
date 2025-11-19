#!/usr/bin/env python3
"""
Regenerate plots for a specific experiment with JPEG format.
"""

import sys
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis'))

from monitoring_data_loader import MonitoringDataLoader
from metrics_calculator import MetricsCalculator
from monitoring_visualizer import MonitoringVisualizer


def regenerate_plots(exp_dir: str, dpi: int = 300):
    """Regenerate all plots for an experiment."""

    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return False

    print("="*70)
    print(f"REGENERATING PLOTS FOR: {exp_path.name}")
    print("="*70)
    print(f"Location: {exp_path}")
    print(f"Format: JPEG (image/jpeg)")
    print(f"DPI: {dpi}")

    # Load data
    print("\n[1/3] Loading monitoring data...")
    loader = MonitoringDataLoader()
    try:
        dfs = loader.load_experiment(
            str(exp_path),
            resample_freq='1s',
            validate=False
        )
        print(f"✅ Loaded {len(dfs)} datasets")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False

    # Compute metrics
    print("\n[2/3] Computing metrics...")
    calc = MetricsCalculator(verbose=False)
    try:
        metrics = calc.compute_all_metrics(dfs, exp_dir=str(exp_path))
        print(f"✅ Metrics computed")
    except Exception as e:
        print(f"❌ Failed to compute metrics: {e}")
        return False

    # Generate plots
    print("\n[3/3] Generating plots...")
    output_dir = exp_path / 'analysis' / 'plots'
    visualizer = MonitoringVisualizer(
        output_dir=str(output_dir),
        dpi=dpi,
        image_format='jpeg'  # JPEG format to fix API error
    )

    try:
        plot_paths = visualizer.generate_all_plots(dfs, metrics)

        print(f"\n{'='*70}")
        print(f"✅ Successfully generated {len(plot_paths)} plots:")
        print(f"{'='*70}")

        for plot_name, plot_path in plot_paths.items():
            file_size = Path(plot_path).stat().st_size / 1024  # KB
            print(f"  ✅ {plot_name:35s}: {file_size:7.1f} KB")

        print(f"\n📁 Output directory: {output_dir}")
        return True

    except Exception as e:
        print(f"❌ Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        # Default experiment
        exp_dir = "results/oph/monitoring/exp_20251118_103745"

    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    success = regenerate_plots(exp_dir, dpi)
    sys.exit(0 if success else 1)
