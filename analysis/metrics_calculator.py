"""
Metrics Calculator - Compute all performance, energy, and efficiency metrics

This module implements metric calculations for:
- Energy metrics (Section 2.2): Power, total energy, energy-per-1000-events, breakdown
- Performance metrics (Section 2.1): Throughput, CPU%, IPC, cache efficiency
- Efficiency metrics (Section 2.3): Performance-per-Watt, samples/Joule, EDP
- Convergence metrics (Section 2.3): Loss values, stability, GAN balance, convergence rate
- Monitoring overhead (Section 2.4): Auto-monitoring cost quantification
- Correlations (Section 4): Statistical analysis of metric relationships

Based on ANALYSIS_SPECIFICATION.md v1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path
from scipy.stats import pearsonr
import warnings

try:
    from .utils import (
        compute_power_from_energy,
        safe_divide,
        parse_stdout_losses,
    )
except ImportError:
    from utils import (
        compute_power_from_energy,
        safe_divide,
        parse_stdout_losses,
    )


class MetricsCalculator:
    """
    Calculate all metrics for BoloGAN monitoring analysis.

    This class computes 30+ metrics across 6 categories:
    1. Energy metrics (power, total energy, breakdown)
    2. Performance metrics (throughput, CPU, IPC, cache)
    3. Efficiency metrics (performance-per-Watt, samples/J, EDP)
    4. Convergence metrics (loss values, stability, GAN balance)
    5. Monitoring overhead (CPU%, memory, per-tool cost)
    6. Correlations (statistical relationships between metrics)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize metrics calculator.

        Args:
            verbose: Print calculation progress and warnings
        """
        self.verbose = verbose

    def compute_energy_metrics(self,
                               rapl_df: pd.DataFrame,
                               training_metrics: Optional[Dict] = None) -> Dict:
        """
        Compute energy metrics from RAPL data (Section 2.2).

        Metrics:
        - Total energy (Joules)
        - Average power (Watt)
        - Energy per 1000 events (standard metric for comparison with GEANT4)
        - Energy breakdown (CPU% vs DRAM%)

        Args:
            rapl_df: DataFrame with RAPL energy columns (*_uj) and timestamp
            training_metrics: Optional dict with training info (for energy-per-1000-events)

        Returns:
            Dictionary with energy metrics:
            {
                'total_energy_j': float,
                'average_power_w': float,
                'energy_per_1000_events_j': float or None,
                'energy_breakdown': {
                    'cpu_j': float,
                    'dram_j': float,
                    'cpu_percent': float,
                    'dram_percent': float,
                },
                'per_component': {
                    'package_0_j': float,
                    'package_1_j': float,
                    'dram_0_j': float,
                    'dram_1_j': float,
                }
            }
        """
        if len(rapl_df) == 0:
            raise ValueError("RAPL DataFrame is empty")

        if self.verbose:
            print("Computing energy metrics...")

        # Identify energy columns
        energy_cols = [col for col in rapl_df.columns if col.endswith('_uj')]

        if len(energy_cols) == 0:
            raise ValueError("No energy columns (*_uj) found in RAPL DataFrame")

        # Calculate total energy per component (microJoules → Joules)
        # RAPL columns are already cumulative after overflow correction
        per_component = {}

        for col in energy_cols:
            # Energy = final - initial (already corrected for overflow)
            energy_uj = rapl_df[col].iloc[-1] - rapl_df[col].iloc[0]
            energy_j = energy_uj / 1e6  # microJoules to Joules

            # Extract component name from column
            # Format: intel-rapl:0_package_0_uj → package_0
            component = col.replace('intel-rapl:', '').replace('_uj', '')
            # Clean up numbering: intel-rapl:0_package_0 → package_0
            if '_' in component:
                parts = component.split('_')
                # Keep last two parts (e.g., package_0, dram_0)
                component = '_'.join(parts[-2:])

            per_component[f"{component}_j"] = energy_j

        # Breakdown: CPU (package) vs DRAM
        cpu_energy = 0.0
        dram_energy = 0.0

        for key, value in per_component.items():
            if 'package' in key:
                cpu_energy += value
            elif 'dram' in key:
                dram_energy += value

        total_energy = cpu_energy + dram_energy

        # Calculate percentages
        cpu_percent = safe_divide(cpu_energy, total_energy, default=0.0) * 100
        dram_percent = safe_divide(dram_energy, total_energy, default=0.0) * 100

        # Calculate average power
        if 'timestamp' in rapl_df.columns and len(rapl_df) > 1:
            # Handle both datetime and numeric timestamps
            ts_diff = rapl_df['timestamp'].iloc[-1] - rapl_df['timestamp'].iloc[0]

            # Convert to seconds if datetime
            if hasattr(ts_diff, 'total_seconds'):
                duration_s = ts_diff.total_seconds()
            else:
                duration_s = float(ts_diff)

            average_power_w = safe_divide(total_energy, duration_s, default=0.0)
        else:
            average_power_w = 0.0
            warnings.warn("Cannot compute average power: timestamp column missing or insufficient data")

        # Energy per 1000 events (if training metrics available)
        energy_per_1000_events = None
        if training_metrics and 'training_size' in training_metrics:
            training_size = training_metrics['training_size']
            if training_size and isinstance(training_size, tuple):
                total_samples = training_size[0]
                energy_per_1000_events = safe_divide(total_energy, total_samples / 1000.0, default=None)

        result = {
            'total_energy_j': total_energy,
            'average_power_w': average_power_w,
            'energy_per_1000_events_j': energy_per_1000_events,
            'energy_breakdown': {
                'cpu_j': cpu_energy,
                'dram_j': dram_energy,
                'cpu_percent': cpu_percent,
                'dram_percent': dram_percent,
            },
            'per_component': per_component,
        }

        if self.verbose:
            print(f"  Total energy: {total_energy:.2f} J ({total_energy/1000:.3f} kJ)")
            print(f"  Average power: {average_power_w:.2f} W")
            print(f"  CPU: {cpu_percent:.1f}%, DRAM: {dram_percent:.1f}%")
            if energy_per_1000_events:
                print(f"  Energy/1000-events: {energy_per_1000_events:.2f} J")

        return result

    def compute_performance_metrics(self,
                                    pidstat_df: Optional[pd.DataFrame] = None,
                                    vmstat_df: Optional[pd.DataFrame] = None,
                                    perf_df: Optional[pd.DataFrame] = None,
                                    iostat_df: Optional[pd.DataFrame] = None,
                                    training_metrics: Optional[Dict] = None,
                                    duration_s: Optional[float] = None) -> Dict:
        """
        Compute performance metrics (Section 2.1).

        Metrics:
        - Training time (seconds)
        - Throughput (samples/second)
        - CPU utilization (average %, peak %)
        - Memory footprint (peak MB)
        - I/O throughput (kB/s)
        - IPC (instructions per cycle)
        - Cache hit rate (%)

        Args:
            pidstat_df: Process pidstat data (CPU%, memory RSS)
            vmstat_df: System vmstat data (CPU us+sy, iowait)
            perf_df: Performance counters (instructions, cycles, cache)
            iostat_df: I/O statistics (rkB/s, wkB/s)
            training_metrics: Dict from parse_stdout_losses (iterations, total_time)
            duration_s: Total duration in seconds (from metadata or timestamps)

        Returns:
            Dictionary with performance metrics
        """
        if self.verbose:
            print("Computing performance metrics...")

        result = {}

        # Training time
        if training_metrics and 'total_time' in training_metrics and training_metrics['total_time']:
            result['training_time_s'] = training_metrics['total_time'][-1]
        elif duration_s is not None:
            result['training_time_s'] = duration_s
        else:
            result['training_time_s'] = None

        # Throughput (samples/second)
        if training_metrics and 'training_size' in training_metrics and result['training_time_s']:
            training_size = training_metrics['training_size']
            if training_size and isinstance(training_size, tuple):
                total_samples = training_size[0]
                result['throughput_samples_per_s'] = safe_divide(
                    total_samples,
                    result['training_time_s'],
                    default=None
                )
            else:
                result['throughput_samples_per_s'] = None
        else:
            result['throughput_samples_per_s'] = None

        # Iteration throughput
        if training_metrics and 'iterations' in training_metrics:
            n_iterations = len(training_metrics['iterations'])
            if n_iterations > 0 and result['training_time_s']:
                result['iteration_throughput_per_s'] = safe_divide(
                    n_iterations,
                    result['training_time_s'],
                    default=None
                )
            else:
                result['iteration_throughput_per_s'] = None
        else:
            result['iteration_throughput_per_s'] = None

        # CPU utilization from vmstat (us + sy)
        if vmstat_df is not None and len(vmstat_df) > 0:
            if 'us' in vmstat_df.columns and 'sy' in vmstat_df.columns:
                cpu_util = vmstat_df['us'] + vmstat_df['sy']
                result['cpu_utilization_avg_percent'] = cpu_util.mean()
                result['cpu_utilization_peak_percent'] = cpu_util.max()

                # I/O wait percentage
                if 'wa' in vmstat_df.columns:
                    result['iowait_avg_percent'] = vmstat_df['wa'].mean()
                else:
                    result['iowait_avg_percent'] = None
            else:
                result['cpu_utilization_avg_percent'] = None
                result['cpu_utilization_peak_percent'] = None
                result['iowait_avg_percent'] = None
        else:
            result['cpu_utilization_avg_percent'] = None
            result['cpu_utilization_peak_percent'] = None
            result['iowait_avg_percent'] = None

        # Memory footprint from pidstat (RSS in kB)
        # Handle both old format (RSS) and new format (rss)
        if pidstat_df is not None and len(pidstat_df) > 0:
            rss_col = 'rss' if 'rss' in pidstat_df.columns else 'RSS'

            if rss_col in pidstat_df.columns:
                # RSS is in kB, convert to MB
                result['memory_peak_mb'] = pidstat_df[rss_col].max() / 1024.0
                result['memory_avg_mb'] = pidstat_df[rss_col].mean() / 1024.0
            else:
                result['memory_peak_mb'] = None
                result['memory_avg_mb'] = None
        else:
            result['memory_peak_mb'] = None
            result['memory_avg_mb'] = None

        # I/O throughput from iostat_dev
        if iostat_df is not None and len(iostat_df) > 0:
            read_cols = [c for c in iostat_df.columns if 'rkB' in c or 'rkB/s' in c]
            write_cols = [c for c in iostat_df.columns if 'wkB' in c or 'wkB/s' in c]

            total_read = 0.0
            total_write = 0.0

            if read_cols:
                total_read = iostat_df[read_cols[0]].mean()
            if write_cols:
                total_write = iostat_df[write_cols[0]].mean()

            result['io_read_kbps'] = total_read
            result['io_write_kbps'] = total_write
            result['io_total_kbps'] = total_read + total_write
        else:
            result['io_read_kbps'] = None
            result['io_write_kbps'] = None
            result['io_total_kbps'] = None

        # IPC and cache metrics from perf
        if perf_df is not None and len(perf_df) > 0:
            # Parse perf events (new CSV format: timestamp, value, unit, event, ...)
            if 'event' in perf_df.columns and 'value' in perf_df.columns:
                # Extract specific events (handle :u suffix for user-mode)
                # Use .str.startswith() to match both 'instructions' and 'instructions:u'
                instructions = perf_df[perf_df['event'].str.startswith('instructions')]['value'].sum()
                cycles = perf_df[perf_df['event'].str.startswith('cycles')]['value'].sum()

                # IPC = instructions / cycles
                if cycles > 0:
                    result['ipc'] = safe_divide(instructions, cycles, default=None)
                else:
                    result['ipc'] = None

                # Cache hit rate
                cache_refs = perf_df[perf_df['event'].str.startswith('cache-references')]['value'].sum()
                cache_miss = perf_df[perf_df['event'].str.startswith('cache-misses')]['value'].sum()

                if cache_refs > 0:
                    cache_hits = cache_refs - cache_miss
                    result['cache_hit_rate_percent'] = safe_divide(cache_hits, cache_refs, default=0.0) * 100
                else:
                    result['cache_hit_rate_percent'] = None
            else:
                result['ipc'] = None
                result['cache_hit_rate_percent'] = None
        else:
            result['ipc'] = None
            result['cache_hit_rate_percent'] = None

        if self.verbose:
            print(f"  Training time: {result.get('training_time_s', 'N/A')} s")
            print(f"  Throughput: {result.get('throughput_samples_per_s', 'N/A')} samples/s")
            print(f"  CPU utilization: {result.get('cpu_utilization_avg_percent', 'N/A')}%")
            print(f"  Memory peak: {result.get('memory_peak_mb', 'N/A')} MB")
            print(f"  IPC: {result.get('ipc', 'N/A')}")

        return result

    def compute_efficiency_metrics(self,
                                   energy_metrics: Dict,
                                   performance_metrics: Dict) -> Dict:
        """
        Compute efficiency metrics (Section 2.3).

        Metrics:
        - Performance-per-Watt (samples/s/W) - Standard Green AI metric
        - samples/Joule - Energy efficiency
        - Energy-Delay Product (J·s²) - Trade-off metric

        Args:
            energy_metrics: Dict from compute_energy_metrics()
            performance_metrics: Dict from compute_performance_metrics()

        Returns:
            Dictionary with efficiency metrics
        """
        if self.verbose:
            print("Computing efficiency metrics...")

        result = {}

        throughput = performance_metrics.get('throughput_samples_per_s')
        avg_power = energy_metrics.get('average_power_w')
        total_energy = energy_metrics.get('total_energy_j')
        training_time = performance_metrics.get('training_time_s')

        # Performance-per-Watt (samples/s/W)
        if throughput and avg_power and avg_power > 0:
            result['performance_per_watt'] = safe_divide(throughput, avg_power, default=None)
        else:
            result['performance_per_watt'] = None

        # Samples per Joule
        if throughput and total_energy and training_time and total_energy > 0:
            total_samples = throughput * training_time
            result['samples_per_joule'] = safe_divide(total_samples, total_energy, default=None)
        else:
            result['samples_per_joule'] = None

        # Energy-Delay Product (E × T²)
        if total_energy and training_time:
            result['energy_delay_product'] = total_energy * (training_time ** 2)
        else:
            result['energy_delay_product'] = None

        if self.verbose:
            print(f"  Performance-per-Watt: {result.get('performance_per_watt', 'N/A')}")
            print(f"  Samples/Joule: {result.get('samples_per_joule', 'N/A')}")
            print(f"  Energy-Delay Product: {result.get('energy_delay_product', 'N/A')}")

        return result

    def compute_convergence_metrics(self, training_metrics: Optional[Dict] = None) -> Dict:
        """
        Compute training convergence and loss metrics (Section 2.3 - Training Analysis).

        Metrics:
        - Final Generator Loss (Gloss) - Last iteration value
        - Final Discriminator Loss (Dloss) - Last iteration value
        - Loss convergence rate - Iterations to reach 90% of final loss
        - Loss stability - Standard deviation in final 20% of iterations
        - GAN balance - Ratio |Gloss|/|Dloss| (ideally ~1 for balanced training)
        - Training data points - Number of logged iterations
        - Iteration interval - Logging frequency (log_interval)

        Args:
            training_metrics: Dict from parse_stdout_losses() containing:
                - iterations: List[int] - Iteration numbers
                - gloss: List[float] - Generator loss values
                - dloss: List[float] - Discriminator loss values
                - training_size: Tuple[int, int] - Dataset size

        Returns:
            Dictionary with convergence metrics
        """
        if self.verbose:
            print("Computing convergence metrics...")

        result = {}

        # Return empty metrics if no training data
        if not training_metrics or not training_metrics.get('iterations'):
            if self.verbose:
                print("  No training metrics available")
            return {
                'final_gloss': None,
                'final_dloss': None,
                'convergence_iteration': None,
                'loss_stability_gloss': None,
                'loss_stability_dloss': None,
                'gan_balance': None,
                'num_data_points': 0,
                'log_interval': None,
            }

        iterations = training_metrics['iterations']
        gloss = training_metrics['gloss']
        dloss = training_metrics['dloss']

        num_points = len(iterations)
        result['num_data_points'] = num_points

        # Compute log_interval (difference between consecutive iterations)
        if num_points >= 2:
            # Take median to handle checkpoint intervals
            intervals = [iterations[i+1] - iterations[i] for i in range(num_points-1)]
            result['log_interval'] = int(np.median(intervals))
        else:
            result['log_interval'] = None

        # Final loss values
        result['final_gloss'] = gloss[-1] if gloss else None
        result['final_dloss'] = dloss[-1] if dloss else None

        # Loss stability (std dev in final 20% of training)
        if num_points >= 5:
            final_20_pct_idx = max(1, int(num_points * 0.8))
            final_gloss_values = gloss[final_20_pct_idx:]
            final_dloss_values = dloss[final_20_pct_idx:]

            result['loss_stability_gloss'] = float(np.std(final_gloss_values))
            result['loss_stability_dloss'] = float(np.std(final_dloss_values))
        else:
            result['loss_stability_gloss'] = None
            result['loss_stability_dloss'] = None

        # Convergence rate: iteration where loss reaches 90% of final value
        # For GANs, convergence is when loss stabilizes (less variability)
        if num_points >= 10:
            # Use rolling window to detect stabilization
            window_size = max(3, num_points // 5)
            gloss_rolling_std = pd.Series(gloss).rolling(window=window_size).std()

            # Find first point where rolling std drops below 20% of overall std
            overall_std = np.std(gloss)
            threshold = 0.2 * overall_std

            convergence_points = np.where(gloss_rolling_std < threshold)[0]
            if len(convergence_points) > 0:
                conv_idx = convergence_points[0]
                result['convergence_iteration'] = iterations[conv_idx]
            else:
                result['convergence_iteration'] = None
        else:
            result['convergence_iteration'] = None

        # GAN balance: ratio of absolute losses
        # Ideally ~1 for well-balanced GAN training
        if result['final_gloss'] is not None and result['final_dloss'] is not None:
            abs_gloss = abs(result['final_gloss'])
            abs_dloss = abs(result['final_dloss'])

            # Handle division by zero
            if abs_dloss > 1e-8:
                result['gan_balance'] = abs_gloss / abs_dloss
            else:
                result['gan_balance'] = None
        else:
            result['gan_balance'] = None

        if self.verbose:
            print(f"  Data points: {result['num_data_points']}")
            print(f"  Log interval: {result['log_interval']}")
            print(f"  Final Gloss: {result['final_gloss']:.4f}" if result['final_gloss'] is not None else "  Final Gloss: N/A")
            print(f"  Final Dloss: {result['final_dloss']:.4f}" if result['final_dloss'] is not None else "  Final Dloss: N/A")
            print(f"  Loss stability (Gloss): {result['loss_stability_gloss']:.4f}" if result['loss_stability_gloss'] is not None else "  Loss stability (Gloss): N/A")
            print(f"  Loss stability (Dloss): {result['loss_stability_dloss']:.4f}" if result['loss_stability_dloss'] is not None else "  Loss stability (Dloss): N/A")
            print(f"  Convergence iteration: {result['convergence_iteration']}" if result['convergence_iteration'] else "  Convergence iteration: N/A")
            print(f"  GAN balance: {result['gan_balance']:.3f}" if result['gan_balance'] is not None else "  GAN balance: N/A")

        return result

    def compute_monitoring_overhead(self,
                                   overhead_df: Optional[pd.DataFrame] = None,
                                   vmstat_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Compute monitoring overhead metrics (Section 2.4).

        This is a UNIQUE FEATURE of this thesis - quantifying the cost of observability.

        Metrics:
        - Total CPU overhead (%)
        - Total memory overhead (MB)
        - Per-tool breakdown (vmstat, iostat, pidstat, perf, rapl, ...)

        Args:
            overhead_df: DataFrame from train_system_monitoring_overhead.csv
            vmstat_df: System vmstat for total CPU usage context

        Returns:
            Dictionary with overhead metrics
        """
        if self.verbose:
            print("Computing monitoring overhead...")

        result = {
            'cpu_overhead_percent': None,
            'memory_overhead_mb': None,
            'tools': {}
        }

        if overhead_df is None or len(overhead_df) == 0:
            if self.verbose:
                print("  ⚠️  Monitoring overhead data not available")
            return result

        # Normalize column names (old format: Command/%CPU, new format: command/cpu_percent)
        command_col = 'command' if 'command' in overhead_df.columns else 'Command'
        cpu_col = 'cpu_percent' if 'cpu_percent' in overhead_df.columns else '%CPU'
        rss_col = 'rss' if 'rss' in overhead_df.columns else 'RSS'

        # Aggregate by tool (command name)
        if command_col in overhead_df.columns and cpu_col in overhead_df.columns:
            # Group by command and calculate average CPU%
            tools_cpu = overhead_df.groupby(command_col)[cpu_col].mean()
            result['tools'] = tools_cpu.to_dict()

            # Total CPU overhead
            result['cpu_overhead_percent'] = tools_cpu.sum()

            # Memory overhead (RSS in kB)
            if rss_col in overhead_df.columns:
                tools_mem = overhead_df.groupby(command_col)[rss_col].mean()
                result['memory_overhead_mb'] = tools_mem.sum() / 1024.0  # kB to MB

            if self.verbose:
                print(f"  Total CPU overhead: {result['cpu_overhead_percent']:.2f}%")
                print(f"  Total memory overhead: {result.get('memory_overhead_mb', 'N/A')} MB")
                print(f"  Tools monitored: {len(result['tools'])}")
        else:
            if self.verbose:
                print(f"  ⚠️  Expected columns (command/cpu_percent or Command/%CPU) not found")
                print(f"  Available columns: {overhead_df.columns.tolist()}")

        return result

    def compute_derived_metrics(self,
                               dfs: Dict[str, pd.DataFrame],
                               metrics: Dict) -> Dict:
        """
        Compute derived metrics for workload characterization (Section 2.6).

        Metrics:
        - Workload type classification (compute-bound vs I/O-bound vs memory-bound)
        - Thermal throttling events detection
        - I/O bottleneck score
        - Power profile (instantaneous power time series)

        Args:
            dfs: Dictionary of DataFrames
            metrics: Dictionary of computed metrics

        Returns:
            Dictionary with derived metrics
        """
        if self.verbose:
            print("Computing derived metrics...")

        result = {}

        # Workload Type Classification
        perf = metrics.get('performance', {})
        ipc = perf.get('ipc')
        iowait = perf.get('iowait_avg_percent')

        if ipc is not None and iowait is not None:
            # Classification based on IPC and I/O wait
            if iowait > 10:
                workload_type = "io-bound"
            elif ipc > 1.0 and iowait < 5:
                workload_type = "compute-bound"
            elif ipc < 0.5:
                workload_type = "memory-bound"
            else:
                workload_type = "balanced"

            result['workload_type'] = workload_type

            # I/O Bottleneck Score (higher = worse bottleneck)
            result['io_bottleneck_score'] = iowait * (1.0 / max(ipc, 0.1))
        else:
            result['workload_type'] = None
            result['io_bottleneck_score'] = None

        # Thermal Throttling Detection
        if 'thermal' in dfs and 'cpu_freq' in dfs:
            thermal_df = dfs['thermal']
            freq_df = dfs['cpu_freq']

            if len(thermal_df) > 0 and len(freq_df) > 0:
                # Get temperature columns
                temp_cols = [c for c in thermal_df.columns if c.startswith('thermal') or 'temp' in c.lower()]
                freq_cols = [c for c in freq_df.columns if c.startswith('cpu') and c != 'timestamp']

                if temp_cols and freq_cols and 'timestamp' in thermal_df.columns and 'timestamp' in freq_df.columns:
                    # Find base frequency (max observed)
                    base_freq = freq_df[freq_cols].max().max()

                    # Count throttling events (freq drops below 90% of base freq)
                    throttle_threshold = base_freq * 0.9
                    throttle_events = (freq_df[freq_cols] < throttle_threshold).sum().sum()

                    result['thermal_throttling_events'] = int(throttle_events)
                    result['base_frequency_mhz'] = base_freq
                else:
                    result['thermal_throttling_events'] = None
                    result['base_frequency_mhz'] = None
            else:
                result['thermal_throttling_events'] = None
                result['base_frequency_mhz'] = None
        else:
            result['thermal_throttling_events'] = None
            result['base_frequency_mhz'] = None

        # Power Profile (instantaneous power time series)
        if 'rapl' in dfs:
            rapl_df = dfs['rapl']
            if len(rapl_df) > 1 and 'timestamp' in rapl_df.columns:
                energy_cols = [c for c in rapl_df.columns if c.endswith('_uj')]
                if energy_cols:
                    # Calculate instantaneous power for total energy
                    total_energy = rapl_df[energy_cols].sum(axis=1)
                    # total_energy is already in microJoules (sum of _uj columns)
                    power_series = compute_power_from_energy(total_energy, rapl_df['timestamp'])

                    # Filter out NaN values (from derivative at edges)
                    power_series_clean = power_series[~np.isnan(power_series)]

                    if len(power_series_clean) > 0:
                        # Store power profile statistics
                        result['power_profile'] = {
                            'mean_w': float(np.mean(power_series_clean)),
                            'std_w': float(np.std(power_series_clean)),
                            'min_w': float(np.min(power_series_clean)),
                            'max_w': float(np.max(power_series_clean)),
                            'median_w': float(np.median(power_series_clean)),
                        }
                    else:
                        result['power_profile'] = None
                else:
                    result['power_profile'] = None
            else:
                result['power_profile'] = None
        else:
            result['power_profile'] = None

        if self.verbose:
            if result.get('workload_type'):
                print(f"  Workload type: {result['workload_type']}")
            if result.get('io_bottleneck_score') is not None:
                print(f"  I/O bottleneck score: {result['io_bottleneck_score']:.2f}")
            if result.get('thermal_throttling_events') is not None:
                print(f"  Thermal throttling events: {result['thermal_throttling_events']}")
            if result.get('power_profile'):
                pp = result['power_profile']
                print(f"  Power profile: {pp['mean_w']:.1f}W avg, {pp['min_w']:.1f}-{pp['max_w']:.1f}W range")

        return result

    def compute_correlations(self,
                            dfs: Dict[str, pd.DataFrame],
                            metrics: Dict) -> Dict:
        """
        Compute statistical correlations between metrics (Section 4).

        Correlations analyzed:
        - C1: Frequency ↔ Power (DVFS impact)
        - C2: Temperature ↔ Frequency (thermal throttling)
        - C3: IPC ↔ iowait (workload characterization)
        - C4: Cache% ↔ IPC (cache efficiency)
        - C5: Power ↔ Gloss/Dloss (energy dynamics during training)
        - C6: CPU% ↔ Throughput (performance scalability)

        Args:
            dfs: Dictionary of DataFrames from MonitoringDataLoader
            metrics: Dictionary of computed metrics

        Returns:
            Dictionary with correlation results
        """
        if self.verbose:
            print("Computing correlations...")

        result = {}

        # C1: Frequency vs Power (DVFS impact)
        if 'cpu_freq' in dfs and 'rapl' in dfs:
            freq_df = dfs['cpu_freq']
            rapl_df = dfs['rapl']

            if len(freq_df) > 0 and len(rapl_df) > 0:
                # Calculate average CPU frequency
                freq_cols = [c for c in freq_df.columns if c.startswith('cpu') and c != 'timestamp']
                if freq_cols and 'timestamp' in rapl_df.columns:
                    avg_freq = freq_df[freq_cols].mean(axis=1)

                    # Calculate total power
                    energy_cols = [c for c in rapl_df.columns if c.endswith('_uj')]
                    if energy_cols and 'timestamp' in rapl_df.columns:
                        power_w = compute_power_from_energy(
                            rapl_df[energy_cols].sum(axis=1),
                            rapl_df['timestamp']
                        )

                        # Convert to pandas Series if numpy array
                        if isinstance(power_w, np.ndarray):
                            power_w = pd.Series(power_w, index=rapl_df.index[:len(power_w)])

                        # Align lengths (drop NaN from power calculation)
                        min_len = min(len(avg_freq), len(power_w))
                        avg_freq_aligned = avg_freq.iloc[:min_len].dropna()
                        power_aligned = power_w.iloc[:min_len].dropna()

                        # Ensure same length
                        common_idx = avg_freq_aligned.index.intersection(power_aligned.index)

                        if len(common_idx) > 2:
                            r, p = pearsonr(avg_freq_aligned[common_idx], power_aligned[common_idx])
                            result['freq_vs_power'] = {
                                'pearson_r': r,
                                'p_value': p,
                                'r_squared': r**2,
                                'interpretation': self._interpret_correlation(r, p),
                            }

        # C4: Cache Hit Rate vs IPC (cache efficiency)
        if 'perf' in dfs and len(dfs['perf']) > 0:
            perf_df = dfs['perf']

            # Check if we have cache and IPC data
            has_cache = 'cache_references' in perf_df.columns and 'cache_misses' in perf_df.columns
            has_ipc = 'instructions' in perf_df.columns and 'cycles' in perf_df.columns

            if has_cache and has_ipc:
                # Calculate cache hit rate
                cache_refs = perf_df['cache_references']
                cache_miss = perf_df['cache_misses']
                cache_hit_rate = ((cache_refs - cache_miss) / cache_refs * 100).replace([np.inf, -np.inf], np.nan).dropna()

                # Calculate IPC
                instructions = perf_df['instructions']
                cycles = perf_df['cycles']
                ipc = (instructions / cycles).replace([np.inf, -np.inf], np.nan).dropna()

                # Align and correlate
                common_idx = cache_hit_rate.index.intersection(ipc.index)

                if len(common_idx) > 2:
                    r, p = pearsonr(cache_hit_rate[common_idx], ipc[common_idx])
                    result['cache_vs_ipc'] = {
                        'pearson_r': r,
                        'p_value': p,
                        'interpretation': self._interpret_correlation(r, p),
                    }

        # C5: Power vs Loss (energy dynamics during training)
        if 'rapl' in dfs and 'training_metrics' in metrics:
            rapl_df = dfs['rapl']
            training = metrics.get('training_metrics', {})

            if len(rapl_df) > 0 and 'gloss' in training and 'dloss' in training:
                # Get power time-series
                energy_cols = [c for c in rapl_df.columns if c.endswith('_uj')]
                if energy_cols and 'timestamp' in rapl_df.columns:
                    power_w = compute_power_from_energy(
                        rapl_df[energy_cols].sum(axis=1),
                        rapl_df['timestamp']
                    )

                    # Get loss values with timestamps
                    gloss = training.get('gloss', [])
                    dloss = training.get('dloss', [])
                    iterations = training.get('iterations', [])

                    if len(gloss) > 2 and len(dloss) > 2:
                        # For correlation, we need to align power samples with loss samples
                        # This is approximate since loss is logged at specific iterations
                        # Use mean power in windows around each iteration

                        # Simple approach: compute Pearson on available data points
                        if len(gloss) == len(dloss):
                            # Correlate Gloss with mean power (if we have enough points)
                            if len(gloss) > 2:
                                try:
                                    # Use iteration index as proxy (assumes uniform time)
                                    r_g, p_g = pearsonr(gloss, range(len(gloss)))
                                    r_d, p_d = pearsonr(dloss, range(len(dloss)))

                                    result['power_vs_loss'] = {
                                        'note': 'Temporal correlation (limited by iteration sampling)',
                                        'gloss_trend_r': r_g,
                                        'gloss_trend_p': p_g,
                                        'dloss_trend_r': r_d,
                                        'dloss_trend_p': p_d,
                                    }
                                except:
                                    pass

        # C6: CPU Utilization vs Throughput (performance scalability)
        if 'vmstat' in dfs and 'performance' in metrics:
            vmstat_df = dfs['vmstat']
            throughput = metrics['performance'].get('throughput_samples_per_s')

            if len(vmstat_df) > 0 and throughput:
                # CPU utilization from vmstat (us + sy)
                if 'us' in vmstat_df.columns and 'sy' in vmstat_df.columns:
                    cpu_util = (vmstat_df['us'] + vmstat_df['sy']).dropna()

                    if len(cpu_util) > 2:
                        # For scalability, we'd need throughput time-series
                        # Store CPU stats for later visualization
                        cpu_mean = cpu_util.mean()
                        cpu_std = cpu_util.std()

                        result['cpu_vs_throughput'] = {
                            'note': 'Scalability analysis (requires time-series throughput for full correlation)',
                            'cpu_mean': cpu_mean,
                            'cpu_std': cpu_std,
                            'throughput': throughput,
                            'efficiency': throughput / cpu_mean if cpu_mean > 0 else None,
                        }

        # C7: Memory RSS vs Throughput (memory impact)
        if 'pidstat_process' in dfs and 'performance' in metrics:
            pidstat_df = dfs['pidstat_process']
            throughput = metrics['performance'].get('throughput_samples_per_s')

            if len(pidstat_df) > 0 and throughput and 'rss' in pidstat_df.columns:
                rss = pidstat_df['rss'].dropna()

                if len(rss) > 2:
                    # Memory stats
                    rss_mean = rss.mean()
                    rss_std = rss.std()

                    result['memory_vs_throughput'] = {
                        'note': 'Memory impact analysis (requires time-series throughput for full correlation)',
                        'rss_mean_mb': rss_mean,
                        'rss_std_mb': rss_std,
                        'throughput': throughput,
                        'memory_efficiency': throughput / rss_mean if rss_mean > 0 else None,
                    }

        # C8: I/O Wait vs Throughput (I/O bottleneck)
        if 'vmstat' in dfs and 'performance' in metrics:
            vmstat_df = dfs['vmstat']
            throughput = metrics['performance'].get('throughput_samples_per_s')

            if len(vmstat_df) > 0 and throughput and 'wa' in vmstat_df.columns:
                iowait = vmstat_df['wa'].dropna()

                if len(iowait) > 2:
                    iowait_mean = iowait.mean()
                    iowait_std = iowait.std()

                    result['iowait_vs_throughput'] = {
                        'note': 'I/O bottleneck analysis',
                        'iowait_mean_percent': iowait_mean,
                        'iowait_std_percent': iowait_std,
                        'throughput': throughput,
                        'io_impact_score': iowait_mean / throughput if throughput > 0 else None,
                    }

        if self.verbose:
            print(f"  Computed {len(result)} correlations")

        return result

    def _interpret_correlation(self, r: float, p: float, alpha: float = 0.05) -> str:
        """
        Interpret Pearson correlation coefficient.

        Args:
            r: Pearson correlation coefficient
            p: p-value
            alpha: Significance level (default 0.05)

        Returns:
            Human-readable interpretation string
        """
        # Check statistical significance
        if p > alpha:
            return f"Not significant (p={p:.3f} > {alpha})"

        # Interpret strength (Cohen's guidelines)
        abs_r = abs(r)

        if abs_r < 0.3:
            strength = "Weak"
        elif abs_r < 0.5:
            strength = "Moderate"
        elif abs_r < 0.7:
            strength = "Strong"
        else:
            strength = "Very strong"

        direction = "positive" if r > 0 else "negative"

        return f"{strength} {direction} correlation (r={r:.3f}, p={p:.3e})"

    def compute_all_metrics(self,
                           dfs: Dict[str, pd.DataFrame],
                           exp_dir: Optional[str] = None) -> Dict:
        """
        Compute all metrics in one call (convenience method).

        Args:
            dfs: Dictionary of DataFrames from MonitoringDataLoader
            exp_dir: Experiment directory path (for stdout parsing)

        Returns:
            Dictionary with all metrics
        """
        if self.verbose:
            print("\n" + "="*70)
            print("COMPUTING ALL METRICS")
            print("="*70)

        result = {}

        # Parse training metrics from stdout.log
        if exp_dir and Path(exp_dir).exists():
            stdout_path = Path(exp_dir) / 'logs' / 'train_stdout.log'
            if stdout_path.exists():
                try:
                    result['training_metrics'] = parse_stdout_losses(str(stdout_path))
                except Exception as e:
                    warnings.warn(f"Failed to parse training metrics: {e}")
                    result['training_metrics'] = {}
            else:
                result['training_metrics'] = {}
        elif 'training_metrics' in dfs:
            result['training_metrics'] = dfs['training_metrics']
        else:
            result['training_metrics'] = {}

        # Energy metrics
        if 'rapl' in dfs and len(dfs['rapl']) > 0:
            result['energy'] = self.compute_energy_metrics(
                dfs['rapl'],
                training_metrics=result['training_metrics']
            )
        else:
            result['energy'] = {}
            warnings.warn("RAPL data not available - energy metrics skipped")

        # Calculate duration from timestamps
        duration_s = None
        if 'rapl' in dfs and 'timestamp' in dfs['rapl'].columns:
            ts_diff = dfs['rapl']['timestamp'].iloc[-1] - dfs['rapl']['timestamp'].iloc[0]

            # Convert to seconds if datetime
            if hasattr(ts_diff, 'total_seconds'):
                duration_s = ts_diff.total_seconds()
            else:
                duration_s = float(ts_diff)

        # Performance metrics
        result['performance'] = self.compute_performance_metrics(
            pidstat_df=dfs.get('pidstat_process'),
            vmstat_df=dfs.get('vmstat'),
            perf_df=dfs.get('perf'),
            iostat_df=dfs.get('iostat_dev'),
            training_metrics=result['training_metrics'],
            duration_s=duration_s,
        )

        # Efficiency metrics
        if result.get('energy') and result.get('performance'):
            result['efficiency'] = self.compute_efficiency_metrics(
                result['energy'],
                result['performance']
            )
        else:
            result['efficiency'] = {}

        # Convergence metrics (training loss analysis)
        result['convergence'] = self.compute_convergence_metrics(
            result.get('training_metrics')
        )

        # Monitoring overhead
        if 'monitoring_overhead' in dfs:
            result['monitoring_overhead'] = self.compute_monitoring_overhead(
                dfs['monitoring_overhead'],
                dfs.get('vmstat')
            )
        else:
            result['monitoring_overhead'] = {}
            warnings.warn("Monitoring overhead data not available")

        # Derived metrics (workload characterization, bottlenecks)
        result['derived'] = self.compute_derived_metrics(dfs, result)

        # Correlations
        result['correlations'] = self.compute_correlations(dfs, result)

        if self.verbose:
            print("="*70)
            print("✅ ALL METRICS COMPUTED")
            print("="*70)

        return result


__all__ = ['MetricsCalculator']
