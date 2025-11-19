"""
MonitoringVisualizer: Generate comprehensive plots for BoloGAN monitoring analysis.

This module implements all 11 required plots from ANALYSIS_SPECIFICATION.md:
    1. Power Profile (time-series)
    2. Energy Breakdown (pie chart)
    3. Frequency vs Power (scatter + regression)
    4. Memory Usage (stacked area)
    5. Thermal Analysis (multi-axis time-series)
    6. Monitoring Overhead (bar chart)
    7. Loss Evolution (time-series)
    8. Power vs Loss Evolution (dual-axis time-series)
    9. Workload Characterization (2D scatter, IPC vs iowait)
    10. Energy-Performance Trade-off (Pareto frontier)
    11. Performance Scalability (scatter + regression)

Author: Claude Code
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from scipy.stats import linregress
import seaborn as sns


class MonitoringVisualizer:
    """
    Visualizer for BoloGAN monitoring data analysis.

    Generates publication-quality plots (300 DPI, 1920x1080) for thesis.
    """

    def __init__(self, output_dir: str = "analysis/plots", dpi: int = 300,
                 figsize: Tuple[float, float] = (12, 6), style: str = 'seaborn-v0_8-darkgrid',
                 image_format: str = 'jpeg'):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving plots
            dpi: DPI for saved figures (300 for publication)
            figsize: Figure size in inches (width, height)
            style: Matplotlib style
            image_format: Image format for saved plots ('jpeg' or 'png', default: 'jpeg')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self.image_format = image_format.lower()
        self.image_ext = 'jpg' if self.image_format == 'jpeg' else self.image_format

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')

        # Color palette
        self.colors = {
            'power_total': '#e74c3c',      # Red
            'power_cpu': '#3498db',        # Blue
            'power_dram': '#2ecc71',       # Green
            'gloss': '#3498db',            # Blue
            'dloss': '#2ecc71',            # Green
            'temp': '#e67e22',             # Orange
            'freq': '#9b59b6',             # Purple
            'memory_used': '#e74c3c',
            'memory_buff': '#f39c12',
            'memory_free': '#95a5a6'
        }

    def _standardize_rapl_columns(self, rapl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize RAPL column names and compute power from energy if needed.

        Handles both raw format (intel-rapl:*_uj) and processed format (power_*).
        """
        rapl_df = rapl_df.copy()

        # Check if already in standardized format
        if 'power_package_0' in rapl_df.columns:
            return rapl_df

        # Map raw column names to standard names
        col_mapping = {}
        for col in rapl_df.columns:
            if 'package_0' in col and 'dram' not in col:
                col_mapping[col] = 'energy_package_0_uj'
            elif 'package_1' in col and 'dram' not in col:
                col_mapping[col] = 'energy_package_1_uj'
            elif 'dram' in col and ('rapl:0' in col or 'dram_0' in col):
                col_mapping[col] = 'energy_dram_0_uj'
            elif 'dram' in col and ('rapl:1' in col or 'dram_1' in col):
                col_mapping[col] = 'energy_dram_1_uj'

        rapl_df = rapl_df.rename(columns=col_mapping)

        # Convert energy (microjoules) to power (watts)
        # P = ΔE / Δt
        for component in ['package_0', 'package_1', 'dram_0', 'dram_1']:
            energy_col = f'energy_{component}_uj'
            power_col = f'power_{component}'

            if energy_col in rapl_df.columns:
                # Compute power from energy difference
                energy_j = rapl_df[energy_col] / 1e6  # microjoules to joules
                time_diff = rapl_df['timestamp_rel'].diff().fillna(1.0)  # seconds
                rapl_df[power_col] = energy_j.diff() / time_diff
                # Fill first NaN with second value
                rapl_df[power_col] = rapl_df[power_col].bfill()
                # Clip negative values (overflow artifacts)
                rapl_df[power_col] = rapl_df[power_col].clip(lower=0)

        return rapl_df

    def plot_power_profile(self, rapl_df: pd.DataFrame, detailed: bool = False) -> str:
        """
        Plot 1: Power Profile (time-series).

        Args:
            rapl_df: DataFrame with power or energy columns (will be standardized)
            detailed: If True, also plot CPU/DRAM breakdown

        Returns:
            Path to saved plot
        """
        # Ensure timestamp_rel exists
        rapl_df = self._ensure_timestamp_rel(rapl_df)

        # Standardize RAPL columns
        rapl_df = self._standardize_rapl_columns(rapl_df)

        # Increase figure width for detailed plot to reduce crowding
        if detailed:
            figsize = (20, 6)
        else:
            figsize = (16, 6)
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate total, CPU, DRAM power
        rapl_df['power_total'] = (
            rapl_df['power_package_0'] + rapl_df['power_package_1'] +
            rapl_df['power_dram_0'] + rapl_df['power_dram_1']
        )
        rapl_df['power_cpu'] = rapl_df['power_package_0'] + rapl_df['power_package_1']
        rapl_df['power_dram'] = rapl_df['power_dram_0'] + rapl_df['power_dram_1']

        # Plot
        if detailed:
            # Detailed: only CPU and DRAM breakdown (total shown in other plot)
            # Use solid capstyle and joinstyle for smoother appearance
            ax.plot(rapl_df['timestamp_rel'], rapl_df['power_cpu'],
                   label='CPU Power', color=self.colors['power_cpu'], linewidth=2.5, alpha=0.8,
                   solid_capstyle='round', solid_joinstyle='round')
            ax.plot(rapl_df['timestamp_rel'], rapl_df['power_dram'],
                   label='DRAM Power', color=self.colors['power_dram'], linewidth=2.5, alpha=0.8,
                   solid_capstyle='round', solid_joinstyle='round')
            filename = f'power_profile_detailed.{self.image_ext}'
        else:
            # Simple: only total power with thinner line
            ax.plot(rapl_df['timestamp_rel'], rapl_df['power_total'],
                   label='Total Power', color=self.colors['power_total'], linewidth=2)
            filename = f'power_profile.{self.image_ext}'

        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power (Watt)', fontsize=12, fontweight='bold')
        ax.set_title('Power Profile', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Statistics annotation
        avg_power = rapl_df['power_total'].mean()
        peak_power = rapl_df['power_total'].max()
        min_power = rapl_df['power_total'].min()

        if detailed:
            # Add note about oscillations for detailed plot
            power_std = rapl_df['power_total'].std()
            ax.text(0.02, 0.98,
                   f'Avg: {avg_power:.1f} W\nPeak: {peak_power:.1f} W\nMin: {min_power:.1f} W\nStd: {power_std:.1f} W\n\nOscillations due to variable\ncomputational load per iteration',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.02, 0.98, f'Avg: {avg_power:.1f} W\nPeak: {peak_power:.1f} W',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_energy_breakdown(self, energy_metrics: Dict) -> str:
        """
        Plot 2: Energy Breakdown (pie chart).

        Args:
            energy_metrics: Dict with 'energy_breakdown' key containing either:
                           - Aggregated: {'cpu_j', 'dram_j'}
                           - Per-component: {'package_0_j', 'package_1_j', 'dram_0_j', 'dram_1_j'}

        Returns:
            Path to saved plot
        """
        breakdown = energy_metrics.get('energy_breakdown', {})

        if not breakdown:
            print("⚠️  Warning: No energy breakdown data available")
            return ""

        # Check format: aggregated or per-component
        if 'cpu_j' in breakdown and 'dram_j' in breakdown:
            # Aggregated format
            labels = ['CPU', 'DRAM']
            values = [breakdown['cpu_j'], breakdown['dram_j']]
        elif 'package_0_j' in breakdown:
            # Per-component format
            labels = ['CPU Package 0', 'CPU Package 1', 'DRAM 0', 'DRAM 1']
            values = [
                breakdown.get('package_0_j', 0),
                breakdown.get('package_1_j', 0),
                breakdown.get('dram_0_j', 0),
                breakdown.get('dram_1_j', 0)
            ]
        else:
            print("⚠️  Warning: Unknown energy breakdown format")
            return ""

        # Check for valid data
        if sum(values) == 0 or any(np.isnan(v) or v is None for v in values):
            print("⚠️  Warning: Invalid energy breakdown data (NaN or zero)")
            return ""

        # Convert to kJ for readability
        values_kj = [v / 1000 for v in values]

        # Colors
        colors_pie = ['#3498db', '#2980b9', '#2ecc71', '#27ae60']

        fig, ax = plt.subplots(figsize=(10, 8))

        # Pie chart
        wedges, texts, autotexts = ax.pie(
            values_kj, labels=labels, autopct='%1.1f%%',
            colors=colors_pie, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )

        # Enhance autotext
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # Add absolute values in legend
        legend_labels = [f'{label}: {value:.2f} kJ' for label, value in zip(labels, values_kj)]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

        ax.set_title('Energy Breakdown by Component', fontsize=14, fontweight='bold')

        # Calculate CPU vs DRAM percentages
        if len(values) == 2:
            # Aggregated format
            cpu_percent = values[0] / sum(values) * 100
            dram_percent = values[1] / sum(values) * 100
        else:
            # Per-component format
            cpu_percent = (values[0] + values[1]) / sum(values) * 100
            dram_percent = (values[2] + values[3]) / sum(values) * 100

        # Interpretation annotation
        if cpu_percent > 70:
            workload = "Compute-intensive workload"
        elif dram_percent > 40:
            workload = "Memory-intensive workload"
        else:
            workload = "Balanced workload"

        ax.text(0.5, -0.1, f'CPU: {cpu_percent:.1f}% | DRAM: {dram_percent:.1f}%\n{workload}',
                transform=ax.transAxes, fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        output_path = self.output_dir / f'energy_breakdown.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_frequency_vs_power(self, freq_df: pd.DataFrame, rapl_df: pd.DataFrame) -> str:
        """
        Plot 3: Frequency vs Power (scatter + linear regression).

        Args:
            freq_df: DataFrame with frequency data
            rapl_df: DataFrame with power or energy data

        Returns:
            Path to saved plot
        """
        # Ensure timestamp_rel exists
        freq_df = self._ensure_timestamp_rel(freq_df)
        rapl_df = self._ensure_timestamp_rel(rapl_df)

        # Calculate average frequency if not present
        if 'avg_freq_mhz' not in freq_df.columns:
            # Find all cpu*_mhz columns
            cpu_cols = [col for col in freq_df.columns if col.startswith('cpu') and col.endswith('_mhz')]
            if len(cpu_cols) > 0:
                freq_df['avg_freq_mhz'] = freq_df[cpu_cols].mean(axis=1)
            else:
                print("⚠️  Warning: No frequency columns found")
                return ""

        # Standardize RAPL columns
        rapl_df = self._standardize_rapl_columns(rapl_df)

        # Calculate power_total if not present
        if 'power_total' not in rapl_df.columns:
            rapl_df['power_total'] = (
                rapl_df['power_package_0'] + rapl_df['power_package_1'] +
                rapl_df['power_dram_0'] + rapl_df['power_dram_1']
            )

        # Merge on timestamp
        merged = pd.merge_asof(
            freq_df[['timestamp_rel', 'avg_freq_mhz']].sort_values('timestamp_rel'),
            rapl_df[['timestamp_rel', 'power_total']].sort_values('timestamp_rel'),
            on='timestamp_rel',
            direction='nearest',
            tolerance=1.0
        ).dropna()

        if len(merged) == 0:
            print("⚠️  Warning: No overlapping data for frequency vs power plot")
            return ""

        freq = merged['avg_freq_mhz'].values
        power = merged['power_total'].values

        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(freq, power)
        r_squared = r_value ** 2

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)

        # Scatter
        ax.scatter(freq, power, alpha=0.6, color='#3498db', s=30, label='Measurements')

        # Regression line
        freq_range = np.linspace(freq.min(), freq.max(), 100)
        power_pred = slope * freq_range + intercept
        ax.plot(freq_range, power_pred, 'r--', linewidth=2.5,
               label=f'Linear Fit: P = {slope:.4f}×f + {intercept:.1f}')

        # Formatting
        ax.set_xlabel('Average CPU Frequency (MHz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Power (Watt)', fontsize=12, fontweight='bold')
        ax.set_title('DVFS Analysis: Frequency vs Power', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # R² annotation - simple, no explanation
        note_text = f'R² = {r_squared:.4f}\np-value = {p_value:.2e}\n\nTemporal evolution'

        ax.text(0.05, 0.95, note_text,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        output_path = self.output_dir / f'frequency_vs_power.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_memory_usage(self, mem_df: pd.DataFrame) -> str:
        """
        Plot 4: Memory Usage (stacked area time-series).

        Args:
            mem_df: DataFrame with columns: timestamp, total, used, free, buff/cache (in MB)

        Returns:
            Path to saved plot
        """
        # Create time axis (seconds from start)
        mem_df = mem_df.copy()
        if 'timestamp_rel' not in mem_df.columns:
            if 'timestamp' in mem_df.columns:
                mem_df['timestamp_rel'] = (mem_df['timestamp'] - mem_df['timestamp'].iloc[0]).dt.total_seconds()
            else:
                mem_df['timestamp_rel'] = np.arange(len(mem_df))

        # Convert MB to GB
        used_gb = mem_df['used'] / 1024
        buff_cache_gb = mem_df['buff/cache'] / 1024
        free_gb = mem_df['free'] / 1024
        total_gb = mem_df['total'] / 1024

        fig, ax = plt.subplots(figsize=self.figsize)

        # Stacked area
        ax.fill_between(mem_df['timestamp_rel'], 0, used_gb,
                        label='Used', color=self.colors['memory_used'], alpha=0.7)
        ax.fill_between(mem_df['timestamp_rel'], used_gb,
                        used_gb + buff_cache_gb,
                        label='Buff/Cache', color=self.colors['memory_buff'], alpha=0.7)
        ax.fill_between(mem_df['timestamp_rel'],
                        used_gb + buff_cache_gb,
                        total_gb,
                        label='Free', color=self.colors['memory_free'], alpha=0.7)

        # Total RAM line
        ax.plot(mem_df['timestamp_rel'], total_gb,
               color='black', linewidth=2, linestyle='--', label='Total RAM')

        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
        ax.set_title('Memory Usage Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, total_gb.max() * 1.05)

        # Memory leak detection
        used_trend = np.polyfit(mem_df['timestamp_rel'], used_gb, 1)
        if used_trend[0] > 0.01:  # Significant growth (>0.01 GB/s)
            ax.text(0.98, 0.02, '⚠️  Potential memory leak detected',
                   transform=ax.transAxes, fontsize=10, ha='right',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        else:
            # Add explanation of buff/cache
            avg_buff_cache = buff_cache_gb.mean()
            ax.text(0.98, 0.02,
                   f'Buff/Cache: Linux filesystem cache\nusing free RAM ({avg_buff_cache:.1f} GB avg)',
                   transform=ax.transAxes, fontsize=9, ha='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        output_path = self.output_dir / f'memory_usage.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_thermal_analysis(self, thermal_df: pd.DataFrame, rapl_df: pd.DataFrame,
                             freq_df: pd.DataFrame) -> str:
        """
        Plot 5: Thermal Analysis (multi-axis time-series).

        Args:
            thermal_df: DataFrame with thermal data
            rapl_df: DataFrame with power or energy data
            freq_df: DataFrame with frequency data

        Returns:
            Path to saved plot
        """
        # Ensure timestamp_rel exists
        thermal_df = self._ensure_timestamp_rel(thermal_df)
        rapl_df = self._ensure_timestamp_rel(rapl_df)
        freq_df = self._ensure_timestamp_rel(freq_df)

        # Convert thermal data to Celsius if in milliCelsius
        if 'thermal_zone0_milliC' in thermal_df.columns:
            thermal_df['thermal_zone0_c'] = thermal_df['thermal_zone0_milliC'] / 1000
        if 'thermal_zone1_milliC' in thermal_df.columns:
            thermal_df['thermal_zone1_c'] = thermal_df['thermal_zone1_milliC'] / 1000

        # Calculate average frequency if needed
        if 'avg_freq_mhz' not in freq_df.columns:
            cpu_cols = [col for col in freq_df.columns if col.startswith('cpu') and col.endswith('_mhz')]
            if len(cpu_cols) > 0:
                freq_df['avg_freq_mhz'] = freq_df[cpu_cols].mean(axis=1)

        # Standardize RAPL columns
        rapl_df = self._standardize_rapl_columns(rapl_df)

        # Calculate power_total if not present
        if 'power_total' not in rapl_df.columns:
            rapl_df['power_total'] = (
                rapl_df['power_package_0'] + rapl_df['power_package_1'] +
                rapl_df['power_dram_0'] + rapl_df['power_dram_1']
            )

        # Use very wide figsize for maximum clarity
        fig, ax1 = plt.subplots(figsize=(24, 6))

        # Temperature (primary y-axis) - Using more distinct colors
        color_temp0 = '#ff6b6b'  # Bright red
        color_temp1 = '#ff8c42'  # Orange
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax1.plot(thermal_df['timestamp_rel'], thermal_df['thermal_zone0_c'],
                color=color_temp0, linewidth=2.5, label='Temp Zone 0', alpha=0.9)
        if 'thermal_zone1_c' in thermal_df.columns:
            # Both zones solid line (no dashes)
            ax1.plot(thermal_df['timestamp_rel'], thermal_df['thermal_zone1_c'],
                    color=color_temp1, linewidth=2.5, linestyle='-', label='Temp Zone 1', alpha=0.8)
        ax1.tick_params(axis='y')
        ax1.grid(True, alpha=0.3)

        # Power (secondary y-axis 1)
        ax2 = ax1.twinx()
        color_power = '#3498db'  # Blue for power
        ax2.set_ylabel('Power (Watt)', fontsize=12, fontweight='bold')
        ax2.plot(rapl_df['timestamp_rel'], rapl_df['power_total'],
                color=color_power, linewidth=2, label='Total Power', alpha=0.7)
        ax2.tick_params(axis='y')

        # Frequency (secondary y-axis 2) - Smoothed
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        color_freq = '#9b59b6'  # Purple for frequency
        ax3.set_ylabel('Frequency (MHz)', fontsize=12, fontweight='bold')

        # Apply rolling average to smooth frequency
        freq_smoothed = freq_df['avg_freq_mhz'].rolling(window=5, center=True).mean().fillna(freq_df['avg_freq_mhz'])
        ax3.plot(freq_df['timestamp_rel'], freq_smoothed,
                color=color_freq, linewidth=2, label='Avg CPU Freq (smoothed)', alpha=0.8)
        ax3.tick_params(axis='y')

        ax1.set_title('Thermal Analysis: Temperature, Power, Frequency', fontsize=14, fontweight='bold')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=9)

        # Throttling detection (enhanced)
        throttle_threshold = 80  # °C
        max_temp = thermal_df['thermal_zone0_c'].max()
        throttling_detected = False

        if max_temp > throttle_threshold:
            throttling_detected = True
            ax1.axhline(y=throttle_threshold, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Throttle Threshold')

            # Detect frequency drops correlated with temp spikes
            # Check if frequency drops when temperature is high
            high_temp_mask = thermal_df['thermal_zone0_c'] > throttle_threshold
            if high_temp_mask.sum() > 0 and len(freq_df) > 0:
                # Get timestamps where temp > threshold
                high_temp_times = thermal_df.loc[high_temp_mask, 'timestamp_rel'].values

                # Check if frequency dropped during those times (simplified check)
                # Compare avg freq during high temp vs overall avg
                overall_avg_freq = freq_smoothed.mean()
                freq_during_hot = []

                for t in high_temp_times[:min(10, len(high_temp_times))]:  # Sample first 10 events
                    closest_idx = (np.abs(freq_df['timestamp_rel'].values - t)).argmin()
                    freq_during_hot.append(freq_smoothed.iloc[closest_idx])

                if len(freq_during_hot) > 0:
                    avg_freq_hot = np.mean(freq_during_hot)
                    freq_drop_pct = (overall_avg_freq - avg_freq_hot) / overall_avg_freq * 100

                    if freq_drop_pct > 5:  # Significant drop
                        warning_text = f'⚠️  Thermal throttling detected\n(Freq drop: {freq_drop_pct:.1f}%)'
                    else:
                        warning_text = f'⚠️  High temperature detected\n(No significant freq drop)'
                else:
                    warning_text = '⚠️  High temperature detected'
            else:
                warning_text = '⚠️  High temperature detected'

            ax1.text(0.02, 0.98, warning_text,
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

        plt.tight_layout()
        output_path = self.output_dir / f'thermal_analysis.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_monitoring_overhead(self, overhead_metrics: Dict) -> str:
        """
        Plot 6: Monitoring Overhead (horizontal bar chart).

        Args:
            overhead_metrics: Dict with 'per_tool_breakdown' key containing
                             {'tool_name': {'cpu_percent': float}}

        Returns:
            Path to saved plot
        """
        breakdown = overhead_metrics.get('per_tool_breakdown', {})

        if not breakdown:
            print("⚠️  Warning: No monitoring overhead data available")
            return ""

        # Prepare data
        tools = []
        cpu_percents = []
        for tool, metrics in breakdown.items():
            tools.append(tool)
            cpu_percents.append(metrics.get('cpu_percent', 0))

        # Sort by CPU% descending
        sorted_indices = np.argsort(cpu_percents)[::-1]
        tools = [tools[i] for i in sorted_indices]
        cpu_percents = [cpu_percents[i] for i in sorted_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(tools))
        colors_bar = plt.cm.YlOrRd(np.linspace(0.3, 0.8, len(tools)))

        ax.barh(y_pos, cpu_percents, color=colors_bar, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tools, fontsize=11)
        ax.invert_yaxis()  # Highest on top
        ax.set_xlabel('Average CPU Usage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Monitoring Overhead by Tool', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(cpu_percents):
            ax.text(v + 0.05, i, f'{v:.2f}%', va='center', fontsize=10)

        # Total overhead annotation
        total_overhead = sum(cpu_percents)
        ax.text(0.98, 0.98, f'Total Overhead: {total_overhead:.2f}%',
                transform=ax.transAxes, fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        output_path = self.output_dir / f'monitoring_overhead.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_loss_evolution(self, training_metrics: Dict) -> str:
        """
        Plot 7: Loss Evolution (time-series).

        Args:
            training_metrics: Dict with 'iterations', 'gloss', 'dloss' keys

        Returns:
            Path to saved plot
        """
        iterations = training_metrics.get('iterations', [])
        gloss = training_metrics.get('gloss', [])
        dloss = training_metrics.get('dloss', [])

        if len(iterations) == 0:
            print("⚠️  Warning: No training loss data available")
            return ""

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot losses
        ax.plot(iterations, gloss, label='Generator Loss (Gloss)',
               color=self.colors['gloss'], linewidth=2.5, marker='o', markersize=5)
        ax.plot(iterations, dloss, label='Discriminator Loss (Dloss)',
               color=self.colors['dloss'], linewidth=2.5, linestyle='--', marker='s', markersize=5)

        # Formatting
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('GAN Training: Loss Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Final values annotation
        final_gloss = gloss[-1]
        final_dloss = dloss[-1]
        ax.text(0.98, 0.98, f'Final Gloss: {final_gloss:.4f}\nFinal Dloss: {final_dloss:.4f}',
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        output_path = self.output_dir / f'loss_evolution.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_power_vs_loss(self, rapl_df: pd.DataFrame, training_metrics: Dict) -> str:
        """
        Plot 8: Power vs Loss Evolution (dual-axis time-series).

        Args:
            rapl_df: DataFrame with power or energy data
            training_metrics: Dict with 'timestamps', 'gloss', 'dloss'

        Returns:
            Path to saved plot
        """
        # Ensure timestamp_rel exists
        rapl_df = self._ensure_timestamp_rel(rapl_df)

        # Standardize RAPL columns
        rapl_df = self._standardize_rapl_columns(rapl_df)

        # Calculate power_total if not present
        if 'power_total' not in rapl_df.columns:
            rapl_df['power_total'] = (
                rapl_df['power_package_0'] + rapl_df['power_package_1'] +
                rapl_df['power_dram_0'] + rapl_df['power_dram_1']
            )

        iterations = training_metrics.get('iterations', [])
        timestamps = training_metrics.get('timestamps', [])
        gloss = training_metrics.get('gloss', [])
        dloss = training_metrics.get('dloss', [])

        # If no timestamps, try to infer from iterations and RAPL data
        if len(timestamps) == 0:
            if len(iterations) > 0 and len(rapl_df) > 0:
                # Map iterations to approximate timestamps
                # Assume training spans the RAPL monitoring period
                training_start = rapl_df['timestamp_rel'].min()
                training_duration = rapl_df['timestamp_rel'].max() - training_start
                max_iter = max(iterations)

                if max_iter > 0:
                    # Linear interpolation: timestamp = start + (iter/max_iter) * duration
                    timestamps = [training_start + (it / max_iter) * training_duration for it in iterations]
                    print(f"  ℹ️  Inferred timestamps from iterations (approx)")
                else:
                    timestamps = [training_start] * len(iterations)
            else:
                print("⚠️  Warning: Cannot create timestamps for power vs loss plot")
                return ""

        if len(timestamps) == 0 or len(gloss) == 0:
            print("⚠️  Warning: Insufficient training data for power vs loss plot")
            return ""

        # Calculate Pearson correlation between power and losses
        # Interpolate power values at loss timestamps for alignment
        from scipy.stats import pearsonr

        power_at_loss_times = np.interp(timestamps, rapl_df['timestamp_rel'], rapl_df['power_total'])

        # Correlation: Power vs Gloss
        if len(power_at_loss_times) > 2 and len(gloss) > 2:
            r_gloss, p_gloss = pearsonr(power_at_loss_times, gloss)
        else:
            r_gloss, p_gloss = np.nan, np.nan

        # Correlation: Power vs Dloss
        if len(power_at_loss_times) > 2 and len(dloss) > 2:
            r_dloss, p_dloss = pearsonr(power_at_loss_times, dloss)
        else:
            r_dloss, p_dloss = np.nan, np.nan

        # Even wider figsize for clarity
        fig, ax1 = plt.subplots(figsize=(20, 6))

        # Power (primary y-axis) - thinner line
        color_power = self.colors['power_total']
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Power (Watt)', color=color_power, fontsize=12, fontweight='bold')
        ax1.plot(rapl_df['timestamp_rel'], rapl_df['power_total'],
                color=color_power, linewidth=2, label='Total Power', alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color_power)
        ax1.grid(True, alpha=0.3)

        # Loss (secondary y-axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.plot(timestamps, gloss, label='Gloss',
                color=self.colors['gloss'], linewidth=2, marker='o', markersize=6)
        ax2.plot(timestamps, dloss, label='Dloss',
                color=self.colors['dloss'], linewidth=2, linestyle='--', marker='s', markersize=6)

        ax1.set_title('Energy Dynamics During Training: Power vs Loss', fontsize=14, fontweight='bold')

        # Combined legend - bottom right
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

        # Correlation annotation box
        if not np.isnan(r_gloss):
            corr_text = f'Pearson Correlation:\n'
            corr_text += f'Power ↔ Gloss: r={r_gloss:.3f}, p={p_gloss:.3e}\n'
            corr_text += f'Power ↔ Dloss: r={r_dloss:.3f}, p={p_dloss:.3e}\n\n'

            # Interpretation
            if abs(r_gloss) > 0.7:
                interp_gloss = "Strong"
            elif abs(r_gloss) > 0.4:
                interp_gloss = "Moderate"
            else:
                interp_gloss = "Weak"

            corr_text += f'Power-Gloss: {interp_gloss} correlation'

            ax1.text(0.02, 0.98, corr_text,
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))

        plt.tight_layout()
        output_path = self.output_dir / f'power_vs_loss_evolution.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_workload_characterization(self, vmstat_df: pd.DataFrame,
                                      perf_df: Optional[pd.DataFrame],
                                      performance_metrics: Dict,
                                      rapl_df: Optional[pd.DataFrame] = None) -> str:
        """
        Plot 9: Workload Characterization (2D bubble chart: IPC vs iowait).

        Args:
            vmstat_df: DataFrame with vmstat data
            perf_df: DataFrame with perf data (optional)
            performance_metrics: Dict with 'ipc' and 'throughput_samples_per_s'
            rapl_df: DataFrame with RAPL power data (optional, for bubble size)

        Returns:
            Path to saved plot
        """
        # Ensure timestamp_rel exists
        vmstat_df = self._ensure_timestamp_rel(vmstat_df)

        # Get IPC
        if perf_df is not None and len(perf_df) > 0:
            perf_df = self._ensure_timestamp_rel(perf_df)
            perf_df = perf_df.copy()

            print(f"  [DEBUG] Perf data columns: {perf_df.columns.tolist()}")
            print(f"  [DEBUG] Perf data shape: {perf_df.shape}")

            # Check if perf data is in long format (event column) or wide format (instructions column)
            if 'event' in perf_df.columns:
                # Long format: IPC is already in metric_value for instructions:u event
                # The perf output already calculates IPC as "insn per cycle"
                instr_rows = perf_df[perf_df['event'] == 'instructions:u'].copy()

                if len(instr_rows) > 0 and 'metric_value' in instr_rows.columns:
                    # metric_value for instructions:u already contains IPC!
                    perf_df = instr_rows[['timestamp_rel', 'metric_value']].copy()
                    perf_df = perf_df.rename(columns={'metric_value': 'ipc'})
                    print(f"  [DEBUG] Extracted IPC from {len(perf_df)} instruction samples")
                    print(f"  [DEBUG] IPC range: [{perf_df['ipc'].min():.2f}, {perf_df['ipc'].max():.2f}]")
                else:
                    print(f"  ⚠️  No 'instructions:u' event found in perf data")
                    perf_df = None
            elif 'instructions' in perf_df.columns and 'cycles' in perf_df.columns:
                # Wide format: calculate IPC
                perf_df['ipc'] = perf_df['instructions'] / perf_df['cycles']
                print(f"  [DEBUG] Calculated IPC from wide format")
            else:
                print(f"  ⚠️  Unknown perf data format")
                perf_df = None

            if perf_df is not None:
                perf_df = perf_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ipc'])
                print(f"  [DEBUG] Perf data after IPC extraction: {len(perf_df)} rows")

            # Merge with vmstat
            # IMPORTANT: Start from perf (878 samples) and find vmstat, not the opposite!
            # This ensures we use all IPC samples instead of repeating just a few
            # Use direction='nearest' without tolerance to always find the closest vmstat sample
            merged = pd.merge_asof(
                perf_df[['timestamp_rel', 'ipc']].sort_values('timestamp_rel'),
                vmstat_df[['timestamp_rel', 'wa']].sort_values('timestamp_rel'),
                on='timestamp_rel',
                direction='nearest'  # No tolerance - always match to nearest
            ).dropna()

            print(f"  [DEBUG] Merged {len(merged)} perf samples with vmstat (used {len(merged)}/{len(perf_df)} perf samples)")

            if len(merged) == 0:
                print("⚠️  Warning: No overlapping perf/vmstat data for workload characterization")
                return ""

            print(f"  [DEBUG] Merged data points: {len(merged)} (vmstat: {len(vmstat_df)}, perf: {len(perf_df)})")

            ipc_values = merged['ipc'].values
            iowait_values = merged['wa'].values

            # Check for unique values to understand data distribution
            unique_ipc = len(np.unique(ipc_values))
            unique_iowait = len(np.unique(iowait_values))
            print(f"  [DEBUG] Unique IPC values: {unique_ipc}, Unique I/O Wait values: {unique_iowait}")

            # Show value distribution
            print(f"  [DEBUG] IPC value counts: {dict(zip(*np.unique(ipc_values, return_counts=True)))}")
            print(f"  [DEBUG] I/O Wait value counts: {dict(zip(*np.unique(iowait_values, return_counts=True)))}")

        else:
            # Fallback: use average IPC from metrics
            ipc_avg = performance_metrics.get('ipc', 1.0)
            if ipc_avg is None:
                ipc_avg = 1.0  # Default value when perf data unavailable
            ipc_values = np.array([ipc_avg])
            iowait_values = np.array([vmstat_df['wa'].mean()])

        throughput = performance_metrics.get('throughput_samples_per_s', 0)
        if throughput is None:
            throughput = 0

        # Get power data for bubble sizing (if RAPL available)
        power_values = None
        timestamps = None
        if rapl_df is not None and len(rapl_df) > 0:
            rapl_df = self._ensure_timestamp_rel(rapl_df)
            rapl_df = self._standardize_rapl_columns(rapl_df)

            # Calculate total power if not present
            if 'power_total' not in rapl_df.columns:
                rapl_df['power_total'] = (
                    rapl_df['power_package_0'] + rapl_df['power_package_1'] +
                    rapl_df['power_dram_0'] + rapl_df['power_dram_1']
                )

            # Merge with existing data to get power per point
            if perf_df is not None and len(merged) > 0:
                # Add timestamp_rel to merged for power merge
                merged_with_power = pd.merge_asof(
                    merged.sort_values('timestamp_rel'),
                    rapl_df[['timestamp_rel', 'power_total']].sort_values('timestamp_rel'),
                    on='timestamp_rel',
                    direction='nearest',
                    tolerance=2.0
                ).dropna()

                if len(merged_with_power) > 0:
                    power_values = merged_with_power['power_total'].values
                    timestamps = merged_with_power['timestamp_rel'].values
                    # Update ipc/iowait arrays to match
                    ipc_values = merged_with_power['ipc'].values
                    iowait_values = merged_with_power['wa'].values

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Bubble chart with power-based sizing and temporal color gradient
        if power_values is not None and timestamps is not None:
            # Normalize power for bubble size (50-500 range)
            power_normalized = 50 + 450 * (power_values - power_values.min()) / (power_values.max() - power_values.min() + 1e-6)

            # Color by temporal evolution (blue → red gradient)
            scatter = ax.scatter(ipc_values, iowait_values,
                                s=power_normalized,
                                c=timestamps,
                                cmap='viridis',
                                alpha=0.6,
                                edgecolors='black',
                                linewidth=0.8)

            # Add colorbar for temporal evolution
            cbar = plt.colorbar(scatter, ax=ax, label='Time (s)')
            cbar.ax.tick_params(labelsize=9)
        else:
            # Fallback: simple scatter without power sizing
            ax.scatter(ipc_values, iowait_values,
                      s=150, alpha=0.5, color='#3498db', edgecolors='black', linewidth=1)

        # Add mean working point marker - modern pin style (single marker)
        avg_ipc = np.mean(ipc_values)
        avg_iowait = np.mean(iowait_values)

        # Debug: print actual values being plotted
        print(f"  [DEBUG] Workload characterization: avg_ipc={avg_ipc:.4f}, avg_iowait={avg_iowait:.4f}%")
        print(f"  [DEBUG] IPC range: [{ipc_values.min():.4f}, {ipc_values.max():.4f}]")
        print(f"  [DEBUG] I/O Wait range: [{iowait_values.min():.4f}, {iowait_values.max():.4f}]%")

        ax.scatter(avg_ipc, avg_iowait, s=400, marker='D', color='#e74c3c',
                  edgecolors='black', linewidth=2.5, zorder=10)

        # Set Y-axis to show conceptual regions while keeping data visible
        iowait_min = iowait_values.min()
        iowait_max = iowait_values.max()

        # Use a fixed range that shows conceptual regions (I/O BOUND, COMPUTE BOUND)
        # while keeping data points well visible
        # I/O BOUND region should be visible at top, COMPUTE BOUND at bottom

        # Add small negative margin for points at y=0
        y_min = -0.5

        # Set max to show I/O BOUND region (high I/O wait)
        # Use max between data-driven and minimum conceptual range
        if iowait_max < 5:
            # Low I/O wait data: extend to ~20% to show I/O BOUND region
            y_max = 20
        elif iowait_max < 20:
            # Medium I/O wait: extend a bit more
            y_max = max(25, iowait_max * 1.5)
        else:
            # High I/O wait: use data-driven range
            y_max = min(100, iowait_max * 1.3)

        ax.set_ylim(y_min, y_max)

        # Annotate working point - pointing UPWARD
        ax.annotate('Working Point',
                   xy=(avg_ipc, avg_iowait),
                   xytext=(15, 40), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.7, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        # Region annotations - positioned to indicate conceptual extremes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # I/O BOUND (top-left corner) - HIGH I/O wait, LOW IPC region
        # Position at top-left to indicate where I/O bound workloads would be
        ax.text(xlim[0] + 0.02*(xlim[1]-xlim[0]), ylim[1] - 0.08*(ylim[1]-ylim[0]),
               'I/O BOUND', fontsize=12, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
               verticalalignment='top', horizontalalignment='left')

        # COMPUTE BOUND (bottom-right corner) - LOW I/O wait, HIGH IPC region
        # Position at bottom-right to indicate where compute bound workloads would be
        ax.text(xlim[1] - 0.02*(xlim[1]-xlim[0]), ylim[0] + 0.08*(ylim[1]-ylim[0]),
               'COMPUTE BOUND', fontsize=12, fontweight='bold', color='green',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6),
               verticalalignment='bottom', horizontalalignment='right')

        # Formatting
        ax.set_xlabel('IPC (Instructions Per Cycle)', fontsize=12, fontweight='bold')
        ax.set_ylabel('I/O Wait (%)', fontsize=12, fontweight='bold')
        ax.set_title('Workload Characterization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Workload classification - consider BOTH IPC and I/O wait
        avg_ipc = np.mean(ipc_values)
        avg_iowait = np.mean(iowait_values)

        # Classification logic:
        # - High I/O wait (>5%) = I/O BOUND regardless of IPC
        # - Low I/O wait (<2%) + Good IPC (>0.7) = COMPUTE BOUND (CPU-intensive)
        # - Low I/O wait + Low IPC (<0.5) = MEMORY BOUND (cache misses, memory stalls)
        # - Otherwise = BALANCED
        if avg_iowait > 5:
            workload_type = "I/O BOUND"
            color_box = 'red'
        elif avg_iowait < 2.0 and avg_ipc > 0.7:
            workload_type = "COMPUTE BOUND"
            color_box = 'green'
        elif avg_iowait < 2.0 and avg_ipc < 0.5:
            workload_type = "MEMORY BOUND"
            color_box = 'orange'
        else:
            workload_type = "BALANCED"
            color_box = 'blue'

        # Position metrics box in top-right to avoid covering region labels
        ax.text(0.98, 0.98, f'Avg IPC: {avg_ipc:.2f}\nAvg I/O Wait: {avg_iowait:.2f}%\nType: {workload_type}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.3))

        plt.tight_layout()
        output_path = self.output_dir / f'workload_characterization.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_energy_performance_tradeoff(self, rapl_df: pd.DataFrame,
                                        performance_metrics: Dict,
                                        window_size: int = 10) -> str:
        """
        Plot 10: Energy-Performance Trade-off (Pareto frontier).

        Args:
            rapl_df: DataFrame with power or energy data
            performance_metrics: Dict with throughput and efficiency metrics
            window_size: Seconds for temporal snapshots

        Returns:
            Path to saved plot
        """
        # Ensure timestamp_rel exists
        rapl_df = self._ensure_timestamp_rel(rapl_df)

        # Standardize RAPL columns
        rapl_df = self._standardize_rapl_columns(rapl_df)

        # Calculate power_total if not present
        if 'power_total' not in rapl_df.columns:
            rapl_df['power_total'] = (
                rapl_df['power_package_0'] + rapl_df['power_package_1'] +
                rapl_df['power_dram_0'] + rapl_df['power_dram_1']
            )

        # Calculate windowed metrics
        rapl_df = rapl_df.copy()
        rapl_df['window'] = (rapl_df['timestamp_rel'] // window_size).astype(int)

        throughputs = []
        efficiencies = []

        for window_id, group in rapl_df.groupby('window'):
            avg_power = group['power_total'].mean()
            if avg_power > 0:
                # Approximate throughput from iteration metrics
                # (In real scenario, would need per-window iteration counts)
                throughput = performance_metrics.get('throughput_samples_per_s', 0)
                efficiency = throughput / avg_power if avg_power > 0 else 0

                throughputs.append(throughput)
                efficiencies.append(efficiency)

        if len(throughputs) == 0:
            # Fallback: single point
            throughputs = [performance_metrics.get('throughput_samples_per_s', 0) or 0]
            efficiencies = [performance_metrics.get('samples_per_joule', 0) or 0]

        throughputs = np.array(throughputs)
        efficiencies = np.array(efficiencies)

        # Compute Pareto frontier
        pareto_indices = self._compute_pareto_frontier(throughputs, efficiencies)

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)

        # All points
        ax.scatter(throughputs, efficiencies, alpha=0.6, s=80,
                  color='#3498db', label='Temporal Snapshots', edgecolors='black', linewidth=1)

        # Pareto frontier - more visible
        if len(pareto_indices) > 1:
            pareto_throughput = throughputs[pareto_indices]
            pareto_efficiency = efficiencies[pareto_indices]
            sorted_idx = np.argsort(pareto_throughput)
            ax.plot(pareto_throughput[sorted_idx], pareto_efficiency[sorted_idx],
                   'r-', linewidth=4, marker='o', markersize=12,
                   label='Pareto Frontier', alpha=0.9, zorder=3)

        # Formatting
        ax.set_xlabel('Throughput (samples/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy Efficiency (samples/Joule)', fontsize=12, fontweight='bold')
        ax.set_title('Energy-Performance Trade-off', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Optimal point annotation (max efficiency × throughput product) - modern marker
        if len(throughputs) > 0:
            product = throughputs * efficiencies
            best_idx = np.argmax(product)
            ax.scatter(throughputs[best_idx], efficiencies[best_idx],
                      s=350, color='#e74c3c', marker='D', edgecolors='black',
                      linewidth=2.5, zorder=5, label='Optimal Point')
            # Position annotation below point to avoid title
            ax.annotate('Optimal\nPoint',
                       xy=(throughputs[best_idx], efficiencies[best_idx]),
                       xytext=(15, -40), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.7, edgecolor='black'),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

        plt.tight_layout()
        output_path = self.output_dir / f'energy_performance_tradeoff.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def plot_performance_scalability(self, vmstat_df: pd.DataFrame,
                                    performance_metrics: Dict) -> str:
        """
        Plot 11: Performance Scalability (CPU utilization vs throughput).

        Args:
            vmstat_df: DataFrame with vmstat data
            performance_metrics: Dict with performance metrics

        Returns:
            Path to saved plot
        """
        # Ensure timestamp_rel exists
        vmstat_df = self._ensure_timestamp_rel(vmstat_df)

        vmstat_df = vmstat_df.copy()
        vmstat_df['cpu_util'] = vmstat_df['us'] + vmstat_df['sy']

        # Approximate iteration throughput (constant for this experiment)
        throughput = performance_metrics.get('iteration_throughput_per_s', 0)
        if throughput is None:
            throughput = 0

        cpu_util = vmstat_df['cpu_util'].values
        throughput_arr = np.full(len(cpu_util), throughput)

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)

        # Scatter
        ax.scatter(cpu_util, throughput_arr, alpha=0.6, s=60,
                  color='#3498db', edgecolors='black', linewidth=0.5)

        # Linear regression
        if len(cpu_util) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(cpu_util, throughput_arr)
            cpu_range = np.linspace(cpu_util.min(), cpu_util.max(), 100)
            throughput_pred = slope * cpu_range + intercept
            ax.plot(cpu_range, throughput_pred, 'r--', linewidth=2.5,
                   label=f'Linear Fit (slope={slope:.4f})')

        # Formatting
        ax.set_xlabel('CPU Utilization (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (iterations/s)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Scalability', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Zoom in on y-axis to show small variations
        y_min = throughput_arr.min()
        y_max = throughput_arr.max()
        y_range = y_max - y_min
        if y_range < 0.001:  # Very small variation
            y_center = (y_min + y_max) / 2
            ax.set_ylim(y_center - 0.0005, y_center + 0.0005)
        else:
            # Add 10% padding to focus on variations
            padding = y_range * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)

        # Simple annotation - no explanation
        avg_cpu = vmstat_df['cpu_util'].mean()
        efficiency = throughput / avg_cpu if avg_cpu > 0 else 0

        ax.text(0.02, 0.98, f'Avg CPU: {avg_cpu:.1f}%\nThroughput: {throughput:.4f} iter/s\nEfficiency: {efficiency:.4f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        output_path = self.output_dir / f'performance_scalability.{self.image_ext}'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
        plt.close()

        return str(output_path)

    def _ensure_timestamp_rel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has timestamp_rel column (seconds from start).

        Args:
            df: DataFrame with either 'timestamp_rel' or 'timestamp' column

        Returns:
            DataFrame with timestamp_rel column added if needed
        """
        df = df.copy()
        if 'timestamp_rel' not in df.columns:
            if 'timestamp' in df.columns:
                # BACKWARD COMPATIBILITY: Check if timestamp is already relative (old perf data)
                # New data from fixed monitoring scripts will have Unix timestamps
                if pd.api.types.is_numeric_dtype(df['timestamp']):
                    # Numeric dtype - check if relative or Unix timestamp
                    if df['timestamp'].max() < 1000000:
                        # Legacy relative timestamps (e.g., old perf data: 1.00, 2.00, ...)
                        df['timestamp_rel'] = df['timestamp']
                        return df

                # Convert to datetime if not already (Unix timestamps)
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timestamp_rel'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            else:
                # Fallback: use index as seconds
                df['timestamp_rel'] = np.arange(len(df))
        return df

    def _compute_pareto_frontier(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute Pareto frontier (non-dominated points).

        Args:
            X: First objective (maximize)
            Y: Second objective (maximize)

        Returns:
            Array of indices of Pareto-optimal points
        """
        is_pareto = np.ones(len(X), dtype=bool)
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j:
                    if X[j] >= X[i] and Y[j] >= Y[i]:
                        if X[j] > X[i] or Y[j] > Y[i]:
                            is_pareto[i] = False
                            break
        return np.where(is_pareto)[0]

    def generate_all_plots(self, dfs: Dict[str, pd.DataFrame],
                          metrics: Dict) -> Dict[str, str]:
        """
        Generate all 11 plots.

        Args:
            dfs: Dict of DataFrames from MonitoringDataLoader
            metrics: Dict of computed metrics from MetricsCalculator

        Returns:
            Dict mapping plot names to file paths
        """
        plot_paths = {}

        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)

        # Plot 1: Power Profile
        if 'rapl' in dfs and len(dfs['rapl']) > 0:
            print("\n[1/11] Power Profile...")
            rapl_df = dfs['rapl'].copy()

            path = self.plot_power_profile(rapl_df, detailed=False)
            if path:
                plot_paths['power_profile'] = path
                print(f"  ✅ Saved: {path}")

            # Detailed version
            path_detailed = self.plot_power_profile(rapl_df, detailed=True)
            if path_detailed:
                plot_paths['power_profile_detailed'] = path_detailed
                print(f"  ✅ Saved: {path_detailed}")
        else:
            print("\n[1/11] Power Profile... ⚠️  Skipped (no RAPL data)")

        # Plot 2: Energy Breakdown
        if 'energy' in metrics and metrics['energy']:
            print("\n[2/11] Energy Breakdown...")
            path = self.plot_energy_breakdown(metrics['energy'])
            if path:
                plot_paths['energy_breakdown'] = path
                print(f"  ✅ Saved: {path}")
        else:
            print("\n[2/11] Energy Breakdown... ⚠️  Skipped (no RAPL data)")

        # Plot 3: Frequency vs Power
        freq_key = 'cpu_freq' if 'cpu_freq' in dfs else 'freq'
        if freq_key in dfs and 'rapl' in dfs and len(dfs['rapl']) > 0:
            print("\n[3/11] Frequency vs Power...")
            path = self.plot_frequency_vs_power(dfs[freq_key], dfs['rapl'])
            if path:
                plot_paths['frequency_vs_power'] = path
                print(f"  ✅ Saved: {path}")
        else:
            print("\n[3/11] Frequency vs Power... ⚠️  Skipped (missing freq or RAPL data)")

        # Plot 4: Memory Usage
        if 'free_mem' in dfs:
            print("\n[4/11] Memory Usage...")
            path = self.plot_memory_usage(dfs['free_mem'])
            if path:
                plot_paths['memory_usage'] = path
                print(f"  ✅ Saved: {path}")

        # Plot 5: Thermal Analysis
        freq_key = 'cpu_freq' if 'cpu_freq' in dfs else 'freq'
        if 'thermal' in dfs and 'rapl' in dfs and freq_key in dfs and len(dfs['rapl']) > 0:
            print("\n[5/11] Thermal Analysis...")
            path = self.plot_thermal_analysis(dfs['thermal'], dfs['rapl'], dfs[freq_key])
            if path:
                plot_paths['thermal_analysis'] = path
                print(f"  ✅ Saved: {path}")
        else:
            missing = []
            if 'thermal' not in dfs:
                missing.append('thermal')
            if 'rapl' not in dfs or len(dfs.get('rapl', [])) == 0:
                missing.append('rapl')
            if freq_key not in dfs:
                missing.append('freq')
            print(f"\n[5/11] Thermal Analysis... ⚠️  Skipped (missing: {', '.join(missing)})")

        # Plot 6: Monitoring Overhead
        if 'monitoring_overhead' in metrics:
            print("\n[6/11] Monitoring Overhead...")
            path = self.plot_monitoring_overhead(metrics['monitoring_overhead'])
            if path:
                plot_paths['monitoring_overhead'] = path
                print(f"  ✅ Saved: {path}")

        # Plot 7: Loss Evolution
        if 'training_metrics' in metrics:
            print("\n[7/11] Loss Evolution...")
            path = self.plot_loss_evolution(metrics['training_metrics'])
            if path:
                plot_paths['loss_evolution'] = path
                print(f"  ✅ Saved: {path}")

        # Plot 8: Power vs Loss
        if 'rapl' in dfs and len(dfs['rapl']) > 0 and 'training_metrics' in metrics:
            print("\n[8/11] Power vs Loss Evolution...")
            path = self.plot_power_vs_loss(dfs['rapl'], metrics['training_metrics'])
            if path:
                plot_paths['power_vs_loss'] = path
                print(f"  ✅ Saved: {path}")
        else:
            print("\n[8/11] Power vs Loss Evolution... ⚠️  Skipped (missing data)")

        # Plot 9: Workload Characterization
        if 'vmstat' in dfs:
            print("\n[9/11] Workload Characterization...")
            perf_df = dfs.get('perf', None)
            rapl_df = dfs.get('rapl', None)
            path = self.plot_workload_characterization(
                dfs['vmstat'], perf_df, metrics.get('performance', {}), rapl_df
            )
            if path:
                plot_paths['workload_characterization'] = path
                print(f"  ✅ Saved: {path}")

        # Plot 10: Energy-Performance Trade-off
        if 'rapl' in dfs and len(dfs['rapl']) > 0:
            print("\n[10/11] Energy-Performance Trade-off...")
            path = self.plot_energy_performance_tradeoff(
                dfs['rapl'], metrics.get('performance', {})
            )
            if path:
                plot_paths['energy_performance_tradeoff'] = path
                print(f"  ✅ Saved: {path}")
        else:
            print("\n[10/11] Energy-Performance Trade-off... ⚠️  Skipped (no RAPL data)")

        # Plot 11: Performance Scalability
        if 'vmstat' in dfs:
            print("\n[11/11] Performance Scalability...")
            path = self.plot_performance_scalability(
                dfs['vmstat'], metrics.get('performance', {})
            )
            if path:
                plot_paths['performance_scalability'] = path
                print(f"  ✅ Saved: {path}")

        print("\n" + "="*70)
        print(f"PLOTS GENERATED: {len(plot_paths)}/11")
        print("="*70)

        return plot_paths


if __name__ == "__main__":
    # Example usage
    print("MonitoringVisualizer module loaded successfully")
    print("Use: visualizer = MonitoringVisualizer(output_dir='analysis/plots')")
    print("     plot_paths = visualizer.generate_all_plots(dfs, metrics)")
