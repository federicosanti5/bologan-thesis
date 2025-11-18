"""
BoloGAN Energy & Performance Analysis Suite

A comprehensive Python suite for analyzing monitoring data from BoloGAN training experiments.

Modules:
    - utils: Helper functions (RAPL overflow, parsing, resampling)
    - monitoring_data_loader: CSV loading and preprocessing
    - metrics_calculator: Metric computation (energy, performance, efficiency)
    - visualizer: Plot generation (power profiles, correlations, etc.)
    - report_generator: Report and LaTeX table generation

Author: Analysis Suite
Version: 1.0
Date: 2025-11-18
"""

__version__ = "1.0.0"
__author__ = "BoloGAN Analysis Suite"

from .utils import (
    handle_rapl_overflow,
    parse_stdout_losses,
    resample_to_grid,
    safe_divide,
    compute_power_from_energy,
)

from .monitoring_data_loader import MonitoringDataLoader
from .metrics_calculator import MetricsCalculator

__all__ = [
    "handle_rapl_overflow",
    "parse_stdout_losses",
    "resample_to_grid",
    "safe_divide",
    "compute_power_from_energy",
    "MonitoringDataLoader",
    "MetricsCalculator",
]
