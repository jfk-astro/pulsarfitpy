"""
pulsarfitpy - Polynomial approximation and Physics-Informed Neural Networks for pulsar data.

This package provides tools for analyzing pulsar data from the ATNF catalogue using:
- Polynomial approximation (PulsarApproximation)
- Physics-Informed Neural Networks (PulsarPINN)
- Logging utilities (configure_logging)
- Quantitative metrics for objective model assessment (RMSE, MAE, reduced χ²)
"""

from .utils import configure_logging
from .approximation import PulsarApproximation
from .pinn import PulsarPINN
from .pinn_visualizer import VisualizePINN
from .export_solutions import ExportPINN

__all__ = [
    "configure_logging",
    "PulsarApproximation",
    "PulsarPINN",
    "print_metrics_guide",
]
