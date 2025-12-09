"""
Visualization utilities for Physics-Informed Neural Networks (PINNs)

This module provides plotting and visualization functions for PINN models.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class VisualizePINN:
    """
    Visualization class for models trained with the PulsarPINN class.
    
    Provides plotting methods for loss curves, predictions, and model diagnostics.
    """
    
    def __init__(self, pinn_model):
        """
        Initialize visualizer with a PulsarPINN model.
        
        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - pinn_model : PulsarPINN
            The input PINN model from the PulsarPINN class.
    
        """
        self.pinn = pinn_model
    
    def plot_predictions_vs_data(
        self,
        x_values: Optional[np.ndarray] = None,
        y_predictions: Optional[np.ndarray] = None,
        x_axis: str = None,
        y_axis: str = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
    ) -> None:
        """
        Create comparison plot of model predictions vs. raw data.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - x_values : Optional[np.ndarray]
            X values for prediction curve.
            If None, generates extended range.

        - y_predictions : Optional[np.ndarray]
            Predicted y values. If None, generates from x_values or extended range.

        - save_path : Optional[str]
            If provided, saves figure to this path.
            Default: None (display only).

        - figsize : Tuple[int, int]
            Figure size in inches (width, height).
            Default: (12, 8).

        - title : Optional[str]
            Plot title. If None, generates default title.
            Default: None.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        None
            Displays or saves matplotlib figure.
        """

        # Generate predictions if not provided (uses extended range for smooth curve)
        if x_values is None or y_predictions is None:
            x_values, y_predictions = self.pinn.predict_extended(extend=0.5, n_points=300)

        # Create figure and axes (FIXED: properly unpack figure and axes)
        fig, ax = plt.subplots(figsize=figsize)

        # Plot training data points (blue circles)
        ax.scatter(
            self.pinn.x_train,
            self.pinn.y_train,
            c="blue",
            alpha=0.4,
            s=30,
            label="Train Data",
            marker="o",
        )

        # Plot validation data points (orange circles)
        ax.scatter(
            self.pinn.x_val,
            self.pinn.y_val,
            c="orange",
            alpha=0.4,
            s=30,
            label="Validation Data",
            marker="o",
        )

        # Plot test data points (red circles)
        ax.scatter(
            self.pinn.x_test,
            self.pinn.y_test,
            c="red",
            alpha=0.4,
            s=30,
            label="Test Data",
            marker="o",
        )

        # Plot model prediction curve (green line, drawn on top with zorder=5)
        ax.plot(
            x_values,
            y_predictions,
            "g-",
            linewidth=2.5,
            label="PINN Prediction",
            zorder=5,
        )

        # Set axis labels using the symbolic variable names
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

        # Set title (use default with solution name if available)
        if title is None:
            title = f"PINN Predictions vs. Data"
            if self.pinn.solution_name:
                title += f" ({self.pinn.solution_name})"
        ax.set_title(title)

        # Add legend and grid
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add R² score annotation in top-left corner if available
        if self.pinn.test_metrics and "test_r2" in self.pinn.test_metrics:
            r2_text = f"Test R² = {self.pinn.test_metrics['test_r2']:.4f}"
            ax.text(
                0.02,
                0.98,
                r2_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save to file if path provided, otherwise display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_loss_curves(self, log_scale: bool = True) -> None:
        """
        Plot training and validation loss curves.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - log_scale : bool
            Use logarithmic y-axis. Default: True.
        """
        # Create figure with two subplots side-by-side (FIXED: properly unpack figure and axes)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Create array of epoch numbers for x-axis (0, 1, 2, ..., num_epochs-1)
        epochs = np.arange(len(self.pinn.loss_log["total"]))

        # LEFT PLOT: Total Loss (Training vs Validation)
        # Plot training total loss as solid line
        ax1.plot(epochs, self.pinn.loss_log["total"], label="Train", linewidth=2)

        # Plot validation total loss if it exists (checked every val_interval epochs)
        if self.pinn.loss_log["val_total"]:
            # Calculate epoch numbers where validation was performed
            val_epochs = np.linspace(
                0, len(epochs) - 1, len(self.pinn.loss_log["val_total"])
            )
            ax1.plot(
                val_epochs,
                self.pinn.loss_log["val_total"],
                label="Validation",
                marker="o",
                markersize=4,
            )

        # Configure left plot appearance
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Total Loss")
        ax1.set_title("Total Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if log_scale:
            ax1.set_yscale("log")

        # RIGHT PLOT: Loss Components (Physics vs Data)
        # Plot training physics and data losses as dashed lines
        ax2.plot(epochs, self.pinn.loss_log["physics"], label="Physics", linestyle="--")
        ax2.plot(epochs, self.pinn.loss_log["data"], label="Data", linestyle="--")

        # Plot validation component losses if they exist
        if self.pinn.loss_log["val_physics"]:
            ax2.plot(
                val_epochs,
                self.pinn.loss_log["val_physics"],
                label="Val Physics",
                linestyle=":",
                marker="s",
                markersize=3,
            )
            ax2.plot(
                val_epochs,
                self.pinn.loss_log["val_data"],
                label="Val Data",
                linestyle=":",
                marker="s",
                markersize=3,
            )

        # Configure right plot appearance
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss Components")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if log_scale:
            ax2.set_yscale("log")

        # Adjust spacing between subplots and display
        plt.tight_layout()
        plt.show()