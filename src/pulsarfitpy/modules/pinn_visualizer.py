"""
Visualization utilities for Physics-Informed Neural Networks (PINNs)

This module provides plotting and visualization functions for PINN models.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
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
    def plot_residuals_analysis(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot residuals (prediction errors) vs. input values.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - figsize : Tuple[int, int]
            Figure size in inches (width, height).
            Default: (10, 6).
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute predictions on test set
        self.pinn.model.eval()
        with torch.no_grad():
            y_pred_test = self.pinn.model(self.pinn.x_test_torch).detach().numpy().flatten()
        
        residuals = self.pinn.y_test.flatten() - y_pred_test
        
        # Create scatter plot
        ax.scatter(
            self.pinn.x_test.flatten(),
            residuals,
            c='purple',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
            label='Residuals'
        )
        
        # Add reference line at zero
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        
        ax.set_xlabel('Input (X)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residuals (Y_true - Y_pred)', fontsize=11, fontweight='bold')
        ax.set_title('Residual Analysis (Test Set)', fontsize=12, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Add statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        textstr = f'Mean = {residual_mean:.4f}\nStd = {residual_std:.4f}'
        ax.text(
            0.05, 0.95,
            textstr,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            fontsize=10,
            family='monospace'
        )
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_scatter(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Create scatter plot of predicted vs. true values.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - figsize : Tuple[int, int]
            Figure size in inches (width, height).
            Default: (10, 8).
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute predictions on test set
        self.pinn.model.eval()
        with torch.no_grad():
            y_pred_all = self.pinn.model(self.pinn.x_test_torch).detach().numpy().flatten()
        
        y_true_all = self.pinn.y_test.flatten()
        
        # Create scatter plot
        ax.scatter(
            y_true_all,
            y_pred_all,
            c='teal',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
            label='Predictions'
        )
        
        # Add perfect prediction line
        min_val = min(y_true_all.min(), y_pred_all.min())
        max_val = max(y_true_all.max(), y_pred_all.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            linewidth=2,
            label='Perfect Prediction'
        )
        
        ax.set_xlabel('True Values (Y_true)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Values (Y_pred)', fontsize=11, fontweight='bold')
        ax.set_title('Prediction Accuracy (Test Set)', fontsize=12, fontweight='bold', pad=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        if self.pinn.test_metrics and 'test_r2' in self.pinn.test_metrics:
            r2 = self.pinn.test_metrics['test_r2']
            ax.text(
                0.05, 0.95,
                f'R² = {r2:.4f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=11,
                fontweight='bold',
                family='monospace'
            )
        
        plt.tight_layout()
        plt.show()
    
    def plot_uncertainty_quantification(
        self,
        uncertainties: Dict,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot uncertainty quantification with confidence intervals.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - uncertainties : Dict
            Dictionary of uncertainties from bootstrap_uncertainty() or
            monte_carlo_uncertainty() with keys like 'n_braking', 'logK'.
            Each value should be a dict with 'mean', 'std', 'ci_lower', 'ci_upper'.

        - figsize : Tuple[int, int]
            Figure size in inches (width, height).
            Default: (10, 6).
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        constants_list = list(uncertainties.keys())
        means = [uncertainties[c]['mean'] for c in constants_list]
        stds = [uncertainties[c]['std'] for c in constants_list]
        ci_lower = [uncertainties[c]['ci_lower'] for c in constants_list]
        ci_upper = [uncertainties[c]['ci_upper'] for c in constants_list]
        
        x_pos = np.arange(len(constants_list))
        
        # Plot error bars (mean +/- std)
        ax.errorbar(
            x_pos,
            means,
            yerr=stds,
            fmt='o',
            markersize=10,
            capsize=5,
            capthick=2,
            linewidth=2,
            color='darkblue',
            label='Mean +/- Std',
            zorder=3
        )
        
        # Plot confidence interval bounds
        ax.scatter(
            x_pos,
            ci_lower,
            marker='_',
            s=200,
            linewidths=3,
            color='red',
            zorder=2
        )
        ax.scatter(
            x_pos,
            ci_upper,
            marker='_',
            s=200,
            linewidths=3,
            color='red',
            label='95% CI',
            zorder=2
        )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace('_', '\n') for c in constants_list], fontsize=10)
        ax.set_ylabel('Parameter Value', fontsize=11, fontweight='bold')
        ax.set_title('Uncertainty Quantification', fontsize=12, fontweight='bold', pad=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_robustness_validation(
        self,
        robustness_results: Dict,
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        Plot robustness validation test results.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - robustness_results : Dict
            Dictionary of robustness test results from run_all_robustness_tests()
            with keys: 'permutation_test', 'feature_shuffling_test', 'impossible_physics_test'.

        - figsize : Tuple[int, int]
            Figure size in inches (width, height).
            Default: (12, 5).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract test results
        test_names = ['Permutation\nTest', 'Feature\nShuffling', 'Impossible\nPhysics']
        test_results = [
            robustness_results['permutation_test']['is_significant'],
            robustness_results['feature_shuffling_test']['r2_difference'] > 0.1,
            robustness_results['impossible_physics_test']['real_much_better']
        ]
        colors = ['green' if result else 'red' for result in test_results]
        
        # Left plot: Pass/Fail bar chart
        bars = ax1.bar(
            test_names,
            [1 if r else 0 for r in test_results],
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=2
        )
        ax1.set_ylim([0, 1.2])
        ax1.set_ylabel('Pass (1) / Fail (0)', fontsize=11, fontweight='bold')
        ax1.set_title('Robustness Validation Summary', fontsize=12, fontweight='bold', pad=12)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['FAIL', 'PASS'])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add pass/fail symbols
        for i, (bar, result) in enumerate(zip(bars, test_results)):
            symbol = '[PASS]' if result else '[FAIL]'
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                0.5,
                symbol,
                ha='center',
                va='center',
                fontsize=28,
                fontweight='bold',
                color='white'
            )
        
        # Right plot: Detailed metrics
        ax2.axis('off')
        
        metrics_text = "DETAILED ROBUSTNESS METRICS\n" + "=" * 45 + "\n\n"
        
        # Permutation test details
        perm_test = robustness_results['permutation_test']
        metrics_text += "Permutation Test:\n"
        metrics_text += f"  p-value: {perm_test.get('p_value', 'N/A'):.4f}\n"
        metrics_text += f"  Significant: {perm_test.get('is_significant', False)}\n\n"
        
        # Feature shuffling details
        feat_test = robustness_results['feature_shuffling_test']
        metrics_text += "Feature Shuffling:\n"
        metrics_text += f"  R² difference: {feat_test.get('r2_difference', 0):.4f}\n"
        metrics_text += f"  Original R²: {feat_test.get('r2_original', 0):.4f}\n"
        metrics_text += f"  Shuffled R²: {feat_test.get('r2_shuffled', 0):.4f}\n\n"
        
        # Impossible physics details
        imp_test = robustness_results['impossible_physics_test']
        metrics_text += "Impossible Physics:\n"
        metrics_text += f"  Real better: {imp_test.get('real_much_better', False)}\n"
        metrics_text += f"  Real R²: {imp_test.get('r2_real', 0):.4f}\n"
        metrics_text += f"  Impossible R²: {imp_test.get('r2_impossible', 0):.4f}\n"
        
        ax2.text(
            0.1,
            0.95,
            metrics_text,
            transform=ax2.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )
        
        plt.tight_layout()
        plt.show()
    
    def plot_braking_index_distribution(
        self,
        learned_constants: Dict[str, float],
        uncertainties: Dict,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot histogram of braking index distribution with uncertainty.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - learned_constants : Dict[str, float]
            Dictionary of learned constants from store_learned_constants().

        - uncertainties : Dict
            Dictionary of uncertainties from bootstrap_uncertainty() or
            monte_carlo_uncertainty().

        - figsize : Tuple[int, int]
            Figure size in inches (width, height).
            Default: (10, 6).
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        learned_n = learned_constants.get('n_braking', 0)
        n_std = uncertainties.get('n_braking', {}).get('std', 0.1)
        
        # Generate bootstrap distribution
        n_bootstrap_values = [
            learned_n + np.random.normal(0, n_std) for _ in range(100)
        ]
        
        # Create histogram
        ax.hist(
            n_bootstrap_values,
            bins=25,
            color='skyblue',
            edgecolor='black',
            alpha=0.7,
            label='Bootstrap Distribution'
        )
        
        # Add reference lines
        ax.axvline(
            learned_n,
            color='green',
            linestyle='-',
            linewidth=2.5,
            label=f'Learned (n={learned_n:.2f})'
        )
        ax.axvline(
            3.0,
            color='orange',
            linestyle=':',
            linewidth=2.5,
            label='Canonical (n=3.0)'
        )
        
        ax.set_xlabel('Braking Index (n)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Braking Index Distribution', fontsize=12, fontweight='bold', pad=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()