"""
CSV Export utilities for Physics-Informed Neural Networks (PINNs)

This module provides CSV export functionality for PINN models.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ExportPINN:
    """
    CSV export class for PulsarPINN models.
    
    Provides methods to export predictions, metadata, and raw data to CSV files.
    """
    
    def __init__(self, pinn_model):
        """
        Initialize exporter with a PulsarPINN model.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - pinn_model : PulsarPINN
            The PINN model to export data from.
        """
        self.pinn = pinn_model
    
    def save_predictions_to_csv(
        self,
        filepath: str,
        x_value_name: str,
        y_value_name: str,
        x_values: Optional[np.ndarray] = None,
        y_predictions: Optional[np.ndarray] = None,
        include_raw_data: bool = True,
        include_test_metrics: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model predictions and metadata to CSV file.

        This method exports the model's predictions along with optional raw data,
        test metrics, and learned constants for comparison with ATNF or other
        reference data.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - filepath : str
            Output CSV file path. Parent directories will be created if needed.

        - x_value_name : str
            Column name for x values in CSV.

        - y_value_name : str
            Column name for y values in CSV.

        - x_values : Optional[np.ndarray]
            X values for predictions. If None, uses extended prediction range.
            Shape: (n_points,)

        - y_predictions : Optional[np.ndarray]
            Predicted y values. If None and x_values provided, generates predictions.
            If both None, uses predict_extended(). Shape: (n_points,)

        - include_raw_data : bool
            If True, includes original data points in output.
            Default: True.

        - include_test_metrics : bool
            If True, adds test metrics as part of data.
            Default: True.

        - additional_metadata : Optional[Dict[str, Any]]
            Additional metadata to include in header comments.
            Default: None.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        None
            Writes CSV file to specified filepath.

        -------------------------------------------------------------------
                                      NOTES
        -------------------------------------------------------------------
        CSV Structure:
        - Header comments (lines starting with #) contain metadata
        - First data section: Model predictions
        - Second section: Original train/val/test data if include_raw_data=True
        """

        # Create output directory if needed
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        # Generate predictions if not provided
        if x_values is None or y_predictions is None:
            logger.info("Generating extended predictions for CSV export")
            x_values, y_predictions = self.pinn.predict_extended(extend=0.5, n_points=300)
        elif x_values is not None and y_predictions is None:
            # If user provided x but not y, generate predictions
            self.pinn.model.eval()
            with self.pinn.torch.no_grad():
                x_torch = self.pinn._numpy_array_to_tensor(x_values)
                y_torch = self.pinn.model(x_torch)
                y_predictions = y_torch.numpy().flatten()

        # Prepare metadata for exporting
        metadata_lines = self._generate_metadata_lines(
            include_test_metrics, additional_metadata
        )

        # Create predictions dataframe
        predictions_df = pd.DataFrame(
            {
                x_value_name: x_values,
                f"{y_value_name}_predicted": y_predictions,
                "data_type": "model_prediction",
            }
        )

        # Add raw data if requested
        if include_raw_data:
            raw_data_df = self._build_raw_data_dataframe(x_value_name, y_value_name)
            combined_df = pd.concat([predictions_df, raw_data_df], ignore_index=True)
        else:
            combined_df = predictions_df

        # Write to CSV
        with open(filepath, "w", encoding="utf-8") as f:
            # Write metadata as comments
            for line in metadata_lines:
                f.write(f"# {line}\n")
            f.write("#\n")

            # Write data
            combined_df.to_csv(f, index=False)

        logger.info(f"Predictions saved to {filepath}")
        logger.info(f"  - Model predictions: {len(predictions_df)} points")
        if include_raw_data:
            logger.info(f"  - Raw data points: {len(raw_data_df)} points")

    def _generate_metadata_lines(
        self, include_test_metrics: bool, additional_metadata: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate metadata header lines for CSV output."""
        lines = []

        lines.append("=" * 70)
        lines.append("PHYSICS-INFORMED NEURAL NETWORK PREDICTIONS")
        lines.append("=" * 70)

        lines.append(f"Model: {self.pinn.__class__.__name__}")
        lines.append(f"Input Variable: {self.pinn.x_sym}")
        lines.append(f"Output Variable: {self.pinn.y_sym}")
        lines.append(f"Differential Equation: {self.pinn.differential_eq}")

        # Architecture
        lines.append(
            f"Network Architecture: {self.pinn.input_layer} → {self.pinn.hidden_layers} → {self.pinn.output_layer}"
        )
        lines.append(
            f"Total Parameters: {sum(p.numel() for p in self.pinn.model.parameters())}"
        )

        # Data splits
        lines.append("")
        lines.append("Data Splits:")
        lines.append(
            f"  Train: {len(self.pinn.x_train)} samples ({self.pinn.train_split:.1%})"
        )
        lines.append(
            f"  Validation: {len(self.pinn.x_val)} samples ({self.pinn.val_split:.1%})"
        )
        lines.append(
            f"  Test: {len(self.pinn.x_test)} samples ({self.pinn.test_split:.1%})"
        )

        # Learned constants
        lines.append("")
        lines.append("Learned Physical Constants:")
        for name, param in self.pinn.learnable_params.items():
            lines.append(f"  {name} = {param.item():.8f}")

        # Test metrics
        if include_test_metrics and self.pinn.test_metrics:
            lines.append("")
            lines.append("Test Set Performance:")
            lines.append(
                f"  R² Score: {self.pinn.test_metrics.get('test_r2', float('nan')):.6f}"
            )
            lines.append(
                f"  RMSE: {self.pinn.test_metrics.get('test_rmse', float('nan')):.6e}"
            )
            lines.append(
                f"  MAE: {self.pinn.test_metrics.get('test_mae', float('nan')):.6e}"
            )
            lines.append(
                f"  Reduced χ²: {self.pinn.test_metrics.get('test_chi2_reduced', float('nan')):.6f}"
            )

        # Additional metadata
        if additional_metadata:
            lines.append("")
            lines.append("Additional Metadata:")
            for key, value in additional_metadata.items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 70)

        return lines

    def _build_raw_data_dataframe(self, x_name: str, y_name: str) -> pd.DataFrame:
        """Build DataFrame containing original raw data with split labels."""

        # List to hold all data rows (each row is a dictionary)
        data_rows = []

        # Add all training data points with "raw_train" label
        for i in range(len(self.pinn.x_train)):
            data_rows.append(
                {
                    x_name: self.pinn.x_train[i],
                    f"{y_name}_predicted": self.pinn.y_train[i],
                    "data_type": "raw_train",
                }
            )

        # Add all validation data points with "raw_validation" label
        for i in range(len(self.pinn.x_val)):
            data_rows.append(
                {
                    x_name: self.pinn.x_val[i],
                    f"{y_name}_predicted": self.pinn.y_val[i],
                    "data_type": "raw_validation",
                }
            )

        # Add all test data points with "raw_test" label
        for i in range(len(self.pinn.x_test)):
            data_rows.append(
                {
                    x_name: self.pinn.x_test[i],
                    f"{y_name}_predicted": self.pinn.y_test[i],
                    "data_type": "raw_test",
                }
            )

        # Convert list of dictionaries to pandas DataFrame
        raw_data_df = pd.DataFrame(data_rows)

        return raw_data_df

    def save_learned_constants_to_csv(
        self,
        filepath: str,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save only learned constants to a separate CSV file.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - filepath : str
            Output CSV file path.

        - additional_info : Optional[Dict[str, Any]]
            Additional information to include alongside constants.
            Default: None.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        None
            Writes CSV file with learned constants.
        """

        # Create output directory if needed
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        # Build data for CSV
        constants_data = []
        for name, param in self.pinn.learnable_params.items():
            row = {
                "constant_name": name,
                "value": param.item(),
                "equation": str(self.pinn.differential_eq),
            }
            if additional_info:
                row.update(additional_info)
            constants_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(constants_data)

        # Write to CSV
        df.to_csv(filepath, index=False)

        logger.info(f"Learned constants saved to {filepath}")
        logger.info(f"  - Number of constants: {len(constants_data)}")

    def save_metrics_to_csv(
        self,
        filepath: str,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save evaluation metrics to a separate CSV file.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - filepath : str
            Output CSV file path.

        - additional_info : Optional[Dict[str, Any]]
            Additional information to include with metrics.
            Default: None.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        None
            Writes CSV file with evaluation metrics.
        """

        # Create output directory if needed
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        # Check if metrics have been computed
        if not self.pinn.test_metrics:
            logger.warning("No test metrics available. Run evaluate_test_set() first.")
            return

        # Build data for CSV
        metrics_data = []
        for metric_name, metric_value in self.pinn.test_metrics.items():
            row = {
                "metric_name": metric_name,
                "value": metric_value,
            }
            if additional_info:
                row.update(additional_info)
            metrics_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(metrics_data)

        # Write to CSV
        df.to_csv(filepath, index=False)

        logger.info(f"Evaluation metrics saved to {filepath}")
        logger.info(f"  - Number of metrics: {len(metrics_data)}")

    def save_loss_history_to_csv(
        self,
        filepath: str,
    ) -> None:
        """
        Save training loss history to CSV file.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - filepath : str
            Output CSV file path.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        None
            Writes CSV file with loss history.
        """

        # Create output directory if needed
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        # Build DataFrame from loss log
        epochs = list(range(len(self.pinn.loss_log["total"])))

        loss_data = {
            "epoch": epochs,
            "total_loss": self.pinn.loss_log["total"],
            "physics_loss": self.pinn.loss_log["physics"],
            "data_loss": self.pinn.loss_log["data"],
        }

        # Add validation losses if available
        if self.pinn.loss_log["val_total"]:
            # Validation was only computed at certain epochs
            val_epochs = np.linspace(
                0, len(epochs) - 1, len(self.pinn.loss_log["val_total"])
            ).astype(int)

            # Create full arrays with NaN where validation wasn't computed
            val_total = np.full(len(epochs), np.nan)
            val_physics = np.full(len(epochs), np.nan)
            val_data = np.full(len(epochs), np.nan)

            # Fill in validation values at the epochs where they were computed
            for i, epoch_idx in enumerate(val_epochs):
                val_total[epoch_idx] = self.pinn.loss_log["val_total"][i]
                val_physics[epoch_idx] = self.pinn.loss_log["val_physics"][i]
                val_data[epoch_idx] = self.pinn.loss_log["val_data"][i]

            loss_data["val_total_loss"] = val_total
            loss_data["val_physics_loss"] = val_physics
            loss_data["val_data_loss"] = val_data

        # Create DataFrame
        df = pd.DataFrame(loss_data)

        # Write to CSV
        df.to_csv(filepath, index=False)

        logger.info(f"Loss history saved to {filepath}")
        logger.info(f"  - Number of epochs: {len(epochs)}")