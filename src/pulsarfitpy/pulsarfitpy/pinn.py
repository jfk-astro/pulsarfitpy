"""
Physics-Informed Neural Networks (PINNs) for Pulsar Analysis

This module implements a PINN for learning physical constants from pulsar data
while enforcing physics constraints through differential equations.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

# TODO: Fix PINN training progress logger

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class PulsarPINN:
    """
    Physics-Informed Neural Network (PINN) for pulsar model analysis.

    Trains a PINN through a differential equation model w/ physics constraints to infer
    physical constants from pulsar observations.

    --------------------------------------------------------------------------------
                                       PARAMETERS
    --------------------------------------------------------------------------------

    - differential_eq : sympy.Eq
        Symbolic differential equation representing the analyzed physical model.
        Example: sympy.Eq(logB, logR + 0.5*logP + 0.5*logPDOT)

    - x_sym : sympy.Symbol
        Independent variable symbol for the neural network input.
        This is the variable the NN predicts FROM.
        Example: logP (period)

    - y_sym : sympy.Symbol
        Dependent variable symbol for the neural network output.
        This is the variable the NN predicts TO.
        Example: logB (magnetic field)

    - learn_constants : Dict[sympy.Symbol, float]
        Dictionary of unknown constants to learn with initial guesses.
        Example: {logR: 18.0}

    - fixed_inputs : Dict[sympy.Symbol: np.ndarray]
        Dictionary of fixed data arrays for all variables in the equation.
        Must include data for BOTH x_sym and y_sym, plus any other variables.
        Example: {logP: logP_data, logPDOT: logPDOT_data, logB: logB_data}

    - log_scale : bool, optional
        Whether data is already in log scale. Default: True.

    - input_layer : int, optional
        Neural network architecture for the first input layer

    - hidden_layers : List[int], optional
        Neural network architecture as list of hidden layer sizes.
        Example: [32, 16, 8] creates 3 hidden layers. Default: [32, 16].

    - output_layer : int, optional
        Neural network architecture for the last output layer

    - train_split : float, optional
        Fraction of data for training (0.0-1.0). Default: 0.70.

    - val_split : float, optional
        Fraction of data for validation (0.0-1.0). Default: 0.15.

    - test_split : float, optional
        Fraction of data for testing (0.0-1.0). Default: 0.15.
        Note: train_split + val_split + test_split must equal 1.0.

    - random_seed : int, optional
        Random seed for reproducible data splitting. Default: 42.

    - solution_name : Optional[str], optional
        Descriptive name for the solution/model being computed.
        Used in CSV output for identification. Default: None.

    --------------------------------------------------------------------------------
                                       ATTRIBUTES
    --------------------------------------------------------------------------------

    - model : nn.Sequential
        The neural network model with Tanh activations.

    - learnable_params : Dict[str, nn.Parameter]
        Dictionary of trainable physical constants.

    - loss_log : Dict[str, List[float]]
        Training history with keys: 'total', 'physics', 'data',
        'val_total', 'val_physics', 'val_data'.

    - test_metrics : Dict[str, float]
        Evaluation metrics including R², RMSE, MAE, reduced χ².
    """

    def __init__(
        self,
        differential_eq: sp.Eq,
        x_sym: sp.Symbol,
        y_sym: sp.Symbol,
        learn_constants: Dict[sp.Symbol, float],
        fixed_inputs: Dict[sp.Symbol, np.ndarray],
        log_scale: bool = True,
        input_layer: int = 1,
        hidden_layers: Optional[List[int]] = None,
        output_layer: int = 1,
        train_split: float = 0.70,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
        solution_name: Optional[str] = None,
    ):

        # Inputs
        self.differential_eq = differential_eq
        self.x_sym = x_sym
        self.y_sym = y_sym
        self.learn_constants = learn_constants
        self.fixed_inputs = fixed_inputs
        self.log_scale = log_scale
        self.solution_name = solution_name

        # Neural Network configurations
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers or [32, 16]
        self.output_layer = output_layer

        # Data splitting
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        # Initialize state
        self.model: Optional[nn.Sequential] = None
        self.learnable_params: Dict[str, nn.Parameter] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.physics_residual_fn = None

        # Initialize loss & test metrics
        self.loss_log = {
            "total": [],
            "physics": [],
            "data": [],
            "val_total": [],
            "val_physics": [],
            "val_data": [],
        }
        self.test_metrics: Dict[str, float] = {}

        # Data assignments to prepare data for PINN training
        self.x_raw: Optional[np.ndarray] = None
        self.y_raw: Optional[np.ndarray] = None
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Torch tensors for PyTorch system
        self.x_train_torch: Optional[torch.Tensor] = None
        self.y_train_torch: Optional[torch.Tensor] = None
        self.x_val_torch: Optional[torch.Tensor] = None
        self.y_val_torch: Optional[torch.Tensor] = None
        self.x_test_torch: Optional[torch.Tensor] = None
        self.y_test_torch: Optional[torch.Tensor] = None

        # Assign fixed input tensors for each split
        self.fixed_torch_train: Dict[str, torch.Tensor] = {}
        self.fixed_torch_val: Dict[str, torch.Tensor] = {}
        self.fixed_torch_test: Dict[str, torch.Tensor] = {}

        # Call methods & build the PINN
        self._prepare_data()
        self._build_neural_network()
        self._create_physics_residual()

        # Confirm Initialization
        logger.info(f"PulsarPINN initialized.")

    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================

    def _prepare_data(self):
        """Extract and randomly split data into train/validation/test sets."""

        # Extract raw x and y data
        self.x_raw = self.fixed_inputs[self.x_sym].copy()
        self.y_raw = self.fixed_inputs[self.y_sym].copy()

        # Calculate split sizes
        total_data_points = len(self.x_raw)
        num_train_points = int(total_data_points * self.train_split)
        num_val_points = int(total_data_points * self.val_split)

        # Create fair distribution
        np.random.seed(self.random_seed)
        shuffled_indices = np.random.permutation(total_data_points)

        # Divide indices into three groups
        train_indices = shuffled_indices[:num_train_points]
        val_indices = shuffled_indices[
            num_train_points : num_train_points + num_val_points
        ]
        test_indices = shuffled_indices[num_train_points + num_val_points :]

        # Split raw data using the indices
        self.x_train = self.x_raw[train_indices]
        self.y_train = self.y_raw[train_indices]
        self.x_val = self.x_raw[val_indices]
        self.y_val = self.y_raw[val_indices]
        self.x_test = self.x_raw[test_indices]
        self.y_test = self.y_raw[test_indices]

        # Convert to PyTorch tensors for training
        self.x_train_torch = self._numpy_array_to_tensor(self.x_train)
        self.y_train_torch = self._numpy_array_to_tensor(self.y_train)
        self.x_val_torch = self._numpy_array_to_tensor(self.x_val)
        self.y_val_torch = self._numpy_array_to_tensor(self.y_val)
        self.x_test_torch = self._numpy_array_to_tensor(self.x_test)
        self.y_test_torch = self._numpy_array_to_tensor(self.y_test)

        # Takes the items from fixed_inputs dictionary and split each variables using the same indices
        for symbol, data_array in self.fixed_inputs.items():
            variable_name = str(symbol)
            self.fixed_torch_train[variable_name] = self._numpy_array_to_tensor(
                data_array[train_indices]
            )
            self.fixed_torch_val[variable_name] = self._numpy_array_to_tensor(
                data_array[val_indices]
            )
            self.fixed_torch_test[variable_name] = self._numpy_array_to_tensor(
                data_array[test_indices]
            )

    def _numpy_array_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        return torch.tensor(array, dtype=torch.float64).view(-1, 1)

    def _build_neural_network(self):
        """Constructs the physics-informed neural network."""

        layers = []

        # Add first input layer
        input_dim = self.input_layer

        # Build hidden layers with Tanh activation functions based on hidden_layers
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))  # Connect layers
            layers.append(nn.Tanh())  # Add non-linearity
            input_dim = hidden_dim

        # Add final output layer
        layers.append(nn.Linear(input_dim, self.output_layer))

        # Combine all layers into sequential model
        self.model = nn.Sequential(*layers).double()

        # Recognize trainable physical constants in differential equation (e.g., logR)
        self.learnable_params = {
            str(sym): nn.Parameter(torch.tensor([val], dtype=torch.float64))
            for sym, val in self.learn_constants.items()
        }

        # Setup optimizer to train both network weights and physical constants
        all_params = list(self.model.parameters()) + list(
            self.learnable_params.values()
        )
        self.optimizer = torch.optim.Adam(all_params, lr=1e-3)
        logger.info(f"Network architecture: {self.hidden_layers}")

    def _create_physics_residual(self):
        """Convert symbolic differential equation to callable residual function."""

        # Calculate residual during training (should equal zero when physics is satisfied)
        residual_expression = self.differential_eq.lhs - self.differential_eq.rhs

        # Start collecting all symbols: input variable, output variable, and learnable constants
        symbols = [self.x_sym, self.y_sym] + list(self.learn_constants.keys())

        # Add any remaining fixed input variables (avoid duplicates)
        for sym in self.fixed_inputs.keys():
            if sym not in symbols:
                symbols.append(sym)

        # Convert symbolic equation to labdified function for PINN use (try PyTorch first, fallback to NumPy)
        try:
            lambdify_expression = sp.lambdify(
                symbols, residual_expression, modules="torch"
            )
        except Exception:
            lambdify_expression = sp.lambdify(
                symbols, residual_expression, modules="numpy"
            )

        # Computes physics residual for PINN system
        def residual_fn(
            x_tensor: torch.Tensor,
            y_tensor: torch.Tensor,
            fixed_dict: Dict[str, torch.Tensor],
        ) -> torch.Tensor:

            # Build argument list that lambdify_expression needs
            args = []

            # Match each symbol to its corresponding data source
            for sym in symbols:
                name = str(sym)

                # Append input variable data (what neural network takes in)
                if sym == self.x_sym:
                    args.append(x_tensor)

                # Append output variable data (what neural network predicts)
                elif sym == self.y_sym:
                    args.append(y_tensor)

                # Append learnable constant (e.g., stellar radius we're trying to find)
                elif name in self.learnable_params:
                    args.append(self.learnable_params[name])

                # Append fixed data we already know (e.g., period derivative)
                elif name in fixed_dict:
                    args.append(fixed_dict[name])

                # Raise error if we're missing required data
                else:
                    raise ValueError(f"Missing data for symbol {name}")

            # Call physics equation with all arguments to get residual (how much physics is violated)
            return lambdify_expression(*args)

        # Store residual function to be used during training
        self.physics_residual_fn = residual_fn

    # =========================================================================
    # TRAINING METHODS
    # =========================================================================

    def train(
        self,
        epochs: int = 3000,
        training_reports: int = 100,
        physics_weight: float = 1.0,
        data_weight: float = 1.0,
    ) -> None:
        """
        Trains the PINN.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------

        - epochs : int
            Number of training iterations. Default: 3000.

        - training_reports : int
            Frequency of checking training progress as the PINN trains. Default: 100.

        - physics_weight : float
            Weight for physics loss component.
            Default: 1.0.

        - data_weight : float
            Weight for data loss component.
            Default: 1.0.
        """
        # Inform user that training starts
        logger.info(f"Starting training for {epochs} epochs")

        # What each epoch does
        for epoch in range(epochs):
            # Training step
            train_loss = self._train_step(physics_weight, data_weight)

            # Log training losses
            self.loss_log["total"].append(train_loss["total"])
            self.loss_log["physics"].append(train_loss["physics"])
            self.loss_log["data"].append(train_loss["data"])

            # Prints progress & metric logs in terminal for every interval
            if epoch % training_reports == 0:

                # Evaluate & store loss based on training sets
                val_loss = self._validation_step(physics_weight, data_weight)

                self.loss_log["val_total"].append(val_loss["total"])
                self.loss_log["val_physics"].append(val_loss["physics"])
                self.loss_log["val_data"].append(val_loss["data"])

                learned_constants = ", ".join(
                    f"{name}={value.item():.4f}"
                    for name, value in self.learnable_params.items()
                )

                # Print progress report to console
                logger.info(
                    f"Epoch {epoch:5d}: Train={train_loss['total']:.6e}, "
                    f"Val={val_loss['total']:.6e} | {learned_constants}"
                )

        # Print final learned constants after PINN training is done
        print("LEARNED CONSTANTS: ")
        for name, param in self.learnable_params.items():
            print(f"  {name:10s} = {param.item():.8f}")

    def _train_step(
        self, physics_weight: float, data_weight: float
    ) -> Dict[str, float]:
        """Executes one epoch"""
        self.model.train()

        # Predict the y value based on x values in a Torch tensor
        predicted_y = self.model(self.x_train_torch)

        # Compute losses during training
        physics_residual = self.physics_residual_fn(
            self.x_train_torch, predicted_y, self.fixed_torch_train
        )
        loss_physics = torch.mean(physics_residual**2)
        loss_data = torch.mean((predicted_y - self.y_train_torch) ** 2)
        loss_total = physics_weight * loss_physics + data_weight * loss_data

        # Backward pass
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return {
            "total": loss_total.item(),
            "physics": loss_physics.item(),
            "data": loss_data.item(),
        }

    def _validation_step(
        self, physics_weight: float, data_weight: float
    ) -> Dict[str, float]:
        """Execute validation step."""

        # Set model to evaluation mode temporarily
        self.model.eval()

        # Disable gradient calculation
        with torch.no_grad():

            # Make predictions on y data
            predicted_y = self.model(self.x_val_torch)

            # Compute all losses for this iteration during training so far
            physics_residual = self.physics_residual_fn(
                self.x_val_torch, predicted_y, self.fixed_torch_val
            )
            loss_physics = torch.mean(physics_residual**2)
            loss_data = torch.mean((predicted_y - self.y_val_torch) ** 2)
            loss_total = physics_weight * loss_physics + data_weight * loss_data

        return {
            "total": loss_total.item(),
            "physics": loss_physics.item(),
            "data": loss_data.item(),
        }

    # =========================================================================
    # EVALUATION METHODS
    # =========================================================================

    def evaluate_test_set(self, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model on test set with corresponding metrics.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        verbose : bool
            If True, print detailed evaluation report. Default: True.

        -------------------------------------------------------------------
                                      RETURNS
        -------------------------------------------------------------------
        Dict[str, float]
            Dictionary containing evaluation metrics for train/val/test splits.
        """
        self.model.eval()

        with torch.no_grad():
            # Compute all metrics for each splits (train, val, & test)
            train_metrics = self._compute_metrics(
                self.x_train_torch, self.y_train_torch, self.fixed_torch_train, "train"
            )
            val_metrics = self._compute_metrics(
                self.x_val_torch, self.y_val_torch, self.fixed_torch_val, "val"
            )
            test_metrics = self._compute_metrics(
                self.x_test_torch, self.y_test_torch, self.fixed_torch_test, "test"
            )

        # Combine all metrics
        self.test_metrics = {**train_metrics, **val_metrics, **test_metrics}

        if verbose:
            self._print_evaluation_report()

        return self.test_metrics

    def _compute_metrics(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        fixed_dict: Dict[str, torch.Tensor],
        prefix: str,
    ) -> Dict[str, float]:
        """Compute evaluation metrics for a data split."""

        # Store current predicted_y for metric evaluation
        predicted_y = self.model(x)

        # Compute residuals
        physics_residual = self.physics_residual_fn(x, predicted_y, fixed_dict)
        loss_physics = torch.mean(physics_residual**2)
        loss_data = torch.mean((predicted_y - y_true) ** 2)
        loss_total = loss_physics + loss_data

        # R² score
        y_mean = torch.mean(y_true)
        total_variance = torch.sum((y_true - y_mean) ** 2)
        residual_variance = torch.sum((y_true - predicted_y) ** 2)
        r2 = 1 - (residual_variance / total_variance)

        # Root Mean Squared Error (RMSE) & Mean Absolute Error (MAE) values
        rmse = torch.sqrt(torch.mean((predicted_y - y_true) ** 2))
        mae = torch.mean(torch.abs(predicted_y - y_true))

        # Reduced χ²
        n_samples = len(y_true)
        n_params = sum(p.numel() for p in self.model.parameters()) + len(
            self.learnable_params
        )
        degrees_of_freedom = max(n_samples - n_params, 1)
        chi2_reduced = residual_variance / degrees_of_freedom

        return {
            f"{prefix}_loss_total": loss_total.item(),
            f"{prefix}_loss_physics": loss_physics.item(),
            f"{prefix}_loss_data": loss_data.item(),
            f"{prefix}_r2": r2.item(),
            f"{prefix}_rmse": rmse.item(),
            f"{prefix}_mae": mae.item(),
            f"{prefix}_chi2_reduced": chi2_reduced.item(),
        }

    def _print_evaluation_report(self) -> None:
        """Print formatted evaluation report with overfitting detection."""

        print("\n" + "=" * 70)
        print("MODEL EVALUATION - QUANTITATIVE METRICS")
        print("=" * 70)

        # Print metrics for each data split (test first as most important)
        for split_name in ["test", "val", "train"]:

            # Get number of samples in this split (e.g., self.x_test, self.x_val, self.x_train)
            num_samples = len(getattr(self, f"x_{split_name}"))

            # Print header for this split
            print(f"\n{split_name.upper()} SET (n={num_samples}):")

            # Print all metrics with consistent formatting
            print(
                f"  Loss (Total):      {self.test_metrics[f'{split_name}_loss_total']:.6e}"
            )
            print(
                f"  Loss (Physics):    {self.test_metrics[f'{split_name}_loss_physics']:.6e}"
            )
            print(
                f"  Loss (Data):       {self.test_metrics[f'{split_name}_loss_data']:.6e}"
            )
            print(f"  R² Score:          {self.test_metrics[f'{split_name}_r2']:.6f}")
            print(f"  RMSE:              {self.test_metrics[f'{split_name}_rmse']:.6e}")
            print(f"  MAE:               {self.test_metrics[f'{split_name}_mae']:.6e}")

            # Only print reduced chi-squared for test set
            if split_name == "test":
                print(
                    f"  Reduced χ²:        {self.test_metrics[f'{split_name}_chi2_reduced']:.6f}"
                )

        # Calculate difference between training and test R² scores
        train_test_r2_difference = (
            self.test_metrics["train_r2"] - self.test_metrics["test_r2"]
        )

        # If train R² is much higher than test R², model likely overfitted (memorized training data)
        if train_test_r2_difference > 0.1:
            print(f" WARNING: Possible overfitting detected.")
            print(f"  Train R² - Test R² = {train_test_r2_difference:.4f}")
        else:
            # Small difference means model generalizes well to new data
            print(f"Good generalization: ΔR² = {train_test_r2_difference:.4f}")

        print("=" * 70)

    # =========================================================================
    # PREDICTION AND VISUALIZATION
    # =========================================================================

    def predict_extended(
        self, extend: float = 0.5, n_points: int = 300
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions over extended range.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - extend : float
            Prediction extension beyond data range by the PINN. Default: 0.5.

        - n_points : int
            Number of prediction points within extension interval. Default: 300.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        - x_values : np.ndarray
            Shape (n_points,) array of x values of predicted PINN extensions.

        - y_predictions : np.ndarray
            Shape (n_points,) array of predicted y values of predicted PINN extensions.
        """
        self.model.eval()

        # Disable gradient tracking
        with torch.no_grad():

            # Find max & min values of original data
            x_min = self.x_raw.min()
            x_max = self.x_raw.max()

            # Created extended x value range
            x_extended = torch.linspace(
                x_min - extend, x_max + extend, n_points, dtype=torch.float64
            ).view(-1, 1)

            # Generate respective extended y value range
            y_extended = self.model(x_extended)

        return x_extended.numpy().flatten(), y_extended.numpy().flatten()

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
        epochs = np.arange(len(self.loss_log["total"]))

        # LEFT PLOT: Total Loss (Training vs Validation)
        # Plot training total loss as solid line
        ax1.plot(epochs, self.loss_log["total"], label="Train", linewidth=2)

        # Plot validation total loss if it exists (checked every val_interval epochs)
        if self.loss_log["val_total"]:
            # Calculate epoch numbers where validation was performed
            val_epochs = np.linspace(
                0, len(epochs) - 1, len(self.loss_log["val_total"])
            )
            ax1.plot(
                val_epochs,
                self.loss_log["val_total"],
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
        ax2.plot(epochs, self.loss_log["physics"], label="Physics", linestyle="--")
        ax2.plot(epochs, self.loss_log["data"], label="Data", linestyle="--")

        # Plot validation component losses if they exist
        if self.loss_log["val_physics"]:
            ax2.plot(
                val_epochs,
                self.loss_log["val_physics"],
                label="Val Physics",
                linestyle=":",
                marker="s",
                markersize=3,
            )
            ax2.plot(
                val_epochs,
                self.loss_log["val_data"],
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

        # Decide to do log scaling based on class input
        if log_scale:
            ax2.set_yscale("log")

        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_data(
        self,
        x_values: Optional[np.ndarray] = None,
        y_predictions: Optional[np.ndarray] = None,
        x_name: str = None,
        y_name: str = None,
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
            x_values, y_predictions = self.predict_extended(extend=0.5, n_points=300)

        # Create figure and axes (FIXED: properly unpack figure and axes)
        fig, ax = plt.subplots(figsize=figsize)

        # Plot training data points (blue circles)
        ax.scatter(
            self.x_train,
            self.y_train,
            c="blue",
            alpha=0.4,
            s=30,
            label="Train Data",
            marker="o",
        )

        # Plot validation data points (orange squares)
        ax.scatter(
            self.x_val,
            self.y_val,
            c="orange",
            alpha=0.5,
            s=40,
            label="Validation Data",
            marker="s",
        )

        # Plot test data points (red triangles)
        ax.scatter(
            self.x_test,
            self.y_test,
            c="red",
            alpha=0.6,
            s=50,
            label="Test Data",
            marker="^",
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
        ax.set_xlabel(str(self.x_sym), fontsize=12, fontweight="bold")
        ax.set_ylabel(str(self.y_sym), fontsize=12, fontweight="bold")

        # Set title (use default with solution name if available)
        if title is None:
            title = f"PINN Predictions vs. Data"
            if self.solution_name:
                title += f" ({self.solution_name})"
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add legend and grid
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add R² score annotation in top-left corner if available
        if self.test_metrics and "test_r2" in self.test_metrics:
            r2_text = f"Test R² = {self.test_metrics['test_r2']:.4f}"
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

    # =========================================================================
    # CSV EXPORT METHODS
    # =========================================================================

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

        - Writes CSV file to specified filepath.

        -------------------------------------------------------------------
                                      NOTES
        -------------------------------------------------------------------

        CSV Structure:
        - Header comments (lines starting with #) contain metadata
        - First data section: Model predictions
        - Second section: Original train/val/test data if true
        """

        # Create output directory if needed
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        # Generate predictions if not provided
        if x_values is None or y_predictions is None:
            logger.info("Generating extended predictions for CSV export")
            x_values, y_predictions = self.predict_extended(extend=0.5, n_points=300)
        elif x_values is not None and y_predictions is None:
            # If user provided x but not y, generate predictions
            self.model.eval()
            with torch.no_grad():
                x_torch = self._numpy_array_to_tensor(x_values)
                y_torch = self.model(x_torch)
                y_predictions = y_torch.numpy().flatten()

        # Prepare metadata for exporting
        metadata_lines = self._generate_metadata_lines(
            include_test_metrics, additional_metadata
        )

        predictions_df = pd.DataFrame(
            {
                x_value_name: x_values,
                f"{y_value_name}_predicted": y_predictions,
                "data_type": "model_prediction",
            }
        )

        if self.solution_name:
            predictions_df["solution_name"] = self.solution_name

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

        # Solution identification
        if self.solution_name:
            lines.append(f"Solution Name: {self.solution_name}")

        lines.append(f"Model: {self.__class__.__name__}")
        lines.append(f"Input Variable: {self.x_sym}")
        lines.append(f"Output Variable: {self.y_sym}")
        lines.append(f"Differential Equation: {self.differential_eq}")

        # Architecture
        lines.append(f"Network Architecture: 1 → {self.hidden_layers} → 1")
        lines.append(
            f"Total Parameters: {sum(p.numel() for p in self.model.parameters())}"
        )

        # Data splits
        lines.append("")
        lines.append("Data Splits:")
        lines.append(f"  Train: {len(self.x_train)} samples ({self.train_split:.1%})")
        lines.append(f"  Validation: {len(self.x_val)} samples ({self.val_split:.1%})")
        lines.append(f"  Test: {len(self.x_test)} samples ({self.test_split:.1%})")

        # Learned constants
        lines.append("")
        lines.append("Learned Physical Constants:")
        for name, param in self.learnable_params.items():
            lines.append(f"  {name} = {param.item():.8f}")

        # Test metrics
        if include_test_metrics and self.test_metrics:
            lines.append("")
            lines.append("Test Set Performance:")
            lines.append(
                f"  R² Score: {self.test_metrics.get('test_r2', float('nan')):.6f}"
            )
            lines.append(
                f"  RMSE: {self.test_metrics.get('test_rmse', float('nan')):.6e}"
            )
            lines.append(
                f"  MAE: {self.test_metrics.get('test_mae', float('nan')):.6e}"
            )
            lines.append(
                f"  Reduced χ²: {self.test_metrics.get('test_chi2_reduced', float('nan')):.6f}"
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
        for i in range(len(self.x_train)):
            data_rows.append(
                {
                    x_name: self.x_train[i],  # Input variable value
                    f"{y_name}_predicted": self.y_train[
                        i
                    ],  # True output value (not predicted!)
                    "data_type": "raw_train",  # Label as training data
                }
            )

        # Add all validation data points with "raw_validation" label
        for i in range(len(self.x_val)):
            data_rows.append(
                {
                    x_name: self.x_val[i],  # Input variable value
                    f"{y_name}_predicted": self.y_val[i],  # True output value
                    "data_type": "raw_validation",  # Label as validation data
                }
            )

        # Add all test data points with "raw_test" label
        for i in range(len(self.x_test)):
            data_rows.append(
                {
                    x_name: self.x_test[i],  # Input variable value
                    f"{y_name}_predicted": self.y_test[i],  # True output value
                    "data_type": "raw_test",  # Label as test data
                }
            )

        # Convert list of dictionaries to pandas DataFrame
        raw_data_df = pd.DataFrame(data_rows)

        # Add solution name column if one was provided (for tracking different experiments)
        if self.solution_name:
            raw_data_df["solution_name"] = self.solution_name

        return raw_data_df

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def store_learned_constants(self) -> Dict[str, float]:
        """
        Get learned physical constants from the PINN model.

        -------------------------------------------------------------------
                                    RETURNS
        -------------------------------------------------------------------
        - Dict[str, float]
            Dictionary mapping constant names to learned values.
        """
        return {name: param.item() for name, param in self.learnable_params.items()}

    def set_learn_constants(self, new_constants: Dict[str, float]) -> None:
        """
        Update learnable constants with new values added to the differential equation

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - new_constants : Dict[str, float]
            Dictionary of constants to update or add.
        """
        for name, value in new_constants.items():
            if name in self.learnable_params:
                self.learnable_params[name].data = torch.tensor(
                    [value], dtype=torch.float64
                )
                logger.info(f"Updated {name} = {value:.6f}")
            else:
                self.learnable_params[name] = nn.Parameter(
                    torch.tensor([value], dtype=torch.float64)
                )
                logger.info(f"Added new constant {name} = {value:.6f}")

        # Reinitialize optimizer
        all_params = list(self.model.parameters()) + list(
            self.learnable_params.values()
        )
        self.optimizer = torch.optim.Adam(all_params, lr=1e-3)
