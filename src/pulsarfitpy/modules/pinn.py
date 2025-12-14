"""
Physics-Informed Neural Networks (PINNs) for Pulsar Analysis

This module implements a PINN for learning physical constants from pulsar data
while enforcing physics constraints through differential equations.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

# TODO: Fix PINN training progress logger
# TODO: Add custom input & output layers

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from typing import Dict, List, Optional, Tuple
import logging

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
        Evaluation metrics including Rsquared, RMSE, MAE, reduced chi squared (x^2).
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
    ):

        # Inputs
        self.differential_eq = differential_eq
        self.x_sym = x_sym
        self.y_sym = y_sym
        self.learn_constants = learn_constants
        self.fixed_inputs = fixed_inputs
        self.log_scale = log_scale

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
        tensor = torch.tensor(array, dtype=torch.float64)

        # Only reshape if 1D array, otherwise preserve shape
        if tensor.dim() == 1:
            tensor = tensor.view(-1, 1)
        return tensor

    def _build_neural_network(self):
        """Constructs the physics-informed neural network."""

        layers = []

        # Add first input layer (start with the specified input dimension)
        input_dim = self.input_layer

        # Build hidden layers with Tanh activation functions based on hidden_layers
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))  # Connect layers
            layers.append(nn.Tanh())  # Add non-linearity
            input_dim = hidden_dim

        # Add final output layer (use the specified output dimension)
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
        logger.info(f"Network architecture: {self.input_layer}    {self.hidden_layers}    {self.output_layer}")

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

                # Print all losses at the current epoch
                logger.info(f"Epoch {epoch:5d} Losses:")
                logger.info(f"  Total Loss: {train_loss['total']:.6e}")
                logger.info(f"  Physics Loss: {train_loss['physics']:.6e}")
                logger.info(f"  Data Loss: {train_loss['data']:.6e}")

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

        # Rsquared score
        y_mean = torch.mean(y_true)
        total_variance = torch.sum((y_true - y_mean) ** 2)
        residual_variance = torch.sum((y_true - predicted_y) ** 2)
        r2 = 1 - (residual_variance / total_variance)

        # Root Mean Squared Error (RMSE) & Mean Absolute Error (MAE) values
        rmse = torch.sqrt(torch.mean((predicted_y - y_true) ** 2))
        mae = torch.mean(torch.abs(predicted_y - y_true))

        # Reduced chi square
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
            print(f"  Rsquared Score:          {self.test_metrics[f'{split_name}_r2']:.6f}")
            print(f"  RMSE:              {self.test_metrics[f'{split_name}_rmse']:.6e}")
            print(f"  MAE:               {self.test_metrics[f'{split_name}_mae']:.6e}")

            # Only print reduced chi-squared for test set
            if split_name == "test":
                print(
                    f"  Reduced chi squared:        {self.test_metrics[f'{split_name}_chi2_reduced']:.6f}"
                )

        # Calculate difference between training and test Rsquared scores
        train_test_r2_difference = (
            self.test_metrics["train_r2"] - self.test_metrics["test_r2"]
        )

        # If train Rsquared is much higher than test Rsquared, model likely overfitted (memorized training data)
        if train_test_r2_difference > 0.1:
            print(f" WARNING: Possible overfitting detected.")
            print(f"  Train Rsquared - Test Rsquared = {train_test_r2_difference:.4f}")
        else:
            # Small difference means model generalizes well to new data
            print(f"Good generalization: Rsquared = {train_test_r2_difference:.4f}")

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

    # =========================================================================
    # OTHER UTILITY METHODS
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

    def bootstrap_uncertainty(
        self,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.8,
        epochs: int = 1000,
        confidence_level: float = 0.95,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate uncertainty in learned constants using bootstrap resampling.

        Performs bootstrap resampling to quantify uncertainty in learned physical
        constants. For each bootstrap iteration, randomly samples training data,
        retrains the model, and records the learned constant values.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - n_bootstrap : int
            Number of bootstrap iterations. Default: 100.

        - sample_fraction : float
            Fraction of training data to sample in each iteration (0.0-1.0).
            Default: 0.8 (80% of training data).

        - epochs : int
            Number of training epochs per bootstrap iteration. Default: 1000.

        - confidence_level : float
            Confidence level for intervals (e.g., 0.95 for 95% CI). Default: 0.95.

        - verbose : bool
            Whether to print progress messages. Default: True.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        - Dict[str, Dict[str, float]]
            Nested dictionary with uncertainty statistics for each constant:
            {
                'constant_name': {
                    'mean': mean_value,
                    'std': standard_deviation,
                    'ci_lower': lower_confidence_bound,
                    'ci_upper': upper_confidence_bound,
                    'original': original_fitted_value
                }
            }

        -------------------------------------------------------------------
                                      EXAMPLE
        -------------------------------------------------------------------
        >>> pinn.train(epochs=5000)
        >>> uncertainties = pinn.bootstrap_uncertainty(n_bootstrap=100)
        >>> print(f"logR = {uncertainties['logR']['mean']:.3f} plus minus {uncertainties['logR']['std']:.3f}")
        >>> print(f"95% CI: [{uncertainties['logR']['ci_lower']:.3f}, {uncertainties['logR']['ci_upper']:.3f}]")
        """
        if verbose:
            print("=" * 70)
            print("BOOTSTRAP UNCERTAINTY ESTIMATION")
            print("=" * 70)
            print(f"Bootstrap iterations: {n_bootstrap}")
            print(f"Sample fraction: {sample_fraction:.1%}")
            print(f"Training epochs per iteration: {epochs}")
            print(f"Confidence level: {confidence_level:.1%}")
            print()

        # Store original model state
        original_constants = self.store_learned_constants()
        original_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Storage for bootstrap samples
        bootstrap_constants = {name: [] for name in self.learnable_params.keys()}

        # Perform bootstrap iterations
        for i in range(n_bootstrap):
            if verbose and (i + 1) % max(1, n_bootstrap // 10) == 0:
                print(f"  Bootstrap iteration {i + 1}/{n_bootstrap}...")

            # Randomly sample training data with replacement
            # Calculate the number of samples for bootstrap resampling
            n_samples = int(len(self.x_train) * sample_fraction)
            indices = np.random.choice(len(self.x_train), size=n_samples, replace=True)

            # Create bootstrap sample
            boot_x = self.x_train[indices]
            boot_y = self.y_train[indices]

            # Convert bootstrap samples to tensors
            boot_x_torch = self._numpy_array_to_tensor(boot_x)
            boot_y_torch = self._numpy_array_to_tensor(boot_y)

            # Temporarily replace training data with bootstrap sample
            original_x_train_torch = self.x_train_torch
            original_y_train_torch = self.y_train_torch
            self.x_train_torch = boot_x_torch
            self.y_train_torch = boot_y_torch

            # Train on bootstrap sample (suppress training output)
            self.train(epochs=epochs, training_reports=epochs + 1, physics_weight=1.0, data_weight=1.0)

            # Store learned constants from this iteration
            learned = self.store_learned_constants()
            for name, value in learned.items():
                bootstrap_constants[name].append(value)

            # Restore original training data
            self.x_train_torch = original_x_train_torch
            self.y_train_torch = original_y_train_torch

        # Restore original model state
        self.model.load_state_dict(original_model_state)
        self.set_learn_constants(original_constants)

        # Compute statistics
        alpha = 1 - confidence_level
        results = {}

        for name, values in bootstrap_constants.items():
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)  # Sample standard deviation
            ci_lower = np.percentile(values_array, 100 * alpha / 2)
            ci_upper = np.percentile(values_array, 100 * (1 - alpha / 2))

            results[name] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'original': original_constants[name]
            }

            if verbose:
                print(f"\n{name}:")
                print(f"  Original fitted value: {original_constants[name]:.6f}")
                print(f"  Bootstrap mean:        {mean_val:.6f}")
                print(f"  Bootstrap std dev:     {std_val:.6f}")
                print(f"  {confidence_level:.0%} CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

        if verbose:
            print("\n" + "=" * 70)

        return results

    def monte_carlo_uncertainty(
        self,
        n_simulations: int = 1000,
        noise_level: float = 0.01,
        confidence_level: float = 0.95,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate uncertainty using Monte Carlo simulation with data perturbation.

        Adds Gaussian noise to the input data and re-evaluates the trained model
        to assess sensitivity of learned constants to data perturbations.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - n_simulations : int
            Number of Monte Carlo simulations. Default: 1000.

        - noise_level : float
            Standard deviation of Gaussian noise relative to data std dev.
            Default: 0.01 (1% noise).

        - confidence_level : float
            Confidence level for intervals (e.g., 0.95 for 95% CI). Default: 0.95.

        - verbose : bool
            Whether to print progress messages. Default: True.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        - Dict[str, Dict[str, float]]
            Nested dictionary with uncertainty statistics for each constant:
            {
                'constant_name': {
                    'mean': mean_value,
                    'std': standard_deviation,
                    'ci_lower': lower_confidence_bound,
                    'ci_upper': upper_confidence_bound,
                    'original': original_fitted_value
                }
            }

        -------------------------------------------------------------------
                                      EXAMPLE
        -------------------------------------------------------------------
        >>> pinn.train(epochs=5000)
        >>> uncertainties = pinn.monte_carlo_uncertainty(n_simulations=1000)
        >>> print(f"logR = {uncertainties['logR']['mean']:.3f} plus minus {uncertainties['logR']['std']:.3f}")
        """
        if verbose:
            print("=" * 70)
            print("MONTE CARLO UNCERTAINTY ESTIMATION")
            print("=" * 70)
            print(f"Simulations: {n_simulations}")
            print(f"Noise level: {noise_level:.1%} of data std dev")
            print(f"Confidence level: {confidence_level:.1%}")
            print()

        # Store original constants and model
        original_constants = self.store_learned_constants()
        original_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Storage for Monte Carlo samples
        mc_constants = {name: [] for name in self.learnable_params.keys()}

        # Get data standard deviations for noise scaling
        x_std = self.x_train_torch.std()
        y_std = self.y_train_torch.std()

        for i in range(n_simulations):
            if verbose and (i + 1) % max(1, n_simulations // 10) == 0:
                print(f"  Simulation {i + 1}/{n_simulations}...")

            # Add Gaussian noise to data
            x_noise = torch.randn_like(self.x_train_torch) * (noise_level * x_std)
            y_noise = torch.randn_like(self.y_train_torch) * (noise_level * y_std)

            perturbed_x = self.x_train_torch + x_noise
            perturbed_y = self.y_train_torch + y_noise

            # Temporarily replace training data with perturbed data
            original_x_train = self.x_train_torch
            original_y_train = self.y_train_torch
            self.x_train_torch = perturbed_x
            self.y_train_torch = perturbed_y

            # Quick training (fewer epochs than bootstrap)
            self.train(epochs=500, training_reports=501, physics_weight=1.0, data_weight=1.0)

            # Store learned constants
            learned = self.store_learned_constants()
            for name, value in learned.items():
                mc_constants[name].append(value)

            # Restore original training data
            self.x_train_torch = original_x_train
            self.y_train_torch = original_y_train

        # Restore original model state
        self.model.load_state_dict(original_model_state)
        self.set_learn_constants(original_constants)

        # Compute statistics
        alpha = 1 - confidence_level
        results = {}

        for name, values in mc_constants.items():
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array, ddof=1)
            ci_lower = np.percentile(values_array, 100 * alpha / 2)
            ci_upper = np.percentile(values_array, 100 * (1 - alpha / 2))

            results[name] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'original': original_constants[name]
            }

            if verbose:
                print(f"\n{name}:")
                print(f"  Original fitted value: {original_constants[name]:.6f}")
                print(f"  Monte Carlo mean:      {mean_val:.6f}")
                print(f"  Monte Carlo std dev:   {std_val:.6f}")
                print(f"  {confidence_level:.0%} CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

        if verbose:
            print("\n" + "=" * 70)

        return results

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

    # =========================================================================
    # ROBUSTNESS VALIDATION METHODS
    # =========================================================================

    def validate_with_permutation_test(
        self,
        n_permutations: int = 100,
        epochs: int = 1000,
        significance_level: float = 0.05,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Perform permutation test to validate that model learns real relationships.

        Randomly shuffles target labels and retrains to establish null distribution.
        If the real model significantly outperforms permuted models, the learned
        relationships are likely genuine rather than spurious.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - n_permutations : int
            Number of random permutations to test. Default: 100.

        - epochs : int
            Training epochs for each permutation. Default: 1000.

        - significance_level : float
            Significance threshold (e.g., 0.05 for p < 0.05). Default: 0.05.

        - verbose : bool
            Whether to print progress and results. Default: True.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        - Dict[str, any]
            Dictionary containing:
            {
                'real_r2': R squared from actual trained model,
                'permuted_r2_mean': Mean R squared from permuted models,
                'permuted_r2_std': Std dev of permuted R values,
                'permuted_r2_values': List of all permuted Rsquared scores,
                'p_value': Fraction of permuted models >= real model,
                'is_significant': Whether real model significantly better,
                'significance_level': Threshold used
            }

        -------------------------------------------------------------------
                                      EXAMPLE
        -------------------------------------------------------------------
        >>> pinn.train(epochs=5000)
        >>> validation = pinn.validate_with_permutation_test(n_permutations=100)
        >>> if validation['is_significant']:
        >>>     print("Model learns genuine physical relationships!")
        >>> print(f"p-value: {validation['p_value']:.4f}")
        """
        if verbose:
            print("=" * 70)
            print("PERMUTATION TEST - VALIDATING LEARNED RELATIONSHIPS")
            print("=" * 70)
            print(f"Permutations: {n_permutations}")
            print(f"Epochs per permutation: {epochs}")
            print(f"Significance level: {significance_level}")
            print()

        # Store original model state
        original_constants = self.store_learned_constants()
        original_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Get real model performance
        if not self.test_metrics:
            self.evaluate_test_set(verbose=False)
        real_r2 = self.test_metrics.get('test_r2', 0.0)

        if verbose:
            print(f"Real model Rsquared: {real_r2:.6f}")
            print("\nRunning permutation tests...")

        # Storage for permuted results
        permuted_r2_values = []

        for i in range(n_permutations):
            if verbose and (i + 1) % max(1, n_permutations // 10) == 0:
                print(f"  Permutation {i + 1}/{n_permutations}...")

            # Randomly shuffle target labels (breaks real relationships)
            permuted_indices = np.random.permutation(len(self.y_train_torch))
            permuted_y_train = self.y_train_torch[permuted_indices]

            # Temporarily replace training data with permuted labels
            original_y_train = self.y_train_torch
            self.y_train_torch = permuted_y_train

            # Train on permuted data (suppress output)
            self.train(epochs=epochs, training_reports=epochs + 1, physics_weight=1.0, data_weight=1.0)

            # Evaluate on original test set
            test_metrics = self.evaluate_test_set(verbose=False)
            permuted_r2_values.append(test_metrics.get('test_r2', 0.0))

            # Restore original training data
            self.y_train_torch = original_y_train

        # Restore original model state
        self.model.load_state_dict(original_model_state)
        self.set_learn_constants(original_constants)

        # Compute statistics
        permuted_r2_array = np.array(permuted_r2_values)
        permuted_mean = np.mean(permuted_r2_array)
        permuted_std = np.std(permuted_r2_array, ddof=1)

        # Calculate p-value: fraction of permuted models that perform as well or better
        p_value = np.mean(permuted_r2_array >= real_r2)
        is_significant = p_value < significance_level

        results = {
            'real_r2': real_r2,
            'permuted_r2_mean': permuted_mean,
            'permuted_r2_std': permuted_std,
            'permuted_r2_values': permuted_r2_values,
            'p_value': p_value,
            'is_significant': is_significant,
            'significance_level': significance_level
        }

        if verbose:
            print("\n" + "=" * 70)
            print("PERMUTATION TEST RESULTS")
            print("=" * 70)
            print(f"\nReal model Rsquared:           {real_r2:.6f}")
            print(f"Permuted models Rsquared mean: {permuted_mean:.6f} plus minus {permuted_std:.6f}")
            print(f"p-value:                 {p_value:.4f}")
            print(f"\nSignificance test (alpha = {significance_level}):")
            if is_significant:
                print("   PASSED: Real model significantly better than random")
                print("     Model learns genuine physical relationships")
            else:
                print("    FAILED: Real model not significantly better than random")
                print("     WARNING: Model may be capturing spurious correlations")
            print("=" * 70)

        return results

    def validate_with_feature_shuffling(
        self,
        n_shuffles: int = 50,
        epochs: int = 1000,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Validate model by shuffling input features to break real relationships.

        Randomly shuffles x-values (breaking x-y relationship) and retrains.
        If real model significantly outperforms shuffled models, features contain
        genuine information.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - n_shuffles : int
            Number of feature shuffling iterations. Default: 50.

        - epochs : int
            Training epochs per shuffle. Default: 1000.

        - verbose : bool
            Whether to print progress and results. Default: True.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        - Dict[str, any]
            Dictionary with validation results including Rsquared comparisons.

        -------------------------------------------------------------------
                                      EXAMPLE
        -------------------------------------------------------------------
        >>> pinn.train(epochs=5000)
        >>> validation = pinn.validate_with_feature_shuffling(n_shuffles=50)
        >>> print(f"Real vs shuffled difference: {validation['r2_difference']:.4f}")
        """
        if verbose:
            print("=" * 70)
            print("FEATURE SHUFFLING TEST - VALIDATING INPUT IMPORTANCE")
            print("=" * 70)
            print(f"Shuffles: {n_shuffles}")
            print(f"Epochs per shuffle: {epochs}")
            print()

        # Store original model state
        original_constants = self.store_learned_constants()
        original_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Get real model performance
        if not self.test_metrics:
            self.evaluate_test_set(verbose=False)
        real_r2 = self.test_metrics.get('test_r2', 0.0)

        if verbose:
            print(f"Real model Rsquared: {real_r2:.6f}")
            print("\nRunning feature shuffling tests...")

        # Storage for shuffled results
        shuffled_r2_values = []

        for i in range(n_shuffles):
            if verbose and (i + 1) % max(1, n_shuffles // 10) == 0:
                print(f"  Shuffle {i + 1}/{n_shuffles}...")

            # Randomly shuffle input features (breaks x-y relationship)
            shuffled_indices = np.random.permutation(len(self.x_train_torch))
            shuffled_x_train = self.x_train_torch[shuffled_indices]

            # Temporarily replace training data with shuffled features
            original_x_train = self.x_train_torch
            self.x_train_torch = shuffled_x_train

            # Train on shuffled data
            self.train(epochs=epochs, training_reports=epochs + 1, physics_weight=1.0, data_weight=1.0)

            # Evaluate
            test_metrics = self.evaluate_test_set(verbose=False)
            shuffled_r2_values.append(test_metrics.get('test_r2', 0.0))

            # Restore original training data
            self.x_train_torch = original_x_train

        # Restore original model state
        self.model.load_state_dict(original_model_state)
        self.set_learn_constants(original_constants)

        # Compute statistics
        shuffled_r2_array = np.array(shuffled_r2_values)
        shuffled_mean = np.mean(shuffled_r2_array)
        shuffled_std = np.std(shuffled_r2_array, ddof=1)
        r2_difference = real_r2 - shuffled_mean

        results = {
            'real_r2': real_r2,
            'shuffled_r2_mean': shuffled_mean,
            'shuffled_r2_std': shuffled_std,
            'shuffled_r2_values': shuffled_r2_values,
            'r2_difference': r2_difference,
            'improvement_factor': real_r2 / max(shuffled_mean, 1e-10)
        }

        if verbose:
            print("\n" + "=" * 70)
            print("FEATURE SHUFFLING TEST RESULTS")
            print("=" * 70)
            print(f"\nReal model Rsquared:           {real_r2:.6f}")
            print(f"Shuffled models Rsquared mean: {shuffled_mean:.6f} plus minus {shuffled_std:.6f}")
            print(f"Improvement:             {r2_difference:.6f}")
            print(f"Factor improvement:      {results['improvement_factor']:.2f}x")
            
            if r2_difference > 0.1:
                print("\n   PASSED: Real features significantly better than shuffled")
                print("     Input features contain genuine information")
            else:
                print("\n   WARNING: Real features not much better than shuffled")
                print("     Features may not contain meaningful signal")
            print("=" * 70)

        return results

    def validate_with_impossible_physics(
        self,
        epochs: int = 2000,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Test model with physically impossible parameter combinations.

        Inverts the differential equation or swaps variable roles to create
        physically meaningless relationships. A robust model should perform
        poorly on impossible physics.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - epochs : int
            Training epochs for impossible physics test. Default: 2000.

        - verbose : bool
            Whether to print results. Default: True.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        - Dict[str, any]
            Dictionary with comparison between real and impossible physics.

        -------------------------------------------------------------------
                                      EXAMPLE
        -------------------------------------------------------------------
        >>> pinn.train(epochs=5000)
        >>> validation = pinn.validate_with_impossible_physics()
        >>> if validation['real_much_better']:
        >>>     print("Model correctly rejects impossible physics!")
        """
        if verbose:
            print("=" * 70)
            print("IMPOSSIBLE PHYSICS TEST - VALIDATING PHYSICS CONSTRAINTS")
            print("=" * 70)
            print(f"Training epochs: {epochs}")
            print()

        # Store original model state
        original_constants = self.store_learned_constants()
        original_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Get real model performance
        if not self.test_metrics:
            self.evaluate_test_set(verbose=False)
        real_r2 = self.test_metrics.get('test_r2', 0.0)
        real_loss = self.test_metrics.get('test_loss_total', float('inf'))

        if verbose:
            print(f"Real physics model Rsquared: {real_r2:.6f}")
            print(f"Real physics loss:     {real_loss:.6e}")
            print("\nTesting with inverted physics (swapped inputs/outputs)...")

        # Swap training and test data (physically meaningless)
        original_x_train = self.x_train_torch
        original_y_train = self.y_train_torch
        original_x_test = self.x_test_torch
        original_y_test = self.y_test_torch

        # Swap training data roles
        self.x_train_torch = original_y_train
        self.y_train_torch = original_x_train
        self.x_test_torch = original_y_test
        self.y_test_torch = original_x_test

        # Train on impossible physics
        self.train(epochs=epochs, training_reports=epochs + 1, physics_weight=1.0, data_weight=1.0)

        # Evaluate impossible model
        impossible_metrics = self.evaluate_test_set(verbose=False)
        impossible_r2 = impossible_metrics.get('test_r2', 0.0)
        impossible_loss = impossible_metrics.get('test_loss_total', float('inf'))

        # Restore original data and model
        self.x_train_torch = original_x_train
        self.y_train_torch = original_y_train
        self.x_test_torch = original_x_test
        self.y_test_torch = original_y_test
        self.model.load_state_dict(original_model_state)
        self.set_learn_constants(original_constants)

        # Compute comparison
        r2_difference = real_r2 - impossible_r2
        real_much_better = r2_difference > 0.2  # Real model should be much better

        results = {
            'real_r2': real_r2,
            'impossible_r2': impossible_r2,
            'real_loss': real_loss,
            'impossible_loss': impossible_loss,
            'r2_difference': r2_difference,
            'real_much_better': real_much_better
        }

        if verbose:
            print("\n" + "=" * 70)
            print("IMPOSSIBLE PHYSICS TEST RESULTS")
            print("=" * 70)
            print(f"\nReal physics model:")
            print(f"  Rsquared:   {real_r2:.6f}")
            print(f"  Loss: {real_loss:.6e}")
            print(f"\nImpossible physics model (swapped variables):")
            print(f"  Rsquared:   {impossible_r2:.6f}")
            print(f"  Loss: {impossible_loss:.6e}")
            print(f"\nDifference: {r2_difference:.6f}")
            
            if real_much_better:
                print("\n   PASSED: Real physics significantly better than impossible")
                print("     Model respects physical constraints")
            else:
                print("\n    WARNING: Real physics not much better than impossible")
                print("     Model may not be learning genuine physics")
            print("=" * 70)

        return results

    def run_all_robustness_tests(
        self,
        n_permutations: int = 100,
        n_shuffles: int = 50,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Run complete suite of robustness validation tests.

        Executes permutation test, feature shuffling, and impossible physics
        validation to comprehensively assess model reliability.

        -------------------------------------------------------------------
                                    PARAMETERS
        -------------------------------------------------------------------
        - n_permutations : int
            Number of permutations for label shuffling test. Default: 100.

        - n_shuffles : int
            Number of shuffles for feature shuffling test. Default: 50.

        - verbose : bool
            Whether to print detailed results. Default: True.

        -------------------------------------------------------------------
                                     RETURNS
        -------------------------------------------------------------------
        - Dict[str, any]
            Dictionary containing all test results and overall assessment.

        -------------------------------------------------------------------
                                      EXAMPLE
        -------------------------------------------------------------------
        >>> pinn.train(epochs=5000)
        >>> robustness = pinn.run_all_robustness_tests()
        >>> if robustness['all_tests_passed']:
        >>>     print("Model passed all robustness checks!")
        """
        if verbose:
            print("\n" + "=" * 70)
            print("COMPREHENSIVE ROBUSTNESS VALIDATION SUITE")
            print("=" * 70)
            print()

        # Run all tests
        permutation_results = self.validate_with_permutation_test(
            n_permutations=n_permutations,
            epochs=1000,
            verbose=verbose
        )

        if verbose:
            print("\n")

        feature_results = self.validate_with_feature_shuffling(
            n_shuffles=n_shuffles,
            epochs=1000,
            verbose=verbose
        )

        if verbose:
            print("\n")

        physics_results = self.validate_with_impossible_physics(
            epochs=2000,
            verbose=verbose
        )

        # Overall assessment
        all_tests_passed = (
            permutation_results['is_significant'] and
            feature_results['r2_difference'] > 0.1 and
            physics_results['real_much_better']
        )

        results = {
            'permutation_test': permutation_results,
            'feature_shuffling_test': feature_results,
            'impossible_physics_test': physics_results,
            'all_tests_passed': all_tests_passed
        }

        if verbose:
            print("\n" + "=" * 70)
            print("OVERALL ROBUSTNESS ASSESSMENT")
            print("=" * 70)
            print(f"\nPermutation test:      {' PASS' if permutation_results['is_significant'] else '  FAIL'}")
            print(f"Feature shuffling:     {' PASS' if feature_results['r2_difference'] > 0.1 else '  FAIL'}")
            print(f"Impossible physics:    {' PASS' if physics_results['real_much_better'] else '  FAIL'}")
            print(f"\n{'='*70}")
            if all_tests_passed:
                print("VERDICT: Model demonstrates robust learning of genuine physics")
                print("         Safe to use for scientific inference")
            else:
                print("VERDICT: Model shows signs of spurious correlations")
                print("         Use caution in scientific interpretation")
            print("=" * 70)

        return results