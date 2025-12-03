"""
Physics-Informed Neural Networks (PINNs) for pulsar data analysis.

This module implements the PulsarPINN class for learning physical constants
from pulsar data while enforcing physics constraints through differential equations.
Uses PyTorch for automatic differentiation and neural network training.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import sympy as sp
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from psrqpy import QueryATNF

logger = logging.getLogger(__name__)


class PulsarPINN:
    """
    Physics-Informed Neural Network (PINN) for learning physical relationships from pulsar data.
    
    This class combines neural network predictions with physics-based constraints (differential
    equations) to learn physical constants from the ATNF pulsar catalogue while respecting
    known physical laws. The model automatically splits data into training, validation, and
    test sets for robust evaluation.
    
    Parameters
    ----------
    x_param : str
        Name of the x-axis parameter from ATNF catalogue (e.g., 'P0' for period).
    y_param : str
        Name of the y-axis parameter from ATNF catalogue (e.g., 'P1' for period derivative).
    differential_eq : sympy.Eq
        Symbolic differential equation representing the physics constraint.
        Example: sp.Eq(logPdot, logC + (2 - n) * logP)
    x_sym : sympy.Symbol
        Symbolic variable corresponding to x_param in the differential equation.
    y_sym : sympy.Symbol
        Symbolic variable corresponding to y_param in the differential equation.
    learn_constants : dict, optional
        Dictionary mapping symbolic constants to their initial guesses.
        Example: {logC: -14.75, n: 0.25}
        Default: None (empty dictionary).
    log_scale : bool, optional
        Whether to apply log10 transformation to the data. Default: True.
    psrqpy_filter_fn : callable, optional
        Custom filter function: filter_fn(x_array, y_array) -> boolean_mask.
        Applied after basic filtering for positive values. Default: None.
    fixed_inputs : dict, optional
        Dictionary mapping symbolic variables to fixed data arrays.
        Used for multi-parameter equations where some variables are known.
        Example: {logPDOT: logPDOT_data_array}
        Arrays must match the length of the queried data. Default: None.
    train_split : float, optional
        Fraction of data used for training (0.0 to 1.0). Default: 0.70.
    val_split : float, optional
        Fraction of data used for validation (0.0 to 1.0). Default: 0.15.
    test_split : float, optional
        Fraction of data used for testing (0.0 to 1.0). Default: 0.15.
        Note: train_split + val_split + test_split should equal 1.0.
    random_seed : int, optional
        Random seed for reproducible data splitting. Default: 42.
    hidden_layers : list of int, optional
        Neural network architecture specification as list of hidden layer sizes.
        Example: [32, 16] creates a network with two hidden layers.
        Default: [32, 16] (two hidden layers with 32 and 16 neurons).
    
    Attributes
    ----------
    model : torch.nn.Sequential
        The neural network model with Tanh activations.
    learnable_params : dict
        Dictionary of learnable constant parameters as torch.nn.Parameter objects.
    loss_log : dict
        Training history containing:
        - 'train_total', 'train_physics', 'train_data': Training losses per epoch
        - 'val_total', 'val_physics', 'val_data': Validation losses at intervals
        - 'total', 'physics', 'data': Backward compatibility aliases
    test_metrics : dict
        Final evaluation metrics including R² scores and losses for all data splits.
    x_train, y_train : ndarray
        Training data (transformed according to log_scale).
    x_val, y_val : ndarray
        Validation data (transformed according to log_scale).
    x_test, y_test : ndarray
        Test data (transformed according to log_scale).
    x_all, y_all : ndarray
        All data combined (for plotting purposes).
    
    Notes
    -----
    - The neural network uses double precision (float64) for numerical stability.
    - Data is automatically queried from the ATNF pulsar catalogue via psrqpy.
    - Invalid values (NaN, negative when log_scale=True) are automatically filtered.
    - The loss function combines physics residuals and data fitting:
      Loss = MSE(physics_residual) + MSE(prediction - actual)
    - Early stopping can be enabled during training to prevent overfitting.
    
    See Also
    --------
    PulsarApproximation : Polynomial approximation without physics constraints
    configure_logging : Configure logging verbosity for training output
    """
    def __init__(self, x_param: str, y_param: str,
                 differential_eq: sp.Eq,
                 x_sym: sp.Symbol, y_sym: sp.Symbol,
                 learn_constants: dict = None,
                 log_scale=True,
                 psrqpy_filter_fn=None,
                 fixed_inputs: dict = None,
                 train_split=0.70,
                 val_split=0.15,
                 test_split=0.15,
                 random_seed=42,
                 hidden_layers=None):

        self.x_param = x_param
        self.y_param = y_param
        self.differential_eq = differential_eq
        self.x_sym = x_sym
        self.y_sym = y_sym
        self.learn_constants = learn_constants or {}
        self.log_scale = log_scale
        self.psrqpy_filter_fn = psrqpy_filter_fn
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.hidden_layers = hidden_layers or [32, 16]  # Default architecture

        self.fixed_inputs = fixed_inputs or {}
        self.fixed_torch_inputs = {}
        self.fixed_torch_inputs_train = {}
        self.fixed_torch_inputs_val = {}
        self.fixed_torch_inputs_test = {}
        self.learnable_params = {}
        self.loss_log = {
            "train_total": [], "train_physics": [], "train_data": [],
            "val_total": [], "val_physics": [], "val_data": [],
            "total": [], "physics": [], "data": []  # For backward compatibility
        }
        self.test_metrics = {}

        self._prepare_data()
        self._build_model()
        self._convert_symbolic_to_residual()

    def _prepare_data(self):
        """
        Fetch pulsar data from ATNF catalogue and prepare train/validation/test splits.
        
        This internal method:
        1. Queries the ATNF pulsar catalogue for the specified parameters
        2. Filters out invalid values (NaN, non-positive if log_scale=True)
        3. Applies optional custom filtering
        4. Transforms data to log scale if requested
        5. Randomly splits data into train/val/test sets
        6. Converts data to PyTorch tensors with gradient tracking
        7. Processes fixed inputs for each data split
        """
        query = QueryATNF(params=[self.x_param, self.y_param])
        table = query.table

        x = table[self.x_param].data
        y = table[self.y_param].data

        mask = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0) & (y > 0)
        x, y = x[mask], y[mask]

        if self.psrqpy_filter_fn:
            keep = self.psrqpy_filter_fn(x, y)
            x, y = x[keep], y[keep]

        self.x_raw = np.log10(x) if self.log_scale else x
        self.y_raw = np.log10(y) if self.log_scale else y

        # Shuffle and split data
        np.random.seed(self.random_seed)
        n_total = len(self.x_raw)
        indices = np.random.permutation(n_total)
        
        n_train = int(self.train_split * n_total)
        n_val = int(self.val_split * n_total)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Store all data for reference and plotting
        self.x_all = self.x_raw
        self.y_all = self.y_raw
        
        # Training set
        self.x_train = self.x_raw[train_idx]
        self.y_train = self.y_raw[train_idx]
        self.x_torch = torch.tensor(self.x_train, dtype=torch.float64).view(-1, 1).requires_grad_(True)
        self.y_torch = torch.tensor(self.y_train, dtype=torch.float64).view(-1, 1)
        
        # Validation set
        self.x_val = self.x_raw[val_idx]
        self.y_val = self.y_raw[val_idx]
        self.x_val_torch = torch.tensor(self.x_val, dtype=torch.float64).view(-1, 1).requires_grad_(True)
        self.y_val_torch = torch.tensor(self.y_val, dtype=torch.float64).view(-1, 1)
        
        # Test set
        self.x_test = self.x_raw[test_idx]
        self.y_test = self.y_raw[test_idx]
        self.x_test_torch = torch.tensor(self.x_test, dtype=torch.float64).view(-1, 1).requires_grad_(True)
        self.y_test_torch = torch.tensor(self.y_test, dtype=torch.float64).view(-1, 1)
        
        logger.info(f"Data split: Train={len(train_idx)} ({100*len(train_idx)/n_total:.1f}%), "
              f"Val={len(val_idx)} ({100*len(val_idx)/n_total:.1f}%), "
              f"Test={len(test_idx)} ({100*len(test_idx)/n_total:.1f}%)")

        # Convert fixed inputs to torch tensors for each split
        for symbol, array in self.fixed_inputs.items():
            array = np.asarray(array)
            if len(array) != len(self.x_raw):
                raise ValueError(f"Length mismatch for fixed input '{symbol}'")
            
            tensor_train = torch.tensor(array[train_idx], dtype=torch.float64).view(-1, 1)
            tensor_val = torch.tensor(array[val_idx], dtype=torch.float64).view(-1, 1)
            tensor_test = torch.tensor(array[test_idx], dtype=torch.float64).view(-1, 1)
            
            self.fixed_torch_inputs_train[str(symbol)] = tensor_train
            self.fixed_torch_inputs_val[str(symbol)] = tensor_val
            self.fixed_torch_inputs_test[str(symbol)] = tensor_test
            
        # Set default to training set for backward compatibility
        self.fixed_torch_inputs = self.fixed_torch_inputs_train

    def _build_model(self):
        """
        Construct the neural network architecture dynamically.
        
        Builds a feedforward neural network with:
        - Input layer: 1 neuron (x parameter)
        - Hidden layers: Specified by hidden_layers parameter with Tanh activations
        - Output layer: 1 neuron (y parameter prediction)
        
        All parameters use double precision (float64). Also initializes learnable
        physical constants and the Adam optimizer.
        """
        # Build network layers dynamically based on hidden_layers
        layers = []
        input_size = 1
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, 1))
        
        self.model = nn.Sequential(*layers).double()
        
        logger.info(f"Built neural network with architecture: [1] -> {self.hidden_layers} -> [1]")

        self.learnable_params = {
            str(k): torch.nn.Parameter(torch.tensor([v], dtype=torch.float64))
            for k, v in self.learn_constants.items()
        }

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.learnable_params.values()),
            lr=1e-3
        )

    def _convert_symbolic_to_residual(self):
        """
        Convert symbolic differential equation to a callable residual function.
        
        Transforms the SymPy equation into a PyTorch-compatible function that
        can compute the physics residual during training. The residual is
        (LHS - RHS) of the differential equation and should be minimized to zero.
        
        Uses sympy.lambdify to convert symbolic expressions to numerical functions,
        preferring PyTorch operations for automatic differentiation.
        """
        residual_expr = sp.simplify(self.differential_eq.lhs - self.differential_eq.rhs)
        symbols = [self.x_sym, self.y_sym] + list(self.learn_constants.keys()) + list(self.fixed_inputs.keys())

        try:
            expr_fn = sp.lambdify(symbols, residual_expr, modules="torch")
        except Exception:
            expr_fn = sp.lambdify(symbols, residual_expr, modules="numpy")

        def residual_fn(x_tensor, y_tensor, fixed_inputs_dict=None):
            if fixed_inputs_dict is None:
                fixed_inputs_dict = self.fixed_torch_inputs
                
            subs = {
                str(self.x_sym): x_tensor,
                str(self.y_sym): y_tensor,
            }

            for k, param in self.learnable_params.items():
                subs[str(k)] = param

            for k, tensor in fixed_inputs_dict.items():
                subs[k] = tensor

            inputs = [subs[str(s)] for s in symbols]
            return expr_fn(*inputs)

        self.physics_residual_fn = residual_fn

    def train(self, epochs=3000, val_interval=100, early_stopping_patience=None):
        """
        Train the Physics-Informed Neural Network.
        
        Optimizes both the neural network parameters and learnable physical constants
        by minimizing a composite loss function that includes both data fitting and
        physics constraint violations.
        
        Parameters
        ----------
        epochs : int, optional
            Number of training iterations. Default: 3000.
        val_interval : int, optional
            Frequency (in epochs) for validation evaluation and logging.
            Default: 100 (validate every 100 epochs).
        early_stopping_patience : int, optional
            Number of validation checks without improvement before stopping.
            If None, training continues for all epochs. Default: None.
        
        Notes
        -----
        The loss function is: Loss = MSE(physics_residual) + MSE(data_error)
        where:
        - physics_residual = differential_eq.lhs - differential_eq.rhs
        - data_error = neural_network_prediction - actual_data
        
        Training progress is logged at validation intervals showing:
        - Training loss (on training set)
        - Validation loss (on validation set)
        - Current values of learnable constants
        
        After training completes, the learned constant values are printed.
        """
        logger.info("Training PINN...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()

            y_PINN = self.model(self.x_torch)
            residual = self.physics_residual_fn(self.x_torch, y_PINN, self.fixed_torch_inputs_train)
            loss_phys = torch.mean(residual ** 2)
            loss_data = torch.mean((y_PINN - self.y_torch) ** 2)
            loss = loss_phys + loss_data

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_log["train_total"].append(loss.item())
            self.loss_log["train_physics"].append(loss_phys.item())
            self.loss_log["train_data"].append(loss_data.item())
            # Backward compatibility
            self.loss_log["total"].append(loss.item())
            self.loss_log["physics"].append(loss_phys.item())
            self.loss_log["data"].append(loss_data.item())

            # Validation
            if epoch % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(self.x_val_torch)
                    residual_val = self.physics_residual_fn(self.x_val_torch, y_val_pred, self.fixed_torch_inputs_val)
                    val_loss_phys = torch.mean(residual_val ** 2)
                    val_loss_data = torch.mean((y_val_pred - self.y_val_torch) ** 2)
                    val_loss = val_loss_phys + val_loss_data

                self.loss_log["val_total"].append(val_loss.item())
                self.loss_log["val_physics"].append(val_loss_phys.item())
                self.loss_log["val_data"].append(val_loss_data.item())

                const_str = ", ".join(
                    f"{k}={v.item():.4f}" for k, v in self.learnable_params.items()
                )
                logger.info(f"Epoch {epoch}: Train Loss = {loss.item():.6e}, Val Loss = {val_loss.item():.6e} | {const_str}")
                
                # Check for early stops
                if early_stopping_patience is not None:
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            break
            elif epoch % 1000 == 0 and epoch % val_interval != 0:
                const_str = ", ".join(
                    f"{k}={v.item():.4f}" for k, v in self.learnable_params.items()
                )
                logger.debug(f"Epoch {epoch}: Train Loss = {loss.item():.6e} | {const_str}")

        # Display results..
        result = {k: v.item() for k, v in self.learnable_params.items()}
        msg = ", ".join(f"{k} = {v:.25f}" for k, v in result.items())
        print(f"\nLearned constants: {msg}")

    def evaluate_test_set(self, verbose=True):
        """
        Evaluate the trained model on the held-out test set.
        
        Computes comprehensive evaluation metrics on all data splits (train, validation, test)
        to assess model performance and detect potential overfitting. Includes quantitative
        metrics such as RMSE, MAE, and reduced χ² for objective model assessment.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, prints a detailed evaluation report. Default: True.
        
        Returns
        -------
        dict
            Dictionary containing evaluation metrics:
            - 'test_loss_total': Total loss on test set
            - 'test_loss_physics': Physics residual loss on test set
            - 'test_loss_data': Data fitting loss on test set
            - 'test_r2': R² score on test set
            - 'test_rmse': Root Mean Squared Error on test set
            - 'test_mae': Mean Absolute Error on test set
            - 'test_chi2_reduced': Reduced χ² on test set
            - 'val_r2', 'val_rmse', 'val_mae': Validation set metrics
            - 'train_r2', 'train_rmse', 'train_mae': Training set metrics
        
        Notes
        -----
        The R² score measures the proportion of variance explained by the model,
        where 1.0 is perfect prediction and negative values indicate poor performance.
        
        RMSE (Root Mean Squared Error) penalizes larger errors more heavily than MAE.
        MAE (Mean Absolute Error) gives equal weight to all errors.
        Reduced χ² measures goodness of fit (values close to 1.0 indicate good fit).
        
        A large difference between train_r2 and test_r2 (> 0.1) suggests overfitting,
        indicating the model memorized training data rather than learning generalizable patterns.
        """
        self.model.eval()
        with torch.no_grad():
            y_test_pred = self.model(self.x_test_torch)
            residual_test = self.physics_residual_fn(self.x_test_torch, y_test_pred, self.fixed_torch_inputs_test)
            test_loss_phys = torch.mean(residual_test ** 2)
            test_loss_data = torch.mean((y_test_pred - self.y_test_torch) ** 2)
            test_loss_total = test_loss_phys + test_loss_data
            
            # R^2
            y_test_mean = torch.mean(self.y_test_torch)
            ss_tot = torch.sum((self.y_test_torch - y_test_mean) ** 2)
            ss_res = torch.sum((self.y_test_torch - y_test_pred) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            
            # RMSE (Root Mean Squared Error)
            test_rmse = torch.sqrt(torch.mean((y_test_pred - self.y_test_torch) ** 2))
            
            # MAE (Mean Absolute Error)
            test_mae = torch.mean(torch.abs(y_test_pred - self.y_test_torch))
            
            # Reduced Chi-Squared (χ²)
            n_test = len(self.y_test_torch)
            n_params = sum(p.numel() for p in self.model.parameters()) + len(self.learnable_params)
            dof = max(n_test - n_params, 1)  # degrees of freedom
            chi2 = torch.sum((y_test_pred - self.y_test_torch) ** 2)
            test_chi2_reduced = chi2 / dof
            
            # Validation Metrics
            y_val_pred = self.model(self.x_val_torch)
            y_val_mean = torch.mean(self.y_val_torch)
            ss_tot_val = torch.sum((self.y_val_torch - y_val_mean) ** 2)
            ss_res_val = torch.sum((self.y_val_torch - y_val_pred) ** 2)
            r2_val = 1 - (ss_res_val / ss_tot_val)
            val_rmse = torch.sqrt(torch.mean((y_val_pred - self.y_val_torch) ** 2))
            val_mae = torch.mean(torch.abs(y_val_pred - self.y_val_torch))
            
            # Training Metrics
            y_train_pred = self.model(self.x_torch)
            y_train_mean = torch.mean(self.y_torch)
            ss_tot_train = torch.sum((self.y_torch - y_train_mean) ** 2)
            ss_res_train = torch.sum((self.y_torch - y_train_pred) ** 2)
            r2_train = 1 - (ss_res_train / ss_tot_train)
            train_rmse = torch.sqrt(torch.mean((y_train_pred - self.y_torch) ** 2))
            train_mae = torch.mean(torch.abs(y_train_pred - self.y_torch))
        
        self.test_metrics = {
            'test_loss_total': test_loss_total.item(),
            'test_loss_physics': test_loss_phys.item(),
            'test_loss_data': test_loss_data.item(),
            'test_r2': r2_score.item(),
            'test_rmse': test_rmse.item(),
            'test_mae': test_mae.item(),
            'test_chi2_reduced': test_chi2_reduced.item(),
            'val_r2': r2_val.item(),
            'val_rmse': val_rmse.item(),
            'val_mae': val_mae.item(),
            'train_r2': r2_train.item(),
            'train_rmse': train_rmse.item(),
            'train_mae': train_mae.item()
        }
        
        if verbose:
            print("\n" + "="*70)
            print("FINAL MODEL EVALUATION - QUANTITATIVE METRICS")
            print("="*70)
            print(f"\nTest Set Performance (unseen data, n={len(self.x_test)}):")
            print(f"  Total Loss:        {self.test_metrics['test_loss_total']:.6e}")
            print(f"  Physics Loss:      {self.test_metrics['test_loss_physics']:.6e}")
            print(f"  Data Loss:         {self.test_metrics['test_loss_data']:.6e}")
            print(f"\n  Goodness of Fit Metrics:")
            print(f"  R² Score:          {self.test_metrics['test_r2']:.6f}")
            print(f"  RMSE:              {self.test_metrics['test_rmse']:.6e}")
            print(f"  MAE:               {self.test_metrics['test_mae']:.6e}")
            print(f"  Reduced χ²:        {self.test_metrics['test_chi2_reduced']:.6f}")
            
            print(f"\nValidation Set Performance (n={len(self.x_val)}):")
            print(f"  R² Score:          {self.test_metrics['val_r2']:.6f}")
            print(f"  RMSE:              {self.test_metrics['val_rmse']:.6e}")
            print(f"  MAE:               {self.test_metrics['val_mae']:.6e}")
            
            print(f"\nTraining Set Performance (n={len(self.x_train)}):")
            print(f"  R² Score:          {self.test_metrics['train_r2']:.6f}")
            print(f"  RMSE:              {self.test_metrics['train_rmse']:.6e}")
            print(f"  MAE:               {self.test_metrics['train_mae']:.6e}")
            
            # Interpretation Guide
            print(f"\n" + "-"*70)
            print("METRIC INTERPRETATION:")
            print(f"  • R² (Coefficient of Determination): {self.test_metrics['test_r2']:.4f}")
            print(f"    → Closer to 1.0 is better (explains {100*self.test_metrics['test_r2']:.2f}% of variance)")
            print(f"  • RMSE (Root Mean Squared Error): {self.test_metrics['test_rmse']:.6e}")
            print(f"    → Lower is better, penalizes large errors")
            print(f"  • MAE (Mean Absolute Error): {self.test_metrics['test_mae']:.6e}")
            print(f"    → Lower is better, average prediction error")
            print(f"  • Reduced χ²: {self.test_metrics['test_chi2_reduced']:.4f}")
            print(f"    → Values close to 1.0 indicate good fit")
            if self.test_metrics['test_chi2_reduced'] < 1.0:
                print(f"    → Model may be overfitting the data")
            elif self.test_metrics['test_chi2_reduced'] > 2.0:
                print(f"    → Model may be underfitting or systematic errors present")
            else:
                print(f"    → Good model fit achieved")
            print("-"*70)
            
            # Check for overfitting
            if self.test_metrics['train_r2'] - self.test_metrics['test_r2'] > 0.1:
                print(f"\n⚠ WARNING: Possible overfitting detected!")
                print(f"  Training R² - Test R² = {self.test_metrics['train_r2'] - self.test_metrics['test_r2']:.4f}")
                print(f"  Consider: reducing model complexity, adding regularization, or")
                print(f"           collecting more training data")
            else:
                print(f"\n✓ Good generalization: Train/Test R² difference = {self.test_metrics['train_r2'] - self.test_metrics['test_r2']:.4f}")
                
            print("="*70)
        
        return self.test_metrics

    def predict_extended(self, extend=0.5, n_points=300):
        """
        Generate smooth predictions over an extended range.
        
        Creates predictions beyond the training data range for visualization
        and extrapolation purposes.
        
        Parameters
        ----------
        extend : float, optional
            How much to extend beyond the data range (in log units if log_scale=True).
            Default: 0.5.
        n_points : int, optional
            Number of evenly-spaced prediction points. Default: 300.
        
        Returns
        -------
        x_values : ndarray
            Shape (n_points,) array of x coordinates.
        y_predictions : ndarray
            Shape (n_points,) array of predicted y values.
        """
        with torch.no_grad():
            x_min, x_max = self.x_torch.min().item(), self.x_torch.max().item()
            x_PINN = torch.linspace(x_min - extend, x_max + extend, n_points, dtype=torch.float64).view(-1, 1)
            y_PINN = self.model(x_PINN).numpy()
        return x_PINN.numpy(), y_PINN

    def store_learned_constants(self):
        """
        Retrieve the learned physical constant values.
        
        Returns
        -------
        dict
            Dictionary mapping constant names (as strings) to their learned values (as floats).
        """
        result = {k: v.item() for k, v in self.learnable_params.items()}
        return result

    def set_learn_constants(self, new_constants: dict):
        """
        Update or add learnable constants with new initial values.
        
        Useful for resuming training with different initial guesses or adding
        new constants to learn. Reinitializes the optimizer to include new parameters.
        
        Parameters
        ----------
        new_constants : dict
            Dictionary mapping constant names (str) to new values (float).
            Can add new constants or update existing ones.
        """
        for k, v in new_constants.items():
            if k not in self.learnable_params:
                param = torch.nn.Parameter(torch.tensor([v], dtype=torch.float64))
                self.learnable_params[k] = param
                logger.info(f"Added new learnable constant: {k} = {v:.6f}")
            else:
                self.learnable_params[k].data = torch.tensor([v], dtype=torch.float64)
                logger.info(f"Updated constant: {k} = {v:.6f}")

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.learnable_params.values()),
            lr=1e-3
        )

    def recommend_initial_guesses(self, method="mean"):
        """
        Recommend initial guesses for learnable constants based on training data.
        
        Analyzes the training data to suggest reasonable starting values for
        physical constants, which can improve convergence.
        
        Parameters
        ----------
        method : str, optional
            Method for generating recommendations:
            - 'mean': Use mean(y) / mean(x) as scale factor
            - 'regression': Linear regression to estimate slope/intercept
            - 'ols_loglog': Ordinary least squares in log-log space
            - 'zero': Initialize all constants to 0.0
            Default: 'mean'.
        
        Returns
        -------
        dict
            Dictionary mapping constant names to recommended initial values.
        
        Notes
        -----
        Recommendations are printed to stdout and returned as a dictionary.
        These are suggestions only; the actual initial values are set during
        instantiation via the learn_constants parameter.
        """
        x = self.x_train
        y = self.y_train
        recommended = {}

        if method == "mean":
            scale_factor = np.mean(y) / np.mean(x)
            for k in self.learn_constants:
                recommended[k] = scale_factor

        elif method == "regression":
            model = LinearRegression().fit(x.reshape(-1, 1), y)
            slope = model.coef_[0]
            intercept = model.intercept_
            for k in self.learn_constants:
                name = str(k).lower()
                recommended[k] = slope if "slope" in name or "n" in name else intercept

        elif method == "ols_loglog":
            X = np.vstack([x, np.ones_like(x)]).T
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            slope, intercept = coeffs
            for k in self.learn_constants:
                name = str(k).lower()
                recommended[k] = slope if "slope" in name or "n" in name else intercept

        elif method == "zero":
            for k in self.learn_constants:
                recommended[k] = 0.0

        else:
            raise ValueError(f"Unknown method '{method}'.")

        print("Recommended initial guesses (based on training data only):")
        for k, v in recommended.items():
            print(f"  {k} ≈ {v:.6e}")
        return recommended

    def plot_PINN_loss(self, log=True):
        """
        Plot training and validation loss curves.
        
        Creates a two-panel figure showing:
        - Left: Total loss (training and validation)
        - Right: Loss components (physics and data losses)
        
        Parameters
        ----------
        log : bool, optional
            If True, use logarithmic scale for y-axis. Default: True.
            Recommended for visualizing loss over many orders of magnitude.
        
        Notes
        -----
        This plot helps diagnose training behavior:
        - Decreasing losses indicate successful learning
        - Diverging train/val losses suggest overfitting
        - Oscillating losses may indicate learning rate is too high
        - Dominant physics loss means the model struggles to satisfy the equation
        - Dominant data loss means predictions don't match observations
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total Loss
        epochs_train = range(len(self.loss_log["train_total"]))
        ax1.plot(epochs_train, self.loss_log["train_total"], label='Train Total', linewidth=2, alpha=0.8)
        
        if self.loss_log["val_total"]:
            val_interval = len(self.loss_log["train_total"]) // len(self.loss_log["val_total"])
            epochs_val = range(0, len(self.loss_log["train_total"]), val_interval)
            epochs_val = list(epochs_val)[:len(self.loss_log["val_total"])]

            ax1.plot(epochs_val, self.loss_log["val_total"], label='Val Total', linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        if log:
            ax1.set_yscale('log')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Total Loss: Train vs Validation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Physics & Data Loss
        ax2.plot(epochs_train, self.loss_log["train_physics"], label='Train Physics', linestyle='--', alpha=0.7)
        ax2.plot(epochs_train, self.loss_log["train_data"], label='Train Data', linestyle='--', alpha=0.7)
        
        if self.loss_log["val_physics"]:
            ax2.plot(epochs_val, self.loss_log["val_physics"], label='Val Physics', linestyle=':', marker='s', markersize=3, alpha=0.7)
            ax2.plot(epochs_val, self.loss_log["val_data"], label='Val Data', linestyle=':', marker='s', markersize=3, alpha=0.7)
        
        if log:
            ax2.set_yscale('log')

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_PINN(self, show_splits=True):
        """
        Plot PINN predictions against pulsar data.
        
        Visualizes the trained model's predictions overlaid on the actual pulsar
        data from the ATNF catalogue.
        
        Parameters
        ----------
        show_splits : bool, optional
            If True, color-code data points by split (train/val/test).
            If False, show all data as a single series. Default: True.
        
        Notes
        -----
        The plot includes:
        - Scatter points: Actual pulsar data
        - Red line: PINN model predictions (extended beyond data range)
        - Legend with sample sizes
        - Test R² score in title (if evaluate_test_set() was called)
        
        When show_splits=True:
        - Blue points: Training data (used for fitting)
        - Orange points: Validation data (monitored during training)
        - Green points: Test data (held out for final evaluation)
        """
        x_PINN, y_PINN = self.predict_extended()
        
        plt.figure(figsize=(10, 6))
        
        if show_splits:
            plt.scatter(self.x_train, self.y_train, label=f'Train (n={len(self.x_train)})', s=10, alpha=0.5, c='blue')
            plt.scatter(self.x_val, self.y_val, label=f'Val (n={len(self.x_val)})', s=10, alpha=0.5, c='orange')
            plt.scatter(self.x_test, self.y_test, label=f'Test (n={len(self.x_test)})', s=10, alpha=0.5, c='green')
        else:
            plt.scatter(self.x_all, self.y_all, label='ATNF Data', s=10, alpha=0.5)
            
        plt.plot(x_PINN, y_PINN, color='red', label='PINN Prediction', linewidth=2)
        plt.xlabel(f"log10({self.x_param})" if self.log_scale else self.x_param)
        plt.ylabel(f"log10({self.y_param})" if self.log_scale else self.y_param)
        
        title = 'PINN Prediction vs Pulsar Data'
        if hasattr(self, 'test_metrics') and self.test_metrics:
            title += f"\nTest R² = {self.test_metrics['test_r2']:.4f}"

        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
