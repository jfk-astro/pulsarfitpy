import numpy as np
import torch
import torch.nn as nn
import sympy as sp
import matplotlib.pyplot as plt
import logging

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from psrqpy import QueryATNF

logger = logging.getLogger(__name__)

def configure_logging(level='INFO', format_string=None, log_file=None):
    """
    Configure logging for pulsarfitpy.
    
    Parameters
    ----------
    level : str, optional
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        Default is 'INFO'.

    format_string : str, optional
        Custom format string for log messages.
        Default is '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    log_file : str, optional
        Path to log file. If None, logs to console only.
    """

    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    pulsarfitpy_logger = logging.getLogger('pulsarfitpy')
    pulsarfitpy_logger.setLevel(log_level)
    
    pulsarfitpy_logger.handlers.clear()
    
    formatter = logging.Formatter(format_string)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    pulsarfitpy_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        pulsarfitpy_logger.addHandler(file_handler)
        pulsarfitpy_logger.info(f"Logging to file: {log_file}")
    
    pulsarfitpy_logger.info(f"Logging configured at level: {level.upper()}")

# Pulsar Polynomial Approximation class
class PulsarApproximation:
    def __init__(self, query: QueryATNF, x_param: str, y_param: str, test_degree: int, log_x=True, log_y=True):
        self.query = query
        self.x_param = x_param
        self.y_param = y_param
        self.test_degree = test_degree
        self.log_x = log_x
        self.log_y = log_y

        self.query_table = None
        self.x_data = None
        self.y_data = None
        self.model = None
        self.best_degree = None
        self.coefficients = None
        self.intercept = None
        self.predicted_x = None
        self.predicted_y = None
        self.r2_scores = {}

        self._process_query_data()

    def _process_query_data(self):
        table = self.query.table
        x_vals = np.array(table[self.x_param], dtype=float)
        y_vals = np.array(table[self.y_param], dtype=float)

        # Filter out invalid values
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if self.log_x:
            mask &= x_vals > 0

        if self.log_y:
            mask &= y_vals > 0

        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

        if len(x_vals) == 0:
            raise ValueError("No valid data points found.")

        if self.log_x:
            x_vals = np.log10(x_vals)

        if self.log_y:
            y_vals = np.log10(y_vals)

        self.x_data = x_vals.reshape(-1, 1)
        self.y_data = y_vals
        self.query_table = table[mask]

    def fit_polynomial(self, verbose=True):
        if verbose:
            logger.info("Fitting Polynomial Approximation...")
        best_score = float('-inf')

        for degree in range(1, self.test_degree + 1):
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('reg', LinearRegression())
            ])
            pipeline.fit(self.x_data, self.y_data)
            y_PINN = pipeline.predict(self.x_data)
            
            score = r2_score(self.y_data, y_PINN)
            self.r2_scores[degree] = score

            if verbose:
                logger.info(f"Degree {degree} → R² Score: {score:.6f}")

            if score > best_score:
                best_score = score
                self.model = pipeline
                self.best_degree = degree

        self.coefficients = self.model.named_steps['reg'].coef_
        self.intercept = self.model.named_steps['reg'].intercept_

        self.predicted_x = np.linspace(self.x_data.min(), self.x_data.max(), 100).reshape(-1, 1)
        self.predicted_y = self.model.predict(self.predicted_x)

    def get_polynomial_expression(self):
        terms = [f"{self.intercept:.10f}"]
        for i, coef in enumerate(self.coefficients[1:], start=1):
            terms.append(f"{coef:.10f} * x**{i}")
        
        return " + ".join(terms)

    def print_polynomial(self):
        poly_expr = self.get_polynomial_expression()

        print(f"\nBest Polynomial Degree: {self.best_degree}")
        print(f"Approximated Polynomial Function:\nf(x) = {poly_expr}")

    def plot_r2_scores(self):
        if not self.r2_scores:
            raise RuntimeError("Run `fit_polynomial()` first.")
        
        degrees = list(self.r2_scores.keys())
        scores = list(self.r2_scores.values())

        plt.figure(figsize=(8, 5))
        plt.plot(degrees, scores, marker='o', linestyle='-', color='turquoise')
        plt.title("R² Score vs Polynomial Degree")
        plt.xlabel("Polynomial Degree")
        plt.ylabel("R² Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_approximation_curve(self):
        if self.predicted_x is None or self.predicted_y is None:
            raise RuntimeError("Run `fit_polynomial()` first.")

        plt.figure(figsize=(8, 5))
        plt.scatter(self.x_data, self.y_data, s=10, alpha=0.4, label='Pulsars')
        plt.plot(self.predicted_x, self.predicted_y, color='navy', label=f'Degree {self.best_degree} Fit')
        plt.xlabel(f"log({self.x_param})" if self.log_x else self.x_param)
        plt.ylabel(f"log({self.y_param})" if self.log_y else self.y_param)
        plt.title("Polynomial Fit of Pulsar Data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_combined_analysis(self):
        if self.predicted_x is None or self.predicted_y is None:
            raise RuntimeError("Run `fit_polynomial()` first.")
        if not self.r2_scores:
            raise RuntimeError("R² scores are empty. Run `fit_polynomial()` first.")

        degrees = list(self.r2_scores.keys())
        scores = list(self.r2_scores.values())

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Polynomial Fit
        axs[0].scatter(self.x_data, self.y_data, s=10, alpha=0.5, label='Pulsars')
        axs[0].plot(self.predicted_x, self.predicted_y, color='navy', label=f'Degree {self.best_degree} Fit')
        axs[0].set_xlabel(f"log({self.x_param})" if self.log_x else self.x_param)
        axs[0].set_ylabel(f"log({self.y_param})" if self.log_y else self.y_param)
        axs[0].set_title("Polynomial Fit of Pulsar Data")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: R² vs Degree
        axs[1].plot(degrees, scores, marker='o', linestyle='-', color='turquoise')
        axs[1].set_xlabel("Polynomial Degree")
        axs[1].set_ylabel("R² Score")
        axs[1].set_title("R² Score vs Polynomial Degree")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

# Pulsar PINN Prediction Class + Framework
class PulsarPINN:
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
                 hidden_layers=None):  # NEW PARAMETER

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
        """Evaluate the trained model on the held-out test set."""
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
            
            # Validation Metrics
            y_val_pred = self.model(self.x_val_torch)
            y_val_mean = torch.mean(self.y_val_torch)
            ss_tot_val = torch.sum((self.y_val_torch - y_val_mean) ** 2)
            ss_res_val = torch.sum((self.y_val_torch - y_val_pred) ** 2)
            r2_val = 1 - (ss_res_val / ss_tot_val)
            
            # Training Metrics
            y_train_pred = self.model(self.x_torch)
            y_train_mean = torch.mean(self.y_torch)
            ss_tot_train = torch.sum((self.y_torch - y_train_mean) ** 2)
            ss_res_train = torch.sum((self.y_torch - y_train_pred) ** 2)
            r2_train = 1 - (ss_res_train / ss_tot_train)
        
        self.test_metrics = {
            'test_loss_total': test_loss_total.item(),
            'test_loss_physics': test_loss_phys.item(),
            'test_loss_data': test_loss_data.item(),
            'test_r2': r2_score.item(),
            'val_r2': r2_val.item(),
            'train_r2': r2_train.item()
        }
        
        if verbose:
            print("\n" + "="*60)
            print("FINAL MODEL EVALUATION")
            print("="*60)
            print(f"\nTest Set Performance (unseen data, n={len(self.x_test)}):")
            print(f"  Total Loss:     {self.test_metrics['test_loss_total']:.6e}")
            print(f"  Physics Loss:   {self.test_metrics['test_loss_physics']:.6e}")
            print(f"  Data Loss:      {self.test_metrics['test_loss_data']:.6e}")
            print(f"  R² Score:       {self.test_metrics['test_r2']:.6f}")
            print(f"\nValidation Set Performance (n={len(self.x_val)}):")
            print(f"  R² Score:       {self.test_metrics['val_r2']:.6f}")
            print(f"\nTraining Set Performance (n={len(self.x_train)}):")
            print(f"  R² Score:       {self.test_metrics['train_r2']:.6f}")
            
            # Check for overfitting
            if self.test_metrics['train_r2'] - self.test_metrics['test_r2'] > 0.1:
                print(f"\n⚠ WARNING: Possible overfitting detected!")
                print(f"  Training R² - Test R² = {self.test_metrics['train_r2'] - self.test_metrics['test_r2']:.4f}")
            else:
                print(f"\n✓ Good generalization: Train/Test R² difference = {self.test_metrics['train_r2'] - self.test_metrics['test_r2']:.4f}")
                
            print("="*60)
        
        return self.test_metrics

    def predict_extended(self, extend=0.5, n_points=300):
        with torch.no_grad():
            x_min, x_max = self.x_torch.min().item(), self.x_torch.max().item()
            x_PINN = torch.linspace(x_min - extend, x_max + extend, n_points, dtype=torch.float64).view(-1, 1)
            y_PINN = self.model(x_PINN).numpy()
        return x_PINN.numpy(), y_PINN

    def store_learned_constants(self):
        result = {k: v.item() for k, v in self.learnable_params.items()}
        return result

    def set_learn_constants(self, new_constants: dict):
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
        """Plot training and validation loss curves."""
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
        """Plot PINN predictions with optional train/val/test split visualization."""
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