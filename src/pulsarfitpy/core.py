import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from psrqpy import QueryATNF
import matplotlib.pyplot as plt

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
            print("\nFitting Polynomial Approximation...")
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
                print(f"Degree {degree} → R² Score: {score:.6f}")

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
                 fixed_inputs: dict = None):

        self.x_param = x_param
        self.y_param = y_param
        self.differential_eq = differential_eq
        self.x_sym = x_sym
        self.y_sym = y_sym
        self.learn_constants = learn_constants or {}
        self.log_scale = log_scale
        self.psrqpy_filter_fn = psrqpy_filter_fn

        self.fixed_inputs = fixed_inputs or {}         # symbolic: np arrays
        self.fixed_torch_inputs = {}                   # str(symbolic): torch tensors
        self.learnable_params = {}
        self.loss_log = {"total": [], "physics": [], "data": []}

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

        self.x_torch = torch.tensor(self.x_raw, dtype=torch.float64).view(-1, 1).requires_grad_(True)
        self.y_torch = torch.tensor(self.y_raw, dtype=torch.float64).view(-1, 1)

        # Convert fixed inputs to torch tensors
        for symbol, array in self.fixed_inputs.items():
            array = np.asarray(array)
            if len(array) != len(self.x_raw):
                raise ValueError(f"Length mismatch for fixed input '{symbol}'")
            tensor = torch.tensor(array, dtype=torch.float64).view(-1, 1)
            self.fixed_torch_inputs[str(symbol)] = tensor

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).double()

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

        def residual_fn(x_tensor, y_tensor):
            subs = {
                str(self.x_sym): x_tensor,
                str(self.y_sym): y_tensor,
            }

            for k, param in self.learnable_params.items():
                subs[str(k)] = param

            for k, tensor in self.fixed_torch_inputs.items():
                subs[k] = tensor

            expr_fn = sp.lambdify(symbols, residual_expr, modules="torch")
            inputs = [subs[str(s)] for s in symbols]
            return expr_fn(*inputs)

        self.physics_residual_fn = residual_fn

    def train(self, epochs=3000):
        print("Training PINN...\n")
        for epoch in range(epochs):
            y_PINN = self.model(self.x_torch)
            residual = self.physics_residual_fn(self.x_torch, y_PINN)
            loss_phys = torch.mean(residual ** 2)
            loss_data = torch.mean((y_PINN - self.y_torch) ** 2)
            loss = loss_phys + loss_data

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_log["total"].append(loss.item())
            self.loss_log["physics"].append(loss_phys.item())
            self.loss_log["data"].append(loss_data.item())

            if epoch % 1000 == 0:
                const_str = ", ".join(
                    f"{k}={v.item():.4f}" for k, v in self.learnable_params.items()
                )
                print(f"Epoch {epoch}: Loss = {loss.item():.10e} | {const_str}")

        # Print Result        
        result = {k: v.item() for k, v in self.learnable_params.items()}
        msg = ", ".join(f"{k} = {v:.25f}" for k, v in result.items())
        print(f"\nLearned constants: {msg}")

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
                print(f"Added new learnable constant: {k} = {v:.6f}")
            else:
                self.learnable_params[k].data = torch.tensor([v], dtype=torch.float64)
                print(f"Updated constant: {k} = {v:.6f}")

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.learnable_params.values()),
            lr=1e-3
        )

    def recommend_initial_guesses(self, method="mean"):
        x = self.x_raw
        y = self.y_raw
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

        print("Recommended initial guesses:")
        for k, v in recommended.items():
            print(f"  {k} ≈ {v:.6e}")
        return recommended

    def plot_PINN_loss(self, log=True):
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_log["total"], label='Total Loss', linewidth=2)
        plt.plot(self.loss_log["physics"], label='Physics Loss', linestyle='--')
        plt.plot(self.loss_log["data"], label='Data Loss', linestyle='--')
        if log:
            plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_PINN(self):
        x_PINN, y_PINN = self.predict_extended()
        x_data = self.x_raw
        y_data = self.y_raw

        plt.figure(figsize=(8, 5))
        plt.scatter(x_data, y_data, label='ATNF Data', s=10, alpha=0.5)
        plt.plot(x_PINN, y_PINN, color='red', label='PINN Prediction', linewidth=2)
        plt.xlabel(f"log10({self.x_param})" if self.log_scale else self.x_param)
        plt.ylabel(f"log10({self.y_param})" if self.log_scale else self.y_param)
        plt.title('PINN Prediction vs Pulsar Data')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()