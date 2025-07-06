import numpy as np
import torch
import torch.nn as nn
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from psrqpy import QueryATNF

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

        self._process_query_data()

    # Filters query for sklearn approximations
    def _process_query_data(self):
        table = self.query.table
        x_vals = np.array(table[self.x_param], dtype=float)
        y_vals = np.array(table[self.y_param], dtype=float)

        # Masking for filtering values
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if self.log_x:
            mask &= x_vals > 0
        if self.log_y:
            mask &= y_vals > 0

        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

        if len(x_vals) == 0:
            raise ValueError("No valid data points found: check your query parameters.")

        if self.log_x:
            x_vals = np.log(x_vals)
        if self.log_y:
            y_vals = np.log(y_vals)

        self.x_data = x_vals.reshape(-1, 1)
        self.y_data = y_vals
        self.query_table = table[mask]

    # Fits polynomial to a logarithmic scaled graph
    def fit_polynomial(self):
        best_score = float('-inf')
        best_model = None
        best_degree = None

        for degree in range(1, self.test_degree + 1):
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('reg', LinearRegression())
            ])
            pipeline.fit(self.x_data, self.y_data)
            y_pred = pipeline.predict(self.x_data)
            score = r2_score(self.y_data, y_pred)

            print(f"Degree {degree} → R² Score: {score:.6f}")

            if score > best_score:
                best_score = score
                best_model = pipeline
                best_degree = degree

        self.model = best_model
        self.best_degree = best_degree
        self.coefficients = self.model.named_steps['reg'].coef_
        self.intercept = self.model.named_steps['reg'].intercept_

        self.predicted_x = np.linspace(self.x_data.min(), self.x_data.max(), 100).reshape(-1, 1)
        self.predicted_y = self.model.predict(self.predicted_x)

    # Generates polynomial expression
    def get_polynomial_expression(self):
        terms = [f"{self.intercept:.10f}"]
        for i, coef in enumerate(self.coefficients[1:], start=1):
            terms.append(f"{coef:.10f} * x**{i}")
        return " + ".join(terms)

    # Prints polynomial expression
    def print_polynomial(self):
        poly_expr = self.get_polynomial_expression()
        print(f"\nBest Polynomial Degree: {self.best_degree}")
        print(f"\nApproximated Polynomial Function:\nf(x) = {poly_expr}")

# Pulsar PINN Prediction Class + Framework
class PulsarPINN:
    def __init__(self, x_param: str, y_param: str,
                 physics_eq: sp.Eq,
                 x_sym: sp.Symbol, y_sym: sp.Symbol,
                 learn_constant: dict = None,
                 log_scale=True,
                 psrqpy_filter_fn=None):
        """
        Parameters:
        - x_param, y_param: ATNF pulsar parameters (e.g., 'P0', 'EDOT')
        - physics_eq: sympy equation like Eq(lhs, rhs) to represent physics
        - x_sym, y_sym: symbols used for input/output (e.g., logP, logEDOT)
        - learn_constant: dict of {symbol: initial_guess} for learnable constants
        - log_scale: whether to use log10 scaling
        """
        self.x_param = x_param
        self.y_param = y_param
        self.physics_eq = physics_eq
        self.x_sym = x_sym
        self.y_sym = y_sym
        self.learn_constant = learn_constant or {}
        self.log_scale = log_scale
        self.psrqpy_filter_fn = psrqpy_filter_fn

        self.learnable_params = {}
        self._prepare_data()
        self._build_model()
        self._convert_symbolic_to_residual()

    # Filter data for PyTorch use
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

    # Add layers
    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).double()

        # Create learnable torch parameters based on missing constants
        self.learnable_params = {
            str(k): torch.nn.Parameter(torch.tensor([v], dtype=torch.float64))
            for k, v in self.learn_constant.items()
        }

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.learnable_params.values()),
            lr=1e-3
        )

    # Converts the symbolic sympy equation into a callable PyTorch residual function
    def _convert_symbolic_to_residual(self):
        residual_expr = sp.simplify(self.physics_eq.lhs - self.physics_eq.rhs)
        symbols = [self.x_sym, self.y_sym] + list(self.learn_constant.keys())

        def residual_fn(x_tensor, y_tensor):
            subs = {
                str(self.x_sym): x_tensor,
                str(self.y_sym): y_tensor,
            }
            for k, param in self.learnable_params.items():
                subs[k] = param

            # Convert symbolic expression to a PyTorch-callable function
            expr_fn = sp.lambdify(symbols, residual_expr, modules="torch")
            inputs = [subs[str(s)] for s in symbols]
            return expr_fn(*inputs)

        self.physics_residual_fn = residual_fn

    # Train the model
    def train(self, epochs=3000):
        for epoch in range(epochs):
            y_pred = self.model(self.x_torch)
            residual = self.physics_residual_fn(self.x_torch, y_pred)
            loss_phys = torch.mean(residual**2)
            loss_data = torch.mean((y_pred - self.y_torch)**2)
            loss = loss_phys + loss_data

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 1000 == 0:
                const_str = ", ".join(
                    f"{k}={v.item():.4f}" for k, v in self.learnable_params.items()
                )
                print(f"Epoch {epoch}: Loss = {loss.item():.10e} | {const_str}")

    # Predict extended pattern of model to continue graph
    def predict_extended(self, extend=0.5, n_points=300):
        with torch.no_grad():
            x_min, x_max = self.x_torch.min().item(), self.x_torch.max().item()
            x_grid = torch.linspace(x_min - extend, x_max + extend, n_points, dtype=torch.float64).view(-1, 1)
            y_pred = self.model(x_grid).numpy()
        return x_grid.numpy(), y_pred

    # Print learned constants in terminal
    def show_learned_constants(self):
        result = {k: v.item() for k, v in self.learnable_params.items()}
        msg = ", ".join(f"{k} = {v:.10f}" for k, v in result.items())
        print(f"\nLearned constants: {msg}")
        return result
    
# Example Usage

# Define alternate symbolic model
logP2, logEDOT2, logC2 = sp.symbols("logP2 logEDOT2 logC2")
equation2 = sp.Eq(logEDOT2, logC2 - 4 * logP2)

# Run the model 
pinn2 = PulsarPINN(
    x_param="P0",
    y_param="EDOT",
    physics_eq=equation2,
    x_sym=logP2,
    y_sym=logEDOT2,
    learn_constant={logC2: 38},  # Starting Guess
    log_scale=True
)

pinn2.train(epochs=3000)                    # Train model
x2_ext, y2_ext = pinn2.predict_extended()   # Predict extended pattern based on PINN
pinn2.show_learned_constants()              # Print learned constant

# Plot result
plt.figure(figsize=(8, 6))
plt.scatter(pinn2.x_raw, pinn2.y_raw, s=10, alpha=0.5, label="Pulsars")
plt.plot(x2_ext, y2_ext, label="PINN Fit", color="green", linewidth=2)
plt.xlabel("log10(Period [s])")
plt.ylabel("log10(Spin-down Power [erg/s])")
plt.title("PINN Model (Dipole Radiation) with Learned logC2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()