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

    # Plots query
    def _process_query_data(self):
        table = self.query.table
        x_vals = np.array(table[self.x_param], dtype=float)
        y_vals = np.array(table[self.y_param], dtype=float)

        # Masking for log values
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

    # Fits polynomial to a log-log graph
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

    # Prints polynomial
    def print_polynomial(self):
        poly_expr = self.get_polynomial_expression()
        print(f"\nBest Polynomial Degree: {self.best_degree}")
        print(f"\nApproximated Polynomial Function:\nf(x) = {poly_expr}")

# sympy Variables to PyTorch residuals

def symbolic_torch(sym_eq: sp.Eq, x_sym, y_sym):
    """
    Converts a sympy equation Eq(lhs, rhs) to a PyTorch autograd-compatible residual function.

    Parameters:
    - sym_eq: sympy Eq(lhs, rhs)
    - x_sym: sympy Symbol for input (e.g. logP)
    - y_sym: sympy Symbol for predicted output (e.g. logEDOT)

    Returns:
    - A function of (x_tensor, y_predicted_tensor) → residual_tensor
    """
    # Ensure lhs - rhs = 0 format
    residual_expr = sp.simplify(sym_eq.lhs - sym_eq.rhs)

    # Convert SymPy to Python function using PyTorch
    def residual_fn(x_tensor, y_tensor):
        x = x_tensor.clone().detach().requires_grad_(True)
        y = y_tensor.clone().detach().requires_grad_(True)

        # Define symbolic expressions in PyTorch manually
        # Use torch.log10, torch.pow, etc., and substitute values
        # Assume scalar broadcasting
        locals_dict = {
            str(x_sym): x,
            str(y_sym): y,
            "log": torch.log10,
            "pi": torch.tensor(np.pi, dtype=torch.float64),
        }

        # Recompile SymPy expression into PyTorch
        residual_lambda = sp.lambdify((x_sym, y_sym), residual_expr, modules='torch')
        residual = residual_lambda(x, y)

        return residual

    return residual_fn

# Pulsar PINN class + framework
class PulsarPINN:
    def __init__(self, x_param: str, y_param: str, physics_residual_fn, log_scale=True, psrqpy_filter_fn=None):
        """
        Parameters:
        - x_param, y_param: ATNF pulsar parameters (e.g., 'P0', 'EDOT')
        - physics_residual_fn: function of (x_tensor, y_predicted_tensor) → residual_tensor
        - log_scale: whether to log10 transform input/output
        - psrqpy_filter_fn: optional filter function on raw x/y
        """
        self.x_param = x_param
        self.y_param = y_param
        self.physics_residual_fn = physics_residual_fn
        self.log_scale = log_scale
        self.psrqpy_filter_fn = psrqpy_filter_fn

        self._prepare_data()
        self._build_model()

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

        self.pinnX = np.log10(x) if self.log_scale else x
        self.pinnY = np.log10(y) if self.log_scale else y

        self.x_torch = torch.tensor(self.pinnX, dtype=torch.float64).view(-1, 1).requires_grad_(True)
        self.y_torch = torch.tensor(self.pinnY, dtype=torch.float64).view(-1, 1)

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).double()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

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

            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Total Loss = {loss.item():.4e}")

    def predict_extended(self, extend=0.5, n_points=300):
        with torch.no_grad():
            x_min, x_max = self.x_torch.min().item(), self.x_torch.max().item()
            x_grid = torch.linspace(x_min - extend, x_max + extend, n_points, dtype=torch.float64).view(-1, 1)
            y_pred = self.model(x_grid).numpy()
        return x_grid.numpy(), y_pred

# EXAMPLE PINN USAGE

# Symbolic setup for residual: logPdot = logEDOT - logC + 3*logP
logP, logEDOT = sp.symbols("logP logEDOT")
I = 1e45
C = 4 * np.pi**2 * I
logC = np.log10(C)

logPdot_expr = logEDOT - logC + 3 * logP
residual_eq = sp.Eq(logP, logPdot_expr)

# Convert symbolic equation to torch autograd-compatible residual
torch_residual_fn = symbolic_torch(residual_eq, logP, logEDOT)

# Create and train the PINN
pinn = PulsarPINN("P0", "EDOT", physics_residual_fn=torch_residual_fn)
pinn.train(epochs=10000)

# Predict extended range
x_ext, y_ext = pinn.predict_extended(extend=0.5)

# User-defined plot
plt.figure(figsize=(10, 6))
plt.scatter(pinn.pinnX, pinn.pinnY, s=10, alpha=0.5, label="Observed")
plt.plot(x_ext, y_ext, label="PINN", color="green", linewidth=2)
plt.xlabel("log(Period) [log(s)]")
plt.ylabel("log(Spin-down Energy Loss) [log(erg/s)])")
plt.title("PINN model of Spin-Down Luminosity vs. Period")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()