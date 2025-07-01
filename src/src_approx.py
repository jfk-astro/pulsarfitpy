import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from psrqpy import QueryATNF

# PULSAR POLYNOMIAL APPROXIMATION
class psrApprox:
    def __init__(self, query: QueryATNF, x_param: str, y_param: str, test_degree: int, log_x=False, log_y=False):
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

    def _process_query_data(self):
        table = self.query.table
        x_vals = np.array(table[self.x_param], dtype=float)
        y_vals = np.array(table[self.y_param], dtype=float)

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

    def get_polynomial_expression(self):
        terms = [f"{self.intercept:.10f}"]
        for i, coef in enumerate(self.coefficients[1:], start=1):
            terms.append(f"{coef:.10f} * x**{i}")
        return " + ".join(terms)

# PULSAR PLOTTING CLASS
class pulsar_plotter:
    def __init__(self, approximator: psrApprox):
        self.approximator = approximator

    def plot(self, show_data=True, show_poly=True, annotate=True):
        plt.figure(figsize=(10, 6))

        if show_data:
            plt.scatter(self.approximator.x_data, self.approximator.y_data, s=10, alpha=0.6, label='Pulsars')

        if show_poly:
            plt.plot(
                self.approximator.predicted_x,
                self.approximator.predicted_y,
                color='red',
                label=f'Polynomial Approx (deg={self.approximator.best_degree})'
            )

        x_label = f"log({self.approximator.x_param})" if self.approximator.log_x else self.approximator.x_param
        y_label = f"log({self.approximator.y_param})" if self.approximator.log_y else self.approximator.y_param

        plt.title(f'{y_label} vs. {x_label}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()

        if annotate:
            poly_expr = self.approximator.get_polynomial_expression()
            print(f"\nBest Polynomial Degree: {self.approximator.best_degree}")
            print(f"\nApproximated Polynomial Function:\nf(x) = {poly_expr}")

        plt.show()

# Example usage
query = QueryATNF(params=['JNAME', 'P0', 'P1'], 
                  condition='exist(P0) && exist(BSURF)')

approximator = psrApprox(query=query,x_param='P0',
                         y_param='P1',
                         test_degree=10,
                         log_x=True,
                         log_y=True)

approximator.fit_polynomial()
plotter = pulsar_plotter(approximator)
plotter.plot()