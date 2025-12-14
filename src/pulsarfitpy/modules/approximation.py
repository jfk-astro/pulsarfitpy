"""
Polynomial approximation for pulsar data.

This module provides the PulsarApproximation class for fitting polynomial models
to pulsar parameter relationships using scikit-learn.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from psrqpy import QueryATNF

logger = logging.getLogger(__name__)


class PulsarApproximation:
    """
    Polynomial approximation for pulsar parameter relationships.
    
    This class fits polynomial models to pulsar data from the ATNF catalogue
    and provides visualization and analysis tools.
    
    Parameters
    ----------
    query : QueryATNF
        ATNF pulsar catalogue query object
    x_param : str
        Name of the x-axis parameter from ATNF catalogue
    y_param : str
        Name of the y-axis parameter from ATNF catalogue
    test_degree : int
        Maximum polynomial degree to test
    log_x : bool, optional
        Whether to use log10 scale for x parameter (default: True)
    log_y : bool, optional
        Whether to use log10 scale for y parameter (default: True)
        
    Attributes
    ----------
    model : Pipeline
        Best-fit polynomial model (scikit-learn Pipeline)
    best_degree : int
        Degree of the best-fit polynomial
    r2_scores : dict
        R² scores for each tested polynomial degree
    coefficients : ndarray
        Coefficients of the best-fit polynomial
    intercept : float
        Intercept of the best-fit polynomial
    """
    
    def __init__(self, query: QueryATNF, x_param: str, y_param: str, 
                 test_degree: int, log_x=True, log_y=True):
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
        """Process and filter ATNF query data."""
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
        """
        Fit polynomials of varying degrees and select the best one.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to log progress (default: True)
        """
        if verbose:
            logger.info("Fitting Polynomial Approximation...")
        best_score = float('-inf')

        for degree in range(1, self.test_degree + 1):
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('reg', LinearRegression())
            ])
            pipeline.fit(self.x_data, self.y_data)
            y_pred = pipeline.predict(self.x_data)
            score = r2_score(self.y_data, y_pred)
            self.r2_scores[degree] = score

            if verbose:
                logger.info(f"Degree {degree} -> R² Score: {score:.6f}")

            if score > best_score:
                best_score = score
                self.model = pipeline
                self.best_degree = degree

        self.coefficients = self.model.named_steps['reg'].coef_
        self.intercept = self.model.named_steps['reg'].intercept_

        self.predicted_x = np.linspace(self.x_data.min(), self.x_data.max(), 100).reshape(-1, 1)
        self.predicted_y = self.model.predict(self.predicted_x)

    def compute_metrics(self, verbose=True):
        """
        Compute quantitative metrics for the fitted polynomial model.
        
        Calculates RMSE, MAE, and reduced chi-squared to objectively assess model fit.
        These metrics provide quantitative measures beyond visual comparison.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, prints detailed metrics report. Default: True.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'r2': R² score (coefficient of determination)
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'chi2_reduced': Reduced chi-squared
            - 'n_samples': Number of data points
            - 'n_params': Number of model parameters
        
        Notes
        -----
        - RMSE penalizes larger errors more heavily than MAE
        - MAE gives equal weight to all errors
        - Reduced chi-squared close to 1.0 indicates good fit
        - Reduced chi-squared close to 1.0 indicates good fit
        - R² close to 1.0 indicates the model explains most variance
        """
        if self.model is None or self.predicted_y is None:
            raise RuntimeError("Run `fit_polynomial()` first.")
        
        # Get predictions for all data points
        y_pred = self.model.predict(self.x_data)
        
        # R² Score
        r2 = r2_score(self.y_data, y_pred)
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((y_pred - self.y_data) ** 2))
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_pred - self.y_data))
        
        # Reduced Chi-Squared
        n_samples = len(self.y_data)
        n_params = len(self.coefficients) + 1  # coefficients + intercept
        dof = max(n_samples - n_params, 1)  # degrees of freedom
        chi2 = np.sum((y_pred - self.y_data) ** 2)
        chi2_reduced = chi2 / dof
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'chi2_reduced': chi2_reduced,
            'n_samples': n_samples,
            'n_params': n_params
        }
        
        if verbose:
            print("\n" + "="*70)
            print("POLYNOMIAL FIT - QUANTITATIVE METRICS")
            print("="*70)
            print(f"\nModel: Polynomial Degree {self.best_degree}")
            print(f"Data Points: {n_samples}")
            print(f"Parameters: {n_params}")
            print(f"\nGoodness of Fit Metrics:")
            print(f"  R² Score:          {r2:.6f}")
            print(f"  RMSE:              {rmse:.6e}")
            print(f"  MAE:               {mae:.6e}")
            print(f"  Reduced chi-squared:        {chi2_reduced:.6f}")
            
            print(f"\n" + "-"*70)
            print("METRIC INTERPRETATION:")
            print(f"  • R² (Coefficient of Determination): {r2:.4f}")
            print(f"    - Model explains {100*r2:.2f}% of the variance in the data")
            print(f"  • RMSE (Root Mean Squared Error): {rmse:.6e}")
            print(f"    - Average prediction error (penalizes large deviations)")
            print(f"  • MAE (Mean Absolute Error): {mae:.6e}")
            print(f"    - Average absolute prediction error")
            print(f"  • Reduced chi-squared: {chi2_reduced:.4f}")
            if chi2_reduced < 0.5:
                print(f"    - Model may be overfitting (chi-squared too low)")
            elif chi2_reduced > 2.0:
                print(f"    - Model may be underfitting or systematic errors present")
            else:
                print(f"    - Good model fit achieved")
            print("-"*70)
            print("="*70)
        
        return metrics
    
    def get_polynomial_expression(self):
        """
        Get the polynomial expression as a string.
        
        Returns
        -------
        str
            Polynomial expression in the form "a + b*x + c*x**2 + ..."
        """
        terms = [f"{self.intercept:.10f}"]
        for i, coef in enumerate(self.coefficients[1:], start=1):
            terms.append(f"{coef:.10f} * x**{i}")
        return " + ".join(terms)

    def print_polynomial(self):
        """Print the best-fit polynomial degree and expression."""
        poly_expr = self.get_polynomial_expression()
        print(f"\nBest Polynomial Degree: {self.best_degree}")
        print(f"Approximated Polynomial Function:\nf(x) = {poly_expr}")

    def plot_r2_scores(self):
        """Plot R² scores vs polynomial degree."""
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
        """Plot the data points and best-fit polynomial curve."""
        if self.predicted_x is None or self.predicted_y is None:
            raise RuntimeError("Run `fit_polynomial()` first.")

        plt.figure(figsize=(8, 5))
        plt.scatter(self.x_data, self.y_data, s=10, alpha=0.4, label='Pulsars')
        plt.plot(self.predicted_x, self.predicted_y, color='navy', 
                label=f'Degree {self.best_degree} Fit')
        plt.xlabel(f"log({self.x_param})" if self.log_x else self.x_param)
        plt.ylabel(f"log({self.y_param})" if self.log_y else self.y_param)
        plt.title("Polynomial Fit of Pulsar Data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_combined_analysis(self):
        """Plot combined analysis: polynomial fit and R² scores side-by-side."""
        if self.predicted_x is None or self.predicted_y is None:
            raise RuntimeError("Run `fit_polynomial()` first.")
        if not self.r2_scores:
            raise RuntimeError("R² scores are empty. Run `fit_polynomial()` first.")

        degrees = list(self.r2_scores.keys())
        scores = list(self.r2_scores.values())

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Polynomial Fit
        axs[0].scatter(self.x_data, self.y_data, s=10, alpha=0.5, label='Pulsars')
        axs[0].plot(self.predicted_x, self.predicted_y, color='navy', 
                   label=f'Degree {self.best_degree} Fit')
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
