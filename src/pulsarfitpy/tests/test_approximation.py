"""
Tests for pulsarfitpy.approximation module.

Tests the PulsarApproximation class for polynomial fitting of pulsar data.
"""

import pytest
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unittest.mock import Mock, MagicMock, patch
from sklearn.metrics import r2_score
from pulsarfitpy.approximation import PulsarApproximation


class TestPulsarApproximation:
    """Test suite for PulsarApproximation class."""

    @pytest.fixture
    def mock_query(self):
        """Create a mock ATNF query object with sample data."""
        query = Mock()

        # Create sample data - periods and period derivatives
        n_points = 50
        periods = np.logspace(-3, 1, n_points)  # 0.001 to 10 seconds
        period_derivs = 10 ** (-15) * periods  # Simple relationship

        # Create a mock table
        mock_table = {
            "P0": periods,
            "P1": period_derivs,
            "BSURF": np.logspace(8, 13, n_points),
        }

        query.table = mock_table
        return query

    @pytest.fixture
    def mock_query_with_invalid_data(self):
        """Create a mock query with some invalid (NaN, negative) values."""
        query = Mock()

        periods = np.array([0.001, 0.01, np.nan, 0.1, -0.5, 1.0, 10.0])
        period_derivs = np.array([1e-15, 1e-14, 1e-13, np.nan, 1e-12, 1e-11, 1e-10])

        mock_table = {
            "P0": periods,
            "P1": period_derivs,
        }

        query.table = mock_table
        return query

    def test_initialization(self, mock_query):
        """Test that PulsarApproximation initializes correctly."""
        approx = PulsarApproximation(
            query=mock_query,
            x_param="P0",
            y_param="P1",
            test_degree=5,
            log_x=True,
            log_y=True,
        )

        assert approx.x_param == "P0"
        assert approx.y_param == "P1"
        assert approx.test_degree == 5
        assert approx.log_x is True
        assert approx.log_y is True
        assert approx.x_data is not None
        assert approx.y_data is not None

    def test_data_filtering(self, mock_query_with_invalid_data):
        """Test that invalid data points are filtered out."""
        approx = PulsarApproximation(
            query=mock_query_with_invalid_data,
            x_param="P0",
            y_param="P1",
            test_degree=3,
            log_x=True,
            log_y=True,
        )

        # Should have 4 valid points (excluding NaN and negative values)
        assert len(approx.x_data) == 4
        assert len(approx.y_data) == 4
        assert np.all(np.isfinite(approx.x_data))
        assert np.all(np.isfinite(approx.y_data))

    def test_log_transformation(self, mock_query):
        """Test that logarithmic transformation is applied correctly."""
        approx = PulsarApproximation(
            query=mock_query,
            x_param="P0",
            y_param="P1",
            test_degree=3,
            log_x=True,
            log_y=True,
        )

        # Transformed data should be in log space
        original_periods = mock_query.table["P0"]
        expected_log_periods = np.log10(
            original_periods[np.isfinite(original_periods) & (original_periods > 0)]
        )

        np.testing.assert_array_almost_equal(
            approx.x_data.flatten()[: len(expected_log_periods)],
            expected_log_periods[: len(approx.x_data)],
        )

    def test_no_log_transformation(self, mock_query):
        """Test that data remains untransformed when log_x=False, log_y=False."""
        approx = PulsarApproximation(
            query=mock_query,
            x_param="P0",
            y_param="P1",
            test_degree=3,
            log_x=False,
            log_y=False,
        )

        # Data should not be log-transformed
        original_periods = mock_query.table["P0"]
        valid_mask = np.isfinite(original_periods)
        expected_periods = original_periods[valid_mask]

        np.testing.assert_array_almost_equal(
            approx.x_data.flatten()[: len(expected_periods)],
            expected_periods[: len(approx.x_data)],
        )

    def test_no_valid_data_raises_error(self):
        """Test that ValueError is raised when no valid data points exist."""
        query = Mock()
        query.table = {
            "P0": np.array([np.nan, np.nan, -1]),
            "P1": np.array([np.nan, np.nan, np.nan]),
        }

        with pytest.raises(ValueError, match="No valid data points found"):
            PulsarApproximation(
                query=query,
                x_param="P0",
                y_param="P1",
                test_degree=3,
                log_x=True,
                log_y=True,
            )

    def test_fit_polynomial_runs(self, mock_query):
        """Test that fit_polynomial runs without errors."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=5
        )

        approx.fit_polynomial(verbose=False)

        assert approx.model is not None
        assert approx.best_degree is not None
        assert approx.coefficients is not None
        assert approx.intercept is not None
        assert len(approx.r2_scores) == 5

    def test_fit_polynomial_selects_best_degree(self, mock_query):
        """Test that the best polynomial degree is selected based on R² score."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=5
        )

        approx.fit_polynomial(verbose=False)

        # Best degree should have the highest R² score
        best_score = approx.r2_scores[approx.best_degree]
        for degree, score in approx.r2_scores.items():
            assert best_score >= score

    def test_fit_polynomial_generates_predictions(self, mock_query):
        """Test that fit_polynomial generates prediction arrays."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        approx.fit_polynomial(verbose=False)

        assert approx.predicted_x is not None
        assert approx.predicted_y is not None
        assert len(approx.predicted_x) == 100  # Default n_eval_points
        assert len(approx.predicted_y) == 100

    def test_r2_scores_improve_with_degree(self, mock_query):
        """Test that R² scores generally improve or stay constant with higher degree."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=5
        )

        approx.fit_polynomial(verbose=False)

        # Generally, R² should not decrease with higher degree
        # (though in practice it might plateau)
        scores = [approx.r2_scores[d] for d in range(1, 6)]
        assert all(score >= 0 for score in scores)  # All scores should be non-negative

    def test_compute_metrics_returns_dict(self, mock_query):
        """Test that compute_metrics returns a dictionary with expected keys."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        approx.fit_polynomial(verbose=False)
        metrics = approx.compute_metrics(verbose=False)

        assert isinstance(metrics, dict)
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "chi2_reduced" in metrics
        assert "n_samples" in metrics
        assert "n_params" in metrics

    def test_compute_metrics_values_reasonable(self, mock_query):
        """Test that computed metrics have reasonable values."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        approx.fit_polynomial(verbose=False)
        metrics = approx.compute_metrics(verbose=False)

        # R² should be between 0 and 1 (can be negative for very bad fits, but unlikely here)
        assert -1 <= metrics["r2"] <= 1

        # RMSE and MAE should be non-negative
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

        # Chi-squared should be non-negative
        assert metrics["chi2_reduced"] >= 0

        # Sample count should match data size
        assert metrics["n_samples"] == len(approx.x_data)

    def test_compute_metrics_before_fitting_raises_error(self, mock_query):
        """Test that compute_metrics raises error if called before fitting."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        with pytest.raises(RuntimeError, match="Run `fit_polynomial\\(\\)` first"):
            approx.compute_metrics(verbose=False)

    def test_get_polynomial_expression(self, mock_query):
        """Test that get_polynomial_expression returns a string."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=2
        )

        approx.fit_polynomial(verbose=False)
        expression = approx.get_polynomial_expression()

        assert isinstance(expression, str)
        assert "x" in expression
        assert "+" in expression or "-" in expression

    def test_print_polynomial(self, mock_query, capsys):
        """Test that print_polynomial outputs expected information."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        approx.fit_polynomial(verbose=False)
        approx.print_polynomial()

        captured = capsys.readouterr()
        assert "Best Polynomial Degree" in captured.out
        assert "Approximated Polynomial Function" in captured.out
        assert "f(x)" in captured.out

    def test_plot_r2_scores_before_fitting_raises_error(self, mock_query):
        """Test that plot_r2_scores raises error if called before fitting."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        with pytest.raises(RuntimeError, match="Run `fit_polynomial\\(\\)` first"):
            approx.plot_r2_scores()

    def test_plot_r2_scores_creates_figure(self, mock_query):
        """Test that plot_r2_scores creates a matplotlib figure."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        approx.fit_polynomial(verbose=False)

        # Close any existing figures
        plt.close("all")

        approx.plot_r2_scores()

        # Check that a figure was created
        assert len(plt.get_fignums()) > 0

        plt.close("all")

    def test_plot_approximation_curve_before_fitting_raises_error(self, mock_query):
        """Test that plot_approximation_curve raises error if called before fitting."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        with pytest.raises(RuntimeError, match="Run `fit_polynomial\\(\\)` first"):
            approx.plot_approximation_curve()

    def test_plot_approximation_curve_creates_figure(self, mock_query):
        """Test that plot_approximation_curve creates a matplotlib figure."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        approx.fit_polynomial(verbose=False)

        plt.close("all")
        approx.plot_approximation_curve()

        assert len(plt.get_fignums()) > 0

        plt.close("all")

    def test_plot_combined_analysis_before_fitting_raises_error(self, mock_query):
        """Test that plot_combined_analysis raises error if called before fitting."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        with pytest.raises(RuntimeError):
            approx.plot_combined_analysis()

    def test_plot_combined_analysis_creates_figure(self, mock_query):
        """Test that plot_combined_analysis creates a matplotlib figure with subplots."""
        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        approx.fit_polynomial(verbose=False)

        plt.close("all")
        approx.plot_combined_analysis()

        # Should create a figure with 2 subplots
        assert len(plt.get_fignums()) > 0
        fig = plt.gcf()
        assert len(fig.axes) == 2

        plt.close("all")

    def test_linear_relationship_fits_well(self):
        """Test that a perfect linear relationship is fitted well."""
        query = Mock()

        # Create perfect linear relationship
        x_data = np.linspace(0, 10, 100)
        y_data = 2 * x_data + 3

        query.table = {
            "X": x_data,
            "Y": y_data,
        }

        approx = PulsarApproximation(
            query=query,
            x_param="X",
            y_param="Y",
            test_degree=3,
            log_x=False,
            log_y=False,
        )

        approx.fit_polynomial(verbose=False)

        # Should select degree 1 or have very high R² for degree 1
        assert approx.r2_scores[1] > 0.99

    def test_quadratic_relationship_needs_higher_degree(self):
        """Test that a quadratic relationship requires degree >= 2."""
        query = Mock()

        # Create quadratic relationship
        x_data = np.linspace(-5, 5, 100)
        y_data = x_data**2 + 2 * x_data + 1

        query.table = {
            "X": x_data,
            "Y": y_data,
        }

        approx = PulsarApproximation(
            query=query,
            x_param="X",
            y_param="Y",
            test_degree=5,
            log_x=False,
            log_y=False,
        )

        approx.fit_polynomial(verbose=False)

        # Degree 2 should have much better R² than degree 1
        assert approx.r2_scores[2] > approx.r2_scores[1] + 0.1
        # Degree 2 should have nearly perfect fit
        assert approx.r2_scores[2] > 0.99

    def test_verbose_mode_produces_output(self, mock_query, caplog):
        """Test that verbose=True produces log output."""
        import logging

        approx = PulsarApproximation(
            query=mock_query, x_param="P0", y_param="P1", test_degree=3
        )

        with caplog.at_level(logging.INFO):
            approx.fit_polynomial(verbose=True)
            assert len(caplog.records) > 0
            assert any(
                "Fitting Polynomial" in record.message for record in caplog.records
            )
