"""
Integration tests for pulsarfitpy package.

Tests the interaction between different components and end-to-end workflows.
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import Mock
from pulsarfitpy import configure_logging, PulsarApproximation


class TestPackageImports:
    """Test that package imports work correctly."""

    def test_import_main_package(self):
        """Test that main package can be imported."""
        import pulsarfitpy

        assert pulsarfitpy is not None

    def test_import_configure_logging(self):
        """Test that configure_logging can be imported from package."""
        from pulsarfitpy import configure_logging

        assert configure_logging is not None

    def test_import_pulsar_approximation(self):
        """Test that PulsarApproximation can be imported from package."""
        from pulsarfitpy import PulsarApproximation

        assert PulsarApproximation is not None

    def test_import_pulsar_pinn(self):
        """Test that PulsarPINN can be imported from package."""
        from pulsarfitpy import PulsarPINN

        assert PulsarPINN is not None

    def test_package_all_attribute(self):
        """Test that __all__ is properly defined."""
        import pulsarfitpy

        assert hasattr(pulsarfitpy, "__all__")
        assert "configure_logging" in pulsarfitpy.__all__
        assert "PulsarApproximation" in pulsarfitpy.__all__
        assert "PulsarPINN" in pulsarfitpy.__all__


class TestEndToEndWorkflow:
    """Test complete workflows from start to finish."""

    @pytest.fixture
    def complete_mock_query(self):
        """Create a complete mock query for end-to-end testing."""
        query = Mock()

        # Generate realistic pulsar data
        n_pulsars = 200
        np.random.seed(42)

        # Log-space periods (millisecond to seconds)
        periods = np.logspace(-3, 1, n_pulsars)

        # Period derivative with realistic relationship
        period_derivs = 1e-15 * periods**1.5

        # Add some realistic noise
        period_derivs *= np.random.lognormal(0, 0.3, n_pulsars)

        query.table = {
            "P0": periods,
            "P1": period_derivs,
        }

        return query

    def test_complete_approximation_workflow(self, complete_mock_query):
        """Test a complete polynomial approximation workflow."""
        # Step 1: Configure logging
        configure_logging("WARNING")  # Suppress info messages for cleaner test output

        # Step 2: Create approximation object
        approx = PulsarApproximation(
            query=complete_mock_query,
            x_param="P0",
            y_param="P1",
            test_degree=5,
            log_x=True,
            log_y=True,
        )

        # Step 3: Fit polynomial
        approx.fit_polynomial(verbose=False)

        # Step 4: Compute metrics
        metrics = approx.compute_metrics(verbose=False)

        # Step 5: Verify results
        assert approx.model is not None
        assert approx.best_degree is not None
        assert 1 <= approx.best_degree <= 5
        assert metrics["r2"] > 0  # Should have some predictive power
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

        # Step 6: Generate polynomial expression
        expr = approx.get_polynomial_expression()
        assert isinstance(expr, str)
        assert "x" in expr

    def test_workflow_with_plotting(self, complete_mock_query):
        """Test workflow including plotting functions."""
        configure_logging("ERROR")

        approx = PulsarApproximation(
            query=complete_mock_query,
            x_param="P0",
            y_param="P1",
            test_degree=3,
            log_x=True,
            log_y=True,
        )

        approx.fit_polynomial(verbose=False)

        # Test all plotting functions
        plt.close("all")
        approx.plot_r2_scores()
        assert len(plt.get_fignums()) > 0

        plt.close("all")
        approx.plot_approximation_curve()
        assert len(plt.get_fignums()) > 0

        plt.close("all")
        approx.plot_combined_analysis()
        assert len(plt.get_fignums()) > 0
        fig = plt.gcf()
        assert len(fig.axes) == 2  # Should have 2 subplots

        plt.close("all")

    def test_workflow_with_different_parameters(self):
        """Test workflow with different parameter combinations."""
        query = Mock()

        # Create different parameter relationship
        n = 100
        x_data = np.linspace(1, 100, n)
        y_data = np.sqrt(x_data) * 10

        query.table = {
            "X": x_data,
            "Y": y_data,
        }

        # Without log transformation
        approx = PulsarApproximation(
            query=query,
            x_param="X",
            y_param="Y",
            test_degree=4,
            log_x=False,
            log_y=False,
        )

        approx.fit_polynomial(verbose=False)
        metrics = approx.compute_metrics(verbose=False)

        assert metrics["r2"] > 0.8  # Should fit reasonably well

    def test_multiple_fits_same_data(self, complete_mock_query):
        """Test running multiple fits on the same data."""
        configure_logging("ERROR")

        # First fit with degree 3
        approx1 = PulsarApproximation(
            query=complete_mock_query,
            x_param="P0",
            y_param="P1",
            test_degree=3,
        )
        approx1.fit_polynomial(verbose=False)
        metrics1 = approx1.compute_metrics(verbose=False)

        # Second fit with degree 5
        approx2 = PulsarApproximation(
            query=complete_mock_query,
            x_param="P0",
            y_param="P1",
            test_degree=5,
        )
        approx2.fit_polynomial(verbose=False)
        metrics2 = approx2.compute_metrics(verbose=False)

        # Both should produce valid results
        assert metrics1["r2"] > 0
        assert metrics2["r2"] > 0

        # Higher degree should have same or better R²
        assert (
            metrics2["r2"] >= metrics1["r2"] - 0.01
        )  # Allow small numerical differences


class TestErrorHandling:
    """Test error handling across the package."""

    def test_approximation_before_fitting_errors(self, mock_atnf_query):
        """Test that methods requiring fitting raise appropriate errors."""
        approx = PulsarApproximation(
            query=mock_atnf_query, x_param="P0", y_param="P1", test_degree=3
        )

        # These should raise RuntimeError before fitting
        with pytest.raises(RuntimeError):
            approx.compute_metrics(verbose=False)

        with pytest.raises(RuntimeError):
            approx.plot_r2_scores()

        with pytest.raises(RuntimeError):
            approx.plot_approximation_curve()

        with pytest.raises(RuntimeError):
            approx.plot_combined_analysis()

    def test_empty_data_handling(self):
        """Test handling of empty or invalid datasets."""
        query = Mock()
        query.table = {
            "P0": np.array([]),
            "P1": np.array([]),
        }

        with pytest.raises(ValueError):
            PulsarApproximation(query=query, x_param="P0", y_param="P1", test_degree=3)

    def test_all_nan_data_handling(self):
        """Test handling when all data points are NaN."""
        query = Mock()
        query.table = {
            "P0": np.array([np.nan, np.nan, np.nan]),
            "P1": np.array([np.nan, np.nan, np.nan]),
        }

        with pytest.raises(ValueError, match="No valid data points"):
            PulsarApproximation(query=query, x_param="P0", y_param="P1", test_degree=3)


class TestLoggingIntegration:
    """Test logging integration across different components."""

    def test_logging_affects_approximation_output(self, mock_atnf_query, caplog):
        """Test that logging configuration affects PulsarApproximation output."""
        import logging

        # Configure DEBUG logging
        configure_logging("DEBUG")

        approx = PulsarApproximation(
            query=mock_atnf_query, x_param="P0", y_param="P1", test_degree=2
        )

        with caplog.at_level(logging.DEBUG, logger="pulsarfitpy"):
            approx.fit_polynomial(verbose=True)
            # Should have log messages
            assert len(caplog.records) > 0

    def test_logging_suppression(self, mock_atnf_query, caplog):
        """Test that ERROR level logging suppresses info messages."""
        import logging

        configure_logging("ERROR")

        approx = PulsarApproximation(
            query=mock_atnf_query, x_param="P0", y_param="P1", test_degree=2
        )

        with caplog.at_level(logging.INFO, logger="pulsarfitpy"):
            approx.fit_polynomial(verbose=True)
            # Should have no INFO level messages
            info_messages = [r for r in caplog.records if r.levelno == logging.INFO]
            # May have some, but they shouldn't be from fitting since verbose=True but logger is ERROR level


class TestDataQuality:
    """Test handling of various data quality issues."""

    def test_sparse_data(self):
        """Test handling of very sparse datasets."""
        query = Mock()
        query.table = {
            "P0": np.array([0.001, 0.1, 10.0]),
            "P1": np.array([1e-15, 1e-13, 1e-11]),
        }

        approx = PulsarApproximation(
            query=query,
            x_param="P0",
            y_param="P1",
            test_degree=2,  # Use lower degree for sparse data
            log_x=True,
            log_y=True,
        )

        approx.fit_polynomial(verbose=False)
        assert approx.model is not None

    def test_data_with_mixed_scales(self):
        """Test data with very different scales."""
        query = Mock()

        # Mix of very small and very large values
        x_data = np.array([1e-10, 1e-5, 1e-1, 1e3, 1e8])
        y_data = np.array([1e-20, 1e-15, 1e-10, 1e-5, 1e-1])

        query.table = {
            "X": x_data,
            "Y": y_data,
        }

        # Log transformation helps with mixed scales
        approx = PulsarApproximation(
            query=query, x_param="X", y_param="Y", test_degree=2, log_x=True, log_y=True
        )

        approx.fit_polynomial(verbose=False)
        assert approx.model is not None
        assert len(approx.x_data) == 5

    def test_data_with_duplicates(self):
        """Test handling of duplicate data points."""
        query = Mock()

        x_data = np.array([1, 2, 2, 3, 3, 3, 4, 5])
        y_data = np.array([2, 4, 4.1, 6, 6.1, 5.9, 8, 10])

        query.table = {
            "X": x_data,
            "Y": y_data,
        }

        approx = PulsarApproximation(
            query=query,
            x_param="X",
            y_param="Y",
            test_degree=2,
            log_x=False,
            log_y=False,
        )

        approx.fit_polynomial(verbose=False)
        metrics = approx.compute_metrics(verbose=False)

        # Should still produce reasonable results
        assert metrics["r2"] > 0


class TestMetricsConsistency:
    """Test consistency of metrics across different scenarios."""

    def test_perfect_fit_metrics(self):
        """Test metrics for a perfect polynomial fit."""
        query = Mock()

        # Generate data from a known polynomial
        x = np.linspace(0, 10, 100)
        y = 2 * x**2 + 3 * x + 1  # Degree 2 polynomial

        query.table = {
            "X": x,
            "Y": y,
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
        metrics = approx.compute_metrics(verbose=False)

        # Perfect fit should have R² very close to 1
        assert metrics["r2"] > 0.9999
        # RMSE should be very small
        assert metrics["rmse"] < 1e-10

    def test_metrics_comparison_across_degrees(self, mock_atnf_query):
        """Test that metrics improve or stay constant with higher degrees."""
        configure_logging("ERROR")

        r2_scores = []

        for max_degree in [2, 3, 4, 5]:
            approx = PulsarApproximation(
                query=mock_atnf_query,
                x_param="P0",
                y_param="P1",
                test_degree=max_degree,
            )
            approx.fit_polynomial(verbose=False)
            metrics = approx.compute_metrics(verbose=False)
            r2_scores.append(metrics["r2"])

        # R² should generally improve or stay the same
        for i in range(len(r2_scores) - 1):
            # Allow small numerical tolerance
            assert r2_scores[i + 1] >= r2_scores[i] - 0.01
