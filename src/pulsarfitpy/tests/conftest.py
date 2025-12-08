"""
Shared pytest fixtures and configuration for pulsarfitpy tests.
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import Mock


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_pulsar_data():
    """
    Provide sample pulsar data for testing.

    Returns a dictionary with typical ATNF pulsar parameters.
    """
    n_pulsars = 100

    data = {
        "P0": np.random.uniform(0.001, 10.0, n_pulsars),  # Period in seconds
        "P1": np.random.uniform(1e-20, 1e-10, n_pulsars),  # Period derivative
        "BSURF": np.random.uniform(1e8, 1e13, n_pulsars),  # Surface magnetic field
        "EDOT": np.random.uniform(1e30, 1e38, n_pulsars),  # Spin-down energy loss
        "AGE": np.random.uniform(1e3, 1e10, n_pulsars),  # Characteristic age
    }

    return data


@pytest.fixture
def mock_atnf_query(sample_pulsar_data):
    """
    Create a mock psrqpy QueryATNF object.

    Returns a Mock object that simulates the QueryATNF interface.
    """
    query = Mock()
    query.table = sample_pulsar_data
    query.num_pulsars = len(sample_pulsar_data["P0"])

    return query


@pytest.fixture
def linear_data():
    """
    Generate perfect linear relationship data for testing.

    Returns x and y data following y = 2x + 3.
    """
    x = np.linspace(0, 10, 100)
    y = 2 * x + 3

    return {"x": x, "y": y}


@pytest.fixture
def quadratic_data():
    """
    Generate perfect quadratic relationship data for testing.

    Returns x and y data following y = x² + 2x + 1.
    """
    x = np.linspace(-5, 5, 100)
    y = x**2 + 2 * x + 1

    return {"x": x, "y": y}


@pytest.fixture
def noisy_data():
    """
    Generate data with noise for testing robustness.

    Returns x and y data with Gaussian noise added.
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 3 + np.random.normal(0, 0.5, 100)

    return {"x": x, "y": y}


@pytest.fixture
def data_with_outliers():
    """
    Generate data with outliers for testing robustness.

    Returns x and y data with some extreme outlier points.
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 3

    # Add some outliers
    outlier_indices = [10, 30, 70]
    for idx in outlier_indices:
        y[idx] += np.random.choice([-20, 20])

    return {"x": x, "y": y}
