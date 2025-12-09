"""
Integration test for PulsarPINN with ATNF pulsar database.

This test demonstrates the complete workflow of using PulsarPINN to learn
the relationship between pulsar period, period derivative, and surface
magnetic field from real ATNF catalogue data.
"""

import pytest
import sympy as sp
import numpy as np
from psrqpy import QueryATNF

from pulsarfitpy.pinn import PulsarPINN
from pulsarfitpy.pinn_visualizer import VisualizePINN
from pulsarfitpy.export_solutions import ExportPINN


@pytest.fixture
def physics_equation():
    """Define the physics equation for magnetic field calculation."""
    logP, logPDOT, logB = sp.symbols('logP logPDOT logB')
    logR = sp.Symbol('logR')
    
    # Differential equation: logB = logR + 0.5*logP + 0.5*logPDOT
    differential_equation = sp.Eq(logB, logR + 0.5 * logP + 0.5 * logPDOT)
    
    return differential_equation, logP, logPDOT, logB, logR


@pytest.fixture
def atnf_data():
    """Query and prepare ATNF pulsar data."""
    query = QueryATNF(
        params=['P0', 'P1', 'BSURF'], 
        condition='exist(P0) && exist(P1) && exist(BSURF)'
    )
    table = query.table
    
    # Extract and log-transform data
    P = table['P0'].data
    PDOT = table['P1'].data
    BSURF = table['BSURF'].data
    
    logP_data = np.log10(P)
    logPDOT_data = np.log10(PDOT)
    logB_data = np.log10(BSURF)
    
    return logP_data, logPDOT_data, logB_data


def test_pinn_initialization(physics_equation, atnf_data):
    """Test PINN model initialization with real data."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    assert pinn is not None
    assert len(pinn.train_data[0]) > 0
    assert len(pinn.val_data[0]) > 0
    assert len(pinn.test_data[0]) > 0


def test_pinn_training(physics_equation, atnf_data):
    """Test PINN training with reduced epochs for speed."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    # Train with reduced epochs for testing
    pinn.train(
        epochs=100,
        training_reports=50,
        physics_weight=1.0,
        data_weight=1.0
    )
    
    assert len(pinn.train_losses) > 0
    assert len(pinn.val_losses) > 0
    assert pinn.train_losses[-1] < pinn.train_losses[0]  # Loss should decrease


def test_pinn_evaluation(physics_equation, atnf_data):
    """Test PINN evaluation on test set."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    test_metrics = pinn.evaluate_test_set(verbose=False)
    
    assert 'test_loss' in test_metrics
    assert 'r2' in test_metrics
    assert 'mae' in test_metrics
    assert test_metrics['r2'] >= 0  # R² should be meaningful


def test_learned_constants(physics_equation, atnf_data):
    """Test retrieval of learned physical constants."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    learned_constants = pinn.store_learned_constants()
    
    assert 'logR' in learned_constants
    assert isinstance(learned_constants['logR'], float)


def test_predictions(physics_equation, atnf_data):
    """Test generating extended predictions."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    x_extended, y_extended = pinn.predict_extended(extend=0.5, n_points=500)
    
    assert len(x_extended) == 500
    assert len(y_extended) == 500
    assert x_extended.min() < logP_data.min()
    assert x_extended.max() > logP_data.max()


def test_export_predictions(physics_equation, atnf_data, tmp_path):
    """Test exporting predictions to CSV."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    exporter = ExportPINN(pinn_model=pinn)
    
    predictions_file = tmp_path / "predictions.csv"
    constants_file = tmp_path / "constants.csv"
    metrics_file = tmp_path / "metrics.csv"
    
    exporter.save_predictions_to_csv(
        x_value_name="log(Period)",
        y_value_name="log(B-field)",
        filepath=str(predictions_file)
    )
    exporter.save_learned_constants_to_csv(filepath=str(constants_file))
    exporter.save_metrics_to_csv(filepath=str(metrics_file))
    
    assert predictions_file.exists()
    assert constants_file.exists()
    assert metrics_file.exists()


def test_bootstrap_uncertainty(physics_equation, atnf_data):
    """Test bootstrap uncertainty estimation for learned constants."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    # Test with minimal bootstrap iterations for speed
    uncertainties = pinn.bootstrap_uncertainty(
        n_bootstrap=10,
        sample_fraction=0.8,
        epochs=50,
        confidence_level=0.95,
        verbose=False
    )
    
    assert 'logR' in uncertainties
    assert 'mean' in uncertainties['logR']
    assert 'std' in uncertainties['logR']
    assert 'ci_lower' in uncertainties['logR']
    assert 'ci_upper' in uncertainties['logR']
    assert 'original' in uncertainties['logR']
    
    # Check that confidence interval brackets the mean
    assert uncertainties['logR']['ci_lower'] < uncertainties['logR']['mean']
    assert uncertainties['logR']['mean'] < uncertainties['logR']['ci_upper']
    
    # Check that std dev is positive
    assert uncertainties['logR']['std'] > 0


def test_monte_carlo_uncertainty(physics_equation, atnf_data):
    """Test Monte Carlo uncertainty estimation for learned constants."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    # Test with minimal simulations for speed
    uncertainties = pinn.monte_carlo_uncertainty(
        n_simulations=10,
        noise_level=0.01,
        confidence_level=0.95,
        verbose=False
    )
    
    assert 'logR' in uncertainties
    assert 'mean' in uncertainties['logR']
    assert 'std' in uncertainties['logR']
    assert 'ci_lower' in uncertainties['logR']
    assert 'ci_upper' in uncertainties['logR']
    assert 'original' in uncertainties['logR']
    
    # Check that confidence interval brackets the mean
    assert uncertainties['logR']['ci_lower'] < uncertainties['logR']['mean']
    assert uncertainties['logR']['mean'] < uncertainties['logR']['ci_upper']
    
    # Check that std dev is positive
    assert uncertainties['logR']['std'] > 0


def test_permutation_validation(physics_equation, atnf_data):
    """Test permutation test for validating learned relationships."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    # Test with minimal permutations for speed
    results = pinn.validate_with_permutation_test(
        n_permutations=5,
        epochs=50,
        verbose=False
    )
    
    assert 'real_r2' in results
    assert 'permuted_r2_mean' in results
    assert 'p_value' in results
    assert 'is_significant' in results
    
    # Real model should generally outperform random permutations
    assert results['real_r2'] >= results['permuted_r2_mean']


def test_feature_shuffling_validation(physics_equation, atnf_data):
    """Test feature shuffling for validating input importance."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    # Test with minimal shuffles for speed
    results = pinn.validate_with_feature_shuffling(
        n_shuffles=5,
        epochs=50,
        verbose=False
    )
    
    assert 'real_r2' in results
    assert 'shuffled_r2_mean' in results
    assert 'r2_difference' in results
    assert 'improvement_factor' in results
    
    # Real model should be better than shuffled features
    assert results['r2_difference'] >= 0


def test_impossible_physics_validation(physics_equation, atnf_data):
    """Test impossible physics validation."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    results = pinn.validate_with_impossible_physics(
        epochs=100,
        verbose=False
    )
    
    assert 'real_r2' in results
    assert 'impossible_r2' in results
    assert 'r2_difference' in results
    assert 'real_much_better' in results


def test_comprehensive_robustness_suite(physics_equation, atnf_data):
    """Test running all robustness checks together."""
    differential_equation, logP, logPDOT, logB, logR = physics_equation
    logP_data, logPDOT_data, logB_data = atnf_data
    
    architecture_NN = [16, 32, 16]
    learn_constants = {logR: 18.0}
    fixed_data = {
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    }
    
    pinn = PulsarPINN(
        differential_eq=differential_equation,
        x_sym=logP,
        y_sym=logB,
        learn_constants=learn_constants,
        log_scale=True,
        fixed_inputs=fixed_data,
        hidden_layers=architecture_NN,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    pinn.train(epochs=100, training_reports=50, physics_weight=1.0, data_weight=1.0)
    
    # Run with minimal iterations for speed
    results = pinn.run_all_robustness_tests(
        n_permutations=5,
        n_shuffles=5,
        verbose=False
    )
    
    assert 'permutation_test' in results
    assert 'feature_shuffling_test' in results
    assert 'impossible_physics_test' in results
    assert 'all_tests_passed' in results
    assert isinstance(results['all_tests_passed'], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

