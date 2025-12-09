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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
