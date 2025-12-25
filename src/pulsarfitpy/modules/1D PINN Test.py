"""
PINN Test Script for Pulsar Spindown Analysis
Author: Om Kasar & Saumil Sharma
Date: December 2025
"""

import numpy as np
import sympy as sp
from pinn import PulsarPINN
from pinn_visualizer import VisualizePINN
from export_solutions import ExportPINN
from psrqpy import QueryATNF

query = QueryATNF(params=['P0', 'P1', 'ASSOC'], condition='P0 < 0.03')
query_table = query.table

P = np.array(query_table['P0'])
Pdot = np.array(query_table['P1'])

log_P = np.log10(P)
log_Pdot = np.log10(np.abs(Pdot))

valid_data = np.isfinite(log_P) & np.isfinite(log_Pdot)
log_P = log_P[valid_data]
log_Pdot = log_Pdot[valid_data]

print(f"Queried {len(log_P)} millisecond pulsars from ATNF catalog")

logP = sp.Symbol('logP')
logPdot = sp.Symbol('logPdot')
n_braking = sp.Symbol('n_braking')
logK = sp.Symbol('logK')

differential_equation = sp.Eq(logPdot, (n_braking - 1) * logP + logK)

fixed_inputs = {
    logP: log_P,
    logPdot: log_Pdot
}

pinn = PulsarPINN(
    differential_eq=differential_equation,
    x_sym=logP,
    y_sym=logPdot,
    learn_constants={
        n_braking: 2.1,
        logK: -16.0
    },
    fixed_inputs=fixed_inputs,
    log_scale=True,
    input_layer=1,
    hidden_layers=[64, 32],
    output_layer=1,
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    random_seed=42
)

pinn.show_hyperparameters()

pinn.train(
    epochs=6000,
    training_reports=600,
    physics_weight=1.0,
    data_weight=1.0
)

metrics = pinn.evaluate_test_set(verbose=True)

learned_constants = pinn.store_learned_constants()
learned_n = learned_constants['n_braking']
learned_logK = learned_constants['logK']

print("\n" + "=" * 80)
print("LEARNED PHYSICAL CONSTANTS")
print("=" * 80)
print(f"\nBraking Index: n = {learned_n:.6f}")
print(f"Spindown Constant: log(K) = {learned_logK:.6f}")
print("=" * 80)

uncertainties = pinn.bootstrap_uncertainty(
    n_bootstrap=50,
    sample_fraction=0.9,
    epochs=1500,
    confidence_level=0.95,
    verbose=True
)

robustness_results = pinn.run_all_robustness_tests(
    n_permutations=50,
    n_shuffles=50,
    verbose=True
)

cv_results_5fold = pinn.kfold_cross_validation(
    k=5,
    epochs=2000,
    physics_weight=1.0,
    data_weight=1.0,
    verbose=True
)

cv_results_10fold = pinn.kfold_cross_validation(
    k=10,
    epochs=2000,
    physics_weight=1.0,
    data_weight=1.0,
    verbose=True
)

visualizer = VisualizePINN(pinn)

visualizer.plot_predictions_vs_data(
    x_axis='log(P) [Period in seconds]',
    y_axis='log(dP/dt) [Period Derivative]',
    title='P-dP/dt Diagram: Millisecond Pulsar Spindown (ATNF Data)'
)

visualizer.plot_loss_curves(log_scale=True)
visualizer.plot_residuals_analysis()
visualizer.plot_prediction_scatter()

visualizer.plot_braking_index_distribution(
    learned_constants=learned_constants,
    uncertainties=uncertainties
)

visualizer.plot_uncertainty_quantification(uncertainties=uncertainties)
visualizer.plot_robustness_validation(robustness_results=robustness_results)

exporter = ExportPINN(pinn)

exporter.save_predictions_to_csv(
    filepath='data/outputs/pinn_predictions.csv',
    x_value_name='log_period',
    y_value_name='log_period_derivative',
    include_raw_data=True,
    include_test_metrics=True,
    additional_metadata={
        'model_type': 'Millisecond Pulsar Spindown PINN',
        'data_source': 'ATNF Pulsar Catalog',
        'n_pulsars_used': len(log_P)
    }
)

exporter.save_learned_constants_to_csv(
    filepath='data/outputs/learned_constants.csv',
    include_uncertainty=True,
    uncertainty_method='bootstrap',
    n_iterations=50,
    additional_info={
        'model': 'pulsar_spindown',
        'equation': str(differential_equation),
        'training_epochs': 6000
    }
)

exporter.save_metrics_to_csv(
    filepath='data/outputs/evaluation_metrics.csv',
    additional_info={
        'test_set_size': len(pinn.x_test),
        'train_set_size': len(pinn.x_train),
        'val_set_size': len(pinn.x_val),
        'total_samples': len(log_P)
    }
)

exporter.save_loss_history_to_csv(
    filepath='data/outputs/loss_history.csv'
)

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nTest R2: {metrics['test_r2']:.6f}")
print(f"Test RMSE: {metrics['test_rmse']:.6e}")
print(f"Test MAE: {metrics['test_mae']:.6e}")
print(f"\nn = {learned_n:.4f} +/- {uncertainties['n_braking']['std']:.4f}")
print(f"95% CI: [{uncertainties['n_braking']['ci_lower']:.4f}, {uncertainties['n_braking']['ci_upper']:.4f}]")
print(f"log(K) = {learned_logK:.4f} +/- {uncertainties['logK']['std']:.4f}")
print(f"\n5-Fold CV R2: {cv_results_5fold['mean_r2']:.4f} +/- {cv_results_5fold['std_r2']:.4f}")
print(f"10-Fold CV R2: {cv_results_10fold['mean_r2']:.4f} +/- {cv_results_10fold['std_r2']:.4f}")
print("=" * 80)