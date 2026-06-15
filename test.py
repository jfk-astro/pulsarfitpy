"""
PINN Test Script for Pulsar Spindown Analysis
Author: Om Kasar & Saumil Sharma
Date: December 2025
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from src.pulsarfitpy.modules.pinn import PulsarPINN
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
        n_braking: 3.0,
        logK: -16.0
    },
    fixed_inputs=fixed_inputs,
    log_scale=True,
    hidden_layers=[128, 64, 32],  # Deeper network for better learning
    train_split=0.75,
    val_split=0.15,
    test_split=0.10,
    random_seed=42
)

pinn.show_hyperparameters()

pinn.train(
    epochs=20000,
    training_reports=2000,
    physics_weight=0.8,  # Balanced weighting
    data_weight=1.2,     # Slightly favor data fitting
)

metrics = pinn.evaluate_test_set(verbose=True)

learned_constants = pinn.store_learned_constants()
learned_n = learned_constants['n_braking']
learned_logK = learned_constants['logK']

print("\n" + "-" * 80)
print("LEARNED PHYSICAL CONSTANTS")
print("-")
print(f"\nBraking Index: n = {learned_n:.6f}")
print(f"Spindown Constant: log(K) = {learned_logK:.6f}")
print("-" * 80)

uncertainties = pinn.bootstrap_uncertainty(
    n_bootstrap=50,
    sample_fraction=0.9,
    epochs=800,
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
    epochs=1000,
    physics_weight=0.8,
    data_weight=1.2,
    verbose=True
)

cv_results_10fold = pinn.kfold_cross_validation(
    k=10,
    epochs=800,
    physics_weight=0.8,
    data_weight=1.2,
    verbose=True
)

x_extended, y_pred_extended = pinn.predict_extended(extend=0.2, n_points=500)
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(pinn.x_test, pinn.y_test, color='cornflowerblue', alpha=0.6, s=30, label='Test Data', edgecolors='none')
ax.scatter(pinn.x_train, pinn.y_train, color='steelblue', alpha=0.4, s=20, label='Training Data', edgecolors='none')
ax.plot(x_extended, y_pred_extended, linewidth=2.5, label='PINN Model')
ax.set_xlabel('log(P) [log(s)]', fontsize=14, fontweight='bold')
ax.set_ylabel('log(dP/dt) [log(s/s)]', fontsize=14, fontweight='bold')
ax.set_title('Period Derivative vs Period - PINN Fit', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12, 7))
epochs = np.arange(len(pinn.loss_log['total']))

# Plot with better colors and styling
ax.plot(epochs, pinn.loss_log['total'], label='Total Loss', linewidth=2, color='#2E86AB', alpha=0.9)
ax.plot(epochs, pinn.loss_log['physics'], label='Physics Loss', linewidth=2, color='#A23B72', alpha=0.9)
ax.plot(epochs, pinn.loss_log['data'], label='Data Loss', linewidth=2, color='#F18F01', alpha=0.9)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_yscale('log')

# Improved y-axis limits to focus on the stable training region
ax.set_ylim(1e-5, 10)

ax.set_title('PINN Training Loss Convergence', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, framealpha=0.95, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--', which='both')

# Add minor grid lines for log scale
ax.grid(True, which='minor', alpha=0.15, linestyle=':')

plt.tight_layout()
plt.show()

print("\n" + "-" * 80)
print("FINAL SUMMARY")
print("-" * 80)
print(f"\nTest R2: {metrics['test_r2']:.6f}")
print(f"Test RMSE: {metrics['test_rmse']:.6e}")
print(f"Test MAE: {metrics['test_mae']:.6e}")
print(f"\nn = {learned_n:.4f} +/- {uncertainties['n_braking']['std']:.4f}")
print(f"95% CI: [{uncertainties['n_braking']['ci_lower']:.4f}, {uncertainties['n_braking']['ci_upper']:.4f}]")
print(f"log(K) = {learned_logK:.4f} +/- {uncertainties['logK']['std']:.4f}")
print(f"\n5-Fold CV R2: {cv_results_5fold['mean_r2']:.4f} +/- {cv_results_5fold['std_r2']:.4f}")
print(f"10-Fold CV R2: {cv_results_10fold['mean_r2']:.4f} +/- {cv_results_10fold['std_r2']:.4f}")