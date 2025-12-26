"""
PINN Test Script for Pulsar Spindown Analysis
Author: Om Kasar & Saumil Sharma
Date: December 2025
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pinn import PulsarPINN
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
    input_layer=1,
    hidden_layers=[64, 64, 32],
    output_layer=1,
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    random_seed=42
)

pinn.show_hyperparameters()

# Hyperparameters table
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
hyperparams = [
    ['Parameter', 'Value'],
    ['Neural Network Architecture', f'{pinn.input_layer} → {pinn.hidden_layers} → {pinn.output_layer}'],
    ['Total NN Parameters', f'{sum(p.numel() for p in pinn.model.parameters()):,}'],
    ['Training Epochs', '10,000'],
    ['Training Split', f'{pinn.train_split*100:.0f}%'],
    ['Validation Split', f'{pinn.val_split*100:.0f}%'],
    ['Test Split', f'{pinn.test_split*100:.0f}%'],
    ['Total Samples', f'{len(log_P)}'],
    ['Training Samples', f'{len(pinn.x_train)}'],
    ['Validation Samples', f'{len(pinn.x_val)}'],
    ['Test Samples', f'{len(pinn.x_test)}'],
    ['Optimizer', 'Adam'],
    ['Learning Rate', '0.001'],
    ['Physics Weight', '1.0'],
    ['Data Weight', '1.0'],
    ['Random Seed', '42'],
    ['Differential Equation', str(differential_equation)],
    ['Learned Constants', 'n_braking (init: 3.0), logK (init: -16.0)']
]
table = ax.table(cellText=hyperparams, cellLoc='left', loc='center', colWidths=[0.4, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
for i in range(len(hyperparams)):
    if i == 0:
        table[(i, 0)].set_facecolor('#4472C4')
        table[(i, 1)].set_facecolor('#4472C4')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_text_props(weight='bold', color='white')
    else:
        table[(i, 0)].set_facecolor('#D9E2F3')
        table[(i, 1)].set_facecolor('#F2F2F2')
ax.set_title('Model Hyperparameters and Configuration', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

pinn.train(
    epochs=10000,
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

# Predictions vs data
x_extended, y_pred_extended = pinn.predict_extended(extend=0.2, n_points=500)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(pinn.x_test, pinn.y_test, color='cornflowerblue', alpha=0.6, s=30, label='Test Data', edgecolors='none')
ax.scatter(pinn.x_train, pinn.y_train, color='lightsteelblue', alpha=0.4, s=20, label='Training Data', edgecolors='none')
ax.plot(x_extended, y_pred_extended, linewidth=2, label='PINN Model #3 (Our Work)')
ax.set_xlabel('log(P) [log(s)]', fontsize=12)
ax.set_ylabel('log(dP/dt) [log(s/s)]', fontsize=12)
ax.set_title('Period Derivative vs. Period - PINN Model #3', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Loss curves
fig, ax = plt.subplots(figsize=(6, 6))
epochs = np.arange(len(pinn.loss_log['total']))
ax.plot(epochs, pinn.loss_log['total'], label='Total Loss', linewidth=1.5)
ax.plot(epochs, pinn.loss_log['physics'], label='Physics Loss', linewidth=1.5)
ax.plot(epochs, pinn.loss_log['data'], label='Data Loss', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_yscale('log')
ax.set_ylim(bottom=None, top=100000)
ax.set_title('PINN Loss vs. Epoch Curves - Model #3', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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