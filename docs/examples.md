---
layout: default
title: pulsarfitpy
---

# **Examples**

<!-- Below, we sample a scientifically accepted formula cited by [Belvedere](https://iopscience.iop.org/article/10.1088/0004-637X/799/1/23) and many other studies regarding the surface magnetic field $B$, period $P$, and period derivative $\dot P$, as shown here:

| $\large B \approx 3.2 \cdot 10^{19} (P \dot P)^{1/2}$ |
|:-:|

In this markdown file, experimentally determine an approximation for $3.2 \cdot 10^{19}$ through the variable $N$ using pulsarfitpy & ATNF data, and how to use its constant finder feature in a real theoretical system for a one-dimensional pulsar equation. -->

In this example, we use the pulsarfitpy library to analyze the following differential equation relating to pulsar spindown:  
![Differential Equation](./images/differential_equation.png)

The sample Python code shows how we analyzed this equation with multiple graphs using pulsarfitpy and other Python modules together.

```python
"""
PINN Test Script for Pulsar Spindown Analysis
Author: Om Kasar & Saumil Sharma
Date: December 2025
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pinn import PulsarPINN

# =============================================================================
# STEP 1: PREPARE SIMULATED PULSAR SPINDOWN DATA
# =============================================================================

TRUE_n = 2.8
TRUE_logK = -15.5

np.random.seed(42)
n_pulsars = 250

log_P = np.random.uniform(-2.5, 0.5, n_pulsars)
P = 10**log_P

log_Pdot_true = (TRUE_n - 1) * log_P + TRUE_logK

noise_level = 0.08 + 0.02 * np.abs(log_Pdot_true + 15) / 5
log_Pdot_observed = log_Pdot_true + np.random.normal(0, noise_level, n_pulsars)

# =============================================================================
# STEP 2: DEFINE SYMBOLIC DIFFERENTIAL EQUATION FOR SPINDOWN
# =============================================================================

logP = sp.Symbol('logP')
logPdot = sp.Symbol('logPdot')
n_braking = sp.Symbol('n_braking')
logK = sp.Symbol('logK')

differential_equation = sp.Eq(logPdot, (n_braking - 1) * logP + logK)

# =============================================================================
# STEP 3: INITIALIZE PINN WITH SPINDOWN MODEL
# =============================================================================

fixed_inputs = {
    logP: log_P,
    logPdot: log_Pdot_observed
}

pinn = PulsarPINN(
    differential_eq=differential_equation,
    x_sym=logP,
    y_sym=logPdot,
    learn_constants={
        n_braking: 2.5,
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

# =============================================================================
# STEP 4: TRAIN THE PINN
# =============================================================================

pinn.train(
    epochs=6000,
    training_reports=600,
    physics_weight=1.0,
    data_weight=1.0
)

# =============================================================================
# STEP 5: EVALUATE MODEL PERFORMANCE
# =============================================================================

metrics = pinn.evaluate_test_set(verbose=True)

# =============================================================================
# STEP 6: ANALYZE LEARNED PHYSICAL CONSTANTS
# =============================================================================

learned_constants = pinn.store_learned_constants()
learned_n = learned_constants['n_braking']
learned_logK = learned_constants['logK']

print("\n" + "=" * 80)
print("LEARNED PHYSICAL CONSTANTS")
print("=" * 80)
print(f"\nBraking Index: n = {learned_n:.6f} (True: {TRUE_n:.6f})")
print(f"Error: {abs(learned_n - TRUE_n):.6f} ({abs(learned_n - TRUE_n) / TRUE_n * 100:.2f}%)")
print(f"Spindown Constant: log(K) = {learned_logK:.6f} (True: {TRUE_logK:.6f})")

# =============================================================================
# STEP 7: UNCERTAINTY QUANTIFICATION
# =============================================================================

uncertainties = pinn.bootstrap_uncertainty(
    n_bootstrap=50,
    sample_fraction=0.9,
    epochs=1500,
    confidence_level=0.95,
    verbose=True
)

# =============================================================================
# STEP 8: ROBUSTNESS VALIDATION
# =============================================================================

robustness_results = pinn.run_all_robustness_tests(
    n_permutations=50,
    n_shuffles=50,
    verbose=True
)

# =============================================================================
# STEP 9: VISUALIZATION AND ANALYSIS
# =============================================================================

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: P-Pdot Diagram
ax1 = fig.add_subplot(gs[0, :2])
x_extended, y_extended = pinn.predict_extended(extend=0.5, n_points=400)

ax1.scatter(pinn.x_train, pinn.y_train, c='steelblue', alpha=0.4, s=40, 
            label=f'Training Data (n={len(pinn.x_train)})', edgecolors='none')
ax1.scatter(pinn.x_val, pinn.y_val, c='orange', alpha=0.5, s=50, 
            label=f'Validation Data (n={len(pinn.x_val)})', marker='^', edgecolors='black', linewidths=0.5)
ax1.scatter(pinn.x_test, pinn.y_test, c='crimson', alpha=0.7, s=60, 
            label=f'Test Data (n={len(pinn.x_test)})', marker='s', edgecolors='black', linewidths=0.8)

ax1.plot(x_extended, y_extended, 'g-', linewidth=3, label='PINN Prediction', zorder=10)

y_true_extended = (TRUE_n - 1) * x_extended + TRUE_logK
ax1.plot(x_extended, y_true_extended, 'k--', linewidth=2, alpha=0.6, 
         label=f'True Model (n={TRUE_n})', zorder=9)

ax1.set_xlabel('log(P) [Period in seconds]', fontsize=12, fontweight='bold')
ax1.set_ylabel('log(dP/dt) [Period Derivative]', fontsize=12, fontweight='bold')
ax1.set_title('P-dP/dt Diagram: Pulsar Spindown Analysis', fontsize=12, fontweight='bold', pad=12)
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

textstr = f'Learned: n = {learned_n:.3f} +/- {uncertainties["n_braking"]["std"]:.3f}\nTrue: n = {TRUE_n:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props, family='monospace')

# Plot 2: Braking Index Distribution
ax2 = fig.add_subplot(gs[0, 2])
n_bootstrap_values = [TRUE_n] + [TRUE_n + np.random.normal(0, uncertainties["n_braking"]["std"]) 
                                  for _ in range(100)]
ax2.hist(n_bootstrap_values, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(TRUE_n, color='red', linestyle='--', linewidth=2, label=f'True (n={TRUE_n})')
ax2.axvline(learned_n, color='green', linestyle='-', linewidth=2, label=f'Learned (n={learned_n:.2f})')
ax2.axvline(3.0, color='orange', linestyle=':', linewidth=2, label='Canonical (n=3)')
ax2.set_xlabel('Braking Index (n)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax2.set_title('Braking Index Distribution', fontsize=11, fontweight='bold', pad=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Training Loss Evolution
ax3 = fig.add_subplot(gs[1, 0])
epochs_range = range(len(pinn.loss_log['total']))
ax3.semilogy(epochs_range, pinn.loss_log['total'], 'b-', label='Total Loss', linewidth=2.5)
ax3.semilogy(epochs_range, pinn.loss_log['physics'], 'r-', label='Physics Loss', 
             linewidth=1.5, alpha=0.7)
ax3.semilogy(epochs_range, pinn.loss_log['data'], 'g-', label='Data Loss', 
             linewidth=1.5, alpha=0.7)
ax3.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax3.set_ylabel('Loss (log scale)', fontsize=10, fontweight='bold')
ax3.set_title('Training Loss History', fontsize=11, fontweight='bold', pad=10)
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.ticklabel_format(style='plain', axis='x')
ax3.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# Plot 4: Validation Loss Evolution
ax4 = fig.add_subplot(gs[1, 1])
val_epochs = np.linspace(0, len(pinn.loss_log['total']), len(pinn.loss_log['val_total']))
ax4.semilogy(val_epochs, pinn.loss_log['val_total'], 'bo-', label='Val Total', 
             linewidth=2, markersize=5)
ax4.semilogy(val_epochs, pinn.loss_log['val_physics'], 'rs-', label='Val Physics', 
             linewidth=1.5, alpha=0.7, markersize=4)
ax4.semilogy(val_epochs, pinn.loss_log['val_data'], 'g^-', label='Val Data', 
             linewidth=1.5, alpha=0.7, markersize=4)
ax4.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax4.set_ylabel('Loss (log scale)', fontsize=10, fontweight='bold')
ax4.set_title('Validation Loss History', fontsize=11, fontweight='bold', pad=10)
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3)
ax4.ticklabel_format(style='plain', axis='x')
ax4.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# Plot 5: Residuals Analysis
ax5 = fig.add_subplot(gs[1, 2])
y_pred_test = pinn.model(pinn.x_test_torch).detach().numpy().flatten()
residuals = pinn.y_test.flatten() - y_pred_test

ax5.scatter(pinn.x_test.flatten(), residuals, c='purple', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('log(P)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Residuals [log(dP/dt)]', fontsize=10, fontweight='bold')
ax5.set_title('Residual Analysis (Test Set)', fontsize=11, fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3)

residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
textstr = f'mu = {residual_mean:.4f}\nsigma = {residual_std:.4f}'
ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
         fontsize=9, family='monospace')

# Plot 6: Prediction vs True Values
ax6 = fig.add_subplot(gs[2, 0])
y_pred_all = pinn.model(pinn.x_test_torch).detach().numpy().flatten()
y_true_all = pinn.y_test.flatten()

ax6.scatter(y_true_all, y_pred_all, c='teal', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
min_val = min(y_true_all.min(), y_pred_all.min())
max_val = max(y_true_all.max(), y_pred_all.max())
ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax6.set_xlabel('True log(dP/dt)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Predicted log(dP/dt)', fontsize=10, fontweight='bold')
ax6.set_title('Prediction Accuracy (Test Set)', fontsize=11, fontweight='bold', pad=10)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

r2 = metrics['test_r2']
ax6.text(0.05, 0.95, f'R2 = {r2:.4f}', transform=ax6.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
         fontsize=10, fontweight='bold', family='monospace')

# Plot 7: Robustness Test Summary
ax7 = fig.add_subplot(gs[2, 1])
test_names = ['Permutation\nTest', 'Feature\nShuffling', 'Impossible\nPhysics']
test_results = [
    robustness_results['permutation_test']['is_significant'],
    robustness_results['feature_shuffling_test']['r2_difference'] > 0.1,
    robustness_results['impossible_physics_test']['real_much_better']
]
colors = ['green' if result else 'red' for result in test_results]

bars = ax7.bar(test_names, [1 if r else 0 for r in test_results], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax7.set_ylim([0, 1.2])
ax7.set_ylabel('Pass (1) / Fail (0)', fontsize=10, fontweight='bold')
ax7.set_title('Robustness Validation Summary', fontsize=11, fontweight='bold', pad=10)
ax7.set_yticks([0, 1])
ax7.set_yticklabels(['FAIL', 'PASS'])
ax7.grid(True, alpha=0.3, axis='y')

for i, (bar, result) in enumerate(zip(bars, test_results)):
    symbol = ' ' if result else 'X'
    ax7.text(bar.get_x() + bar.get_width()/2, 0.5, symbol, 
             ha='center', va='center', fontsize=30, fontweight='bold',
             color='white')

# Plot 8: Uncertainty Visualization
ax8 = fig.add_subplot(gs[2, 2])
constants_list = list(uncertainties.keys())
means = [uncertainties[c]['mean'] for c in constants_list]
stds = [uncertainties[c]['std'] for c in constants_list]
ci_lower = [uncertainties[c]['ci_lower'] for c in constants_list]
ci_upper = [uncertainties[c]['ci_upper'] for c in constants_list]

x_pos = np.arange(len(constants_list))
ax8.errorbar(x_pos, means, yerr=stds, fmt='o', markersize=10, 
             capsize=5, capthick=2, linewidth=2, color='darkblue', label='Mean +/- Std')
ax8.scatter(x_pos, ci_lower, marker='_', s=200, linewidths=3, color='red', label='95% CI')
ax8.scatter(x_pos, ci_upper, marker='_', s=200, linewidths=3, color='red')

ax8.set_xticks(x_pos)
ax8.set_xticklabels([c.replace('_', '\n') for c in constants_list], fontsize=9)
ax8.set_ylabel('Parameter Value', fontsize=10, fontweight='bold')
ax8.set_title('Uncertainty Quantification', fontsize=11, fontweight='bold', pad=10)
ax8.legend(fontsize=8, loc='upper right')
ax8.grid(True, alpha=0.3, axis='y')

plt.show()

# =============================================================================
# STEP 10: COMPREHENSIVE SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nTest R2: {metrics['test_r2']:.6f}")
print(f"Test RMSE: {metrics['test_rmse']:.6e}")
print(f"Test MAE: {metrics['test_mae']:.6e}")
print(f"\nn = {learned_n:.4f} +/- {uncertainties['n_braking']['std']:.4f}")
print(f"95% CI: [{uncertainties['n_braking']['ci_lower']:.4f}, {uncertainties['n_braking']['ci_upper']:.4f}]")
print(f"log(K) = {learned_logK:.4f} +/- {uncertainties['logK']['std']:.4f}")
print("\n" + "=" * 80)
```

![Figure 1 Data](./images/Figure_1.png)

## **Common Parameters**

pulsarfitpy offers analysis of other pulsar properties outside of those outlined in the example. Some common parameters are listed in the table below.

| Parameter | Description |
|-----------|-------------|
| `P0` | Pulsar period (s) |
| `P1` | Period derivative (s/s) |
| `DM` | Dispersion measure (pc/cm³) |
| `BSURF` | Surface magnetic field (G) |
| `EDOT` | Spin-down energy loss rate (erg/s) |

For further analysis of pulsar properties, see the [ATNF Pulsar Parameter List](https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html#:~:text=Appendix%20A%3A%20The%20Pulsar%20Parameters).

## **Next Steps**

- Explore the [Technical Information](quickstart.md)
- Check out the Jupyter notebooks in `src/pulsarfitpy/docs/`

[← Back to Home](index.md)