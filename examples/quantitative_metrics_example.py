"""
Example demonstrating quantitative metrics for objective model assessment.

This script shows how to use RMSE, MAE, and reduced χ² metrics to objectively
assess model fit, going beyond visual comparison alone.
"""

import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from src.pulsarfitpy import PulsarPINN, PulsarApproximation
import matplotlib.pyplot as plt

print("="*80)
print("QUANTITATIVE METRICS DEMONSTRATION")
print("="*80)
print("\nThis example demonstrates objective model assessment using:")
print("  • RMSE (Root Mean Squared Error)")
print("  • MAE (Mean Absolute Error)")
print("  • Reduced χ² (Chi-Squared)")
print("  • R² (Coefficient of Determination)")
print("="*80)

# =============================================================================
# EXAMPLE 1: PINN Model with Quantitative Metrics
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: Physics-Informed Neural Network (PINN)")
print("="*80)

# Define the differential equation for surface magnetic field
logP, logPDOT, logB = sp.symbols('logP logPDOT logB')
logR = sp.Symbol('logR')
differential_equation = sp.Eq(logB, logR + 0.5 * (logP) + 0.5 * (logPDOT))

# Query ATNF data
query = QueryATNF(params=['P0', 'P1', 'BSURF'], 
                 condition='exist(P0) && exist(P1) && exist(BSURF)')
table = query.table

P = table['P0'].data 
PDOT = table['P1'].data  
BSURF = table['BSURF'].data 
logPDOT_data = np.log10(PDOT)

# Setup PINN
learn_constants = {logR: 18}
fixed_data = {logPDOT: logPDOT_data}

pinn = PulsarPINN(
    x_param='P0',    
    y_param='BSURF',          
    differential_eq=differential_equation,
    x_sym=logP,
    y_sym=logB,
    learn_constants=learn_constants,
    log_scale=True,
    fixed_inputs=fixed_data,
    hidden_layers=[32, 16],
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    random_seed=42
)

print("\nTraining PINN model...")
pinn.train(epochs=5000, val_interval=500)

print("\n" + "-"*80)
print("EVALUATING WITH QUANTITATIVE METRICS")
print("-"*80)
metrics = pinn.evaluate_test_set(verbose=True)

# Display learned constants
print("\n" + "-"*80)
print("LEARNED PHYSICAL CONSTANTS")
print("-"*80)
learned = pinn.store_learned_constants()
for const, value in learned.items():
    print(f"  {const} = {value:.10f}")
    if const == 'logR':
        R_value = 10 ** value
        print(f"  R = 10^{value:.4f} = {R_value:.4e}")
print("-"*80)

# Plot results with metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Model fit with splits
ax1 = axes[0]
ax1.scatter(pinn.x_train, pinn.y_train, s=10, alpha=0.4, c='blue', label=f'Train (n={len(pinn.x_train)})')
ax1.scatter(pinn.x_val, pinn.y_val, s=10, alpha=0.4, c='orange', label=f'Val (n={len(pinn.x_val)})')
ax1.scatter(pinn.x_test, pinn.y_test, s=10, alpha=0.5, c='green', label=f'Test (n={len(pinn.x_test)})')
x_pred, y_pred = pinn.predict_extended()
ax1.plot(x_pred, y_pred, 'r-', linewidth=2, label='PINN Model')
ax1.set_xlabel('log10(Period) [s]')
ax1.set_ylabel('log10(Surface Magnetic Field) [G]')
ax1.set_title(f'PINN Fit\nTest R²={metrics["test_r2"]:.4f}, RMSE={metrics["test_rmse"]:.4e}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss curves
ax2 = axes[1]
epochs = range(len(pinn.loss_log["train_total"]))
ax2.plot(epochs, pinn.loss_log["train_total"], label='Train Total', linewidth=1.5)
if pinn.loss_log["val_total"]:
    val_interval = len(pinn.loss_log["train_total"]) // len(pinn.loss_log["val_total"])
    val_epochs = list(range(0, len(pinn.loss_log["train_total"]), val_interval))[:len(pinn.loss_log["val_total"])]
    ax2.plot(val_epochs, pinn.loss_log["val_total"], 'o-', label='Val Total', linewidth=1.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_yscale('log')
ax2.set_title('Training Progress')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pinn_quantitative_metrics.png', dpi=150, bbox_inches='tight')
print("\n✓ Figure saved as 'pinn_quantitative_metrics.png'")
plt.show()

# =============================================================================
# EXAMPLE 2: Polynomial Approximation with Quantitative Metrics
# =============================================================================
print("\n\n" + "="*80)
print("EXAMPLE 2: Polynomial Approximation")
print("="*80)

# Query data for period vs period derivative
query2 = QueryATNF(params=['P0', 'P1'], 
                   condition='exist(P0) && exist(P1) && P0 > 0 && P1 > 0')

approx = PulsarApproximation(
    query=query2,
    x_param='P0',
    y_param='P1',
    test_degree=5,
    log_x=True,
    log_y=True
)

print("\nFitting polynomial models...")
approx.fit_polynomial(verbose=True)

print("\n" + "-"*80)
print("COMPUTING QUANTITATIVE METRICS")
print("-"*80)
poly_metrics = approx.compute_metrics(verbose=True)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Polynomial fit
ax1 = axes[0]
ax1.scatter(approx.x_data, approx.y_data, s=8, alpha=0.4, label='ATNF Data')
ax1.plot(approx.predicted_x, approx.predicted_y, 'r-', linewidth=2, 
         label=f'Degree {approx.best_degree} Fit')
ax1.set_xlabel('log10(Period) [s]')
ax1.set_ylabel('log10(Period Derivative)')
ax1.set_title(f'Polynomial Fit\nR²={poly_metrics["r2"]:.4f}, RMSE={poly_metrics["rmse"]:.4e}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: R² vs degree
ax2 = axes[1]
degrees = list(approx.r2_scores.keys())
scores = list(approx.r2_scores.values())
ax2.plot(degrees, scores, 'o-', linewidth=2, markersize=8)
ax2.axhline(y=poly_metrics['r2'], color='r', linestyle='--', 
            label=f'Best: Degree {approx.best_degree}')
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('R² Score')
ax2.set_title('Model Selection')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polynomial_quantitative_metrics.png', dpi=150, bbox_inches='tight')
print("\n✓ Figure saved as 'polynomial_quantitative_metrics.png'")
plt.show()

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================
print("\n\n" + "="*80)
print("SUMMARY: QUANTITATIVE METRICS COMPARISON")
print("="*80)
print("\nPINN Model (Test Set):")
print(f"  R² Score:        {metrics['test_r2']:.6f}")
print(f"  RMSE:            {metrics['test_rmse']:.6e}")
print(f"  MAE:             {metrics['test_mae']:.6e}")
print(f"  Reduced χ²:      {metrics['test_chi2_reduced']:.6f}")

print("\nPolynomial Model:")
print(f"  R² Score:        {poly_metrics['r2']:.6f}")
print(f"  RMSE:            {poly_metrics['rmse']:.6e}")
print(f"  MAE:             {poly_metrics['mae']:.6e}")
print(f"  Reduced χ²:      {poly_metrics['chi2_reduced']:.6f}")

print("\n" + "="*80)
print("INTERPRETATION GUIDE:")
print("="*80)
print("""
• R² (Coefficient of Determination):
  - Measures proportion of variance explained by the model
  - Range: [0, 1], closer to 1 is better
  - Values > 0.9 indicate excellent fit

• RMSE (Root Mean Squared Error):
  - Penalizes larger errors more heavily
  - Same units as the target variable
  - Lower values indicate better fit
  - Compare models: lower RMSE = better performance

• MAE (Mean Absolute Error):
  - Average absolute difference between predictions and actual
  - More robust to outliers than RMSE
  - Lower values indicate better fit

• Reduced χ² (Chi-Squared):
  - Goodness-of-fit measure normalized by degrees of freedom
  - Values close to 1.0 indicate good fit
  - < 0.5: possible overfitting
  - > 2.0: possible underfitting or systematic errors
  - Between 0.5-2.0: reasonable fit
""")
print("="*80)
print("\n✓ Analysis complete!")
