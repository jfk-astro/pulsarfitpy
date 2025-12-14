"""
Example: Uncertainty Quantification for PINN Learned Constants

This example demonstrates how to use bootstrap and Monte Carlo methods
to estimate uncertainty and confidence intervals for physical constants
learned by the PINN model.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from modules.pinn import PulsarPINN
from modules.pinn_visualizer import VisualizePINN

print("=" * 80)
print("PINN UNCERTAINTY QUANTIFICATION EXAMPLE")
print("=" * 80)

# =============================================================================
# STEP 1: Define Physics Model
# =============================================================================

print("\n1. Defining physics model...")

logP, logPDOT, logB = sp.symbols('logP logPDOT logB')
logR = sp.Symbol('logR')

# Surface magnetic field equation: logB = logR + 0.5*logP + 0.5*logPDOT
differential_equation = sp.Eq(logB, logR + 0.5 * logP + 0.5 * logPDOT)

print(f"   Equation: {differential_equation}")

# =============================================================================
# STEP 2: Load and Prepare Data
# =============================================================================

print("\n2. Querying ATNF pulsar catalogue...")

query = QueryATNF(
    params=['P0', 'P1', 'BSURF'],
    condition='exist(P0) && exist(P1) && exist(BSURF)'
)
table = query.table

P = table['P0'].data
PDOT = table['P1'].data
BSURF = table['BSURF'].data

logP_data = np.log10(P)
logPDOT_data = np.log10(PDOT)
logB_data = np.log10(BSURF)

print(f"   Retrieved {len(P)} pulsars")

# =============================================================================
# STEP 3: Initialize and Train PINN
# =============================================================================

print("\n3. Initializing PINN...")

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

print("\n4. Training PINN (this may take a minute)...")

pinn.train(
    epochs=5000,
    training_reports=1000,
    physics_weight=1.0,
    data_weight=1.0
)

# =============================================================================
# STEP 4: Get Point Estimate
# =============================================================================

print("\n" + "=" * 80)
print("POINT ESTIMATE (ORIGINAL FIT)")
print("=" * 80)

learned_constants = pinn.store_learned_constants()
print(f"\nLearned constant: logR = {learned_constants['logR']:.6f}")
print(f"Physical interpretation: R ≈ 10^{learned_constants['logR']:.2f} cm")

# =============================================================================
# STEP 5: Bootstrap Uncertainty Analysis
# =============================================================================

print("\n" + "=" * 80)
print("BOOTSTRAP UNCERTAINTY ANALYSIS")
print("=" * 80)
print("This resamples the data and retrains the model multiple times")
print("to estimate uncertainty from data variability.\n")

bootstrap_results = pinn.bootstrap_uncertainty(
    n_bootstrap=100,
    sample_fraction=0.8,
    epochs=1000,
    confidence_level=0.95,
    verbose=True
)

# =============================================================================
# STEP 6: Monte Carlo Uncertainty Analysis
# =============================================================================

print("\n" + "=" * 80)
print("MONTE CARLO UNCERTAINTY ANALYSIS")
print("=" * 80)
print("This adds noise to the data and retrains to estimate")
print("sensitivity to measurement uncertainties.\n")

mc_results = pinn.monte_carlo_uncertainty(
    n_simulations=100,
    noise_level=0.01,
    confidence_level=0.95,
    verbose=True
)

# =============================================================================
# STEP 7: Compare Methods
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON OF UNCERTAINTY ESTIMATES")
print("=" * 80)

for const_name in learned_constants.keys():
    print(f"\n{const_name}:")
    print(f"  Point estimate:        {learned_constants[const_name]:.6f}")
    print(f"\n  Bootstrap method:")
    print(f"    Mean ± Std:          {bootstrap_results[const_name]['mean']:.6f} ± {bootstrap_results[const_name]['std']:.6f}")
    print(f"    95% CI:              [{bootstrap_results[const_name]['ci_lower']:.6f}, {bootstrap_results[const_name]['ci_upper']:.6f}]")
    print(f"    CI width:            {bootstrap_results[const_name]['ci_upper'] - bootstrap_results[const_name]['ci_lower']:.6f}")
    
    print(f"\n  Monte Carlo method:")
    print(f"    Mean ± Std:          {mc_results[const_name]['mean']:.6f} ± {mc_results[const_name]['std']:.6f}")
    print(f"    95% CI:              [{mc_results[const_name]['ci_lower']:.6f}, {mc_results[const_name]['ci_upper']:.6f}]")
    print(f"    CI width:            {mc_results[const_name]['ci_upper'] - mc_results[const_name]['ci_lower']:.6f}")

# =============================================================================
# STEP 8: Recommendations for Reporting
# =============================================================================

print("\n" + "=" * 80)
print("RECOMMENDED REPORTING FORMAT")
print("=" * 80)

for const_name in learned_constants.keys():
    boot_mean = bootstrap_results[const_name]['mean']
    boot_std = bootstrap_results[const_name]['std']
    boot_lower = bootstrap_results[const_name]['ci_lower']
    boot_upper = bootstrap_results[const_name]['ci_upper']
    
    print(f"\n{const_name}:")
    print(f"  Value (bootstrap mean):  {boot_mean:.6f}")
    print(f"  Uncertainty (std):       ±{boot_std:.6f}")
    print(f"  95% Confidence Interval: [{boot_lower:.6f}, {boot_upper:.6f}]")
    print(f"\n  LaTeX format:")
    print(f"  {const_name} = {boot_mean:.3f} \\pm {boot_std:.3f}")
    print(f"  {const_name} \\in [{boot_lower:.3f}, {boot_upper:.3f}] \\text{{(95\\% CI)}}")

print("\n" + "=" * 80)
print("INTERPRETATION NOTES")
print("=" * 80)
print("""
- Bootstrap resampling tests robustness to data selection
- Monte Carlo tests sensitivity to measurement noise
- Both methods should give similar results if model is stable
- Larger uncertainties suggest:
  * Limited data
  * High data variability
  * Model sensitivity to initialization
  
- For publication, report:
  * Point estimate with uncertainty: value ± std
  * 95% confidence interval
  * Method used (bootstrap or Monte Carlo)
  * Number of iterations performed
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
