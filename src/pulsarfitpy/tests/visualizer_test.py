"""
Test script for PulsarPINN with ATNF pulsar database.

This script demonstrates using the refactored PulsarPINN class to learn
the relationship between pulsar period, period derivative, and surface
magnetic field.
"""

import sympy as sp
import numpy as np
from psrqpy import QueryATNF


from ..modules.pinn import PulsarPINN
from ..modules.pinn_visualizer import VisualizePINN

# =============================================================================
# STEP 1: Define Symbolic Variables and Physics Equation
# =============================================================================

logP, logPDOT, logB = sp.symbols('logP logPDOT logB')
logR = sp.Symbol('logR')

# Differential equation: logB = logR + 0.5*logP + 0.5*logPDOT
differential_equation = sp.Eq(logB, logR + 0.5 * logP + 0.5 * logPDOT)

print("Physics Model:")
print(f"  {differential_equation}")

# =============================================================================
# STEP 2: Query ATNF Pulsar Database
# =============================================================================

print("\nQuerying ATNF pulsar catalogue...")
query = QueryATNF(params=['P0', 'P1', 'BSURF'], condition='exist(P0) && exist(P1) && exist(BSURF)')
table = query.table

# Extract raw data
P = table['P0'].data       # Period in seconds
PDOT = table['P1'].data    # Period derivative (dimensionless)
BSURF = table['BSURF'].data  # Surface magnetic field in Gauss

print(f"  Retrieved {len(P)} pulsars from ATNF catalogue")

# =============================================================================
# STEP 3: Prepare Data (Log Transform)
# =============================================================================

logP_data = np.log10(P)
logPDOT_data = np.log10(PDOT)
logB_data = np.log10(BSURF)

# =============================================================================
# STEP 4: Configure PINN
# =============================================================================

# Neural network architecture
architecture_NN = [16, 32, 16]

# Initial guess for unknown constant logR
learn_constants = {logR: 18.0}

# Fixed inputs: ALL variables in the equation
fixed_data = {
    logP: logP_data,
    logPDOT: logPDOT_data,
    logB: logB_data
}

print("\nPINN Configuration: \n")
print(f"  Network architecture: {architecture_NN}")
print(f"  Learning constant: logR (initial guess = {learn_constants[logR]})")

# =============================================================================
# STEP 5: Initialize PINN
# =============================================================================

pinn = PulsarPINN(
    differential_eq=differential_equation,
    x_sym=logP,
    y_sym=logB,
    learn_constants=learn_constants,
    log_scale=True,
    fixed_inputs=fixed_data,
    input_layer=3,
    hidden_layers=architecture_NN,
    output_layer=3,
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    random_seed=42,
    solution_name="Magnetic Field PINN Model"
)

# =============================================================================
# STEP 6: Train the PINN
# =============================================================================

print("TRAINING PHASE: \n")

pinn.train(
    epochs=10000,
    training_reports=500,
    physics_weight=1.0,
    data_weight=1.0
)

# =============================================================================
# STEP 7: Evaluate Model Performance
# =============================================================================

print("EVALUATION PHASE: \n")

test_metrics = pinn.evaluate_test_set(verbose=True)

# =============================================================================
# STEP 8: Access Learned Constants
# =============================================================================

print("LEARNED PHYSICAL CONSTANTS: \n")

learned_constants = pinn.store_learned_constants()
print(learned_constants)

# =============================================================================
# STEP 9: Generate Extended Predictions
# =============================================================================

print("PREDICTION PHASE \n")

x_extended, y_extended = pinn.predict_extended(extend=0.5, n_points=500)
print(f"  Generated {len(x_extended)} prediction points")
print(f"  X range: [{x_extended.min():.3f}, {x_extended.max():.3f}]")
print(f"  Y range: [{y_extended.min():.3f}, {y_extended.max():.3f}]")

# =============================================================================
# STEP 10: Visualize Results
# =============================================================================

print("\n")
print("VISUALIZATION PHASE:")

visualizer = VisualizePINN(pinn_model=pinn)

print("  Plotting loss curves...")
visualizer.plot_loss_curves(log_scale=True)

print("  Plotting predictions vs data...")
visualizer.plot_predictions_vs_data(
    x_values=x_extended,
    y_predictions=y_extended,
    x_axis='log₁₀(Period) [s]',
    y_axis='log₁₀(Surface B-field) [G]',
    figsize=(7, 8),
    title='PINN: Pulsar Magnetic Field vs Period'
)