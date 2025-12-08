"""
Test script for PulsarPINN with ATNF pulsar database.

This script demonstrates using the refactored PulsarPINN class to learn
the relationship between pulsar period, period derivative, and surface
magnetic field.
"""

import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from src.pulsarfitpy.pulsarfitpy.pinn import PulsarPINN
import matplotlib.pyplot as plt
from datetime import datetime

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

print("Querying ATNF pulsar catalogue...")
query = QueryATNF(params=['P0', 'P1', 'BSURF'], condition='exist(P0) && exist(P1) && exist(BSURF)')
table = query.table

# Extract raw data
P = table['P0'].data       # Period in seconds
PDOT = table['P1'].data    # Period derivative (dimensionless)
BSURF = table['BSURF'].data  # Surface magnetic field in Gauss

print(f"  Retrieved {len(P)} pulsars from ATNF catalogue")
print()

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

print("PINN Configuration:")
print(f"  Network architecture: {architecture_NN}")
print(f"  Learning constant: logR (initial guess = {learn_constants[logR]})")
print()

# =============================================================================
# STEP 5: Initialize and Train PINN
# =============================================================================

pinn = PulsarPINN(
    differential_eq=differential_equation,
    x_sym=logP,              # Independent variable (input to NN)
    y_sym=logB,              # Dependent variable (output of NN)
    learn_constants=learn_constants,
    log_scale=True,          # Data already in log scale
    fixed_inputs=fixed_data,
    hidden_layers=architecture_NN,
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    random_seed=42,
    solution_name="Magnetic Field PINN Model"
)

print("="*70)
print("TRAINING PINN")
print("="*70)
pinn.train(epochs=10000, training_reports=500)

# Extract learned constants
learned_values = pinn.store_learned_constants()
print(f"\nFinal learned constant: logR = {learned_values['logR']:.8f}")

# Plotting tests
pinn.plot_predictions_vs_data(x_name="Period [s]", y_name="Surface Magnetic Field [G]")
pinn.plot_loss_curves(log_scale=True)
pinn.save_predictions_to_csv(filepath="C:\\repos\\pulsarfitpy\\results", x_value_name="Period [s]", y_value_name="Surface Magnetic Field [G]")