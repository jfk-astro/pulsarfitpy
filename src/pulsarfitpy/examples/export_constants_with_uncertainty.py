"""
Example: Exporting Constants with Uncertainty Information

Demonstrates how to export learned constants with confidence intervals
computed using bootstrap or Monte Carlo methods.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from pulsarfitpy.pinn import PulsarPINN
from pulsarfitpy.export_solutions import ExportPINN
import pandas as pd

print("=" * 80)
print("EXPORT CONSTANTS WITH UNCERTAINTY")
print("=" * 80)

# Define physics model
logP, logPDOT, logB = sp.symbols('logP logPDOT logB')
logR = sp.Symbol('logR')
differential_equation = sp.Eq(logB, logR + 0.5 * logP + 0.5 * logPDOT)

# Query ATNF data
print("\nQuerying ATNF data...")
query = QueryATNF(
    params=['P0', 'P1', 'BSURF'],
    condition='exist(P0) && exist(P1) && exist(BSURF)'
)
table = query.table

logP_data = np.log10(table['P0'].data)
logPDOT_data = np.log10(table['P1'].data)
logB_data = np.log10(table['BSURF'].data)

print(f"Retrieved {len(logP_data)} pulsars")

# Initialize and train PINN
print("\nInitializing and training PINN...")

pinn = PulsarPINN(
    differential_eq=differential_equation,
    x_sym=logP,
    y_sym=logB,
    learn_constants={logR: 18.0},
    log_scale=True,
    fixed_inputs={
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    },
    hidden_layers=[16, 32, 16],
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    random_seed=42
)

pinn.train(epochs=3000, training_reports=1000, physics_weight=1.0, data_weight=1.0)

# Initialize exporter
exporter = ExportPINN(pinn_model=pinn)

# =============================================================================
# EXAMPLE 1: Export without uncertainty
# =============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: Export constants WITHOUT uncertainty")
print("=" * 80)

exporter.save_learned_constants_to_csv(
    filepath="results/constants_no_uncertainty.csv",
    additional_info={
        "data_source": "ATNF",
        "n_pulsars": len(logP_data),
        "training_epochs": 3000
    }
)

print("\nSaved to: results/constants_no_uncertainty.csv")
df1 = pd.read_csv("results/constants_no_uncertainty.csv")
print("\nContents:")
print(df1.to_string(index=False))

# =============================================================================
# EXAMPLE 2: Export with bootstrap uncertainty
# =============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Export constants WITH bootstrap uncertainty")
print("=" * 80)
print("This will take a few minutes as it retrains the model 100 times...")

exporter.save_learned_constants_to_csv(
    filepath="results/constants_bootstrap_uncertainty.csv",
    additional_info={
        "data_source": "ATNF",
        "n_pulsars": len(logP_data),
        "training_epochs": 3000
    },
    include_uncertainty=True,
    uncertainty_method='bootstrap',
    n_iterations=100
)

print("\nSaved to: results/constants_bootstrap_uncertainty.csv")
df2 = pd.read_csv("results/constants_bootstrap_uncertainty.csv")
print("\nContents:")
print(df2.to_string(index=False))

# =============================================================================
# EXAMPLE 3: Export with Monte Carlo uncertainty
# =============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: Export constants WITH Monte Carlo uncertainty")
print("=" * 80)
print("This will also take a few minutes...")

exporter.save_learned_constants_to_csv(
    filepath="results/constants_monte_carlo_uncertainty.csv",
    additional_info={
        "data_source": "ATNF",
        "n_pulsars": len(logP_data),
        "training_epochs": 3000
    },
    include_uncertainty=True,
    uncertainty_method='monte_carlo',
    n_iterations=100
)

print("\nSaved to: results/constants_monte_carlo_uncertainty.csv")
df3 = pd.read_csv("results/constants_monte_carlo_uncertainty.csv")
print("\nContents:")
print(df3.to_string(index=False))

# =============================================================================
# EXAMPLE 4: Format for publication
# =============================================================================

print("\n" + "=" * 80)
print("PUBLICATION-READY FORMATTING")
print("=" * 80)

for _, row in df2.iterrows():
    name = row['constant_name']
    value = row['value']
    std = row['uncertainty_std']
    ci_lower = row['ci_lower_95']
    ci_upper = row['ci_upper_95']
    
    print(f"\n{name}:")
    print(f"  Value ± Uncertainty:  {value:.4f} ± {std:.4f}")
    print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"\n  Plain text for paper:")
    print(f"  {name} = {value:.4f} ± {std:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"\n  LaTeX format:")
    print(f"  ${name} = {value:.4f} \\pm {std:.4f}$")
    print(f"  ${name} \\in [{ci_lower:.4f}, {ci_upper:.4f}]$ (95\\% CI)")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - results/constants_no_uncertainty.csv")
print("  - results/constants_bootstrap_uncertainty.csv")
print("  - results/constants_monte_carlo_uncertainty.csv")
