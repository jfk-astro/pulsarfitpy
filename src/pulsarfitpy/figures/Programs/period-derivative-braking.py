import sympy as sp
from pulsarfitpy import PulsarPINN
from psrqpy import QueryATNF
import numpy as np

# -----------------------------
# 1. Define symbols (log-space)
# -----------------------------
logP = sp.Symbol("logP")
logPdot = sp.Symbol("logPdot")
logC = sp.Symbol("logC")   # log(C), learnable
n = sp.Symbol("n")         # braking index, learnable

# Logarithmic ODE: logPdot = logC + (2 - n) * logP
residual = sp.Eq(logPdot, logC + (2 - n) * logP)

# -----------------------------
# 2. Load and filter data
# -----------------------------
query = QueryATNF(params=["P0", "P1"], condition="exist(P0) && exist(P1)")
table = query.table

P_data = table["P0"].data
Pdot_data = table["P1"].data

# Filter for positive values
mask = (P_data > 0) & (Pdot_data > 0)
P_data = P_data[mask]
Pdot_data = Pdot_data[mask]

# Take logs
logP_data = np.log(P_data)
logPdot_data = np.log(Pdot_data)

# -----------------------------
# 3. Instantiate and train PINN
# -----------------------------
pinn = PulsarPINN(
    x_param="P0",
    y_param="P1",
    differential_eq=residual,
    x_symbol=logP,
    y_symbol=logPdot,
    learn_constants={logC: -14.7529945391535211030031860, n: 0.2561328891074048375031680},  # Initial guesses
    log_scale=True  # Already working in log-space
)

# Train
pinn.train(epochs=10000)

# Plot
pinn.plot_PINN()
pinn.plot_PINN_loss()

# Report results
constants = pinn.store_learned_constants()
logC_val = constants.get("logC")
n_val = constants.get("n")
C_val = np.exp(logC_val)

print("\n📉 Discovered Model Parameters:")
print(f"  logC ≈ {logC_val:.6f}")
print(f"  n    ≈ {n_val:.6f}")
print(f"  C    ≈ {C_val:.4e} (from exp(logC))")