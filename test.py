import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from psrqpy import QueryATNF
from pulsarfitpy import PulsarPINN

# === Define symbols ===
x = sp.Symbol("x")  # log10(P0)
y = sp.Symbol("y")  # log10(EDOT)
a = sp.Symbol("a")  # slope
b = sp.Symbol("b")  # intercept

# === Physics equation ===
physics_eq = sp.Eq(y, a * x + b)

# === Pull real ATNF pulsar data ===
query = QueryATNF(params=["P0", "EDOT"])
table = query.table

# === Filter valid, positive values ===
p0_vals = np.array(table["P0"], dtype=float)
edot_vals = np.array(table["EDOT"], dtype=float)
mask = np.isfinite(p0_vals) & np.isfinite(edot_vals) & (p0_vals > 0) & (edot_vals > 0)
p0_vals = p0_vals[mask]
edot_vals = edot_vals[mask]

# === Log-transform ===
logP = np.log10(p0_vals)
logEDOT = np.log10(edot_vals)

# === Build and train model ===
model = PulsarPINN(
    x_sym=x,
    y_sym=y,
    differential_equation=physics_eq,
    learn_constant={a: -3.0, b: 35.0},
    x_data=p0_vals,
    y_data=edot_vals,
    log_scale=True
)

model.train(epochs=3000)

# === Get prediction ===
x_grid, y_pred = model.predict_extended()

# === Plot ===
plt.figure(figsize=(10, 6))
plt.scatter(logP, logEDOT, s=10, alpha=0.4, label="ATNF Data", color="blue")
plt.plot(x_grid, y_pred, label="PINN Fit", color="red", linewidth=2)
plt.xlabel("log10(P0) [s]")
plt.ylabel("log10(EDOT) [erg/s]")
plt.title("Pulsar Spin-down: Fitting logEDOT = a·logP0 + b")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Show learned constants ===
constants = model.show_learned_constants()
print(f"\na = {constants['a']:.6f}")
print(f"b = {constants['b']:.6f}")

# === Cite ATNF ===
print("\nData from ATNF Pulsar Catalogue (Manchester et al. 2005): https://www.atnf.csiro.au/research/pulsar/psrcat/")