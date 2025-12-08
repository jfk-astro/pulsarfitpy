import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from psrqpy import QueryATNF
from pulsarfitpy import PulsarPINN

# -------------------------------------------------------
# MODEL 2: Spin-down Luminosity y = logK + a * logP + b * logPDOT
# -------------------------------------------------------
x2 = sp.Symbol("logP")
y2 = sp.Symbol("logEDOT")
z2 = sp.Symbol("logPDOT")
logK = sp.Symbol("logK")

a, b = -3, 1
eq2 = sp.Eq(y2, logK + a * x2 + b * z2)

query2 = QueryATNF(params=["P0", "P1", "EDOT"], condition="exist(P0) && exist(P1) && exist(EDOT)")
table2 = query2.table
P2, PDOT2, EDOT2 = table2["P0"].data, table2["P1"].data, table2["EDOT"].data

mask2 = (P2 > 0) & (PDOT2 > 0) & (EDOT2 > 0)
P2, PDOT2, EDOT2 = P2[mask2], PDOT2[mask2], EDOT2[mask2]

logP2 = np.log10(P2)
logPDOT2 = np.log10(PDOT2)

pinn2 = PulsarPINN(
    x_param="P0",
    y_param="EDOT",
    differential_eq=eq2,
    x_symbol=x2,
    y_symbol=y2,
    learn_constants={logK: 46.5497453422014118018523732},
    log_scale=True,
    fixed_inputs={z2: logPDOT2}
)
pinn2.train(epochs=17000)

# -------------------------------------------------------
# MODEL 3: Logarithmic Braking Law logPdot = logC + (2 - n) * logP
# -------------------------------------------------------
x3 = sp.Symbol("logP")
y3 = sp.Symbol("logPdot")
logC = sp.Symbol("logC")
n = sp.Symbol("n")

eq3 = sp.Eq(y3, logC + (2 - n) * x3)

query3 = QueryATNF(params=["P0", "P1"], condition="exist(P0) && exist(P1)")
table3 = query3.table
P3 = table3["P0"].data
PDOT3 = table3["P1"].data

mask3 = (P3 > 0) & (PDOT3 > 0)
P3, PDOT3 = P3[mask3], PDOT3[mask3]

logP3 = np.log(P3)
logPDOT3 = np.log(PDOT3)

pinn3 = PulsarPINN(
    x_param="P0",
    y_param="P1",
    differential_eq=eq3,
    x_symbol=x3,
    y_symbol=y3,
    learn_constants={logC: -14.75, n: 0.25},
    log_scale=True
)
pinn3.train(epochs=10000)

# -------------------------------------------------------
# PLOTTING: Side-by-side Comparison
# -------------------------------------------------------
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(wspace=0.3)

# --------------------------------------
# LEFT: Spin-down Luminosity Model (Model 2)
# --------------------------------------
ax = axes[0]
ax.set_title("Spin-down Energy Loss vs. Period - PINN Model #2")
ax.set_xlabel("log(Period) [log(s)]")
ax.set_ylabel("log(Spin Down Energy Loss) [log(erg/s)]")
ax.grid(True)

x_data2 = pinn2.x_raw
y_data2 = pinn2.y_raw
x_PINN2, y_PINN2 = pinn2.predict_extended()

ax.scatter(x_data2, y_data2, s=8, alpha=0.4, label="ATNF Pulsars")
ax.plot(x_PINN2, y_PINN2, label="PINN Prediction", color="#d62728", linewidth=2)
ax.legend()

# --------------------------------------
# RIGHT: Braking Law Model (Model 3)
# --------------------------------------
ax = axes[1]
ax.set_title("Period Derivative vs. Period - PINN Model #3")
ax.set_xlabel("log(Period) [log(s)]")
ax.set_ylabel("log(Period Derivative) [log(s/s)]")
ax.grid(True)

x_data3 = pinn3.x_raw
y_data3 = pinn3.y_raw
x_PINN3, y_PINN3 = pinn3.predict_extended()

ax.scatter(x_data3, y_data3, s=8, alpha=0.4, label="ATNF Pulsars")
ax.plot(x_PINN3, y_PINN3, label="PINN Prediction", color="#ff7f0e", linewidth=2)
ax.legend()

# -------------------------------------------------------
# SAVE & SHOW
# -------------------------------------------------------
plt.tight_layout()
plt.savefig("figures/Figure4.png", dpi=600)
plt.show()