import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from pulsarfitpy import PulsarPINN

plt.rcParams.update({'font.size': 12})

# -----------------------------------------------------
# MODEL 1: Surface Magnetic Field vs Period
# -----------------------------------------------------
logP1, logPDOT1, logB1 = sp.symbols('logP logPDOT logB')
logR = sp.Symbol('logR')
eq1 = sp.Eq(logB1, logR + 0.5 * (logP1) + 0.5 * (logPDOT1))

query1 = QueryATNF(params=['P0', 'P1', 'BSURF'], condition='exist(P0) && exist(P1) && exist(BSURF)')
table1 = query1.table
P1, PDOT1, BSURF1 = table1['P0'].data, table1['P1'].data, table1['BSURF'].data

mask1 = (P1 > 0) & (PDOT1 > 0) & (BSURF1 > 0)
P1, PDOT1, BSURF1 = P1[mask1], PDOT1[mask1], BSURF1[mask1]
logPDOT_data1 = np.log10(PDOT1)

pinn1 = PulsarPINN(
    x_param='P0',
    y_param='BSURF',
    differential_eq=eq1,
    x_sym=logP1,
    y_sym=logB1,
    learn_constants={logR: 18},
    log_scale=True,
    fixed_inputs={logPDOT1: logPDOT_data1}
)
pinn1.train(epochs=10000)

# -----------------------------------------------------
# MODEL 2: Spin-down Luminosity vs Period
# -----------------------------------------------------
x2 = sp.Symbol("logP")
y2 = sp.Symbol("logEDOT")
z2 = sp.Symbol("logPDOT")
logK = sp.Symbol("logK")

eq2 = sp.Eq(y2, logK + (-3 * x2) + (1 * z2))

query2 = QueryATNF(params=["P0", "P1", "EDOT"], condition="exist(P0) && exist(P1) && exist(EDOT)")
table2 = query2.table
P2, PDOT2, EDOT2 = table2["P0"].data, table2["P1"].data, table2["EDOT"].data

mask2 = (P2 > 0) & (PDOT2 > 0) & (EDOT2 > 0)
P2, PDOT2, EDOT2 = P2[mask2], PDOT2[mask2], EDOT2[mask2]
logPDOT2 = np.log10(PDOT2)

pinn2 = PulsarPINN(
    x_param="P0",
    y_param="EDOT",
    differential_eq=eq2,
    x_sym=x2,
    y_sym=y2,
    learn_constants={logK: 46.56},
    log_scale=True,
    fixed_inputs={z2: logPDOT2}
)
pinn2.train(epochs=10000)

# -----------------------------------------------------
# MODEL 3: Braking Law logPdot = logC + (2 - n) * logP
# -----------------------------------------------------
x3 = sp.Symbol("logP")
y3 = sp.Symbol("logPdot")
logC = sp.Symbol("logC")
n = sp.Symbol("n")

eq3 = sp.Eq(y3, logC + (2 - n) * x3)

query3 = QueryATNF(params=["P0", "P1"], condition="exist(P0) && exist(P1)")
table3 = query3.table
P3, PDOT3 = table3["P0"].data, table3["P1"].data

mask3 = (P3 > 0) & (PDOT3 > 0)
P3, PDOT3 = P3[mask3], PDOT3[mask3]

pinn3 = PulsarPINN(
    x_param="P0",
    y_param="P1",
    differential_eq=eq3,
    x_sym=x3,
    y_sym=y3,
    learn_constants={logC: -14.75, n: 0.25},
    log_scale=True
)
pinn3.train(epochs=10000)

# -----------------------------------------------------
# PLOT: Total Loss for all 3 Models
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Total Loss vs Epochs - All Models")
ax.set_xlabel("Epochs")
ax.set_ylabel("Total Loss")
ax.set_yscale("log")
ax.grid(True, which="both", ls="--", linewidth=0.5)

# Plot only total loss for each model
ax.plot(pinn1.loss_log["total"], label="PINN Model 1", color="#1000BE", linewidth=2)
ax.plot(pinn2.loss_log["total"], label="PINN Model 2", color="#d62728", linewidth=2)
ax.plot(pinn3.loss_log["total"], label="PINN Model 3", color="#ff7f0e", linewidth=2)

ax.legend()
plt.tight_layout()
plt.savefig("figures/testFig5.png", dpi=600)
plt.show()