import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from pulsarfitpy import PulsarPINN

x = sp.Symbol("logP")
y = sp.Symbol("logEDOT")
z = sp.Symbol("logPDOT")
logK = sp.Symbol("logK")

# Spin-down luminosity model: y = logK + a*x + b*z
a = -3
b = 1
eq = sp.Eq(y, logK + a*x + b*z)

query = QueryATNF(params=["P0", "P1", "EDOT"], condition="exist(P0) && exist(P1) && exist(EDOT)")
table = query.table

P = table["P0"].data
PDOT = table["P1"].data
EDOT = table["EDOT"].data

mask = (P > 0) & (PDOT > 0) & (EDOT > 0)
P = P[mask]
PDOT = PDOT[mask]
EDOT = EDOT[mask]

logP = np.log10(P)
logPDOT = np.log10(PDOT)
logEDOT = np.log10(EDOT)

learn_constants = {logK: 46.5645176814723100733317551}  # initial guess

pinn = PulsarPINN(
    x_param="P0",
    y_param="EDOT",
    differential_eq=eq,
    x_sym=x,
    y_sym=y,
    learn_constants=learn_constants,
    log_scale=True,
    fixed_inputs={z: logPDOT}
)

pinn.train(epochs=9000)
pinn.plot_PINN()
pinn.plot_PINN_loss()

constants = pinn.store_learned_constants()
print(constants)