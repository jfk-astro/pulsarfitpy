import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from pulsarfitpy import PulsarPINN

# ------------------------------------------
# Step 1: Define symbolic variables
# ------------------------------------------
logP, logPDOT, logEDOT = sp.symbols('logP logPDOT logEDOT')
logK = sp.Symbol('logK')  # log(4π²I) ~ 45–50 depending on I

# Rotational energy loss formula in log10 form:
# logEDOT = logK + logPDOT - 3 * logP
energy_eq = sp.Eq(logEDOT, logK + logPDOT - 3 * logP)

# ------------------------------------------
# Step 2: Query ATNF data
# ------------------------------------------
query = QueryATNF(params=['P0', 'P1', 'EDOT'],
                  condition='exist(P0) && exist(P1) && exist(EDOT)')
table = query.table

P = table['P0'].data
PDOT = table['P1'].data
EDOT = table['EDOT'].data

# Filter out non-positive values (log-safe)
mask = (P > 0) & (PDOT > 0) & (EDOT > 0)
P = P[mask]
PDOT = PDOT[mask]
EDOT = EDOT[mask]

logP_data = np.log10(P)
logPDOT_data = np.log10(PDOT)
logEDOT_data = np.log10(EDOT)

# ------------------------------------------
# Step 3: Define learnable constant
# ------------------------------------------
learn_constants = {
    logK: 46.0  # Initial guess for log(4π²I) ~ log10(4π² × 10^45)
}

# ------------------------------------------
# Step 4: Instantiate the PINN
# ------------------------------------------
pinn = PulsarPINN(
    x_param='P0',                   # input: P
    y_param='EDOT',                 # target: Edot
    differential_eq=energy_eq,
    x_sym=logP,
    y_sym=logEDOT,
    learn_constants=learn_constants,
    log_scale=True,
    fixed_inputs={logPDOT: logPDOT_data}
)

# ------------------------------------------
# Step 5: Train the PINN
# ------------------------------------------
pinn.train(epochs=10000)

# ------------------------------------------
# Step 6: Results
# ------------------------------------------
pinn.plot_PINN()
pinn.plot_PINN_loss()

constants = pinn.show_learned_constants()
logK_val = constants['logK']
K_val = 10 ** logK_val
print(f"\nEstimated logK ≈ {logK_val:.6f}")
print(f"Estimated K = 10^{logK_val:.6f} ≈ {K_val:.3e}")
