import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from pulsarfitpy import PulsarPINN  

# ------------------------------------------
# Step 1: Define symbolic variables
# ------------------------------------------
logP, logPDOT, logB = sp.symbols('logP logPDOT logB')
logK = sp.Symbol('logK')

# Surface magnetic field equation (in log form):
# logB = logK + 0.5 * (logP + logPDOT)
physics_eq = sp.Eq(logB, logK + 0.5 * (logP + logPDOT))

# ------------------------------------------
# Step 2: Query ATNF data
# ------------------------------------------
query = QueryATNF(params=['P0', 'P1', 'BSURF'], condition='exist(P) && exist(PDOT) && exist(BSURF)')
table = query.table

P = table['P0'].data      # seconds
PDOT = table['P1'].data   # s/s
BSURF = table['BSURF'].data  # Gauss

# Filter valid rows
mask = (P > 0) & (PDOT > 0) & (BSURF > 0)
P = P[mask]
PDOT = PDOT[mask]
BSURF = BSURF[mask]

# Prepare fixed symbolic input logPDOT
logPDOT_data = np.log10(PDOT)

# ------------------------------------------
# Step 3: Define the learnable constant
# ------------------------------------------
learn_constants = {
    logK: 18.0  # Initial guess (true ≈ 19.505)
}

# ------------------------------------------
# Step 4: Instantiate the PINN
# ------------------------------------------
pinn = PulsarPINN(
    x_param='P0',               # Input: P
    y_param='BSURF',            # Target: B
    differential_eq=physics_eq,
    x_sym=logP,
    y_sym=logB,
    learn_constants=learn_constants,
    log_scale=True,
    fixed_inputs={logPDOT: logPDOT_data}  # Supply logPDOT externally
)

# ------------------------------------------
# Step 5: Recommend better guess (optional)
# ------------------------------------------
pinn.recommend_initial_guesses(method='regression')

# ------------------------------------------
# Step 6: Train the PINN
# ------------------------------------------
pinn.train(epochs=10000)

# ------------------------------------------
# Step 7: Show results
# ------------------------------------------
pinn.plot_PINN()
pinn.plot_PINN_loss_curves()

# ------------------------------------------
# Step 7: Show results
# ------------------------------------------
constants = pinn.show_learned_constants()

# Print K = 10 ** logK
logK_val = constants.get('logK')
K_val = 10 ** logK_val
print(f"Approximate K = 10^({logK_val:.10f}) ≈ {K_val:.10e}")