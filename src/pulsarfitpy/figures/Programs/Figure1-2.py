import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from pulsarfitpy import PulsarPINN
import matplotlib.pyplot as plt

# ----------------------------------
# SYMBOLIC DEFINITIONS
# ----------------------------------
logP, logPDOT, logB = sp.symbols('logP logPDOT logB')
logR = sp.Symbol('logR')

# Differential equation: logB = logR + 0.5 * logP + 0.5 * logPDOT
differential_equation = sp.Eq(logB, logR + 0.5 * logP + 0.5 * logPDOT)

# ----------------------------------
# ATNF QUERY
# ----------------------------------
query = QueryATNF(params=['P0', 'P1', 'BSURF'], condition='exist(P0) && exist(P1) && exist(BSURF)')
table = query.table

P = table['P0'].data 
PDOT = table['P1'].data  
BSURF = table['BSURF'].data 

logPDOT_data = np.log10(PDOT)

# ----------------------------------
# PINN SETUP
# ----------------------------------
learn_constants = {logR: 18}         # Initial guess
fixed_data = {logPDOT: logPDOT_data} # Supply logPDOT externally

pinn = PulsarPINN(
    x_param='P0',    
    y_param='BSURF',          
    differential_eq=differential_equation,
    x_symbol=logP,
    y_symbol=logB,
    learn_constants=learn_constants,
    log_scale=True,
    fixed_inputs=fixed_data
)

# ----------------------------------
# TRAIN PINN
# ----------------------------------
pinn.train(epochs=10000)
learned_values = pinn.store_learned_constants()

# ----------------------------------
# FIGURE 1: Model Prediction
# ----------------------------------
plt.figure(figsize=(6, 5))
plt.title("Surface Magnetic Field vs. Period - PINN Model #1")
plt.xlabel('log(Period) [log(s)]')
plt.ylabel('log(Surface Magnetic Field) [log(G)]')
plt.grid(True)

x_data = pinn.x_raw
y_data = pinn.y_raw
plt.scatter(x_data, y_data, label='ATNF Pulsars', s=8, alpha=0.5)

x_PINN, y_PINN = pinn.predict_extended()
plt.plot(x_PINN, y_PINN, color="#1000BE", label='PINN Model #1 (Our Work)')

plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------
# FIGURE 2: Training Loss
# ----------------------------------
plt.figure(figsize=(6, 5))
plt.title('PINN Loss vs. Epoch Curves - Model #1')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)

plt.plot(pinn.loss_log["total"], label='Total Loss')
plt.plot(pinn.loss_log["physics"], label='Physics Loss')
plt.plot(pinn.loss_log["data"], label='Data Loss')

plt.legend()
plt.tight_layout()
plt.show()