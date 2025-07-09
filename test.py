import sympy as sp
from pulsarfitpy import PulsarPINN

# Define symbols
logP, logPDOT = sp.symbols('logP logPDOT')
logK = sp.Symbol('logK')
n = sp.Symbol('n')

# Define physics equation (rewritten as equality for residual definition)
# logPDOT = logK + (2 - n) * logP
physics_eq = sp.Eq(logPDOT, logK + (2 - n) * logP)

# Define learnable constants with rough guesses
learn_constants = {
    logK: -15,
    n: 3.0
}

# Instantiate the model
pinn = PulsarPINN(
    x_param='P0',           # Spin period
    y_param='P1',           # Period derivative
    physics_eq=physics_eq,
    x_sym=logP,
    y_sym=logPDOT,
    learn_constant=learn_constants,
    log_scale=True
)

# Optional: Recommend better starting points from data
pinn.recommend_initial_guesses(method='regression')

# Train the model
pinn.train(epochs=5000)

# Show learned constants
pinn.show_learned_constants()

# Plot PINN prediction vs actual ATNF pulsar data
pinn.plot_prediction_vs_data()

# Plot loss curves (physics, data, total)
pinn.plot_loss_curve()