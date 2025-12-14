"""
Test script for PulsarPINN2D with comprehensive visualization.

This script demonstrates all visualization features of the enhanced PulsarPINN2D
class by solving the 2D Poisson equation.

Requirements:
    - numpy
    - sympy
    - matplotlib
    - scipy

Usage:
    python test_pinn_visualization.py

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, src_path)

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# Import your PINN modules
# Make sure these files are in the same directory or in your Python path
from ..modules.pinn_2D import PulsarPINN2D, Backend
plt.style.use('default')

"""
Main demonstration function showing all visualization capabilities.
"""

print("\n" + "="*90)
print("PulsarPINN2D - Enhanced Visualization Demonstration")
print("Solving 2D Poisson Equation: Laplacian(u) = -1")
print("Domain: [0,1] x [0,1] with u = 0 on boundary")
print("="*90 + "\n")

# ========================================================================
# STEP 1: Define the PDE
# ========================================================================

print("Step 1: Defining PDE symbolically...")
x, y, u = sp.symbols('x y u')

# Poisson equation: Laplacian(u) + 1 = 0
# This represents: d2u/dx2 + d2u/dy2 + 1 = 0
poisson_eq = sp.diff(u, x, 2) + sp.diff(u, y, 2) + 1

print(f"  PDE: {poisson_eq} = 0")
print(f"  Boundary condition: u = 0 on domain boundary")
print()

# ========================================================================
# STEP 2: Generate training data
# ========================================================================

print("Step 2: Generating training data...")

n_boundary = 40  # Points per boundary edge
n_interior = 400  # Interior collocation points

print(f"  - Boundary points: {4 * n_boundary}")
print(f"  - Interior points: {n_interior}")

# Create boundary of unit square [0,1] x [0,1]
boundary_x = np.concatenate([
    np.linspace(0, 1, n_boundary),  # bottom edge (y=0)
    np.ones(n_boundary),             # right edge (x=1)
    np.linspace(1, 0, n_boundary),  # top edge (y=1)
    np.zeros(n_boundary)             # left edge (x=0)
])
boundary_y = np.concatenate([
    np.zeros(n_boundary),            # bottom edge (y=0)
    np.linspace(0, 1, n_boundary),  # right edge (x=1)
    np.ones(n_boundary),             # top edge (y=1)
    np.linspace(1, 0, n_boundary)   # left edge (x=0)
])
boundary_points = np.column_stack([boundary_x, boundary_y])
boundary_values = np.zeros(len(boundary_x))  # u = 0 on boundary

# Create interior collocation points (random sampling)
np.random.seed(42)  # For reproducibility
interior_points = np.random.rand(n_interior, 2)

print("  [OK] Training data generated\n")

# ========================================================================
# STEP 3: Initialize PINN
# ========================================================================

print("Step 3: Initializing PINN...\n")

pinn = PulsarPINN2D(
    pde_expr=poisson_eq,
    input_dim=2,
    hidden_layers=[32, 32, 32],  # 3 hidden layers with 32 neurons each
    output_dim=1,
    backend=Backend.PYTORCH,
    device='cpu'  # Use 'cuda' if you have GPU
)

# Set training data
pinn.set_training_data(
    collocation_points=interior_points,
    boundary_points=boundary_points,
    boundary_values=boundary_values
)

# ========================================================================
# STEP 4: Train the network
# ========================================================================

print("Step 4: Training the network...\n")

metrics = pinn.train(
    epochs=2000,              # Number of training iterations
    learning_rate=1e-3,       # Adam optimizer learning rate
    callback_interval=500     # Print metrics every 500 epochs
)

print("\n[OK] Training complete!")

# Display training summary
summary = pinn.get_metrics_summary()
print("\n" + "="*90)
print("Training Summary")
print("="*90)
for key, value in summary.items():
    if 'loss' in key.lower():
        print(f"  {key:30s}: {value:.6e}")
    elif 'time' in key.lower():
        print(f"  {key:30s}: {value:.4f} s")
    else:
        print(f"  {key:30s}: {value}")
print("="*90 + "\n")

# ========================================================================
# STEP 5: Generate visualizations
# ========================================================================

print("\n" + "="*90)
print("Generating Visualizations")
print("="*90 + "\n")

# Optional: Configure visualization settings
pinn.set_visualization_config(
    colormap='viridis',  # Options: 'viridis', 'plasma', 'inferno', 'magma', etc.
    dpi=100,
    figsize=(10, 8)
)

# 1. Loss evolution history
print("1. Plotting loss evolution...")
fig1 = pinn.plot_loss_history(
    log_scale=True, 
    separate_losses=True,
    savefig='loss_history.png'  # Optional: save to file
)
print("   [OK] Loss history plotted")

# 2. Convergence rate analysis
print("2. Analyzing convergence rate...")
fig2 = pinn.plot_convergence_rate(
    savefig='convergence_rate.png'
)
print("   [OK] Convergence analysis complete")

# 3. 2D solution contour plot
print("3. Plotting 2D solution...")
fig3 = pinn.plot_solution_2d(
    resolution=100,
    contour_levels=20,
    colormap='viridis',
    savefig='solution_2d.png'
)
print("   [OK] 2D solution plotted")

# 4. 3D surface plot
print("4. Plotting 3D solution...")
fig4 = pinn.plot_solution_3d(
    resolution=50,
    elevation=30,
    azimuth=45,
    colormap='viridis',
    savefig='solution_3d.png'
)
print("   [OK] 3D solution plotted")

# 5. PDE residual distribution
print("5. Analyzing PDE residuals...")
fig5 = pinn.plot_residual_distribution(
    resolution=80,
    savefig='residuals.png'
)
print("   [OK] Residual analysis complete")

# 6. Comparison with analytical solution
print("6. Comparing with analytical solution...")

def analytical_poisson(X, Y):
    """
    Approximate analytical solution for 2D Poisson equation.
    
    For Laplacian(u) = -1 with u = 0 on boundary of unit square,
    this is an approximation using separation of variables.
    """
    # Simplified approximation (not exact)
    return (X * (1 - X) * Y * (1 - Y)) / 8

fig6 = pinn.plot_comparison_with_analytical(
    analytical_solution=analytical_poisson,
    resolution=80,
    savefig='comparison.png'
)
print("   [OK] Comparison complete")

# 7. Comprehensive report (all-in-one dashboard)
print("7. Generating comprehensive report...")
fig7 = pinn.create_comprehensive_report(
    resolution=80,
    savefig='comprehensive_report.png'
)
print("   [OK] Comprehensive report generated")

# ========================================================================
# STEP 6: Additional analysis
# ========================================================================

print("\n" + "="*90)
print("Additional Analysis")
print("="*90 + "\n")

# Test predictions on a specific grid
print("Testing predictions on a 10x10 grid...")
x_test = np.linspace(0, 1, 10)
y_test = np.linspace(0, 1, 10)
X_test, Y_test = np.meshgrid(x_test, y_test)

predictions = pinn.predict(X_test, Y_test)

print(f"  Prediction shape: {predictions.shape}")
print(f"  Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
print(f"  Mean value: {predictions.mean():.6f}")
print(f"  Std deviation: {predictions.std():.6f}")

# Optional: Save model for later use
print("\nSaving trained model...")
try:
    pinn.save_model('trained_pinn_model.pt')
    print("  [OK] Model saved to 'trained_pinn_model.pt'")
except Exception as e:
    print(f"  [WARN] Could not save model: {e}")

# ========================================================================
# STEP 7: Display all plots
# ========================================================================

print("\n" + "="*90)
print("Displaying all visualizations...")
print("Close the plot windows to exit.")
print("="*90 + "\n")

plt.show()

print("\n" + "="*90)
print("Visualization demonstration complete!")
print("Generated files:")
print("  - loss_history.png")
print("  - convergence_rate.png")
print("  - solution_2d.png")
print("  - solution_3d.png")
print("  - residuals.png")
print("  - comparison.png")
print("  - comprehensive_report.png")
print("  - trained_pinn_model.pt")
print("="*90 + "\n")


def quick_test():
    """
    Quick test function for rapid debugging.
    Uses fewer epochs and lower resolution for faster execution.
    """
    print("\n" + "="*90)
    print("QUICK TEST MODE - Reduced epochs and resolution")
    print("="*90 + "\n")

    # Define PDE
    x, y, u = sp.symbols('x y u')
    poisson_eq = sp.diff(u, x, 2) + sp.diff(u, y, 2) + 1

    # Generate minimal training data
    n_boundary = 20
    n_interior = 100

    boundary_x = np.concatenate([
        np.linspace(0, 1, n_boundary),
        np.ones(n_boundary),
        np.linspace(1, 0, n_boundary),
        np.zeros(n_boundary)
    ])
    boundary_y = np.concatenate([
        np.zeros(n_boundary),
        np.linspace(0, 1, n_boundary),
        np.ones(n_boundary),
        np.linspace(1, 0, n_boundary)
    ])
    boundary_points = np.column_stack([boundary_x, boundary_y])
    boundary_values = np.zeros(len(boundary_x))

    np.random.seed(42)
    interior_points = np.random.rand(n_interior, 2)

    # Initialize and train
    pinn = PulsarPINN2D(
        pde_expr=poisson_eq,
        input_dim=2,
        hidden_layers=[16, 16],  # Smaller network
        output_dim=1,
        backend=Backend.PYTORCH,
        device='cpu'
    )

    pinn.set_training_data(interior_points, boundary_points, boundary_values)

    print("Training for 500 epochs...")
    metrics = pinn.train(epochs=500, learning_rate=1e-3, callback_interval=250)

    # Quick visualization
    print("\nGenerating quick visualizations...")
    pinn.plot_loss_history(log_scale=True, separate_losses=False)
    pinn.plot_solution_2d(resolution=50)

    plt.show()

    print("\n[OK] Quick test complete!")