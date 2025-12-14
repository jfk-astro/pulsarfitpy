"""
2D Physics-Informed Neural Network Solvers (PINNs) with Advanced Visualization

Enhanced version with comprehensive visualization and analysis tools for
training diagnostics, solution analysis, and publication-quality figures.

Author: Computational Physics Implementation
License: MIT
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Optional, Tuple
from enum import Enum
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import backends
from pytorch_solver_2D import PyTorchBackend
# from deepxde_solver_2D import DeepXDEBackend


class Backend(Enum):
    """Enumeration of available computational backends."""
    PYTORCH = "pytorch"
    DEEPXDE = "deepxde"


class PulsarPINN2D:
    """
    Professional 2D Physics-Informed Neural Network Solver with Visualization.
    
    This class provides a unified interface for solving 2D partial differential
    equations using physics-informed neural networks with comprehensive
    visualization and analysis capabilities.
    
    New Features:
        - Training loss visualization (evolution, heatmaps, convergence)
        - 2D/3D solution visualization with multiple colormaps
        - Residual analysis and error distribution
        - Comparison with analytical solutions
        - Publication-quality figure generation
        - Animated training progress
    """
    
    def __init__(self,
                 pde_expr: sp.Expr,
                 input_dim: int,
                 hidden_layers: List[int],
                 output_dim: int,
                 backend: Backend = Backend.PYTORCH,
                 device: str = 'cpu',
                 **kwargs):
        """Initialize the PulsarPINN2D solver with visualization capabilities."""
        # Validate inputs
        if not isinstance(pde_expr, sp.Expr):
            raise TypeError("pde_expr must be a SymPy expression")
        
        if input_dim != 2:
            warnings.warn(f"PulsarPINN2D is designed for 2D problems, but input_dim={input_dim}")
        
        self.pde_expr = pde_expr
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.backend_type = backend
        
        # Initialize backend
        self._backend = self._initialize_backend(backend, device, **kwargs)
        self._backend.build_network(input_dim, hidden_layers, output_dim)
        
        # Extract variables from PDE
        self.variables = {str(sym): sym for sym in pde_expr.free_symbols}
        
        # Compile PDE loss
        self._backend.pde_loss_fn = self._backend.compile_pde_loss(
            pde_expr, self.variables
        )
        
        self.metrics_history = []
        
        # Visualization settings
        self.vis_config = {
            'colormap': 'viridis',
            'dpi': 100,
            'figsize': (10, 8),
            'style': 'seaborn-v0_8-darkgrid'
        }
        
        self._print_initialization_banner()
    
    def _initialize_backend(self, backend: Backend, device: str, **kwargs):
        """Factory method for backend initialization."""
        if backend == Backend.PYTORCH:
            return PyTorchBackend(device=device)
        elif backend == Backend.DEEPXDE:
            deepxde_backend = kwargs.get('deepxde_backend', 'pytorch')
            # return DeepXDEBackend(backend=deepxde_backend)
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Choose from {[b.value for b in Backend]}"
            )
    
    def _print_initialization_banner(self) -> None:
        """Print initialization information banner."""
        print(f"\n{'='*90}")
        print(f"{'PulsarPINN2D Initialized':^90}")
        print(f"{'='*90}")
        print(f"Backend:      {self.backend_type.value}")
        print(f"Architecture: {self.input_dim} → "
              f"{' → '.join(map(str, self.hidden_layers))} → {self.output_dim}")
        print(f"PDE:          {self.pde_expr}")
        print(f"Variables:    {', '.join(self.variables.keys())}")
        print(f"{'='*90}\n")
    
    def set_training_data(self,
                         collocation_points: np.ndarray,
                         boundary_points: np.ndarray,
                         boundary_values: np.ndarray) -> None:
        """Set training data for the PINN."""
        if not hasattr(self._backend, 'set_training_data'):
            raise AttributeError(
                f"{self.backend_type.value} backend does not support set_training_data(). "
                f"Use backend-specific methods instead."
            )
        
        # Validate shapes
        if collocation_points.shape[1] != 2:
            raise ValueError("collocation_points must have shape (N, 2)")
        if boundary_points.shape[1] != 2:
            raise ValueError("boundary_points must have shape (M, 2)")
        if len(boundary_values) != len(boundary_points):
            raise ValueError("boundary_values length must match boundary_points")
        
        self._backend.set_training_data(
            collocation_points, 
            boundary_points, 
            boundary_values
        )
    
    def train(self,
              epochs: int,
              learning_rate: float = 1e-3,
              callback_interval: int = 500) -> List:
        """Train the physics-informed neural network."""
        if callback_interval <= 0:
            raise ValueError("callback_interval must be positive")
        
        self.metrics_history = self._backend.train(
            epochs=epochs,
            learning_rate=learning_rate,
            callback_interval=callback_interval
        )
        
        return self.metrics_history
    
    def predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the trained network at specified points."""
        return self._backend.predict(x, y)
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def set_visualization_config(self, **kwargs):
        """
        Configure visualization settings.
        
        Args:
            colormap: Matplotlib colormap name (default: 'viridis')
            dpi: Figure DPI (default: 100)
            figsize: Default figure size (default: (10, 8))
            style: Matplotlib style (default: 'seaborn-v0_8-darkgrid')
        """
        self.vis_config.update(kwargs)
    
    def plot_loss_history(self, 
                         log_scale: bool = True,
                         separate_losses: bool = True,
                         savefig: Optional[str] = None) -> plt.Figure:
        """
        Plot training loss evolution over epochs.
        
        Args:
            log_scale: Use logarithmic scale for y-axis
            separate_losses: Plot PDE and boundary losses separately
            savefig: Optional filepath to save figure
            
        Returns:
            Matplotlib figure object
        """
        if not self.metrics_history:
            raise ValueError("No training metrics available. Train the model first.")
        
        epochs = [m.epoch for m in self.metrics_history]
        total_loss = [m.total_loss for m in self.metrics_history]
        pde_loss = [m.pde_loss for m in self.metrics_history]
        boundary_loss = [m.boundary_loss for m in self.metrics_history]
        
        if separate_losses:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=self.vis_config['dpi'])
            
            # Total loss
            axes[0].plot(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Total Loss Evolution', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            if log_scale:
                axes[0].set_yscale('log')
            
            # PDE loss
            axes[1].plot(epochs, pde_loss, 'r-', linewidth=2, label='PDE Loss')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('PDE Residual Loss', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            if log_scale:
                axes[1].set_yscale('log')
            
            # Boundary loss
            axes[2].plot(epochs, boundary_loss, 'g-', linewidth=2, label='Boundary Loss')
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Loss', fontsize=12)
            axes[2].set_title('Boundary Condition Loss', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            if log_scale:
                axes[2].set_yscale('log')
            
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=self.vis_config['figsize'], 
                                  dpi=self.vis_config['dpi'])
            
            ax.plot(epochs, total_loss, 'b-', linewidth=2, label='Total Loss')
            ax.plot(epochs, pde_loss, 'r--', linewidth=1.5, label='PDE Loss')
            ax.plot(epochs, boundary_loss, 'g--', linewidth=1.5, label='Boundary Loss')
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            if log_scale:
                ax.set_yscale('log')
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_loss_heatmap(self, 
                         window_size: int = 100,
                         savefig: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing loss evolution with rolling statistics.
        
        Args:
            window_size: Rolling window size for smoothing
            savefig: Optional filepath to save figure
            
        Returns:
            Matplotlib figure object
        """
        if not self.metrics_history:
            raise ValueError("No training metrics available.")
        
        epochs = np.array([m.epoch for m in self.metrics_history])
        losses = np.array([
            [m.total_loss, m.pde_loss, m.boundary_loss] 
            for m in self.metrics_history
        ])
        
        # Apply rolling average
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(losses, size=min(window_size, len(losses)//5), 
                                   axis=0, mode='nearest')
        
        fig, ax = plt.subplots(figsize=(12, 4), dpi=self.vis_config['dpi'])
        
        im = ax.imshow(smoothed.T, aspect='auto', cmap='hot', 
                      extent=[epochs[0], epochs[-1], 0, 3],
                      origin='lower', interpolation='bilinear')
        
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['Total', 'PDE', 'Boundary'], fontsize=11)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_title('Loss Evolution Heatmap', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Loss Magnitude', fontsize=11)
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_convergence_rate(self, savefig: Optional[str] = None) -> plt.Figure:
        """
        Analyze and plot convergence rate of training.
        
        Args:
            savefig: Optional filepath to save figure
            
        Returns:
            Matplotlib figure object
        """
        if not self.metrics_history:
            raise ValueError("No training metrics available.")
        
        epochs = np.array([m.epoch for m in self.metrics_history])
        total_loss = np.array([m.total_loss for m in self.metrics_history])
        
        # Compute loss reduction rate
        loss_reduction = -np.diff(np.log(total_loss + 1e-10))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), 
                                       dpi=self.vis_config['dpi'])
        
        # Loss on log scale with trend
        ax1.semilogy(epochs, total_loss, 'b-', linewidth=2, alpha=0.7, label='Loss')
        
        # Fit exponential decay
        if len(epochs) > 10:
            from scipy.optimize import curve_fit
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            try:
                popt, _ = curve_fit(exp_decay, epochs, total_loss, 
                                   p0=[total_loss[0], 0.001, total_loss[-1]],
                                   maxfev=5000)
                ax1.semilogy(epochs, exp_decay(epochs, *popt), 'r--', 
                           linewidth=2, label=f'Fit: {popt[0]:.2e}·exp(-{popt[1]:.2e}·x)')
            except:
                pass
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Total Loss', fontsize=12)
        ax1.set_title('Loss with Exponential Fit', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Convergence rate
        ax2.plot(epochs[1:], loss_reduction, 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Reduction Rate', fontsize=12)
        ax2.set_title('Convergence Rate', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_solution_2d(self,
                        resolution: int = 100,
                        x_range: Tuple[float, float] = (0, 1),
                        y_range: Tuple[float, float] = (0, 1),
                        colormap: Optional[str] = None,
                        show_colorbar: bool = True,
                        contour_levels: int = 15,
                        savefig: Optional[str] = None) -> plt.Figure:
        """
        Visualize 2D solution as a filled contour plot.
        
        Args:
            resolution: Grid resolution for visualization
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            colormap: Colormap name (default from config)
            show_colorbar: Whether to show colorbar
            contour_levels: Number of contour levels
            savefig: Optional filepath to save figure
            
        Returns:
            Matplotlib figure object
        """
        cmap = colormap or self.vis_config['colormap']
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Predict solution
        Z = self.predict(X, Y)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.vis_config['figsize'],
                              dpi=self.vis_config['dpi'])
        
        # Filled contour
        contour = ax.contourf(X, Y, Z, levels=contour_levels, cmap=cmap)
        
        # Contour lines
        ax.contour(X, Y, Z, levels=contour_levels, colors='black', 
                  alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('y', fontsize=13)
        ax.set_title('PINN Solution (2D)', fontsize=15, fontweight='bold')
        ax.set_aspect('equal')
        
        if show_colorbar:
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('u(x, y)', fontsize=12)
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_solution_3d(self,
                        resolution: int = 50,
                        x_range: Tuple[float, float] = (0, 1),
                        y_range: Tuple[float, float] = (0, 1),
                        colormap: Optional[str] = None,
                        elevation: float = 30,
                        azimuth: float = 45,
                        savefig: Optional[str] = None) -> plt.Figure:
        """
        Visualize solution as a 3D surface plot.
        
        Args:
            resolution: Grid resolution for visualization
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            colormap: Colormap name (default from config)
            elevation: View elevation angle (degrees)
            azimuth: View azimuth angle (degrees)
            savefig: Optional filepath to save figure
            
        Returns:
            Matplotlib figure object
        """
        cmap = colormap or self.vis_config['colormap']
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Predict solution
        Z = self.predict(X, Y)
        
        # Create 3D figure
        fig = plt.figure(figsize=self.vis_config['figsize'], 
                        dpi=self.vis_config['dpi'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9,
                              linewidth=0, antialiased=True,
                              edgecolor='none')
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('u(x, y)', fontsize=12)
        ax.set_title('PINN Solution (3D Surface)', fontsize=14, fontweight='bold')
        
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_residual_distribution(self,
                                   resolution: int = 100,
                                   x_range: Tuple[float, float] = (0, 1),
                                   y_range: Tuple[float, float] = (0, 1),
                                   savefig: Optional[str] = None) -> plt.Figure:
        """
        Visualize PDE residual distribution across domain.
        
        Args:
            resolution: Grid resolution
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            savefig: Optional filepath to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute residuals (requires backend support)
        # This is a placeholder - actual implementation depends on backend
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        # For demonstration, compute approximate residuals
        Z = self.predict(X, Y)
        
        # Compute numerical derivatives (finite differences)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        d2u_dx2 = np.gradient(np.gradient(Z, dx, axis=1), dx, axis=1)
        d2u_dy2 = np.gradient(np.gradient(Z, dy, axis=0), dy, axis=0)
        
        # Laplacian
        residual = d2u_dx2 + d2u_dy2
        
        # If PDE has source term, add it
        # For Poisson equation: ∇²u + 1 = 0, residual should include +1
        residual = residual + 1  # Adjust based on your PDE
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                       dpi=self.vis_config['dpi'])
        
        # Residual heatmap
        im1 = ax1.imshow(np.abs(residual), extent=[x_range[0], x_range[1], 
                                                    y_range[0], y_range[1]],
                        origin='lower', cmap='hot', aspect='equal')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title('Absolute PDE Residual', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1)
        
        # Residual histogram
        ax2.hist(residual.ravel(), bins=50, color='steelblue', 
                edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residual Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparison_with_analytical(self,
                                       analytical_solution,
                                       resolution: int = 100,
                                       x_range: Tuple[float, float] = (0, 1),
                                       y_range: Tuple[float, float] = (0, 1),
                                       savefig: Optional[str] = None) -> plt.Figure:
        """
        Compare PINN solution with analytical solution.
        
        Args:
            analytical_solution: Function u_exact(x, y) returning analytical solution
            resolution: Grid resolution
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            savefig: Optional filepath to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # PINN solution
        Z_pinn = self.predict(X, Y)
        
        # Analytical solution
        Z_exact = analytical_solution(X, Y)
        
        # Error
        error = np.abs(Z_pinn - Z_exact)
        
        fig = plt.figure(figsize=(16, 5), dpi=self.vis_config['dpi'])
        gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)
        
        # PINN solution
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.contourf(X, Y, Z_pinn, levels=20, cmap='viridis')
        ax1.set_title('PINN Solution', fontsize=13, fontweight='bold')
        ax1.set_xlabel('x', fontsize=11)
        ax1.set_ylabel('y', fontsize=11)
        plt.colorbar(im1, ax=ax1)
        
        # Analytical solution
        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.contourf(X, Y, Z_exact, levels=20, cmap='viridis')
        ax2.set_title('Analytical Solution', fontsize=13, fontweight='bold')
        ax2.set_xlabel('x', fontsize=11)
        ax2.set_ylabel('y', fontsize=11)
        plt.colorbar(im2, ax=ax2)
        
        # Absolute error
        ax3 = fig.add_subplot(gs[2])
        im3 = ax3.contourf(X, Y, error, levels=20, cmap='hot')
        ax3.set_title('Absolute Error', fontsize=13, fontweight='bold')
        ax3.set_xlabel('x', fontsize=11)
        ax3.set_ylabel('y', fontsize=11)
        plt.colorbar(im3, ax=ax3)
        
        # Error statistics
        ax4 = fig.add_subplot(gs[3])
        ax4.axis('off')
        
        mae = np.mean(error)
        rmse = np.sqrt(np.mean(error**2))
        max_error = np.max(error)
        
        stats_text = f"Error Statistics:\n\n"
        stats_text += f"MAE:  {mae:.6e}\n"
        stats_text += f"RMSE: {rmse:.6e}\n"
        stats_text += f"Max:  {max_error:.6e}\n\n"
        stats_text += f"Relative Error:\n"
        stats_text += f"{100*rmse/np.std(Z_exact):.3f}%"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_report(self,
                                   resolution: int = 100,
                                   x_range: Tuple[float, float] = (0, 1),
                                   y_range: Tuple[float, float] = (0, 1),
                                   savefig: Optional[str] = None) -> plt.Figure:
        """
        Generate a comprehensive visualization report.
        
        Args:
            resolution: Grid resolution
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            savefig: Optional filepath to save figure
            
        Returns:
            Matplotlib figure object with multiple subplots
        """
        if not self.metrics_history:
            raise ValueError("No training metrics available. Train the model first.")
        
        fig = plt.figure(figsize=(18, 12), dpi=self.vis_config['dpi'])
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Loss evolution
        ax1 = fig.add_subplot(gs[0, :])
        epochs = [m.epoch for m in self.metrics_history]
        ax1.semilogy(epochs, [m.total_loss for m in self.metrics_history], 
                    'b-', linewidth=2, label='Total')
        ax1.semilogy(epochs, [m.pde_loss for m in self.metrics_history], 
                    'r--', linewidth=1.5, label='PDE')
        ax1.semilogy(epochs, [m.boundary_loss for m in self.metrics_history], 
                    'g--', linewidth=1.5, label='Boundary')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training Loss Evolution', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2D Solution
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = self.predict(X, Y)
        
        ax2 = fig.add_subplot(gs[1, 0])
        im2 = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax2.set_title('Solution (2D)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x', fontsize=10)
        ax2.set_ylabel('y', fontsize=10)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3D Solution
        ax3 = fig.add_subplot(gs[1, 1], projection='3d')
        surf = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                               linewidth=0, antialiased=True)
        ax3.set_title('Solution (3D)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('x', fontsize=9)
        ax3.set_ylabel('y', fontsize=9)
        ax3.set_zlabel('u', fontsize=9)
        ax3.view_init(elev=30, azim=45)
        
        # Solution cross-sections
        ax4 = fig.add_subplot(gs[1, 2])
        mid_idx = resolution // 2
        ax4.plot(x, Z[mid_idx, :], 'b-', linewidth=2, label=f'y={y[mid_idx]:.2f}')
        ax4.plot(x, Z[:, mid_idx], 'r-', linewidth=2, label=f'x={x[mid_idx]:.2f}')
        ax4.set_xlabel('Position', fontsize=10)
        ax4.set_ylabel('u', fontsize=10)
        ax4.set_title('Cross-sections', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Training metrics summary
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        summary = self.get_metrics_summary()
        summary_text = "Training Summary\n" + "="*25 + "\n\n"
        summary_text += f"Final Total Loss: {summary['final_total_loss']:.6e}\n"
        summary_text += f"Final PDE Loss:   {summary['final_pde_loss']:.6e}\n"
        summary_text += f"Final BC Loss:    {summary['final_boundary_loss']:.6e}\n\n"
        summary_text += f"Training Time:    {summary['training_time']:.2f}s\n"
        summary_text += f"Epochs:           {summary['final_epoch']}\n"
        summary_text += f"Time/Epoch:       {summary['average_time_per_epoch']:.4f}s"
        
        ax5.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Solution statistics
        ax6 = fig.add_subplot(gs[2, 1])
        stats = {
            'Min': Z.min(),
            'Max': Z.max(),
            'Mean': Z.mean(),
            'Std': Z.std()
        }
        bars = ax6.bar(range(len(stats)), list(stats.values()), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax6.set_xticks(range(len(stats)))
        ax6.set_xticklabels(list(stats.keys()), fontsize=10)
        ax6.set_ylabel('Value', fontsize=10)
        ax6.set_title('Solution Statistics', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gradient magnitude
        ax7 = fig.add_subplot(gs[2, 2])
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        du_dx = np.gradient(Z, dx, axis=1)
        du_dy = np.gradient(Z, dy, axis=0)
        grad_mag = np.sqrt(du_dx**2 + du_dy**2)
        
        im7 = ax7.contourf(X, Y, grad_mag, levels=20, cmap='plasma')
        ax7.set_title('Gradient Magnitude', fontsize=12, fontweight='bold')
        ax7.set_xlabel('x', fontsize=10)
        ax7.set_ylabel('y', fontsize=10)
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        fig.suptitle('PulsarPINN2D Comprehensive Analysis Report', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_animation(self,
                               resolution: int = 50,
                               x_range: Tuple[float, float] = (0, 1),
                               y_range: Tuple[float, float] = (0, 1),
                               interval: int = 200,
                               save_path: Optional[str] = None):
        """
        Create an animation of solution evolution during training.
        
        Note: This requires saving model checkpoints during training.
        Currently a placeholder for future implementation.
        
        Args:
            resolution: Grid resolution
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis
            interval: Milliseconds between frames
            save_path: Optional path to save animation (e.g., 'animation.gif')
        """
        warnings.warn(
            "Animation requires checkpoint saving during training. "
            "This feature is planned for future implementation."
        )
    
    # ========================================================================
    # EXISTING METHODS (kept from original)
    # ========================================================================
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Compute summary statistics of training metrics."""
        if not self.metrics_history:
            warnings.warn("No training metrics available. Train the model first.")
            return {}
        
        final_metrics = self.metrics_history[-1]
        
        return {
            'final_epoch': final_metrics.epoch,
            'final_total_loss': final_metrics.total_loss,
            'final_pde_loss': final_metrics.pde_loss,
            'final_boundary_loss': final_metrics.boundary_loss,
            'training_time': final_metrics.elapsed_time,
            'average_time_per_epoch': final_metrics.elapsed_time / final_metrics.epoch
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if hasattr(self._backend, 'save_model'):
            self._backend.save_model(filepath)
        else:
            warnings.warn(
                f"Save not implemented for {self.backend_type.value} backend"
            )
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        if hasattr(self._backend, 'load_model'):
            self._backend.load_model(filepath)
        else:
            warnings.warn(
                f"Load not implemented for {self.backend_type.value} backend"
            )
    
    def get_backend(self):
        """Get direct access to the underlying backend."""
        return self._backend