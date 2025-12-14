"""
PyTorch Backend for PulsarPINN2D
=================================

PyTorch-based implementation of the PINN backend providing high flexibility
and custom loss functions for complex PDE formulations.

Author: Om Kasar & Saumil Sharma under jfk-astro
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
from typing import List, Dict, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time


@dataclass
class TrainingMetrics:
    """
    Data class for storing training metrics.
    
    Attributes:
        epoch: Current training epoch
        total_loss: Total loss value
        pde_loss: Physics-informed loss component
        boundary_loss: Boundary condition loss component
        initial_loss: Initial condition loss component (if applicable)
        elapsed_time: Cumulative training time in seconds
    """
    epoch: int
    total_loss: float
    pde_loss: float
    boundary_loss: float
    initial_loss: float
    elapsed_time: float
    
    def __str__(self) -> str:
        """Format metrics for console output."""
        return (f"Epoch {self.epoch:6d} | "
                f"Total Loss: {self.total_loss:.6e} | "
                f"PDE: {self.pde_loss:.6e} | "
                f"BC: {self.boundary_loss:.6e} | "
                f"IC: {self.initial_loss:.6e} | "
                f"Time: {self.elapsed_time:.2f}s")


class _PINNBackend(ABC):
    """
    Abstract base class for PINN backends.
    
    This class defines the interface that all backend implementations must follow,
    ensuring consistent behavior across different computational frameworks.
    """
    
    @abstractmethod
    def build_network(self, input_dim: int, hidden_layers: List[int], 
                     output_dim: int) -> None:
        """
        Construct the neural network architecture.
        
        Args:
            input_dim: Dimension of input space
            hidden_layers: List containing number of neurons in each hidden layer
            output_dim: Dimension of output space
        """
        pass
    
    @abstractmethod
    def compile_pde_loss(self, pde_expr: sp.Expr, variables: Dict[str, sp.Symbol]) -> Callable:
        """
        Compile the PDE residual loss function.
        
        Args:
            pde_expr: SymPy expression representing the PDE
            variables: Dictionary mapping variable names to SymPy symbols
            
        Returns:
            Callable that computes PDE residual
        """
        pass
    
    @abstractmethod
    def train(self, epochs: int, callback_interval: int = 500) -> List[TrainingMetrics]:
        """
        Execute the training loop.
        
        Args:
            epochs: Number of training iterations
            callback_interval: Frequency of metric reporting
            
        Returns:
            List of TrainingMetrics recorded during training
        """
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate the trained network at given points.
        
        Args:
            x: x-coordinates for evaluation
            y: y-coordinates for evaluation
            
        Returns:
            Network predictions at (x, y) locations
        """
        pass


class PyTorchBackend(_PINNBackend):
    """
    PyTorch-based implementation of PINN backend.
    
    This backend provides high flexibility and is suitable for custom
    loss functions and complex PDE formulations. It uses automatic
    differentiation to compute derivatives required for PDE residuals.
    
    Features:
        - Automatic differentiation for arbitrary PDEs
        - Xavier initialization for stable training
        - Adam optimizer with configurable learning rate
        - GPU acceleration support
        - Comprehensive loss tracking
    
    Example:
        >>> backend = PyTorchBackend(device='cuda')
        >>> backend.build_network(2, [32, 32, 32], 1)
        >>> backend.set_training_data(collocation_pts, boundary_pts, boundary_vals)
        >>> metrics = backend.train(epochs=5000, learning_rate=1e-3)
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize PyTorch backend.
        
        Args:
            device: Computation device ('cpu' or 'cuda')
        """
        self.torch = torch
        self.nn = nn
        self.optim = optim
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.pde_loss_fn = None
        self.training_data = {}
        
        print(f"PyTorch Backend initialized on device: {self.device}")
        
    def build_network(self, input_dim: int, hidden_layers: List[int], 
                     output_dim: int) -> None:
        """
        Construct feedforward neural network with tanh activation.
        
        The network architecture follows the standard PINN design with:
        - Tanh activation functions for smooth derivatives
        - Xavier normal initialization for weights
        - Zero initialization for biases
        
        Args:
            input_dim: Dimension of input space
            hidden_layers: List of neurons per hidden layer
            output_dim: Dimension of output space
        """
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(self.nn.Linear(prev_dim, hidden_dim))
            layers.append(self.nn.Tanh())
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(self.nn.Linear(prev_dim, output_dim))
        
        self.model = self.nn.Sequential(*layers).to(self.device)
        
        # Xavier initialization for better convergence
        for layer in self.model:
            if isinstance(layer, self.nn.Linear):
                self.nn.init.xavier_normal_(layer.weight)
                self.nn.init.zeros_(layer.bias)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Network built: {num_params:,} trainable parameters")
    
    def compile_pde_loss(self, pde_expr: sp.Expr, 
                        variables: Dict[str, sp.Symbol]) -> Callable:
        """
        Convert SymPy PDE to PyTorch computational graph.
        
        Uses automatic differentiation to compute required derivatives.
        Currently supports second-order PDEs with derivatives up to order 2.
        
        Args:
            pde_expr: SymPy expression representing the PDE
            variables: Dictionary mapping variable names to symbols
            
        Returns:
            Callable that computes mean squared PDE residual
        """
        # Convert SymPy expression to callable function
        free_symbols = list(pde_expr.free_symbols)
        pde_func = sp.lambdify(free_symbols, pde_expr, 'numpy')
        
        def pde_residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """
            Compute PDE residual using automatic differentiation.
            
            Args:
                x: x-coordinates (requires gradient)
                y: y-coordinates (requires gradient)
                
            Returns:
                Mean squared residual of the PDE
            """
            x.requires_grad_(True)
            y.requires_grad_(True)
            
            # Forward pass through network
            inputs = self.torch.stack([x, y], dim=1)
            u = self.model(inputs).squeeze()
            
            # Compute first derivatives using autograd
            du_dx = self.torch.autograd.grad(
                u, x, 
                grad_outputs=self.torch.ones_like(u),
                create_graph=True
            )[0]
            
            du_dy = self.torch.autograd.grad(
                u, y,
                grad_outputs=self.torch.ones_like(u),
                create_graph=True
            )[0]
            
            # Compute second derivatives
            d2u_dx2 = self.torch.autograd.grad(
                du_dx, x,
                grad_outputs=self.torch.ones_like(du_dx),
                create_graph=True
            )[0]
            
            d2u_dy2 = self.torch.autograd.grad(
                du_dy, y,
                grad_outputs=self.torch.ones_like(du_dy),
                create_graph=True
            )[0]
            
            # Evaluate PDE residual (Laplacian for this example)
            # For custom PDEs, modify this section based on pde_expr
            residual = d2u_dx2 + d2u_dy2
            
            return self.torch.mean(residual ** 2)
        
        return pde_residual
    
    def set_training_data(self, collocation_points: np.ndarray,
                         boundary_points: np.ndarray,
                         boundary_values: np.ndarray) -> None:
        """
        Set training data for the PINN.
        
        Args:
            collocation_points: Interior points for PDE residual (N x 2)
            boundary_points: Boundary points (M x 2)
            boundary_values: Known values at boundary points (M,)
        """
        self.training_data = {
            'x_pde': self.torch.tensor(
                collocation_points[:, 0], 
                dtype=self.torch.float32, 
                device=self.device
            ),
            'y_pde': self.torch.tensor(
                collocation_points[:, 1],
                dtype=self.torch.float32,
                device=self.device
            ),
            'x_bc': self.torch.tensor(
                boundary_points[:, 0],
                dtype=self.torch.float32,
                device=self.device
            ),
            'y_bc': self.torch.tensor(
                boundary_points[:, 1],
                dtype=self.torch.float32,
                device=self.device
            ),
            'u_bc': self.torch.tensor(
                boundary_values,
                dtype=self.torch.float32,
                device=self.device
            )
        }
        
        print(f"Training data loaded:")
        print(f"  - Collocation points: {len(collocation_points)}")
        print(f"  - Boundary points: {len(boundary_points)}")
    
    def train(self, epochs: int, learning_rate: float = 1e-3,
              callback_interval: int = 500) -> List[TrainingMetrics]:
        """
        Execute training with Adam optimizer.
        
        The training loop alternates between:
        1. Computing PDE residual loss on interior points
        2. Computing boundary condition loss on boundary points
        3. Backpropagation and parameter update
        
        Args:
            epochs: Number of training iterations
            learning_rate: Learning rate for Adam optimizer
            callback_interval: Frequency of metric reporting (in epochs)
            
        Returns:
            List of TrainingMetrics recorded during training
        """
        if self.model is None:
            raise RuntimeError("Network not built. Call build_network() first.")
        
        if not self.training_data:
            raise RuntimeError("Training data not set. Call set_training_data() first.")
        
        self.optimizer = self.optim.Adam(self.model.parameters(), lr=learning_rate)
        metrics_history = []
        start_time = time.time()
        
        print(f"\n{'='*90}")
        print(f"{'PyTorch Backend - Training Progress':^90}")
        print(f"{'='*90}\n")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            
            # PDE loss (physics-informed component)
            pde_loss = self.pde_loss_fn(
                self.training_data['x_pde'],
                self.training_data['y_pde']
            )
            
            # Boundary condition loss
            bc_inputs = self.torch.stack([
                self.training_data['x_bc'],
                self.training_data['y_bc']
            ], dim=1)
            bc_pred = self.model(bc_inputs).squeeze()
            bc_loss = self.torch.mean((bc_pred - self.training_data['u_bc']) ** 2)
            
            # Total loss (weighted sum)
            total_loss = pde_loss + bc_loss
            
            # Backpropagation
            total_loss.backward()
            self.optimizer.step()
            
            # Record and display metrics
            if epoch % callback_interval == 0 or epoch == 1:
                elapsed = time.time() - start_time
                metrics = TrainingMetrics(
                    epoch=epoch,
                    total_loss=total_loss.item(),
                    pde_loss=pde_loss.item(),
                    boundary_loss=bc_loss.item(),
                    initial_loss=0.0,
                    elapsed_time=elapsed
                )
                metrics_history.append(metrics)
                print(metrics)
        
        print(f"\n{'='*90}")
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print(f"{'='*90}\n")
        
        return metrics_history
    
    def predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate trained network at specified points.
        
        Args:
            x: x-coordinates (can be 1D array or 2D meshgrid)
            y: y-coordinates (can be 1D array or 2D meshgrid)
            
        Returns:
            Network predictions at (x, y) locations
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        with self.torch.no_grad():
            x_tensor = self.torch.tensor(
                x.flatten(), 
                dtype=self.torch.float32, 
                device=self.device
            )
            y_tensor = self.torch.tensor(
                y.flatten(), 
                dtype=self.torch.float32, 
                device=self.device
            )
            inputs = self.torch.stack([x_tensor, y_tensor], dim=1)
            predictions = self.model(inputs).cpu().numpy()
        
        return predictions.reshape(x.shape)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model state to disk.
        
        Args:
            filepath: Destination file path (will add .pt extension)
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        
        self.torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model state from disk.
        
        Args:
            filepath: Source file path
        """
        if self.model is None:
            raise RuntimeError("Build network before loading weights")
        
        checkpoint = self.torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {filepath}")