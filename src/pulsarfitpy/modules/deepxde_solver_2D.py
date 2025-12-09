"""
DeepXDE Backend for PulsarPINN2D
=================================

DeepXDE-based implementation leveraging high-level APIs for rapid
prototyping with built-in support for various PDEs.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

import deepxde as dde
import numpy as np
import sympy as sp
from typing import List, Dict, Callable, Tuple
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


class DeepXDEBackend(_PINNBackend):
    """
    DeepXDE-based implementation of PINN backend.
    
    This backend leverages the high-level DeepXDE library for rapid
    prototyping and includes built-in support for various PDEs. DeepXDE
    provides optimized implementations and advanced features like adaptive
    sampling and multi-scale architectures.
    
    Features:
        - High-level API for quick prototyping
        - Built-in geometry and boundary condition handling
        - L-BFGS and Adam optimizers
        - Automatic domain sampling
        - Support for complex geometries
        - Adaptive training strategies
    
    Example:
        >>> backend = DeepXDEBackend(backend='pytorch')
        >>> backend.set_geometry(x_range=(0, 1), y_range=(0, 1))
        >>> backend.build_network(2, [32, 32, 32], 1)
        >>> backend.add_boundary_condition(lambda x: 0)
        >>> backend.create_training_data(num_domain=1000, num_boundary=100)
        >>> backend.compile_model()
        >>> metrics = backend.train(epochs=5000)
    """
    
    def __init__(self, backend: str = 'pytorch'):
        """
        Initialize DeepXDE backend.
        
        Args:
            backend: Underlying framework for DeepXDE ('pytorch' or 'tensorflow')
        """
        self.dde = dde
        self.dde.config.set_default_float('float32')
        
        # Set computational backend
        if backend.lower() == 'pytorch':
            self.dde.backend.set_default_backend('pytorch')
            print("DeepXDE Backend initialized with PyTorch")
        else:
            self.dde.backend.set_default_backend('tensorflow')
            print("DeepXDE Backend initialized with TensorFlow")
        
        self.model = None
        self.net = None
        self.pde_expr = None
        self.geom = None
        self.bc_list = []
        self.data = None
        self.pde_func = None
        
    def build_network(self, input_dim: int, hidden_layers: List[int],
                     output_dim: int) -> None:
        """
        Construct fully connected neural network.
        
        Uses DeepXDE's FNN (Feedforward Neural Network) with:
        - Tanh activation functions
        - Glorot normal (Xavier) initialization
        
        Args:
            input_dim: Dimension of input space
            hidden_layers: List of neurons per hidden layer
            output_dim: Dimension of output space
        """
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        self.net = self.dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")
        
        print(f"Network built: {layer_sizes}")
        print(f"Total layers: {len(layer_sizes)}")
    
    def set_geometry(self, x_range: Tuple[float, float],
                    y_range: Tuple[float, float]) -> None:
        """
        Define computational domain as a rectangle.
        
        Args:
            x_range: (x_min, x_max) domain bounds
            y_range: (y_min, y_max) domain bounds
        """
        self.geom = self.dde.geometry.Rectangle(
            xmin=[x_range[0], y_range[0]],
            xmax=[x_range[1], y_range[1]]
        )
        
        print(f"Geometry defined: [{x_range[0]}, {x_range[1]}] × [{y_range[0]}, {y_range[1]}]")
    
    def add_boundary_condition(self, bc_func: Callable, 
                              on_boundary: Callable = None) -> None:
        """
        Add Dirichlet boundary condition.
        
        Args:
            bc_func: Function that returns boundary values bc_func(x) -> u(x)
                    x is a numpy array of shape (n, 2) where each row is [x_i, y_i]
            on_boundary: Function to identify boundary points (optional)
                        Signature: on_boundary(x, on_boundary) -> bool
        """
        if self.geom is None:
            raise RuntimeError("Geometry not defined. Call set_geometry() first.")
        
        if on_boundary is None:
            # Default: apply on all boundaries
            on_boundary = lambda x, on_boundary: on_boundary
        
        bc = self.dde.icbc.DirichletBC(
            self.geom,
            bc_func,
            on_boundary
        )
        self.bc_list.append(bc)
        
        print(f"Boundary condition added (total: {len(self.bc_list)})")
    
    def compile_pde_loss(self, pde_expr: sp.Expr,
                        variables: Dict[str, sp.Symbol]) -> Callable:
        """
        Create PDE residual function for DeepXDE.
        
        The PDE function uses DeepXDE's automatic differentiation
        to compute derivatives of the neural network output.
        
        Args:
            pde_expr: SymPy expression representing the PDE
            variables: Dictionary mapping variable names to symbols
            
        Returns:
            Callable compatible with DeepXDE's PDE format
        """
        self.pde_expr = pde_expr
        
        def pde(x, u):
            """
            PDE residual function compatible with DeepXDE.
            
            Args:
                x: Input coordinates [batch_size, 2]
                   Each row is [x_i, y_i]
                u: Network output [batch_size, 1]
                   Predicted solution values
                
            Returns:
                PDE residual [batch_size, 1]
                Should be zero when PDE is satisfied
            """
            # Compute second derivatives using DeepXDE's grad module
            du_xx = self.dde.grad.hessian(u, x, i=0, j=0)
            du_yy = self.dde.grad.hessian(u, x, i=1, j=1)
            
            # Example: Poisson/Laplace equation
            # For ∇²u = f, return: du_xx + du_yy - f
            # Modify this based on your specific PDE
            return du_xx + du_yy
        
        self.pde_func = pde
        print("PDE residual function compiled")
        return pde
    
    def create_training_data(self, num_domain: int = 1000, 
                           num_boundary: int = 100,
                           train_distribution: str = 'uniform') -> None:
        """
        Create training data using DeepXDE's data structure.
        
        Args:
            num_domain: Number of collocation points in domain
            num_boundary: Number of points on boundary
            train_distribution: Distribution for sampling ('uniform' or 'pseudo')
        """
        if self.geom is None:
            raise RuntimeError("Geometry not defined. Call set_geometry() first.")
        
        if self.pde_func is None:
            raise RuntimeError("PDE not compiled. Call compile_pde_loss() first.")
        
        self.data = self.dde.data.PDE(
            self.geom,
            self.pde_func,
            self.bc_list,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution
        )
        
        print(f"Training data created:")
        print(f"  - Domain points: {num_domain}")
        print(f"  - Boundary points: {num_boundary}")
        print(f"  - Distribution: {train_distribution}")
    
    def compile_model(self) -> None:
        """
        Compile the DeepXDE model by combining network, data, and PDE.
        
        This step creates the computational graph and prepares the model
        for training.
        """
        if self.net is None:
            raise RuntimeError("Network not built. Call build_network() first.")
        
        if self.data is None:
            raise RuntimeError("Training data not created. Call create_training_data() first.")
        
        self.model = self.dde.Model(self.data, self.net)
        print("Model compiled successfully")
    
    def train(self, epochs: int, learning_rate: float = 1e-3,
              callback_interval: int = 500) -> List[TrainingMetrics]:
        """
        Execute training with L-BFGS or Adam optimizer.
        
        DeepXDE supports multiple optimization strategies:
        - Adam for initial training
        - L-BFGS for fine-tuning (optional)
        
        Args:
            epochs: Number of training iterations
            learning_rate: Learning rate for optimizer
            callback_interval: Frequency of metric reporting (in epochs)
            
        Returns:
            List of TrainingMetrics recorded during training
        """
        if self.model is None:
            raise RuntimeError("Model not compiled. Call compile_model() first.")
        
        metrics_history = []
        start_time = time.time()
        
        print(f"\n{'='*90}")
        print(f"{'DeepXDE Backend - Training Progress':^90}")
        print(f"{'='*90}\n")
        
        # Custom callback for metric recording
        def callback(train_state):
            """Callback function to record metrics during training."""
            if train_state.step % callback_interval == 0 or train_state.step == 1:
                elapsed = time.time() - start_time
                
                # Extract loss components from DeepXDE
                loss_train = train_state.loss_train
                
                # Handle different loss formats
                if hasattr(loss_train, '__len__') and len(loss_train) > 1:
                    total = float(np.sum(loss_train))
                    pde = float(loss_train[0])
                    bc = float(loss_train[1]) if len(loss_train) > 1 else 0.0
                else:
                    total = float(loss_train)
                    pde = total
                    bc = 0.0
                
                metrics = TrainingMetrics(
                    epoch=train_state.step,
                    total_loss=total,
                    pde_loss=pde,
                    boundary_loss=bc,
                    initial_loss=0.0,
                    elapsed_time=elapsed
                )
                metrics_history.append(metrics)
                print(metrics)
        
        # Compile and train model
        self.model.compile("adam", lr=learning_rate)
        
        losshistory, train_state = self.model.train(
            iterations=epochs,
            callbacks=[callback],
            display_every=callback_interval
        )
        
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
        
        # Store original shape
        original_shape = x.shape
        
        # Flatten and stack coordinates
        x_flat = x.flatten()
        y_flat = y.flatten()
        points = np.column_stack([x_flat, y_flat])
        
        # Predict
        predictions = self.model.predict(points)
        
        # Reshape to original shape
        return predictions.reshape(original_shape)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk using DeepXDE's save functionality.
        
        Args:
            filepath: Destination file path (directory will be created)
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Source file path
        """
        if self.model is None:
            raise RuntimeError("Compile model before loading weights")
        
        self.model.restore(filepath)
        print(f"Model loaded from {filepath}")