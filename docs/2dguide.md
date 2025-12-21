---
layout: default
title: pulsarfitpy
---

# **pulsarfitpy Technical Information**

## **Implementing 2D Physics Informed Neural Networks (PINNs) with pulsarfitpy**

pulsarfitpy offers a comprehensive framework for solving 2D partial differential equations using Physics Informed Neural Networks (PINNs). The 2D implementation extends builds on the 1D framework to handle complex physical systems. The 2D PINN framework is particularly useful for studying phenomena that require spatial variation across two dimensions, enabling scientists to test theoretical models of pulsar physics against observational data.

## **Understanding the PulsarPINN2D Framework**

The 2D PINN framework in pulsarfitpy is built on a modular architecture consisting of three main components: the primary PulsarPINN2D solver, the PyTorchBackend for automatic differentiation, and the DeepXDEBackend for high-level PDE solving. This separation of concerns allows flexibility in choosing computational strategies while maintaining a unified user interface.

## **PulsarPINN2D Class Overview**

The PulsarPINN2D class serves as the main interface for solving 2D partial differential equations using physics-informed neural networks. It provides a unified interface that abstracts away backend-specific details while offering comprehensive visualization and analysis capabilities.

The class requires the following parameters:

- **pde_expr**=[sympy.Expr]: The symbolic partial differential equation representing the physical model that the PINN must adhere to. Uses SymPy's symbolic notation for mathematical expressions.  
Example: d2u_dx2 + d2u_dy2 (Laplace equation)

- **input_dim**=[int]: The dimensionality of the input space. For 2D problems, this should be set to 2. Represents spatial dimensions x and y.

- **hidden_layers**=[list]: List defining the PINN architecture as hidden layer sizes. Each integer represents a hidden layer with that many neurons.  
*Default:* [32, 32, 32] (meaning three hidden layers with 32 neurons each).

- **output_dim**=[int]: Size of the output layer (number of output features). For standard scalar-valued PDEs, this is 1.  
*Default:* 1.

- **backend**=[Backend]: Enumeration specifying which computational backend to use. Options are Backend.PYTORCH for automatic differentiation or Backend.DEEPXDE for high-level APIs.  
*Default:* Backend.PYTORCH.

- **device**=[str]: Computation device for PyTorch backend ('cpu' or 'cuda'). Only applies when using Backend.PYTORCH.  
*Default:* 'cpu'.

## **Backend Architecture**

The PulsarPINN2D framework supports two distinct computational backends, each offering different advantages for solving 2D PDEs.

### **PyTorch Backend**

The PyTorch backend (`pytorch_solver_2D.py`) provides a low-level, flexible implementation using automatic differentiation. This backend is ideal for custom PDE formulations and when fine-grained control over the computation is needed.

#### Key Features:

- Automatic differentiation for arbitrary PDEs using PyTorch's autograd system
- Xavier normal initialization for neural network weights ensuring stable training
- Adam optimizer with configurable learning rate for adaptive learning
- GPU acceleration support via CUDA for large-scale problems
- Comprehensive loss tracking with separate physics and boundary condition losses
- Support for second-order partial derivatives required by common PDEs

#### The PyTorchBackend class provides the following core functionality:

`.build_network(input_dim, hidden_layers, output_dim)`: Constructs the neural network architecture with Tanh activation functions for smooth derivatives needed in PDE computations.

`.set_training_data(collocation_points, boundary_points, boundary_values)`: Loads training data consisting of interior collocation points for PDE residuals and boundary points with known values.

`.compile_pde_loss(pde_expr, variables)`: Converts a SymPy PDE expression into a PyTorch computational graph using automatic differentiation.

`.train(epochs, learning_rate, callback_interval)`: Executes the training loop, optimizing network weights and computing loss metrics at regular intervals.

`.predict(x, y)`: Evaluates the trained network at specified 2D coordinates.

### **DeepXDE Backend**

The DeepXDE backend (deepxde_solver_2D.py) provides a high-level API built on the DeepXDE library, which is specifically designed for physics-informed neural networks. This backend is excellent for rapid prototyping and standard PDE problems.

#### Key Features:

- High-level API simplifying PINN implementation for standard PDE types
- Built-in geometry handling for rectangular and complex domains
- Automatic boundary condition management with Dirichlet, Neumann, and Robin conditions
- L-BFGS and Adam optimizer options for different optimization strategies
- Automatic domain sampling with uniform and pseudo-random distributions
- Support for multiple geometry types beyond rectangles
- Adaptive training strategies for enhanced convergence

#### The DeepXDEBackend class provides the following core functionality:

`.build_network(input_dim, hidden_layers, output_dim)`: Constructs the neural network using DeepXDE's FNN class with Tanh activation and Glorot normal initialization.

`.set_geometry(x_range, y_range)`: Defines the rectangular computational domain with specified x and y bounds.

`.add_boundary_condition(bc_func, on_boundary)`: Adds Dirichlet boundary conditions to the domain, with automatic boundary detection.

`.compile_pde_loss(pde_expr, variables)`: Creates a PDE residual function compatible with DeepXDE's automatic differentiation system.

`.create_training_data(num_domain, num_boundary, train_distribution)`: Generates training data by sampling collocation points and boundary points from the domain.

`.compile_model()`: Assembles the network, data, and PDE into a complete model ready for training.

`.train(epochs, learning_rate, callback_interval)`: Executes training with the specified optimizer and convergence criteria.

`.predict(x, y)`: Evaluates the trained network at specified 2D coordinates.

## **PulsarPINN2D Methods**

The primary methods of the PulsarPINN2D class for 2D PDE solving and analysis are as follows:

**1. `.set_training_data(collocation_points, boundary_points, boundary_values):`**

Sets the training data for the PINN model. This method prepares interior points where the PDE residual will be enforced and boundary points with known values.

#### Inputs:

- **collocation_points** \[numpy.ndarray]: Interior points where PDE residuals are enforced. Shape must be (N, 2) where N is the number of points.
- **boundary_points** \[numpy.ndarray]: Boundary points with known solution values. Shape must be (M, 2) where M is the number of boundary points.
- **boundary_values** \[numpy.ndarray]: Known solution values at boundary points. Shape must be (M,) matching the number of boundary points.

#### Outputs:

- None. Data is loaded into the backend for training.

**2. `.train(epochs, learning_rate=1e-3, callback_interval=500):`**

Trains the physics-informed neural network by minimizing combined physics and boundary condition losses.

#### Inputs:

- **epochs** [int]: Number of training iterations.  
*Default:* 5000
- **learning_rate** [float]: Learning rate for the optimizer.  
*Default:* 0.001
- **callback_interval** [int]: Frequency of progress updates during training in epochs.  
*Default:* 500

#### Outputs:

- List of TrainingMetrics objects containing epoch-by-epoch loss history

**3. `.predict(x, y):`**

Evaluates the trained network at specified 2D coordinates to obtain solution predictions.

#### Inputs:

- **x** \[numpy.ndarray]: x-coordinates for evaluation. Can be 1D array or 2D meshgrid array.
- **y** \[numpy.ndarray]: y-coordinates for evaluation. Can be 1D array or 2D meshgrid array.

#### Outputs:

- \[numpy.ndarray]: Network predictions at (x, y) locations, same shape as input arrays.

**4. `.set_visualization_config(colormap=None, dpi=None, figsize=None, style=None):`**

Configures visualization settings for all plotting functions.

#### Inputs:

- **colormap** [str]: Matplotlib colormap name.  
*Default:* 'viridis'
- **dpi** [int]: Figure resolution in dots per inch.  
*Default:* 100
- **figsize** [tuple]: Default figure size as (width, height).  
*Default:* (10, 8)
- **style** [str]: Matplotlib style name.  
*Default:* 'seaborn-v0_8-darkgrid'

#### Outputs:

- None. Updates internal configuration dictionary.

**5. `.plot_loss_history(log_scale=True, separate_losses=True, savefig=None):`**

Visualizes training loss evolution over epochs, showing total loss, PDE residual loss, and boundary condition loss.

#### Inputs:

- **log_scale** [bool]: Whether to use logarithmic scale for loss axis.  
*Default:* True
- **separate_losses** [bool]: If True, creates three separate subplots for each loss component. If False, combines all on one plot.  
*Default:* True
- **savefig** [str]: Optional filepath to save the figure.  
*Default:* None

#### Outputs:

- \[matplotlib.figure.Figure]: Figure object containing the loss history visualization.

**6. `.plot_loss_heatmap(window_size=100, savefig=None):`**

Creates a heatmap visualization showing how different loss components evolve during training with rolling average smoothing.

#### Inputs:

- **window_size** [int]: Rolling window size for loss smoothing.  
*Default:* 100
- **savefig** [str]: Optional filepath to save the figure.  
*Default:* None

#### Outputs:

- \[matplotlib.figure.Figure]: Figure object containing the heatmap visualization.

**7. `.plot_convergence_rate(savefig=None):`**

Analyzes and visualizes the convergence rate of training by fitting exponential decay and computing loss reduction rates.

#### Inputs:

- **savefig** [str]: Optional filepath to save the figure.  
*Default:* None

#### Outputs:

- \[matplotlib.figure.Figure]: Figure object with two subplots showing exponential fit and convergence rate metrics.

**8. `.plot_solution_2d(resolution=100, x_range=(0, 1), y_range=(0, 1), colormap=None, show_colorbar=True, contour_levels=15, savefig=None):`**

Visualizes the 2D PINN solution as a filled contour plot with optional contour lines.

#### Inputs:

- **resolution** [int]: Grid resolution for solution visualization.  
*Default:* 100
- **x_range** [tuple]: (min, max) bounds for x-axis.  
*Default:* (0, 1)
- **y_range** [tuple]: (min, max) bounds for y-axis.  
*Default:* (0, 1)
- **colormap** [str]: Colormap name (uses configuration default if None).  
*Default:* None
- **show_colorbar** [bool]: Whether to display colorbar.  
*Default:* True
- **contour_levels** [int]: Number of contour levels to display.  
*Default:* 15
- **savefig** [str]: Optional filepath to save the figure.  
*Default:* None

#### Outputs:

- \[matplotlib.figure.Figure]: Figure object containing the 2D contour plot.

**9. `.plot_solution_3d(resolution=50, x_range=(0, 1), y_range=(0, 1), colormap=None, elevation=30, azimuth=45, savefig=None):`**

Visualizes the 2D PINN solution as a 3D surface plot with adjustable viewpoint.

#### Inputs:

- **resolution** [int]: Grid resolution for solution visualization.  
*Default:* 50
- **x_range** [tuple]: (min, max) bounds for x-axis.  
*Default:* (0, 1)
- **y_range** [tuple]: (min, max) bounds for y-axis.  
*Default:* (0, 1)
- **colormap** [str]: Colormap name (uses configuration default if None).  
*Default:* None
- **elevation** [float]: View elevation angle in degrees.  
*Default:* 30
- **azimuth** [float]: View azimuth angle in degrees.  
*Default:* 45
- **savefig** [str]: Optional filepath to save the figure.  
*Default:* None

Outputs:

- [matplotlib.figure.Figure]: Figure object containing the 3D surface plot.

**10. `.plot_residual_distribution(resolution=100, x_range=(0, 1), y_range=(0, 1), savefig=None):`**

Visualizes the spatial distribution of PDE residuals across the domain to assess where the model satisfies the physics constraints.

#### Inputs:

- **resolution** [int]: Grid resolution for residual computation.  
*Default:* 100
- **x_range** [tuple]: (min, max) bounds for x-axis.  
*Default:* (0, 1)
- **y_range** [tuple]: (min, max) bounds for y-axis.  
*Default:* (0, 1)
- **savefig** [str]: Optional filepath to save the figure.  
*Default:* None

#### Outputs:

- \[matplotlib.figure.Figure]: Figure object showing residual heatmap and statistics.

**11. `.plot_comparison_with_analytical(analytical_solution, resolution=100, x_range=(0, 1), y_range=(0, 1), savefig=None):`**

Compares PINN solution with an analytical reference solution, computing error metrics and visualizing differences.

#### Inputs:

- **analytical_solution** [callable]: Function that computes analytical solution at (x, y) coordinates.
- **resolution** [int]: Grid resolution for comparison.  
*Default:* 100
- **x_range** [tuple]: (min, max) bounds for x-axis.  
*Default:* (0, 1)
- **y_range** [tuple]: (min, max) bounds for y-axis.  
*Default:* (0, 1)
- **savefig** [str]: Optional filepath to save the figure.  
*Default:* None

#### Outputs:

- \[matplotlib.figure.Figure]: Figure with PINN solution, analytical solution, and error distribution.

**12. `.create_comprehensive_report(resolution=100, x_range=(0, 1), y_range=(0, 1), savefig=None):`**

Generates a comprehensive analysis report combining multiple visualizations including loss history, solution plots, residuals, and convergence metrics.

#### Inputs:

- **resolution** [int]: Grid resolution for visualization.  
*Default:* 100
- **x_range** [tuple]: (min, max) bounds for x-axis.  
*Default:* (0, 1)
- **y_range** [tuple]: (min, max) bounds for y-axis.  
*Default:* (0, 1)
- **savefig** [str]: Optional filepath to save the complete report figure.  
*Default:* None

#### Outputs:

- \[matplotlib.figure.Figure]: Figure object with comprehensive analysis dashboard.

**13. `.plot_training_animation(resolution=50, x_range=(0, 1), y_range=(0, 1), interval=200, save_path=None):`**

Creates an animated visualization of how the PINN solution evolves during training.

#### Inputs:

- **resolution** [int]: Grid resolution for animation frames.  
*Default:* 50
- **x_range** [tuple]: (min, max) bounds for x-axis.  
*Default:* (0, 1)
- **y_range** [tuple]: (min, max) bounds for y-axis.  
*Default:* (0, 1)
- **interval** [int]: Delay between frames in milliseconds.  
*Default:* 200
- **save_path** [str]: Optional filepath to save animation.  
*Default:* None

#### Outputs:

- Animation object displaying solution evolution during training.

**14. `.get_metrics_summary():`**

Retrieves a summary dictionary of key metrics from the most recent training session.

#### Inputs:

- None

#### Outputs:

- \[dict]: Dictionary containing final loss values, convergence information, and timing statistics.

**15. `.save_model(filepath):`**

Saves the trained PINN model to disk for later inference or resumption of training.

#### Inputs:

- **filepath** [str]: Destination file path for model checkpoint.

#### Outputs:

- None. Model state is persisted to disk.

**16. `.load_model(filepath):`**

Loads a previously trained PINN model from disk.

#### Inputs:

- **filepath** [str]: Source file path for model checkpoint.

#### Outputs:

- None. Model is restored and ready for prediction.

## **PulsarPINN2D Key Attributes**

Here we describe the main attributes of the PulsarPINN2D class:

### Model Configuration

- **.pde_expr**: SymPy expression representing the partial differential equation
- **.input_dim**: Dimensionality of input space (typically 2)
- **.hidden_layers**: List of integers specifying neurons in each hidden layer
- **.output_dim**: Dimensionality of output space (typically 1)
- **.backend_type**: Backend enumeration specifying computational framework
- **.variables**: Dictionary mapping variable names to SymPy symbols

### Backend and Computation

- **._backend**: Active backend instance (either PyTorchBackend or DeepXDEBackend)
- **.device**: Computation device for PyTorch ('cpu' or 'cuda')

### Training History

- **.metrics_history**: List of TrainingMetrics objects tracking loss evolution
  - Each metric includes epoch number, total loss, physics loss, boundary loss, and elapsed time

### Visualization Configuration

- **.vis_config**: Dictionary containing visualization settings:
  - **colormap**: Default colormap for plots ('viridis' is default)
  - **dpi**: Figure resolution (100 is default)
  - **figsize**: Default figure dimensions ((10, 8) is default)
  - **style**: Matplotlib style name

## **PyTorch Backend Key Attributes**

The PyTorchBackend class maintains the following important attributes:

### Network Architecture

- **.model**: PyTorch sequential model containing the neural network
- **.device**: Computation device (cpu or cuda)
- **.optimizer**: Adam optimizer instance for parameter updates

### Training Data

- **.training_data**: Dictionary containing all training tensors:
  - **x_pde**: x-coordinates for collocation points
  - **y_pde**: y-coordinates for collocation points
  - **x_bc**: x-coordinates for boundary points
  - **y_bc**: y-coordinates for boundary points
  - **u_bc**: Known solution values at boundary points

### Loss Functions

- **.pde_loss_fn**: Callable that computes PDE residual loss using automatic differentiation

## **DeepXDE Backend Key Attributes**

The DeepXDEBackend class maintains the following important attributes:

### Network and Geometry

- **.net**: DeepXDE neural network (FNN) object
- **.geom**: DeepXDE geometry object representing the domain
- **.dde**: Reference to the DeepXDE library module

### Training Configuration

- **.bc_list**: List of boundary condition objects
- **.data**: DeepXDE data object containing training points
- **.model**: DeepXDE model combining network, data, and PDE

### PDE Definition

- **.pde_expr**: SymPy expression for the PDE
- **.pde_func**: DeepXDE-compatible PDE residual function

## **Training Metrics Dataclass**

Both backends use a TrainingMetrics dataclass to record training progress. Each metric includes:

- **epoch**: Current training epoch number
- **total_loss**: Combined physics and boundary condition loss
- **pde_loss**: Physics constraint violation (PDE residual) loss
- **boundary_loss**: Boundary condition satisfaction loss
- **initial_loss**: Initial condition loss (if applicable)
- **elapsed_time**: Cumulative training time in seconds

## **Exporting Trained Models**

The pulsarfitpy framework provides comprehensive export functionality through the PINNSolutionExporter class. This allows you to save training results and persist trained models for future use.

### Key export methods:

**1. `.save_predictions_to_csv(filepath):`**

Exports model predictions on training, validation, and test sets to a CSV file with input-output pairs and model evaluations.

#### Inputs:

- **filepath** [str]: Path and filename for the output CSV file

**2. `.save_learned_constants_to_csv(filepath):`**

Saves the learned physical constants from the trained model to a CSV file with constant names and their optimized values.

#### Inputs:

- **filepath** [str]: Path and filename for the output CSV file

**3. `.save_metrics_to_csv(filepath):`**

Exports comprehensive evaluation metrics (R^2, RMSE, MAE, chi^2) for all data splits (train/val/test) to a CSV file.

#### Inputs:

- **filepath** [str]: Path and filename for the output CSV file

**4. `.save_loss_history_to_csv(filepath):`**

Saves the complete training loss history including total loss, physics loss, and data loss across all epochs to a CSV file.

#### Inputs:

- **filepath** [str]: Path and filename for the output CSV file

**5. `.save_model_checkpoint(filepath, include_metadata=True):`**

Saves the trained model state to a PyTorch checkpoint file (.pt) for later inference or continued training. This preserves the neural network weights, learned constants, and optionally includes training metadata.

#### Inputs:

- **filepath** [str]: Path and filename for the checkpoint file (typically .pt extension)
- **include_metadata** [bool]: Whether to include loss history, metrics, and hyperparameters in the checkpoint.  
*Default:* True

#### Outputs:

- Checkpoint file containing: model state dictionary, learnable parameters, loss history, test metrics, and model metadata

## **Usage Notes**

The following practical guidance applies when using the 2D PINN framework:

**Physics Loss Computation**: The physics loss measures how well the network solution satisfies the governing partial differential equation through residual computation at interior collocation points.

**Boundary Condition Enforcement**: Boundary condition loss ensures the network output matches known values at domain boundaries, which is essential for well-posed PDE problems.

**Grid Resolution**: Higher resolution in collocation points (set via set_training_data) and visualization (set via plot_ methods) improves accuracy but increases computational cost. Balance is needed based on problem complexity.

**Visualization Interpretation**: The contour plots show solution topology directly, while 3D surface plots provide intuitive understanding of solution magnitude variation. Loss heatmaps reveal which training phase had performance issues.

**Backend Selection**: Choose PyTorchBackend for maximum flexibility with custom PDEs, or DeepXDEBackend for rapid prototyping of standard PDE types with built-in optimizations.

**Training Convergence**: Monitor both physics loss and boundary loss separately using plot_loss_history. If physics loss stagnates while boundary loss decreases, increase physics_weight in training parameters.

**Solution Validation**: When analytical solutions are available, use plot_comparison_with_analytical to quantify solution accuracy and identify remaining errors in the physics approximation.

**Residual Analysis**: Use plot_residual_distribution to locate regions where the learned solution violates physics constraints most significantly. High residuals indicate need for domain refinement or model architecture adjustment.

**Memory Management**: For large grids or high-resolution visualizations, use resolution parameter carefully. Memory usage scales with resolution squared for 2D problems.

**Checkpoint Management**: Use save_model and load_model for long training sessions to preserve progress and enable resumption of training without restarting from scratch.

[← Back to Technical Information Home](technicalinformation.md)