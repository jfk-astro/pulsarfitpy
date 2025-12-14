# Visualizing PINN Solutions and Training Dynamics with pulsarfitpy

After training a Physics-Informed Neural Network (PINN) with pulsarfitpy, comprehensive visualization of model predictions, training dynamics, and diagnostic metrics is essential for understanding model behavior, validating physical consistency, and communicating results. The pulsarfitpy library provides robust visualization functionality through the [VisualizePINN](https://github.com/jfk-astro/pulsarfitpy) class, enabling publication-quality plots and diagnostic visualizations for scientific analysis.

This markdown file demonstrates how to visualize PINN solutions using the VisualizePINN class, including prediction comparisons, loss convergence analysis, residual diagnostics, uncertainty quantification, and robustness validation. We provide practical examples showing how to create publication-ready figures that effectively communicate your PINN results for scientific presentations and peer-reviewed publications.

# Understanding the VisualizePINN Class

>[NOTE]
> The VisualizePINN class is designed to work seamlessly with trained PulsarPINN models. Ensure your PINN has been trained and evaluated before attempting visualization operations.

To visualize results from a trained PINN model, the VisualizePINN class provides intuitive methods for generating matplotlib figures that compare predictions against observational data, analyze training convergence, and diagnose model performance.

The class requires the following parameter:

- pinn_model=[PulsarPINN]: A trained PulsarPINN model instance from which visualizations will be generated.

# VisualizePINN Methods

The core methods of the visualization functionality are as follows:

1. .plot_predictions_vs_data(x_values=None, y_predictions=None, x_axis=None, y_axis=None, save_path=None, figsize=(12, 8), title=None):

Creates a comprehensive comparison plot showing model predictions as a continuous curve overlaid on training, validation, and test data points with color-coded data splits.

Inputs:

- x_values [Optional[np.ndarray]]: X values for prediction curve. If None, generates extended range using predict_extended(). Default: None
- y_predictions [Optional[np.ndarray]]: Predicted y values. If None and x_values provided, generates predictions from model. Default: None
- x_axis [str]: Label for x-axis (e.g., "log_P" or "Period"). Default: None
- y_axis [str]: Label for y-axis (e.g., "log_Pdot" or "Period_Derivative"). Default: None
- save_path [Optional[str]]: If provided, saves figure to this path. Default: None (display only)
- figsize [Tuple[int, int]]: Figure size in inches (width, height). Default: (12, 8)
- title [Optional[str]]: Plot title. If None, generates default title with solution name. Default: None

Outputs:

- Displays or saves matplotlib figure containing:
  - Blue circles: Training data points (70% of dataset, α=0.4)
  - Orange circles: Validation data points (15% of dataset, α=0.4)
  - Red circles: Test data points (15% of dataset, α=0.4)
  - Green line: PINN prediction curve (linewidth=2.5, drawn on top with zorder=5)
  - Annotation box: Test set R² score in upper-left corner (if available)
  - Grid lines with dashed styling (α=0.3)
  - Legend with best positioning

2. .plot_loss_curves(log_scale=True):

Generates dual-panel visualization of training and validation loss evolution across epochs, showing both total loss and component decomposition into physics and data contributions.

Inputs:

- log_scale [bool]: Use logarithmic y-axis for better visualization of loss convergence. Default: True

Outputs:

- Displays matplotlib figure with two side-by-side subplots (14" × 5"):
  - **Left panel - Total Loss**: Training loss as solid line, validation loss as line with circular markers (○)
  - **Right panel - Loss Components**: Physics loss and data loss as dashed lines (--), validation components as dotted lines (:) with square markers (□)
  - Validation metrics plotted at checkpoint intervals determined by val_interval parameter
  - Grid lines with α=0.3 transparency for readability
  - Legend identifying all loss curves

3. .plot_residuals_analysis(figsize=(10, 6)):

Creates diagnostic scatter plot of prediction residuals (errors) versus input values to identify systematic biases, heteroscedasticity, or regions of poor model fit.

Inputs:

- figsize [Tuple[int, int]]: Figure size in inches (width, height). Default: (10, 6)

Outputs:

- Displays matplotlib figure containing:
  - Purple scatter points: Residuals (Y_true - Y_pred) on test set with black edges (α=0.6, s=50)
  - Red dashed reference line: Zero error baseline (linewidth=2)
  - Statistics annotation box: Residual mean and standard deviation in yellow box
  - Grid lines for quantitative assessment (α=0.3)
  - Bold axis labels and title

4. .plot_prediction_scatter(figsize=(10, 8)):

Generates scatter plot comparing predicted values against true values with perfect prediction reference line to assess overall model accuracy and identify systematic deviations.

Inputs:

- figsize [Tuple[int, int]]: Figure size in inches (width, height). Default: (10, 8)

Outputs:

- Displays matplotlib figure containing:
  - Teal scatter points: Predicted vs. true values on test set with black edges (α=0.6, s=50)
  - Red dashed diagonal line: Perfect prediction (Y_pred = Y_true) with linewidth=2
  - R² score annotation: Model accuracy metric in light blue box
  - Grid lines for visual assessment (α=0.3)
  - Bold axis labels and title

5. .plot_uncertainty_quantification(uncertainties, figsize=(10, 6)):

Visualizes uncertainty estimates for learned physical constants with error bars (mean ± standard deviation) and 95% confidence interval bounds.

Inputs:

- uncertainties [Dict]: Dictionary of uncertainties from bootstrap_uncertainty() or monte_carlo_uncertainty() methods. Each key should be a constant name (e.g., "n_braking", "logK") with value containing keys: 'mean', 'std', 'ci_lower', 'ci_upper'
- figsize [Tuple[int, int]]: Figure size in inches (width, height). Default: (10, 6)

Outputs:

- Displays matplotlib figure containing:
  - Dark blue circles with error bars: Mean parameter values ± standard deviation (markersize=10, capsize=5)
  - Red horizontal lines: 95% confidence interval bounds (lower and upper)
  - X-axis: Constant names with underscores replaced by newlines for readability
  - Y-axis: Parameter values with bold labels
  - Legend identifying error bars and confidence intervals
  - Grid lines on y-axis for quantitative reading (α=0.3)

6. .plot_robustness_validation(robustness_results, figsize=(12, 5)):

Creates dual-panel visualization summarizing robustness validation test results with pass/fail indicators and detailed metrics for permutation tests, feature shuffling, and impossible physics validation.

Inputs:

- robustness_results [Dict]: Dictionary of robustness test results from run_all_robustness_tests() containing keys: 'permutation_test', 'feature_shuffling_test', 'impossible_physics_test'
- figsize [Tuple[int, int]]: Figure size in inches (width, height). Default: (12, 5)

Outputs:

- Displays matplotlib figure with two side-by-side panels:
  - **Left panel - Pass/Fail Summary**: Bar chart showing test results with green (pass) or red (fail) coloring, overlaid with [PASS] or [FAIL] symbols
  - **Right panel - Detailed Metrics**: Text box containing:
    - Permutation test: p-value and significance flag
    - Feature shuffling: R² difference, original R², shuffled R²
    - Impossible physics: Real vs. impossible R² comparison
  - Both panels formatted for clear interpretation of model robustness

7. .plot_braking_index_distribution(learned_constants, uncertainties, figsize=(10, 6)):

Generates histogram visualization of braking index distribution from bootstrap sampling, comparing learned value against canonical theoretical prediction (n=3.0).

Inputs:

- learned_constants [Dict[str, float]]: Dictionary of learned constants from store_learned_constants() containing 'n_braking' key
- uncertainties [Dict]: Dictionary of uncertainties from bootstrap_uncertainty() or monte_carlo_uncertainty() containing 'n_braking' key with 'std' value
- figsize [Tuple[int, int]]: Figure size in inches (width, height). Default: (10, 6)

Outputs:

- Displays matplotlib figure containing:
  - Sky blue histogram: Bootstrap distribution of braking index values (bins=25, α=0.7)
  - Green solid line: Learned braking index value (linewidth=2.5)
  - Orange dotted line: Canonical theoretical value (n=3.0, linewidth=2.5)
  - Legend identifying all reference lines
  - Grid lines on y-axis for frequency assessment (α=0.3)
  - Bold axis labels and title

# Example Usage

A typical workflow for visualizing PINN results follows this sequence:

1. Train the PINN model:
```python
pinn = PulsarPINN(
    differential_eq=differential_equation,
    x_sym=logP,
    y_sym=logPdot,
    learn_constants={n_braking: 3.0, logK: -16.0},
    fixed_inputs=fixed_inputs
)
pinn.train(epochs=4000, physics_weight=1.5, data_weight=1.0)
```

2. Evaluate model performance:
```python
metrics = pinn.evaluate_test_set(verbose=True)
```

3. Initialize the visualization class:
```python
from visualize_solutions import VisualizePINN
visualizer = VisualizePINN(pinn)
```

4. Create prediction comparison plot:
```python
visualizer.plot_predictions_vs_data(
    x_axis="log(Period) [s]",
    y_axis="log(Period Derivative) [s/s]",
    save_path="pinn_predictions.png"
)
```

5. Visualize training convergence:
```python
visualizer.plot_loss_curves(log_scale=True)
```

6. Generate diagnostic plots:
```python
# Residual analysis
visualizer.plot_residuals_analysis(figsize=(10, 6))

# Prediction scatter plot
visualizer.plot_prediction_scatter(figsize=(10, 8))
```

7. Visualize uncertainty quantification:
```python
# Compute uncertainties using bootstrap method
uncertainties = pinn.bootstrap_uncertainty(n_iterations=100)

# Plot uncertainty estimates
visualizer.plot_uncertainty_quantification(
    uncertainties=uncertainties,
    figsize=(10, 6)
)
```

8. Validate model robustness:
```python
# Run robustness tests
robustness_results = pinn.run_all_robustness_tests()

# Visualize validation results
visualizer.plot_robustness_validation(
    robustness_results=robustness_results,
    figsize=(12, 5)
)
```

9. Analyze braking index distribution:
```python
# Store learned constants
learned_constants = pinn.store_learned_constants()

# Plot braking index distribution
visualizer.plot_braking_index_distribution(
    learned_constants=learned_constants,
    uncertainties=uncertainties,
    figsize=(10, 6)
)
```

# Visualization Format Reference

## Prediction Comparison Plot Structure

The predictions vs. data plot provides a comprehensive visual comparison between model predictions and observational data:

**Visual Elements:**
- **Data Points**: Three color-coded scatter plots representing train (blue), validation (orange), and test (red) splits with consistent transparency (α=0.4) and marker size (s=30)
- **Prediction Curve**: Smooth green line showing continuous PINN prediction across extended input range, drawn with higher z-order to appear on top of data points
- **Performance Annotation**: R² score displayed in wheat-colored box in upper-left corner (if available from test metrics)
- **Grid and Legend**: Dashed grid lines (α=0.3) for quantitative reading, legend with best automatic positioning

**Interpretation Guidelines:**
- Well-trained models show prediction curves passing through the center of data point clouds
- Systematic deviations indicate potential physics constraint violations or insufficient training
- Test data points (red) should align well with predictions if model generalizes effectively

## Loss Curves Visualization Structure

The dual-panel loss curves plot enables analysis of training dynamics and convergence behavior:

**Left Panel - Total Loss:**
- Solid lines showing total loss evolution for training set
- Circular markers (○) on validation loss curve showing checkpoint evaluations
- Logarithmic y-axis (default) compresses large initial losses while preserving convergence detail

**Right Panel - Loss Components:**
- Dashed lines (--) showing physics and data loss decomposition for training
- Dotted lines (:) with square markers (□) showing validation loss components
- Enables identification of physics-data balance and potential overfitting

**Convergence Indicators:**
- Smooth monotonic decrease indicates healthy training
- Divergence between training and validation suggests overfitting
- Plateau in physics loss while data loss decreases indicates insufficient physics weight

## Residuals Analysis Plot Structure

The residuals plot provides critical diagnostic information about model error patterns:

**Visual Elements:**
- Purple scatter points showing test set residuals (Y_true - Y_pred) versus input values
- Red dashed horizontal line at zero indicating perfect prediction
- Yellow annotation box displaying residual statistics (mean and standard deviation)

**Diagnostic Patterns:**
- **Random scatter around zero**: Indicates well-calibrated model with no systematic bias
- **Curved pattern**: Suggests missing nonlinear effects or incorrect differential equation
- **Heteroscedasticity (funnel shape)**: Indicates input-dependent prediction uncertainty
- **Outliers**: May represent unusual pulsars or data quality issues

**Statistical Benchmarks:**
- Residual mean close to zero (|mean| < 0.01) indicates unbiased predictions
- Small standard deviation indicates tight prediction accuracy
- Residual magnitude should be consistent with test set RMSE

## Prediction Scatter Plot Structure

The predicted vs. true values scatter plot assesses overall model accuracy:

**Visual Elements:**
- Teal scatter points comparing predictions against ground truth
- Red dashed diagonal line showing perfect prediction (Y_pred = Y_true)
- Light blue annotation box displaying R² score

**Interpretation Guidelines:**
- Points clustering tightly along diagonal indicate accurate predictions
- Vertical or horizontal spreads indicate systematic over/under-prediction
- R² score quantifies proportion of variance explained (target: R² > 0.95 for good models)

## Uncertainty Quantification Plot Structure

The uncertainty plot visualizes confidence in learned physical constants:

**Visual Elements:**
- Error bars showing mean ± standard deviation for each constant
- Red horizontal markers indicating 95% confidence interval bounds
- X-axis labels with underscores replaced by newlines for readability

**Interpretation Guidelines:**
- Narrow confidence intervals indicate well-constrained parameters
- Overlapping intervals between constants may suggest parameter correlation
- Constants with large uncertainties may require more data or stronger physics constraints

## Robustness Validation Plot Structure

The dual-panel robustness plot summarizes model validation tests:

**Left Panel - Pass/Fail Summary:**
- Green bars: Tests passed (model demonstrates robustness)
- Red bars: Tests failed (model exhibits concerning behavior)
- Large [PASS]/[FAIL] symbols for immediate visual assessment

**Right Panel - Detailed Metrics:**
- Monospace text displaying quantitative test results
- Three test categories: Permutation, Feature Shuffling, Impossible Physics
- Specific p-values, R² differences, and comparison metrics

**Passing Criteria:**
- **Permutation Test**: p-value < 0.05 (model significantly better than random)
- **Feature Shuffling**: R² difference > 0.1 (model relies on feature relationships)
- **Impossible Physics**: Real R² >> Impossible R² (model learns valid physics)

## Braking Index Distribution Plot Structure

The histogram plot visualizes parameter uncertainty and compares against theoretical predictions:

**Visual Elements:**
- Sky blue histogram showing bootstrap distribution of learned braking index
- Green solid line: Learned value from PINN training
- Orange dotted line: Canonical theoretical prediction (n=3.0)
- Legend identifying all reference lines

**Interpretation Guidelines:**
- Narrow distributions indicate high confidence in learned parameter
- Distance from canonical value (n=3.0) indicates deviation from standard dipole braking
- Distribution shape assesses parameter uncertainty and sampling variability

# Advanced Visualization Examples

## Creating Publication-Ready Figures
```python
visualizer = VisualizePINN(pinn)

# High-resolution prediction plot with custom styling
visualizer.plot_predictions_vs_data(
    x_axis=r"$\log_{10}(P)$ [s]",  # LaTeX formatting
    y_axis=r"$\log_{10}(\dot{P})$ [s/s]",
    title="Pulsar Spin-Down: PINN vs. ATNF Observations",
    figsize=(10, 7),
    save_path="figures/publication_predictions.png"
)

# Note: The saved figure will have 300 DPI resolution suitable for publication
```

## Multi-Panel Diagnostic Dashboard
```python
import matplotlib.pyplot as plt

# Create custom figure with 2x2 grid of diagnostic plots
fig = plt.figure(figsize=(16, 12))

# Manually create subplots and use visualizer methods
# Note: This requires accessing internal plotting logic or 
# creating standalone functions for individual plot components

# Alternative: Generate individual plots sequentially
visualizer.plot_predictions_vs_data(save_path="diag_predictions.png")
visualizer.plot_loss_curves(log_scale=True)
visualizer.plot_residuals_analysis()
visualizer.plot_prediction_scatter()
```

## Comparative Uncertainty Analysis
```python
# Compare bootstrap vs. Monte Carlo uncertainty estimates
bootstrap_unc = pinn.bootstrap_uncertainty(n_iterations=100)
monte_carlo_unc = pinn.monte_carlo_uncertainty(n_iterations=100)

# Visualize both uncertainty estimates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot bootstrap uncertainties
plt.sca(ax1)
visualizer.plot_uncertainty_quantification(
    uncertainties=bootstrap_unc,
    figsize=(8, 6)
)
ax1.set_title("Bootstrap Uncertainty Estimation")

# Plot Monte Carlo uncertainties
plt.sca(ax2)
visualizer.plot_uncertainty_quantification(
    uncertainties=monte_carlo_unc,
    figsize=(8, 6)
)
ax2.set_title("Monte Carlo Uncertainty Estimation")

plt.tight_layout()
plt.show()
```

## Extended Predictions Beyond Data Range
```python
# Generate predictions over extended input range
x_extended, y_extended = pinn.predict_extended(extend=1.5, n_points=500)

# Visualize extrapolation behavior
visualizer.plot_predictions_vs_data(
    x_values=x_extended,
    y_predictions=y_extended,
    x_axis="log(Period) [s]",
    y_axis="log(Period Derivative) [s/s]",
    title="PINN Extrapolation Beyond Observed Data Range",
    save_path="extrapolation_analysis.png"
)
```

## Animated Training Convergence
```python
import matplotlib.animation as animation

# Note: This requires custom implementation
# The following pseudocode illustrates the concept

def create_training_animation(pinn, output_file="training_convergence.mp4"):
    """
    Create animated visualization of training convergence.
    
    Requires saving model checkpoints during training at regular intervals.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    def update_frame(epoch):
        # Load checkpoint from epoch
        # Plot loss curves up to current epoch
        # Plot current predictions
        pass
    
    anim = animation.FuncAnimation(
        fig, update_frame, 
        frames=range(0, pinn.epochs, 100),
        interval=100
    )
    anim.save(output_file, writer='ffmpeg', fps=10)
```

# Integration with Scientific Workflows

## Jupyter Notebook Integration

For interactive analysis in Jupyter notebooks:
```python
# Enable inline plotting
%matplotlib inline

# Initialize visualizer
visualizer = VisualizePINN(pinn)

# Create interactive widget for exploring predictions
from ipywidgets import interact, FloatSlider

def interactive_prediction_plot(extend_factor):
    x_ext, y_ext = pinn.predict_extended(extend=extend_factor, n_points=300)
    visualizer.plot_predictions_vs_data(
        x_values=x_ext,
        y_predictions=y_ext
    )

interact(
    interactive_prediction_plot,
    extend_factor=FloatSlider(min=0.0, max=2.0, step=0.1, value=0.5)
)
```

## Batch Visualization Pipeline

For processing multiple PINN models:
```python
import os

def visualize_all_models(model_directory, output_directory):
    """
    Generate standardized visualizations for all trained models.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    for model_file in os.listdir(model_directory):
        if model_file.endswith('.pkl'):
            # Load PINN model
            pinn = load_pinn_model(os.path.join(model_directory, model_file))
            visualizer = VisualizePINN(pinn)
            
            model_name = model_file.replace('.pkl', '')
            
            # Generate all visualizations
            visualizer.plot_predictions_vs_data(
                save_path=f"{output_directory}/{model_name}_predictions.png"
            )
            visualizer.plot_loss_curves(log_scale=True)
            plt.savefig(f"{output_directory}/{model_name}_losses.png")
            plt.close()
            
            visualizer.plot_residuals_analysis()
            plt.savefig(f"{output_directory}/{model_name}_residuals.png")
            plt.close()
            
            print(f"Visualizations completed for {model_name}")
```

## Integration with ATNF Data Comparison
```python
import pandas as pd

# Load ATNF catalogue data
atnf_data = pd.read_csv("atnf_pulsar_data.csv")

# Generate PINN predictions at ATNF data points
x_atnf = atnf_data['log_P'].values.reshape(-1, 1)
y_pinn_pred = pinn.predict(x_atnf)

# Create custom comparison visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Plot ATNF observations
ax.scatter(
    atnf_data['log_P'],
    atnf_data['log_Pdot'],
    c='blue',
    alpha=0.5,
    s=50,
    label='ATNF Observations',
    marker='o'
)

# Plot PINN predictions
ax.scatter(
    x_atnf,
    y_pinn_pred,
    c='red',
    alpha=0.5,
    s=50,
    label='PINN Predictions',
    marker='x'
)

# Add connecting lines for residuals
for i in range(len(x_atnf)):
    ax.plot(
        [x_atnf[i], x_atnf[i]],
        [atnf_data['log_Pdot'].iloc[i], y_pinn_pred[i]],
        'gray',
        alpha=0.2,
        linewidth=0.5
    )

ax.set_xlabel(r"$\log_{10}(P)$ [s]", fontsize=12, fontweight='bold')
ax.set_ylabel(r"$\log_{10}(\dot{P})$ [s/s]", fontsize=12, fontweight='bold')
ax.set_title("PINN Predictions vs. ATNF Catalogue", fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pinn_atnf_comparison.png", dpi=300)
plt.show()
```

# Best Practices for Visualization

1. **Always visualize loss curves**: Verify training convergence and check for overfitting by examining training vs. validation loss trajectories.

2. **Generate diagnostic plots**: Use residuals analysis and prediction scatter plots to identify systematic errors and assess model calibration.

3. **Include uncertainty estimates**: Visualize parameter uncertainties to communicate confidence levels and guide interpretation of learned constants.

4. **Save high-resolution figures**: Use `save_path` parameter with DPI=300 for publication-quality outputs suitable for journals.

5. **Use descriptive axis labels**: Include units and LaTeX formatting for professional presentation (e.g., r"$\log_{10}(P)$ [s]").

6. **Validate robustness**: Always run and visualize robustness tests to ensure model learns meaningful physical relationships.

7. **Compare against baselines**: Include theoretical predictions or empirical fits as reference lines for context.

8. **Document figure generation**: Keep scripts that generate figures under version control for reproducibility.

# Usage Notes

- All visualization methods display figures by default; use `save_path` parameter to save instead
- Matplotlib figures can be customized further after generation using standard matplotlib API
- Large datasets may require adjusting marker sizes and transparencies for clarity
- Logarithmic scaling (log_scale=True) is recommended for loss curves to visualize convergence clearly
- Color schemes follow standard conventions: blue (train), orange (validation), red (test), green (predictions)
- Grid lines with α=0.3 provide quantitative reference without visual clutter
- Figure sizes are optimized for screen display; adjust for specific publication requirements
- All plots support tight_layout() for automatic spacing optimization