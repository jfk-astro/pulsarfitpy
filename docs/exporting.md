---
layout: default
title: pulsarfitpy
---

# **pulsarfitpy Technical Information**

## **Exporting PINN Solutions and Results with pulsarfitpy**

After training a Physics-Informed Neural Network (PINN) with pulsarfitpy, you may want to save your predictions, learned constants, and evaluation metrics to external files for further analysis, publication, or integration with other workflows. The pulsarfitpy library provides comprehensive export functionality through the [ExportPINN](https://github.com/jfk-astro/pulsarfitpy) class, enabling seamless conversion of PINN results into standardized CSV formats.

This markdown file demonstrates how to export PINN solutions using the ExportPINN class, including predictions, learned physical constants, evaluation metrics, and training loss histories. We will provide practical examples showing how to save your PINN results for use with ATNF data comparison, publication-ready outputs, and reproducible scientific workflows.

## **Understanding the ExportPINN Class**

>[NOTE]
> The ExportPINN class is designed to work seamlessly with trained PulsarPINN models. Ensure your PINN has been trained and evaluated before attempting export operations.

To export results from a trained PINN model, the ExportPINN class provides intuitive methods for converting neural network outputs into structured CSV files suitable for scientific analysis and presentation.

The class requires the following parameter:

- **pinn_model**=[PulsarPINN]: A trained PulsarPINN model instance from which data will be exported.

## **ExportPINN Methods**

The core methods of the export functionality are as follows:

**1. `.save_predictions_to_csv(filepath, x_value_name, y_value_name, x_values=None, y_predictions=None, include_raw_data=True, include_test_metrics=True, additional_metadata=None):`**

  Saves model predictions and metadata to a CSV file with optional inclusion of original data points and test metrics.

#### Inputs:
  - **filepath** [str]: Output CSV file path. Parent directories will be created if needed.
  - **x_value_name** [str]: Column name for independent variable (e.g., "log_P" or "Period")
  - **y_value_name** [str]: Column name for dependent variable (e.g., "log_Pdot" or "Period_Derivative")
  - **x_values** [Optional[np.ndarray]]: X values for predictions. If None, uses extended prediction range.  
   *Default:* None
  - **y_predictions** [Optional[np.ndarray]]: Predicted y values. If None and x_values provided, generates predictions.   
   *Default:* None
  - **include_raw_data** [bool]: If True, includes original data points from train/val/test splits.  
   *Default:* True
  - **include_test_metrics** [bool]: If True, adds model performance metrics to header.  
   *Default:* True
  - **additional_metadata** [Optional[dict]]: Additional metadata to include in CSV header comments.  
   *Default:* None

#### Outputs:

 - CSV file containing model predictions, metadata header, and optionally raw observational data

 CSV Structure:

 - Header comments (lines starting with #) contain model architecture, learned constants, and performance metrics
 - Predictions section contains model-generated solutions over specified range
 - Raw data section (if enabled) contains original training, validation, and test data points with split labels

**2. `.save_learned_constants_to_csv(filepath, additional_info=None, include_uncertainty=False, uncertainty_method='bootstrap', n_iterations=100):`**

 Saves learned physical constants from the PINN to a CSV file with optional uncertainty estimates.

#### Inputs:

 - **filepath** [str]: Output CSV file path.
 - **additional_info** [Optional[dict]]: Additional information to include alongside constants.  
  *Default:* None
 - **include_uncertainty** [bool]: Whether to compute and include uncertainty estimates.  
  *Default:* False
 - **uncertainty_method** [str]: Method for uncertainty estimation ('bootstrap' or 'monte_carlo').  
  *Default:* 'bootstrap'
 - **n_iterations** [int]: Number of bootstrap samples or Monte Carlo simulations for uncertainty.  
  *Default:* 100

#### Outputs:

 - CSV file containing learned constants with the following columns:

 **constant_name**: Name of the physical constant (e.g., "n_braking", "logK")  
 **value**: Learned constant value as a float  
 **equation**: The governing differential equation string  
 **uncertainty_std**: Standard deviation (if include_uncertainty=True)  
 **ci_lower_95**: Lower bound of 95% confidence interval (if include_uncertainty=True)  
 **ci_upper_95**: Upper bound of 95% confidence interval (if include_uncertainty=True)  
 **uncertainty_method**: Method used for uncertainty estimation (if include_uncertainty=True)  
 **n_iterations**: Number of iterations used (if include_uncertainty=True)  

**3. `.save_metrics_to_csv(filepath, additional_info=None):`**

 Saves model evaluation metrics to a CSV file for performance comparison and documentation.

#### Inputs:

 - **filepath** [str]: Output CSV file path.
 - **additional_info** [Optional[dict]]: Additional information to include with metrics.  
 *Default:* None

#### Outputs:

 - CSV file containing evaluation metrics including:

 **R² scores** for train/validation/test splits  
 **RMSE (Root Mean Squared Error) values** for each split  
 **MAE (Mean Absolute Error) values** for each split  
 **Reduced χ² statistics** for each split  
 **Total, physics, and data losses** for each split  

**4. `.save_loss_history_to_csv(filepath):`**

 Saves the complete training loss history to a CSV file for analyzing convergence behavior and training dynamics.

#### Inputs:

 - **filepath** [str]: Output CSV file path.

#### Outputs:

 - CSV file containing epoch-by-epoch loss values with the following columns:

 **epoch**: Training epoch number (0 to total_epochs)  
 **total_loss**: Combined physics and data loss  
 **physics_loss**: Physics constraint violation loss  
 **data_loss**: Data fitting loss  
 **val_total_loss**: Validation total loss (if available)  
 **val_physics_loss**: Validation physics loss (if available)  
 **val_data_loss**: Validation data loss (if available)  

## **Example Usage**

A typical workflow for exporting PINN results follows this sequence:

**1. Train the PINN model:**

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

**2. Evaluate model performance:**

```python
metrics = pinn.evaluate_test_set(verbose=True)
```

**3. Initialize the export class:**

```python
from export_solutions import ExportPINN
exporter = ExportPINN(pinn)
```

**4. Export predictions with metadata:**

```python
exporter.save_predictions_to_csv(
    filepath="pinn_predictions.csv",
    x_value_name="log_P",
    y_value_name="log_Pdot",
    include_raw_data=True,
    include_test_metrics=True
)
```

**5. Export learned constants with uncertainty:**

```python
exporter.save_learned_constants_to_csv(
    filepath="learned_constants.csv",
    include_uncertainty=True,
    uncertainty_method='bootstrap',
    n_iterations=100
)
```

**6. Export evaluation metrics:**

```python
exporter.save_metrics_to_csv(
    filepath="model_metrics.csv"
)
```

**7. Export training history:**

```python
exporter.save_loss_history_to_csv(
    filepath="training_history.csv"
)
```

## **Export Format Reference**

### **CSV Header Format for Predictions**

The predictions CSV file includes a comprehensive header with metadata:

```
# ======================================================================
# PHYSICS-INFORMED NEURAL NETWORK PREDICTIONS
# ======================================================================
# Model: PulsarPINN
# Input Variable: logP
# Output Variable: logPdot
# Differential Equation: Eq(logPdot, (n_braking - 1)*logP + logK)
#
# Network Architecture: 1 → [128, 64, 32, 16] → 1
# Total Parameters: 2847
#
# Data Splits:
#   Train: 175 samples (70.0%)
#   Validation: 37 samples (15.0%)
#   Test: 38 samples (15.0%)
#
# Learned Physical Constants:
#   n_braking = 2.845123
#   logK = -15.234567
#
# Test Set Performance:
#   R² Score: 0.987654
#   RMSE: 1.234e-02
#   MAE: 8.901e-03
#   Reduced χ²: 1.456
# ======================================================================
```

Data rows follow the header:

```
log_P,log_Pdot_predicted,data_type
-2.500000,-16.234567,model_prediction
-2.450000,-16.178901,model_prediction
...
-0.034576,-17.456789,raw_train
-0.012345,-17.123456,raw_validation
0.123456,-16.987654,raw_test
```

### **Constants CSV Format**

The learned constants file provides a structured record of discovered physical parameters:

```
constant_name,value,equation,uncertainty_std,ci_lower_95,ci_upper_95,uncertainty_method,n_iterations
n_braking,2.845123,"Eq(logPdot, (n_braking - 1)*logP + logK)",0.087654,2.673456,3.016789,bootstrap,100
logK,-15.234567,"Eq(logPdot, (n_braking - 1)*logP + logK)",0.125432,-15.479876,-14.989258,bootstrap,100
```

### **Metrics CSV Format**

The metrics file documents model performance across all data splits:

```
metric_name,value
train_loss_total,0.001234
train_loss_physics,0.000567
train_loss_data,0.000667
val_loss_total,0.001456
val_loss_physics,0.000678
val_loss_data,0.000778
test_loss_total,0.001567
test_loss_physics,0.000789
test_loss_data,0.000778
train_r2,0.992345
val_r2,0.989012
test_r2,0.987654
train_rmse,0.012345
val_rmse,0.013456
test_rmse,0.014567
train_mae,0.009012
val_mae,0.010123
test_mae,0.011234
train_chi2_reduced,1.234
val_chi2_reduced,1.345
test_chi2_reduced,1.456
```

### **Loss History CSV Format**

The loss history file tracks training progression:

```
epoch,total_loss,physics_loss,data_loss,val_total_loss,val_physics_loss,val_data_loss
0,102.345678,54.123456,48.222222,98.765432,52.345678,46.419754
1,95.234567,48.567890,46.666677,92.123456,47.234567,44.888889
100,0.012345,0.005678,0.006667,0.014567,0.006789,0.007778
500,0.001456,0.000678,0.000778,0.001567,0.000789,0.000778
1000,0.001234,0.000567,0.000667,0.001456,0.000678,0.000778
```

## **Advanced Export Examples**

### **Exporting with Custom Metadata**

```python
exporter = ExportPINN(pinn)

custom_metadata = {
    "study": "Pulsar Braking Index Analysis",
    "author": "Your Name",
    "date": "2025-01-01",
    "data_source": "ATNF Catalogue",
    "notes": "Optimized architecture with 6000 epochs"
}

exporter.save_predictions_to_csv(
    filepath="results/braking_index_analysis.csv",
    x_value_name="log_P",
    y_value_name="log_Pdot",
    additional_metadata=custom_metadata,
    include_raw_data=True
)
```

### **Exporting Extended Predictions**

```python
# Generate extended predictions beyond data range
x_extended, y_extended = pinn.predict_extended(extend=1.0, n_points=500)

exporter.save_predictions_to_csv(
    filepath="extended_predictions.csv",
    x_value_name="log_P",
    y_value_name="log_Pdot",
    x_values=x_extended,
    y_predictions=y_extended,
    include_raw_data=False  # Only export extended predictions
)
```

### **Complete Analysis Export Pipeline**

```python
# After training and evaluation
exporter = ExportPINN(pinn)

# Create organized output directory structure
import os
output_dir = "pinn_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Export all components
exporter.save_predictions_to_csv(
    filepath=f"{output_dir}/predictions.csv",
    x_value_name="log_P",
    y_value_name="log_Pdot"
)

exporter.save_learned_constants_to_csv(
    filepath=f"{output_dir}/learned_constants.csv",
    include_uncertainty=True,
    uncertainty_method='bootstrap',
    n_iterations=100
)

exporter.save_metrics_to_csv(
    filepath=f"{output_dir}/evaluation_metrics.csv"
)

exporter.save_loss_history_to_csv(
    filepath=f"{output_dir}/training_history.csv"
)

print(f"All results exported to {output_dir}/")
```

## **Integration with ATNF Data**

When comparing PINN predictions with ATNF catalogue data, the exported CSV files can be easily merged for validation:

```python
import pandas as pd

# Load PINN predictions
predictions = pd.read_csv("pinn_predictions.csv", comment='#')

# Load ATNF data (example)
atnf_data = pd.read_csv("atnf_pulsar_data.csv")

# Merge on period
comparison = pd.merge(
    predictions[predictions['data_type'] == 'model_prediction'],
    atnf_data,
    on='log_P',
    suffixes=('_pinn', '_atnf')
)

# Calculate residuals
comparison['residual'] = (
    comparison['log_Pdot_pinn'] - comparison['log_Pdot_atnf']
)

comparison.to_csv("pinn_atnf_comparison.csv", index=False)
```

## **Best Practices for Export**

1. **Always export uncertainty estimates**: Include uncertainty quantification (`include_uncertainty=True`) for learned constants to demonstrate model reliability.

2. **Document metadata**: Use `additional_metadata` to record experiment parameters, training configuration, and computational environment.

3. **Preserve raw data**: Keep `include_raw_data=True` in predictions export for transparency and reproducibility.

4. **Export loss history**: Always save training history to verify convergence and detect potential overfitting.

5. **Version control**: Include date, author, and analysis version in metadata for tracking results across multiple experiments.

6. **Organize outputs**: Use directory structures to organize exports from multiple PINN experiments for easy comparison.

## **Usage Notes**

- All exported CSV files include human-readable header comments starting with `#` for easy interpretation
- Parent directories are automatically created if they do not exist
- Uncertainty estimation can be computationally expensive; use smaller `n_iterations` for quick exports
- Raw data is labeled by split (raw_train, raw_validation, raw_test) for easy filtering
- Floating-point precision is maintained through the export process for scientific accuracy
- CSV files are UTF-8 encoded and compatible with spreadsheet applications and data analysis tools

[← Back to Technical Information Home](technicalinformation.md)