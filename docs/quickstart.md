---
layout: default
title: pulsarfitpy
---

# **pulsarfitpy Technical Information**

## **Understanding the PulsarPINN Class**

>[NOTE]
> Some of the class inputs involve specific parameters to be accessed by psrqpy. Refer to the [psrqpy documentation](https://psrqpy.readthedocs.io/_/downloads/en/latest/pdf/) and the [legend of ATNF parameters](https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html?type=expert#par_list) if needed.

To implement this 1D theoretical model into our library, a proper overview of the PulsarPINN class is detailed here.

The class requires the following parameters:

- **differential_eq**=[sympy.Eq]: The symbolic differential equation representing the physical model that the PINN must adhere to. Uses SymPy's symbolic notation for mathematical expressions.

- **x_sym**=[sympy.Symbol]: The symbolic representation of the independent variable data in a differential equation.

- **y_sym**=[sympy.Symbol]: The symbolic representation of the dependent variable data.

- **learn_constants**=[dict]: Dictionary containing unknown physical constants (as SymPy symbols) in the differential equation. These constants will be learned during training.  
*Example*: {logR: 18.0}

- **fixed_inputs**=[dict]: Dictionary containing all known data arrays for independent & dependent variables in the differential equation. Must include arrays for both x_sym and y_sym, plus any additional variables needed in the differential equation.  
*Example*: {logP: logP_data, logPDOT: logPDOT_data, logB: logB_data} where logP_data, logPDOT_data, logB_data.

- **log_scale**=[bool]: Indicates whether the input data is already in logarithmic scale. Used in premade plotting functions.  
 *Default:* True.

- **input_layer**=[int]: Size of the input layer (number of input features) for the PINN.  
 *Default:* 1.

- **hidden_layers**=[list]: List defining the PINN architecture as hidden layer sizes. Each integer represents a hidden layer with that many neurons.  
 *Default:* [32, 16] (meaning two hidden layers with 32 and 16 neurons).

- **output_layer**=[int]: Size of the output layer (number of output features).  
 *Default:* 1.

- **train_split**=[float]: Fraction of data used for training the PINN (range: 0.0–1.0).  
 *Default:* 0.70 (70%).

- **val_split**=[float]: Fraction of data used for validation of the PINN (range: 0.0–1.0).  
 *Default:* 0.15 (15%).

- **test_split**=[float]: Fraction of data used for testing for the PINN (range: 0.0–1.0).  
 *Default:* 0.15 (15%).

- **random_seed**=[int]: Random seed for reproducible data shuffling before PINN training.  
 *Default:* 42.

## **PulsarPINN Methods**

The core methods of the pulsarfitpy framework is as follows:

**1. `.train(epochs=3000, training_reports=100, physics_weight=1.0, data_weight=1.0):`**

 Trains the PINN model by using specified number of epochs and weight values.

#### Inputs: 

 - **epochs** [int]: Number of training iterations.  
   *Default:* 3000
 - **training_reports** [int]: Frequency of progress updates during training.  
   *Default:* 100
 - **physics_weight** [float]: Weighting factor for physics loss component.  
   *Default:* 1.0
 - **data_weight** [float]: Weighting factor for data fitting loss component.  
   *Default:* 1.0

#### Outputs:

 - Learned physical constants in the 1D differential equation in a dictionary

**2. `.predict_extended(extend=0.5, n_points=300):`**

 Generates model solutions over a range beyond the dataset to capture trends of the differential equation. Primarily used for exploring pulsar dynamics beyond given parameter ranges from the ATNF Catalogue.

#### Inputs:

 - **extend** [float]: Amount to extend prediction range beyond data limits.  
   *Default:* 0.5
 - **n_points** [int]: Number of prediction points to generate.  
   *Default:* 300

#### Outputs:

 -  Tuple of (x_values, y_predictions) as NumPy arrays for matplotlib plotting

**3. `.evaluate_test_set(verbose=True):`**

 Computes evaluation metrics and splits during training of the PINN. Used to determine how well the model has trained and how accurate the solutions are.

#### Inputs:

 - **verbose** [bool]: Chooses whether to print detailed evaluation report.  
   *Default:* True

#### Outputs:

 - Dictionary of metrics & data splits of PINN
 - Metrics include: Total Loss, Physics Loss, Data Loss, R^2 Score, RMSE, MAE, and reduced chi^2
 - Detects potential overfitting based on evaluated metrics

**4. `.store_learned_constants()`**

 Retrieves the learned physical constant values from the trained model. Used to extract final results after training for future experiments.

#### Outputs: 
 
 - Dictionary mapping constant names (strings) to their learned values (floats)

**5. `.set_learn_constants(new_constants)`**

 Updates or adds learnable constants with new initial values mid-workflow, and reinitializes the model to include new parameters

#### Inputs:

 - **new_constants** [dict]: Dictionary of constant names (strings) to new values (floats)

**6. `.bootstrap_uncertainty(n_bootstrap=100, sample_fraction=0.8, epochs=1000, confidence_level=0.95, verbose=True):`**

 Estimates uncertainty during model training through bootstrap iterations. Randomly samples training data, retrains model, and records learned values across multiple iterations of the program.

#### Inputs:

 - **n_bootstrap** [int]: Number of bootstrap iterations.  
   *Default:* 100
 - **sample_fraction** [float]: Fraction of training data to sample (0.0–1.0).  
   *Default:* 0.8
 - **epochs** [int]: Training epochs per bootstrap iteration.  
   *Default:* 1000
 - **confidence_level** [float]: Confidence level for intervals (e.g., 0.95 = 95% CI).  
   *Default:* 0.95
 - **verbose** [bool]: Whether to print progress and results.  
   *Default:* True

#### Outputs:

 - Dictionary containing mean, standard deviation, confidence intervals, and original values per learned constants

**7. `.monte_carlo_uncertainty(n_simulations=1000, noise_level=0.01, confidence_level=0.95, verbose=True)`**

 Another method to test uncertainty by adding Gaussian noise for data inputs and model re-evaluation. Ultimately asesses sensitivity of learned constants for accuracy of the model.

#### Inputs:

 - **n_simulations** [int]: Number of Monte Carlo simulations.  
   *Default:* 1000
 - **noise_level** [float]: Noise standard deviation relative to data std dev.  
   *Default:* 0.01 (1%)
 - **confidence_level** [float]: Confidence level for intervals.  
   *Default:* 0.95
 - **verbose** [bool]: Whether to print progress and results.  
   *Default:* True

#### Outputs:

 - Dictionary with uncertainty statistics for each learned constant.

**8. `.validate_with_permutation_test(n_permutations=100, epochs=1000, significance_level=0.05, verbose=True):`**

 Tests whether the model learns properly by comparing against randomly shuffled target labels. If real model outperforms permuted models, the learned relationships are likely genuine and therefore accurate.

#### Inputs:

 - **n_permutations** [int]: Number of random permutations.  
   *Default:* 100
 - **epochs** [int]: Training epochs per permutation.  
   *Default:* 1000
 - **significance_level** [float]: Statistical significance threshold.  
   *Default:* 0.05 (p < 0.05)
 - **verbose** [bool]: Whether to print results.  
   *Default:* True

#### Outputs:

 - Dictionary containing R^2 comparisons, p-value, and significance assessments

**9. `.validate_with_feature_shuffling(n_shuffles=50, epochs=1000, verbose=True):`**

 Validates input feature importance by shuffling x-values to analyze x-y relationships in the differential equation. Real model should outperform shuffled versions if features have genuine results.

#### Inputs:

 - **n_shuffles** [int]: Number of shuffling iterations.  
   *Default:* 50
 - **epochs** [int]: Training epochs per shuffle.  
   *Default:* 1000
 - **verbose** [bool]: Whether to print results.  
   *Default:* True

#### Outputs:

 - Dictionary with R^2 comparison and improvement metrics

**10. `.validate_with_impossible_physics(epochs=2000, verbose=True):`**

 Tests model robustness by training on external relationships (e.g., swapped input/output) to test impossibility & limits. A good model should perform poorly on this impossible physics test.

#### Inputs:

 - **epochs** [int]: Training epochs for impossible physics test.  
   *Default:* 2000
 - **verbose** [bool]: Whether to print results.  
   *Default:* True

#### Outputs:

 - Dictionary comparing real vs. impossible physics performances

**11. `.run_all_robustness_tests(n_permutations=100, n_shuffles=50, verbose=True):`**

 Executes the complete robustness validation functions above automatically, and provides comprehensive assessment of the PINN model.

#### Inputs:

 - **n_permutations** [int]: Permutations for label shuffling test.  
   *Default:* 100
 - **n_shuffles** [int]: Shuffles for feature test.  
   *Default:* 50
 - **verbose** [bool]: Whether to print detailed results.  
   *Default:* True

#### Outputs:

 - Dictionary with all test results and a final pass/fail decision afterwards

## **PulsarPINN Key Attributes**

Here, we go over the main attributes of the class:

#### Model Components

- **.model**: The trained PyTorch neural network (nn.Sequential) with Tanh activation functions
- **.learnable_params**: Dictionary of trainable physical constants as PyTorch parameters
- **.optimizer**: PyTorch Adam optimizer managing network weights and physical constants

#### Data Storage

- **.x_train, .y_train**: Training data arrays (NumPy)
- **.x_val, .y_val**: Validation data arrays (NumPy)
- **.x_test, .y_test**: Test data arrays (NumPy)
- **.x_train_torch, .y_train_torch**: Training data as PyTorch tensors
- **.x_val_torch, .y_val_torch**: Validation data as PyTorch tensors
- **.x_test_torch, .y_test_torch**: Test data as PyTorch tensors
- **.fixed_torch_train, .fixed_torch_val, .fixed_torch_test**: Dictionaries containing fixed input variables for each data split


#### Training History viewer

- **.loss_log**: Dictionary tracking training progress with the following keys:

  - **\["total"\]**: Total loss profile during training (physics + data loss)  
  - **\["physics"\]**: Physics constraint violation loss during training  
  - **\["data"\]**: Data fitting loss during training  
  - **\["val_total"\]**: Total validation loss at checkpoints  
  - **\["val_physics"\]**: Physics validation loss at checkpoints  
  - **\["val_data"\]**: Data validation loss at checkpoints  


#### Evaluate Results

- **.test_metrics**: Dictionary containing evaluation metrics including:

  - R² scores for train/validation/test splits  
  - RMSE (Root Mean Square Error) values  
  - MAE (Mean Absolute Error) values  
  - Reduced χ² statistic  
  - Total, physics, and data losses for each split  

## **Usage Notes**

- The PINN learns to fit data and satisfy physical constraints through the inputted differential equation
- Physics loss measures how well generated solutions of the model satisfies the governing equations
- Data loss measures how closely predictions match observations with ATNF data points
- The weighting of these losses (physics_weight and data_weight) can be adjusted during training
- Validation metrics should be monitored thoroughly during training to detect and prevent overfitting early
- Uncertainty quantification methods should help assess the reliability of learned constants
- Validation tests help distinguish genuine physical relationships from spurious correlations

[← Back to Technical Information Home](technicalinformation.md)
