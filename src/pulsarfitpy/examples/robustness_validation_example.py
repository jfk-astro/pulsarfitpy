"""
Example: Robustness Validation for PINN Models

Demonstrates validation methods to ensure the PINN learns genuine physical
relationships rather than spurious correlations. Includes:
- Permutation tests (randomized labels)
- Feature shuffling (shuffled inputs)
- Impossible physics tests (swapped variables)

These controls are essential for scientific publication.

Author: Om Kasar & Saumil Sharma under jfk-astro
"""

import sympy as sp
import numpy as np
from psrqpy import QueryATNF
from modules.pinn import PulsarPINN

print("=" * 80)
print("ROBUSTNESS VALIDATION FOR PINN MODELS")
print("=" * 80)
print("\nThis example demonstrates standard controls to validate that the PINN")
print("learns genuine physical relationships, not spurious correlations.")

# =============================================================================
# STEP 1: Setup and Train Real Model
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1: Training Real Physics Model")
print("=" * 80)

# Define physics
logP, logPDOT, logB = sp.symbols('logP logPDOT logB')
logR = sp.Symbol('logR')
differential_equation = sp.Eq(logB, logR + 0.5 * logP + 0.5 * logPDOT)

print(f"\nPhysics equation: {differential_equation}")

# Load data
print("\nLoading ATNF data...")
query = QueryATNF(
    params=['P0', 'P1', 'BSURF'],
    condition='exist(P0) && exist(P1) && exist(BSURF)'
)
table = query.table

logP_data = np.log10(table['P0'].data)
logPDOT_data = np.log10(table['P1'].data)
logB_data = np.log10(table['BSURF'].data)

print(f"Retrieved {len(logP_data)} pulsars")

# Initialize PINN
pinn = PulsarPINN(
    differential_eq=differential_equation,
    x_sym=logP,
    y_sym=logB,
    learn_constants={logR: 18.0},
    log_scale=True,
    fixed_inputs={
        logP: logP_data,
        logPDOT: logPDOT_data,
        logB: logB_data
    },
    hidden_layers=[16, 32, 16],
    train_split=0.70,
    val_split=0.15,
    test_split=0.15,
    random_seed=42
)

print("\nTraining PINN (this may take a minute)...")
pinn.train(epochs=3000, training_reports=1000, physics_weight=1.0, data_weight=1.0)

print("\nEvaluating real model...")
real_metrics = pinn.evaluate_test_set(verbose=True)

learned_constants = pinn.store_learned_constants()
print(f"\nLearned constant: logR = {learned_constants['logR']:.6f}")

# =============================================================================
# STEP 2: Permutation Test (Randomized Labels)
# =============================================================================

print("\n\n" + "=" * 80)
print("STEP 2: Permutation Test - Randomizing Target Labels")
print("=" * 80)
print("""
PURPOSE: Test if the model learns genuine relationships or just memorizes data.

METHOD: Randomly shuffle the target labels (y-values) to break real relationships,
then retrain. If the real model significantly outperforms permuted models,
it's learning genuine physics.

INTERPRETATION:
- p-value < 0.05: Real model is significantly better → genuine learning
- p-value >= 0.05: Real model not better than random → spurious correlations
""")

permutation_results = pinn.validate_with_permutation_test(
    n_permutations=100,
    epochs=1000,
    significance_level=0.05,
    verbose=True
)

# =============================================================================
# STEP 3: Feature Shuffling Test
# =============================================================================

print("\n\n" + "=" * 80)
print("STEP 3: Feature Shuffling Test - Validating Input Importance")
print("=" * 80)
print("""
PURPOSE: Test if input features contain meaningful information.

METHOD: Randomly shuffle the input features (x-values) to break the x-y
relationship, then retrain. Real features should perform much better.

INTERPRETATION:
- R² difference > 0.1: Input features are important
- R² difference ≈ 0: Input features don't help (problem!)
""")

feature_results = pinn.validate_with_feature_shuffling(
    n_shuffles=50,
    epochs=1000,
    verbose=True
)

# =============================================================================
# STEP 4: Impossible Physics Test
# =============================================================================

print("\n\n" + "=" * 80)
print("STEP 4: Impossible Physics Test - Testing Physical Constraints")
print("=" * 80)
print("""
PURPOSE: Test if the model respects physical constraints.

METHOD: Swap the roles of input and output variables to create a physically
meaningless relationship. A robust model should perform poorly on this.

INTERPRETATION:
- Real >> Impossible: Model respects physics
- Real ≈ Impossible: Model ignores physics (problem!)
""")

physics_results = pinn.validate_with_impossible_physics(
    epochs=2000,
    verbose=True
)

# =============================================================================
# STEP 5: Comprehensive Assessment
# =============================================================================

print("\n\n" + "=" * 80)
print("COMPREHENSIVE ROBUSTNESS ASSESSMENT")
print("=" * 80)

print("\nTest Summary:")
print("-" * 80)

# Permutation test
perm_status = "✓ PASS" if permutation_results['is_significant'] else "✗ FAIL"
print(f"\n1. Permutation Test: {perm_status}")
print(f"   Real model R²:     {permutation_results['real_r2']:.6f}")
print(f"   Random mean R²:    {permutation_results['permuted_r2_mean']:.6f}")
print(f"   p-value:           {permutation_results['p_value']:.4f}")
if permutation_results['is_significant']:
    print("   → Model learns genuine relationships")
else:
    print("   → WARNING: Model may capture spurious correlations")

# Feature shuffling test
feat_status = "✓ PASS" if feature_results['r2_difference'] > 0.1 else "✗ FAIL"
print(f"\n2. Feature Shuffling Test: {feat_status}")
print(f"   Real model R²:     {feature_results['real_r2']:.6f}")
print(f"   Shuffled mean R²:  {feature_results['shuffled_r2_mean']:.6f}")
print(f"   Improvement:       {feature_results['r2_difference']:.6f}")
if feature_results['r2_difference'] > 0.1:
    print("   → Input features contain genuine information")
else:
    print("   → WARNING: Features may not be informative")

# Impossible physics test
phys_status = "✓ PASS" if physics_results['real_much_better'] else "✗ FAIL"
print(f"\n3. Impossible Physics Test: {phys_status}")
print(f"   Real physics R²:       {physics_results['real_r2']:.6f}")
print(f"   Impossible physics R²: {physics_results['impossible_r2']:.6f}")
print(f"   Difference:            {physics_results['r2_difference']:.6f}")
if physics_results['real_much_better']:
    print("   → Model respects physical constraints")
else:
    print("   → WARNING: Model may ignore physics")

# Overall verdict
all_passed = (
    permutation_results['is_significant'] and
    feature_results['r2_difference'] > 0.1 and
    physics_results['real_much_better']
)

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

if all_passed:
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("\nConclusion:")
    print("  The model demonstrates robust learning of genuine physical")
    print("  relationships. It is suitable for scientific inference and")
    print("  publication.")
    print("\nRecommendation:")
    print("  ✓ Safe to use fitted constants in scientific analysis")
    print("  ✓ Model predictions are reliable")
    print("  ✓ Include these validation results in publication")
else:
    print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("\nConclusion:")
    print("  The model shows signs of learning spurious correlations")
    print("  or failing to respect physical constraints.")
    print("\nRecommendation:")
    print("  ✗ Use caution with fitted constants")
    print("  ✗ Consider model architecture changes")
    print("  ✗ Check data quality and preprocessing")
    print("  ✗ Try different physics weights or training strategies")

print("\n" + "=" * 80)
print("REPORTING FOR PUBLICATION")
print("=" * 80)
print("""
Include the following in your methods section:

"To validate that the PINN learns genuine physical relationships rather than
spurious correlations, we performed three robustness tests:

1. **Permutation Test**: We randomly shuffled target labels and retrained the
   model 100 times. The real model significantly outperformed permuted models
   (p = {:.4f}), indicating genuine learning.

2. **Feature Shuffling**: We shuffled input features 50 times to test feature
   importance. The real model achieved R² = {:.4f}, compared to shuffled
   R² = {:.4f} (Δ = {:.4f}), confirming meaningful feature relationships.

3. **Impossible Physics**: We swapped input/output variables to create physically
   meaningless relationships. The real model (R² = {:.4f}) substantially
   outperformed the impossible model (R² = {:.4f}), demonstrating physical
   constraint adherence.

These validation tests confirm that the model learns genuine physical
relationships suitable for scientific inference."
""".format(
    permutation_results['p_value'],
    feature_results['real_r2'],
    feature_results['shuffled_r2_mean'],
    feature_results['r2_difference'],
    physics_results['real_r2'],
    physics_results['impossible_r2']
))

print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
