"""
Quick reference for interpreting quantitative metrics in pulsarfitpy.
"""

METRICS_REFERENCE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUANTITATIVE METRICS QUICK REFERENCE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ R² (COEFFICIENT OF DETERMINATION)                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ What it measures:  Proportion of variance explained by the model            │
│ Range:             -∞ to 1.0 (typically 0 to 1)                             │
│ Ideal value:       Close to 1.0                                             │
│ Interpretation:    • R² > 0.95: Excellent fit                               │
│                    • R² > 0.90: Very good fit                               │
│                    • R² > 0.75: Good fit                                    │
│                    • R² > 0.50: Moderate fit                                │
│                    • R² < 0.50: Poor fit                                    │
│ Limitations:       Can be artificially high with overfitting                │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ RMSE (ROOT MEAN SQUARED ERROR)                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ What it measures:  Average prediction error (penalizes large errors)        │
│ Range:             0 to ∞                                                    │
│ Ideal value:       Close to 0                                               │
│ Interpretation:    • Lower values = better fit                              │
│                    • Compare across models (lower RMSE wins)                │
│                    • Units match target variable                            │
│ Advantages:        • Penalizes large errors heavily                         │
│                    • Differentiable (useful for optimization)               │
│ Limitations:       • Sensitive to outliers                                  │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ MAE (MEAN ABSOLUTE ERROR)                                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ What it measures:  Average absolute prediction error                        │
│ Range:             0 to ∞                                                    │
│ Ideal value:       Close to 0                                               │
│ Interpretation:    • Lower values = better fit                              │
│                    • Typical error magnitude                                │
│                    • Units match target variable                            │
│ Advantages:        • More robust to outliers than RMSE                      │
│                    • Easier to interpret (average error)                    │
│ Comparison:        • MAE ≈ RMSE: errors are uniform                         │
│                    • RMSE >> MAE: large outliers present                    │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ REDUCED χ² (CHI-SQUARED)                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ What it measures:  Goodness of fit normalized by degrees of freedom         │
│ Range:             0 to ∞                                                    │
│ Ideal value:       Close to 1.0                                             │
│ Interpretation:    • χ²ᵣ < 0.5:  Possible overfitting                       │
│                    • χ²ᵣ ≈ 1.0:  Good fit                                   │
│                    • χ²ᵣ > 2.0:  Underfitting or systematic errors          │
│ Formula:           χ²ᵣ = Σ(yᵢ - ŷᵢ)² / (n - p)                             │
│                    where n = samples, p = parameters                        │
│ Advantages:        • Accounts for model complexity                          │
│                    • Statistical interpretation                             │
└──────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                           OVERFITTING DETECTION                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Compare Training vs. Test Metrics:

✓ GOOD GENERALIZATION (Model learned patterns):
  • |R²_train - R²_test| < 0.05: Excellent
  • |R²_train - R²_test| < 0.10: Good
  • RMSE_train ≈ RMSE_test
  • MAE_train ≈ MAE_test

⚠ POSSIBLE OVERFITTING (Model memorized data):
  • R²_train - R²_test > 0.10: Warning
  • R²_train - R²_test > 0.20: Severe
  • RMSE_train << RMSE_test
  • χ²ᵣ < 0.5

Solutions for overfitting:
  1. Collect more training data
  2. Reduce model complexity (fewer parameters/layers)
  3. Add regularization (L1, L2, dropout)
  4. Early stopping based on validation loss
  5. Increase validation frequency

╔══════════════════════════════════════════════════════════════════════════════╗
║                        USAGE IN PULSARFITPY                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

For PINNs:
    pinn.train(epochs=5000)
    metrics = pinn.evaluate_test_set(verbose=True)
    
    # Access metrics
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4e}")
    print(f"Test MAE: {metrics['test_mae']:.4e}")
    print(f"Test χ²ᵣ: {metrics['test_chi2_reduced']:.4f}")

For Polynomial Approximations:
    approx.fit_polynomial()
    metrics = approx.compute_metrics(verbose=True)
    
    # Access metrics
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4e}")
    print(f"MAE: {metrics['mae']:.4e}")
    print(f"χ²ᵣ: {metrics['chi2_reduced']:.4f}")

╔══════════════════════════════════════════════════════════════════════════════╗
║                       REPORTING IN PUBLICATIONS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Always report:
  1. All four metrics (R², RMSE, MAE, χ²ᵣ)
  2. Sample sizes (n_train, n_val, n_test)
  3. Data split ratios
  4. Train vs. test comparison
  5. Units for RMSE and MAE

Example statement:
  "The PINN model achieved a test set R² of 0.952 (RMSE = 0.123, MAE = 0.098,
   χ²ᵣ = 1.12) on 267 held-out pulsars, demonstrating good generalization with
   minimal overfitting (R²_train - R²_test = 0.004)."

╔══════════════════════════════════════════════════════════════════════════════╗
║                             KEY TAKEAWAYS                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

✓ Never rely on visual comparison alone
✓ Report multiple metrics for complete assessment
✓ Compare train/validation/test to detect overfitting
✓ Lower RMSE and MAE = better predictions
✓ R² closer to 1.0 = more variance explained
✓ χ²ᵣ close to 1.0 = good statistical fit
✓ Context matters: consider physical units and domain knowledge

"""

def print_reference():
    """Print the metrics reference guide."""
    print(METRICS_REFERENCE)

if __name__ == "__main__":
    print_reference()
