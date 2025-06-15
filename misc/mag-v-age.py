import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from psrqpy import QueryATNF

# Generate data points from psrqpy query
query = QueryATNF(params=['JNAME', 
                          'AGE', 
                          'BSURF'], 
                          condition='exist(AGE) && exist(BSURF)')
table = query.table

ages = np.array(np.log(table['AGE'], dtype=float)).reshape(-1, 1) # Take log versions of data
bsurfs = np.array(np.log(table['BSURF'], dtype=float))            # ''

# Create pipeline for regression
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('reg', LinearRegression())
])

# Search for polynomial degree
param_grid = {
    'poly__degree': list(range(1, 5))  # Test degrees from 1 to 5
}

# Integrates pipeline, tests degrees, and a grid fit for data
grid = GridSearchCV(pipeline, param_grid)
grid.fit(ages, bsurfs)

# Predict on a fine grid for plotting
polynomial_model_X = np.linspace(ages.min(), ages.max(), 100).reshape(-1, 1) # Implement smoother values
polynomial_model_Y = grid.best_estimator_.predict(polynomial_model_X)        # Generate approximate model

# Print Degree
best_degree = grid.best_params_['poly__degree']     # Output best degree
print(f"\nChosen Polynomial Degree: {best_degree}") # Print best degree

coefficients = grid.best_estimator_.named_steps['reg'].coef_      # Gather coefficients into an array
first_number = grid.best_estimator_.named_steps['reg'].intercept_ # Store first number of the polynomial

# Build and print simplified polynomial function
print("\nApproximated Polynomial Function:")
print(f"f(x) = {first_number:.10f}", end='') # Prints each coefficient to 10 decimal places

for i, coef in enumerate(coefficients[1:], start=1): # Extracts degree values (i) and coefficient number (coef) based on coefficients list indexing & enumerate
    print(f" + {coef:.10f} * x**{i}", end='')        # Prints function in that order

# Plot the approximated data points & polynomial regression function
plt.figure(figsize=(10, 6))
plt.scatter(ages, bsurfs, s=10, alpha=0.6, label='Pulsars')
plt.plot(polynomial_model_X, polynomial_model_Y, color='red', label='Polynomial Approximation') # Plot the model
plt.legend()
plt.title('Observed Magnetic Field vs. Characteristic Age')
plt.xlabel('Characteristic Age [log(yrs)]')
plt.ylabel('Magnetic Field [log(G)]')
plt.grid(True)
plt.show()