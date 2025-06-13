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
    'poly__degree': list(range(1, 20))  # Test degrees from 1 to 20
}

# Integrates pipeline, tests degrees, and a grid fit for data
grid = GridSearchCV(pipeline, param_grid)
grid.fit(ages, bsurfs)

# Predict on a fine grid for plotting
polynomial_model_X = np.linspace(ages.min(), ages.max(), 100).reshape(-1, 1) # Implement smoother values
polynomial_model_Y = grid.best_estimator_.predict(polynomial_model_X)        # Generate approximate model

# Print Degree
degree = grid.best_params_['poly__degree']   # Output best degree
print(f"Chosen Polynomial Degree: {degree}") # Print degree

# Plot the approximated data points & polynomial regression function
plt.figure(figsize=(10, 6))
plt.scatter(ages, bsurfs, s=10, alpha=0.6, label='Pulsars')
plt.plot(polynomial_model_X, polynomial_model_Y, color='red', label='Best Polynomial Fit') # Plot the model
plt.legend()
plt.title('Observed Magnetic Field vs. Characteristic Age')
plt.xlabel('log(Characteristic Age) [log(yrs)]')
plt.ylabel('log(Magnetic Field) [log(B)]')
plt.grid(True)
plt.show()