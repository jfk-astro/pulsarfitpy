# test.py

import numpy as np
import matplotlib.pyplot as plt
from psrqpy import QueryATNF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Create graph from Age-vs-BSurf.py
def scatter_graph(x_name, y_name, title, x_coord, y_coord, label):
    plt.figure(figsize=(8, 5))
    plt.xlabel(x_name) # NEW: Added Logarithmic Scaling x & y axes
    plt.ylabel(y_name)       # ''
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.scatter(x_coord, y_coord, s=10, alpha=0.6, label=label)
    
# Ask ATNF for name, age, and magnetic field strength of every pulsar that has the data
query = QueryATNF(params=['JNAME', 'AGE', 'BSURF'], condition='exist(AGE) && exist(BSURF)')
table = query.table

# Put the log of values in arrays
ages = np.array(np.log(table['AGE'], dtype=float))
bsurfs = np.array(np.log(table['BSURF'], dtype=float))

# age_train, age_test, bsurfs_train, bsurfs_test = train_test_split(ages, bsurfs, test_size=0.3)
# rmses = []

# q3, q1 = np.percentile(bsurfs, [75, 25])
# iqr = q3-q1
# fence = (q3 + iqr, q1 - iqr)
# np.less(bsurfs, fence[0])
# np.greater(bsurfs, fence[1])

# define line's start and end
line = np.linspace(min(ages), max(ages))

# pipe = Pipeline([ # This does not work and I don't know how to make it work
#         ('scale', StandardScaler()),
#         ('reduce_dims', PCA(n_components=4)),
#         ('clf', SVC(kernel = 'linear', C = 1))])

# param_grid = {'polynomialfeatures__degree': np.arange(20), # HOW DO YOU WORK
# 'linearregression__fit_intercept': [True, False],
# 'linearregression__normalize': [True, False]} # whyyy was normalize removed

# grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)
# grid.fit(ages, bsurfs)

# Plot everything
model = np.poly1d(np.polyfit(ages, bsurfs, 4))    
scatter_graph('Characteristic Age [log10(yrs)]', 'Magnetic Field [log10(G)]', \
              'Magnetic Field vs Pulsar Age', ages, bsurfs, 'Pulsars')

# bsurfs_test = model.fit(ages, bsurfs).predict(ages_test)
# plt.plot(ages_test.ravel(), bsurfs_test, 'r')
    
model = np.poly1d(np.polyfit(ages, bsurfs, 4))
plt.plot(line, model(line), color="red")
plt.show()

