from psrqpy import QueryATNF
from sympy import symbols
from pulsarfitpy import PulsarApproximation

# Query pulsar parameters from ATNF catalog
query = QueryATNF(params=["AGE", "BSURF"])

approx = PulsarApproximation(query, x_param='AGE', y_param='BSURF', test_degree=10)
approx.fit_polynomial()
approx.print_polynomial()
approx.plot_combined_analysis()