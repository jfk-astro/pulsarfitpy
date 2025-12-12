---
layout: default
title: Examples
---

# Examples

TODO!

<!-- Below, we sample a scientifically accepted formula cited by [Belvedere](https://iopscience.iop.org/article/10.1088/0004-637X/799/1/23) and many other studies regarding the surface magnetic field $B$, period $P$, and period derivative $\dot P$, as shown here:

| $\large B \approx 3.2 \cdot 10^{19} (P \dot P)^{1/2}$ |
|:-:|

In this markdown file, experimentally determine an approximation for $3.2 \cdot 10^{19}$ through the variable $N$ using pulsarfitpy & ATNF data, and how to use its constant finder feature in a real theoretical system for a one-dimensional pulsar equation. -->

## Basic Usage

### Python Library

```python
import pulsarfitpy as pf
from psrqpy import QueryATNF

# Query ATNF database
query = QueryATNF(params=['P0', 'P1'])

# Create polynomial approximation
approx = pf.PulsarApproximation(
    query=query,
    x_param='P0',
    y_param='P1',
    test_degree=5,
    log_x=True,
    log_y=True
)

# Fit and analyze
approx.fit_polynomial()
approx.compute_metrics()
approx.plot_combined_analysis()
```

### Go CLI

```bash
# Run analysis
./bin/pulsar-cli analyze --param1 P0 --param2 P1 --degree 5

# Fit polynomial model
./bin/pulsar-cli fit --param1 P0 --param2 P1

# Query database
./bin/pulsar-cli query --params P0,P1,DM
```

### Python GUI

```bash
python src/pulsarfitpy/src/pulsarfitpy/gui.py
```

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `P0` | Pulsar period (s) |
| `P1` | Period derivative (s/s) |
| `DM` | Dispersion measure (pc/cm³) |
| `BSURF` | Surface magnetic field (G) |
| `EDOT` | Spin-down energy loss rate (erg/s) |

## **Next Steps**

- Explore [Examples](examples.md)
- Read the [API Reference](api.md)
- Check out the Jupyter notebooks in `src/pulsarfitpy/docs/`

[← Back to Home](index.md)