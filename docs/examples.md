---
layout: default
title: Examples
---

# Examples

TODO!

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