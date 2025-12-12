---
layout: default
title: pulsarfitpy
---

# **Installation and Project Setup Guide**

## **Prerequisites**

* Python 3.12 or higher
* Go 1.21 or higher (for CLI)

## **Dependencies**

* psrqpy
* numpy
* torch
* sympy
* typing
* logging
* matplotlib
* enum
* warnings
* sklearn
* deepxde
* abc
* dataclasses
* time
* pathlib
* pandas

You may install these by running either one of the following commands:  
```bash
pip install -r requirements.txt
```
```bash
pip install psrqpy numpy torch sympy typing logging matplotlib enum warnings sklearn deepxde abc dataclasses time pathlib pandas
```

## **Python Installation**

### Via pip

```bash
pip install pulsarfitpy
```

### From Source

```bash
git clone https://github.com/jfk-astro/pulsarfitpy.git
cd pulsarfitpy
pip install -e .
```

## **Go CLI Installation**

## **Dependencies**

Additional dependencies required for the CLI are listed below.  
* sys
* json
* argpause

You may install these by running either one of the following commands:  
```bash
pip install -r cli_requirements.txt
```
```bash
pip install sys json argpause
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/jfk-astro/pulsarfitpy.git
cd pulsarfitpy

# Build the CLI
make build
```

The binary will be available at `bin/pulsar-cli.exe` (Windows) or `bin/pulsar-cli` (Unix).

## **Starting Your First Project Using pulsarfitpy**

### Python Library

Import pulsarfitpy and psrqpy into your project. Query the particular params you want and use the following pulsarfitpy methods to fit and analyze the data.  
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

The following commands are the CLI equivalent to the above Python version.
```bash
# Run analysis
./bin/pulsar-cli analyze --param1 P0 --param2 P1 --degree 5

# Fit polynomial model
./bin/pulsar-cli fit --param1 P0 --param2 P1

# Query database
./bin/pulsar-cli query --params P0,P1,DM
```

### Python GUI

Run the following command to use the GUI.
```bash
python src/pulsarfitpy/src/pulsarfitpy/gui.py
```

## **Next Steps**

- Read the [Technical Information](quickstart.md)
- Explore [Examples](examples.md)
- Read the [API Reference](api.md)
- Check out the Jupyter notebooks in `src/pulsarfitpy/docs/`

[← Back to Home](index.md)