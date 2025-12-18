<h1 align="center">pulsarfitpy</h1>

<div align="center">
  
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
  
pulsarfitpy is a tool intended to streamline analysis of  pulsars, a rapidly rotating neutron star. pulsarfitpy offers an intuitive PyTorch framework using [Physics Informed Neural Networks (PINNs)](https://arxiv.org/abs/1711.10561) to analyze differential equations in comparison to observed ATNF data. The application of the PINN framework provides an additional resource in pulsar analysis to test derived theoretical physics expressions and determine their validity.

</div>

## **Features**
- 1D PINN Support
- 2D PINN Support
- Python Library, Command Line Interface, and Graphical User Interface Options
- Real Time Error Metrics
- Customizable Neural Networks

## **Prerequisites**

* Python 3.12 or higher
* Go 1.21 or higher (for CLI)

## **Dependencies**

To use all features of pulsarfitpy, install the following Python modules.  
* psrqpy
* numpy
* torch
* sympy
* typing
* logging
* matplotlib
* argparse
* scikit-learn
* deepxde
* dataclasses
* pathlib
* pandas

You may install these by running either one of the following commands:  
```bash
pip install -r requirements.txt
```
```bash
pip install psrqpy numpy torch sympy typing logging matplotlib argparse scikit-learn deepxde dataclasses pathlib pandas
```  
For the former, please ensure that you are in the correct directory containing the requirements text file.

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

### Build from Source

```bash
# Clone the repository
git clone https://github.com/jfk-astro/pulsarfitpy.git
cd pulsarfitpy

# Build the CLI
make build
```

The binary will be available at `bin/pulsar-cli.exe` (Windows) or `bin/pulsar-cli` (Unix).

## **Guides**
You may find a compilation of more detailed information about pulsarfitpy at [this website](https://jfk-astro.github.io/pulsarfitpy/).

## **License**
pulsarfitpy is under the GNU GPL v3.0 license, a free, copyleft license published by the Free Software Foundation.

## **Credits**
pulsarfitpy is made by jfk-astro consisting of Om Kasar, Saumil Sharma, Kason Lai, and Jonathan Sorenson.