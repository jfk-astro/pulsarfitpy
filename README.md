<h1 align="center">pulsarfitpy</h1>

<p align="center">An Efficient Physics Informed Framework for Pulsar Analysis</p>

`pulsarfitpy` is a Python library that uses empirical data from the [Australia Telescope National Facility (ATNF)](https://www.atnf.csiro.au/) database and `psrqpy` to predict pulsar behaviors using provided Physics Informed Neural Networks (PINNs). For more data visualization, it also offers accurate polynomial approximations of visualized datasets from two `psrqpy` query parameters using `scikit-learn`.

The research paper for our work can be found [here](https://drive.google.com/file/d/1WiuRLKYMaDqFy5ydgsTYTNRFaLY9k_IX/view?usp=sharing).

## Features
* Graphical Representations of ATNF Pulsar Data
* Approximated Polynomial Models of Data
* AI-Based Predictions Following PINNs

## Prerequisites
In order to use `pulsarfitpy`, you will need to download and install the following resources.

| Required Resource | Where to Find |
| --- | --- |
| Python (Version 3.12 or Higher) | <a href="www.https://www.python.org/downloads/">The Python Website</a> |
| NumPy | Install via pip |
| Matplotlib | Install via pip |
| psrqpy | Install via pip |
| scikit-learn | Install via pip |
| PyTorch | Install via pip |
| SymPy | Install via pip |

To install all necessary libraries, run the following command in the terminal:
```bash
pip install numpy matplotlib psrqpy scikit-learn torch sympy
```

## Installation and Implementation
**To install the library, simply run the following command in the terminal after successfully installing Python:**
``` bash
pip install pulsarfitpy
```

**To use the library in a Python program, import the pulsarfitpy library using the following code:**
```python
import pulsarfitpy as pf
```  
Please refer to the documentation for further usage of the library.

## Using an API
[PulsarsAPI](https://github.com/jfk-astro/PulsarsAPI), another project from the JFK Astronomy Club, has a feature allowing you to access pulsarfitpy from an API. PulsarsAPI offers the ability to retrieve data about pulsars through endpoints such as `/api/pulsars/{name}` and `/api/pulsars/{id}`.

Furthermore, PulsarsAPI allows you to use pulsarfitpy through the `/api/pulsars/fit-function` endpoint. When calling the embedded pulsarfitpy within PulsarsAPI, you will have to pass the following arguments:
| Type | Resource |
| --- | --- |
| String | x parameter |
| String | y parameter |
| Integer | test degree |
| Boolean | log x |
| Boolean | log y|

## Contributing
**Please follow the following steps in order when contributing to the library.**  
1. Fork the repository.
2. Create a new branch with the command: `git checkout -b feature-name`.
3. Make your changes.
4. Push your branch with the command: `git push origin feature-name`.
5. Create a pull request.

## Credits
pulsarfitpy was written by Om Kasar, Saumil Sharma, Jonathan Sorenson, and Kason Lai.

## Contact
For any questions about the repository, please email contact.omkasar@gmail.com.
