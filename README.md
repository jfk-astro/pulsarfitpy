# pulsarfitpy
pulsarfitpy is a Python library that uses empirical data from the [Australia Telescope National Facility (ATNF)](https://www.atnf.csiro.au/) database & psrqpy to predict pulsar behaviors using provided Physics Informed Neural Networks (PINNs). For more data visualization, it also offers accurate polynomial approximations of visualized datasets from two psrqpy query parameters using scikit-learn.

## Prerequisites:

- Python (>=3.12)
- NumPy
- Matplotlib
- psrqpy
- scikit-learn
- PyTorch
- SymPy

Install all dependencies by running the following command in the terminal:
```bash
pip install numpy matplotlib psrqpy scikit-learn torch sympy
```

## Installation
To install the library, simply run the following in the terminal:
``` bash
pip install pulsarfitpy
```
For library usage, import the pulsarfitpy library with:
```python
import pulsarfitpy as pf
```
Refer to the documentation for further usage of the library.

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes.
4. Push your branch: `git push origin feature-name`.
5. Create a pull request.

## Credits
pulsarfitpy was written by Om Kasar, Saumil Sharma, Jonathan Sorenson, and Kason Lai.

## Contact
For any questions about the repository, email contact.omkasar@gmail.com.