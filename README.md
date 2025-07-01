# pulsarfitpy
> pulsarfitpy is a Python library that uses empirical data to approximate visualized polynomial functions for Neutron star parameters through the []Australia Telescope National Facility (ATNF)](https://www.atnf.csiro.au/) pulsar database and psrqpy.

## Installation
> Requirements: []
1. Clone the repository:
``` bash
git clone https://github.com/jfk-astro/pulsarfitpy.git
```

2. 

## Usage
> strings are parameters for x and y, options found in [psrqpy parameters](https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html?type=expert#par_list) 

> integer is maximum polynomial degree searched for
pulsarfitpy.poly_reg(str, str, int)
```bash
pulsarfitpy.poly_reg('AGE', 'BSURF', 5)
```

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes.
4. Push your branch: `git push origin feature-name`.
5. Create a pull request.

## Credits
pulsarfitpy was written by Jonathan Sorenson, Om Kasar, Saumil Sharma, and Kason Lai.
