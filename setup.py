from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='pulsarfitpy',
    version='0.1.0',
    author='Om Kasar, Saumil Sharma, Jonathan Sorenson, Kason Lai',
    author_email='contact.omkasar@gmail.com',
    description='pulsarfitpy is a Python library that uses empirical data from the Australia Telescope National Facility (ATNF) ' \
    'database & psrqpy to predict pulsar behaviors using provided Physics Informed Neural Networks (PINNs). For more data visualization, ' \
    'it also offers accurate polynomial approximations of visualized datasets from two psrqpy query parameters using scikit-learn.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPL-3.0',
    url='https://github.com/jfk-astro/pulsarfitpy',
    package_dir={"": "pulsarfitpy"},
    packages=find_packages(where="pulsarfitpy"),
    python_requires='>=3.12',
    install_requires=[
        # Dependencies
        'numpy',
        'torch',
        'psrqpy',
        'scikit-learn',
        'sympy'
    ],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Scientific :: Astronomy',
    ],
)