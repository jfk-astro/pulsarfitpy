from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='pulsarfitpy',
    version='0.2.4',
    author='Om Kasar, Saumil Sharma, Jonathan Sorenson, Kason Lai',
    author_email='contact.omkasar@gmail.com, sausha310@gmail.com, kasonlai08@gmail.com, jonathan.t.sorenson@gmail.com',
    description='A Python library to assist with data analysis and theoretical physics frameworks of the Australian National Telescope Facility (ATNF) Pulsar Catalogue.',
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
        'sympy',
        'matplotlib'
    ],
)