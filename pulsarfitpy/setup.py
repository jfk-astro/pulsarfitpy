from setuptools import setup, Extension
import sys

pulsars_module = Extension(
    'pulsars',
    sources=['src/pulsars_wrapper.c', 'src/pulsars.c'],
    include_dirs=['src'],
)

setup(
    name='pulsars',
    version='0.1',
    description='C library for approximations',
    ext_modules=[pulsars_module],
    py_modules=['pulsars'],
)
