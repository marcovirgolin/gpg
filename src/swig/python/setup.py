#!/usr/bin/env python

"""
setup.py file for SWIG interface
"""

from setuptools import setup



setup (name = 'pyminigpg',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       packages=['pyminigpg'],
	package_data={'pyminigpg': ['_pyface.so']},
       include_package_data=True,
       install_requires=['scikit-learn', 'sympy'],
       )