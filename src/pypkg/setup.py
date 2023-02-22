#!/usr/bin/env python

"""
setup.py file for SWIG interface
"""

from setuptools import setup


setup (name = 'pygpg',
       version = '0.2',
       author      = "Marco Virgolin",
       packages=['pygpg'],
	package_data={'pygpg': ['_pb_gpg.so']},
       include_package_data=True,
       install_requires=['scikit-learn', 'sympy'],
       )