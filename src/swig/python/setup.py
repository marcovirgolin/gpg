#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


pygpg_module = Extension('_pygpg',
                        sources=['pyface_wrap.cxx', 'pyface.cxx'],
                        )

setup (name = 'example',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [pygpg_module],
       py_modules = ["pygpg"],
       )