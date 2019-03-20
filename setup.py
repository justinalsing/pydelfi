#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

setup(name='pydelfi',
      version='v0.1',
      description='LFI in TensorFlow',
      author='Justin Alsing',
      url='https://github.com/justinalsing/pydelfi',
      packages=find_packages(),
      install_requires=[
          "tensorflow>=v1.1.0",
          "getdist",
          "emcee",
          "mpi4py",
          "scipy"
      ])

