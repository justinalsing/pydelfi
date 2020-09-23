#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

setup(name='pydelfi',
      version='v0.2',
      description='LFI in TensorFlow',
      author='Justin Alsing, Tom Charnock, Stephen Feeney',
      url='https://github.com/justinalsing/pydelfi',
      packages=find_packages(),
      install_requires=[
	  "jupyter",
          "getdist>=1.1.0",
          "emcee>=3.0.2",
          "tqdm>=4.41.1",
          "tensorflow-probability==0.11.0",
          "tensorflow==2.3.0"
      ])
