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
          "tensorflow >=v1.1.0, <v2.0",
          "getdist",
          "emcee>=v3.0.2",
          "mpi4py",
          "scipy",
          "tqdm"
      ],
      extras_require={
        "tf": ["tensorflow>=v1.1.0, <v2.0"],
        "tf_gpu": ["tensorflow-gpu>=v1.1.0, <v2.0"]}
     )

