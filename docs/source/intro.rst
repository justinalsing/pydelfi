Introduction to PyDelfi
=======================

pydelfi is ...
--------------

PyDelfi is Density Estimation with 
Likelihood-Free Inference in Python. It makes use of
neural density estimators and adaptive acquisition of simulations.


Quick start
-----------

In this example we'll create an ensemble of
Neural Density Estimators to do an LFI (likelihood-free inference)
estimate of the density of

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import getdist
    from getdist import plots, MCSamples

    import pydelfi.distributions.priors as priors
    import pydelfi.ndes.ndes as ndes
    import pydelfi as delfi
    import tensorflow as tf
    import pydelfi.simulators.cosmic_shear.cosmic_shear as cosmic_shear
    import pickle
    import pydelfi.compression.score.score as score
    tf.logging.set_verbosity(tf.logging.ERROR)
