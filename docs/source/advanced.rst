Advanced Example
================

In this example we'll create an ensemble of
Neural Density Estimators to do an LFI (likelihood-free inference)
estimate of the density of

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import getdist
    from getdist import plots, MCSamples
    import tensorflow as tf
    import pickle


Now, let's load up the pydelfi library.

.. code:: python

    import pydelfi.distributions.priors as priors
    import pydelfi.ndes.ndes as ndes
    import pydelfi as delfi
    import pydelfi.simulators.cosmic_shear.cosmic_shear as cosmic_shear
    import pydelfi.compression.score.score as score
    tf.logging.set_verbosity(tf.logging.ERROR)

Here, we will focus on the simulator for cosmic shear. First,
let's generate some power spectrum between l=10 and 1000. Here,
pz is the photo-z distribution. Here, we'll set up
:python:`simulator` as a function that calls
the power spectrum simulator.

.. code:: python

   # Set up the tomography simulations
   CosmicShearSimulator = cosmic_shear.TomographicCosmicShear(
       pz = pickle.load(open('pydelfi2/pydelfi/simulators/cosmic_shear/pz_5bin.pkl', 'rb')),
       lmin = 10, lmax = 1000, n_ell_bins = 5, 
       sigma_e = 0.3, nbar = 30, Area = 15000)

   # Simulator function: This must be of the form simulator(theta, seed, args) -> simulated data vector
   def simulator(theta, seed, simulator_args, batch=1):
       return CosmicShearSimulator.simulate(theta, seed)

   # Simulator arguments
   simulator_args = None

Generally you would like a weakly informated prior.
For our NDE, we define truncated Gaussian priors. 
The following variables declare our truncation bounds
as well as the prior mean and convariance, over the following
Cosmological parameters: Omega_M, S_8, Omega_b, h, and n_S.


.. code:: python

   # Define the priors parameters
   lower = np.array([0, 0.4, 0, 0.4, 0.7])
   upper = np.array([1, 1.2, 0.1, 1.0, 1.3])
   prior_mean = np.array([0.3, 0.8, 0.05, 0.70, 0.96])
   prior_covariance = np.eye(5)*np.array([0.1, 0.1, 0.05, 0.3, 0.3])**2

   # Prior
   prior = priors.TruncatedGaussian(prior_mean, prior_covariance, lower, upper)


We work in the idealized case where sampling distribution for
each l bin in our power spectra is Wishart. In other words,
we compress our data
as if it were a Wishart distributed. (Each l-bin is a Wishart distributed.)

.. code:: python

   # Fiducial parameters
   theta_fiducial = np.array([0.3, 0.8, 0.05, 0.70, 0.96])

   # Expected support of Wishart likelihood (fiducial inverse power spectrum)
   C = CosmicShearSimulator.power_spectrum(theta_fiducial)
   Cinv = np.array([np.linalg.inv(C[l,:,:]) for l in range(CosmicShearSimulator.n_ell_bins)])

   # Degrees of freedom (effective number of modes per band power)
   nl = CosmicShearSimulator.nl

   # Calculate derivatives of the expected power spectrum
   step = np.array(abs(theta_fiducial)*np.array([0.05, 0.05, 0.05, 0.05, 0.05]))
   dCdt = CosmicShearSimulator.compute_derivatives(theta_fiducial, step)

   # Define compression as score-MLE of a Wishart likelihood
   Compressor = score.Wishart(theta_fiducial, nl, Cinv, dCdt, prior_mean=prior_mean, prior_covariance=prior_covariance)

   # Pull out Fisher matrix inverse
   Finv = Compressor.Finv

   # Compressor function: This must have the form compressor(data, args) -> compressed summaries (pseudoMLE)
   def compressor(d, compressor_args):
       return Compressor.scoreMLE(d)
   compressor_args = None


Next

.. code:: python

   seed = 0
   data = simulator(theta_fiducial, seed, simulator_args)
   compressed_data = compressor(data, compressor_args)

Next 

.. code:: python

	# Create an ensemble of NDEs
   NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=5, n_data=5, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=0),
	   ndes.MixtureDensityNetwork(n_parameters=5, n_data=5, n_components=1, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=1),
	   ndes.MixtureDensityNetwork(n_parameters=5, n_data=5, n_components=2, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=2),
	   ndes.MixtureDensityNetwork(n_parameters=5, n_data=5, n_components=3, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=3),
	   ndes.MixtureDensityNetwork(n_parameters=5, n_data=5, n_components=4, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=4),
	   ndes.MixtureDensityNetwork(n_parameters=5, n_data=5, n_components=5, n_hidden=[30,30], activations=[tf.tanh, tf.tanh], index=5)]

   # Create the DELFI object
   DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, Finv=Finv, theta_fiducial=theta_fiducial, 
			  param_limits = [lower, upper],
			  param_names = ['\Omega_m', 'S_8', '\Omega_b', 'h', 'n_s'], 
			  results_dir = "pydelfi2/pydelfi/simulators/cosmic_shear/results/",
			  input_normalization='fisher')

Next

.. code:: python

   # Do the Fisher pre-training
   DelfiEnsemble.fisher_pretraining()

Next

.. code:: python

   # Initial samples, batch size for population samples, number of populations
   n_initial = 200
   n_batch = 200
   n_populations = 39

   # Do the SNL training
   DelfiEnsemble.sequential_training(simulator, compressor, n_initial, n_batch, n_populations, patience=10, save_intermediate_posteriors=True)


