import tensorflow as tf
import getdist
from getdist import plots, MCSamples
import emcee
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm.auto import tqdm
import scipy.optimize as optimization
from scipy.stats import multivariate_normal
import pickle
import tensorflow_probability as tfp
tfd = tfp.distributions

# pydelfi imports
from pydelfi import ndes

class Delfi():

    def __init__(self, data, prior, nde, Finv=None, theta_fiducial=None,
                 param_limits=None, param_names=None, nwalkers=100,
                 posterior_chain_length=100, proposal_chain_length=100,
                 rank=0, n_procs=1, comm=None, red_op=None, show_plot=True,
                 results_dir="", filename=None, progress_bar=True, input_normalization=None,
                 save=True, restore=False, **kwargs):

        # Data
        self.data = data
        self.D = len(data)

        # Prior
        self.prior = prior

        # Number of parameters
        self.npar = prior.event_shape[0]

        self.NDEs = ndes.NDE(nde, self.prior, **kwargs)

        #TC - some work...
        # Parameter limits
        if ((not hasattr(self.prior, "low"))
                and (not hasattr(self.prior, "high"))):
            if hasattr(self.prior, "distributions"):
                if hasattr(self.prior.distributions[0], "low"):
                    low = np.zeros((self.npar,), dtype=np.float32)
                    high = np.zeros((self.npar,), dtype=np.float32)
                    for i in range(self.npar):
                        low[i] = self.prior.distributions[i].low
                        high[i] = self.prior.distributions[i].high
                    self.prior.low = tf.convert_to_tensor(low, dtype=tf.float32)
                    self.prior.high = tf.convert_to_tensor(high, dtype=tf.float32)
                elif param_limits is None:
                    raise ValueError("Please provide a prior whose distributions have `low` and `high` limits.")
            elif param_limits is not None:
                # Set to provided prior limits if provided
                self.prior.low = tf.convert_to_tensor(param_limits[0].astype(np.float32), dtype=tf.float32)
                self.prior.high = tf.convert_to_tensor(param_limits[1].astype(np.float32), dtype=tf.float32)
            else:
                # Else set to max and min float32
                self.prior.low = tf.convert_to_tensor(np.ones(self.npar)*np.finfo(np.float32).min, dtype=tf.float32)
                self.prior.high = tf.convert_to_tensor(np.ones(self.npar)*np.finfo(np.float32).max, dtype=tf.float32)

        # Fisher matrix and fiducial parameters
        if Finv is not None:
            self.Finv = Finv.astype(np.float32)
            self.fisher_errors = np.sqrt(np.diag(self.Finv)).astype(np.float32)
            self.theta_fiducial = theta_fiducial.astype(np.float32)
            scale = tf.linalg.cholesky(self.Finv)
            self.asymptotic_posterior = ndes.TruncatedMultivariateNormalTriL(
                loc=self.theta_fiducial, scale_tril=scale,
                low=self.prior.low, high=self.prior.high,
                validate_args=False, allow_nan_stats=True,
                name='AsymptoticPosterior')
        else:
            self.Finv = None
            self.fisher_errors = None
            self.theta_fiducial = None
            self.asymptotic_posterior = None

        # Re-scaling for inputs to NDE
        self.input_normalization = input_normalization
        if input_normalization is None:
            self.data_shift = np.zeros(self.D).astype(np.float32)
            self.data_scale = np.ones(self.D).astype(np.float32)
            self.theta_shift = np.zeros(self.npar).astype(np.float32)
            self.theta_scale = np.ones(self.npar).astype(np.float32)
        elif input_normalization is "fisher":
            self.data_shift = self.theta_fiducial
            self.data_scale = self.fisher_errors
            self.theta_shift = self.theta_fiducial
            self.theta_scale = self.fisher_errors
        elif input_normalization is "auto":
            self.input_normalization_set = False
        else:
            self.data_shift, self.data_scale, self.theta_shift, self.theta_scale = input_normalization

        # Training data [initialize empty]
        self.theta_realizations = np.array([], dtype=np.float32).reshape(0,self.npar)
        self.data_realizations = np.array([], dtype=np.float32).reshape(0,self.D)
        self.theta_train = []
        self.data_train = []
        self.n_sims = 0

        # MCMC chain parameters for EMCEE
        self.nwalkers = nwalkers
        self.posterior_chain_length = posterior_chain_length
        self.proposal_chain_length = proposal_chain_length

        # Initialize MCMC chains for posterior and proposal
        if self.asymptotic_posterior is not None:
            self.posterior_samples = self.asymptotic_posterior.sample(self.nwalkers*self.posterior_chain_length).numpy()
            self.proposal_samples = self.asymptotic_posterior.sample(self.nwalkers*self.proposal_chain_length).numpy()
        else:
            self.posterior_samples = self.prior.sample(self.nwalkers*self.posterior_chain_length).numpy()
            self.proposal_samples = self.prior.sample(self.nwalkers*self.proposal_chain_length).numpy()
        self.posterior_weights = (np.ones(len(self.posterior_samples))*1.0/len(self.posterior_samples)).astype(np.float32)
        self.proposal_weights = (np.ones(len(self.proposal_samples))*1.0/len(self.proposal_samples)).astype(np.float32)
        self.log_posterior_values = (np.ones(len(self.posterior_samples))*1.0/len(self.posterior_samples)).astype(np.float32)
        self.log_proposal_values = (np.ones(len(self.proposal_samples))*1.0/len(self.proposal_samples)).astype(np.float32)

        # Parameter names and ranges for plotting with GetDist
        self.names = param_names
        self.labels = param_names
        self.ranges = dict(zip(param_names, [(self.prior.low[i].numpy().astype(np.float64), self.prior.high[i].numpy().astype(np.float64)) for i in range(self.npar) ]))
        self.show_plot = show_plot

        # Training loss, validation loss
        self.training_loss = np.zeros((0, self.NDEs.n_stack), dtype=np.float32)
        self.validation_loss = np.zeros((0, self.NDEs.n_stack), dtype=np.float32)
        self.stacked_sequential_training_loss = []
        self.stacked_sequential_validation_loss = []
        self.sequential_nsims = []

        # MPI-specific setup
        self.rank = rank
        self.n_procs = n_procs
        if n_procs > 1:
            self.use_mpi = True
            self.comm = comm
            self.red_op = red_op
        else:
            self.use_mpi = False

        # Show progress bars?
        self.progress_bar = progress_bar

        # Filenames for saving/restoring graph and attributes
        self.results_dir = results_dir
        self.filename = filename

        # Save attributes of the ojbect as you go?
        self.save = save

        # Restore the graph and dynamic object attributes if restore = True
        if restore == True:

            # Restore the dynamic object attributes
            self.NDEs.weighting, self.posterior_samples, self.posterior_weights, self.proposal_samples, self.proposal_weights, self.training_loss, self.validation_loss, self.stacked_sequential_training_loss, self.stacked_sequential_validation_loss, self.sequential_nsims, self.theta_realizations, self.data_realizations = pickle.load(open(self.results_dir + "/" + self.filename + ".pkl", 'rb'))

            # Restore the NDE models
            self.NDEs.load_models(self.NDEs.stack, directory=self.results_dir, filename=self.filename)

    # Save object attributes
    def saver(self):

        f = open(self.results_dir + "/" + self.filename + ".pkl", 'wb')
        pickle.dump([self.NDEs.weighting.numpy(), self.posterior_samples, self.posterior_weights, self.proposal_samples, self.proposal_weights, self.training_loss, self.validation_loss, self.stacked_sequential_training_loss, self.stacked_sequential_validation_loss, self.sequential_nsims, self.theta_realizations, self.data_realizations], f)
        f.close()

        self.NDEs.save_models(self.NDEs.stack, directory=self.results_dir, filename=self.filename)

    # Divide list of jobs between MPI processes
    def allocate_jobs(self, n_jobs):
        n_j_allocated = 0
        for i in range(self.n_procs):
            n_j_remain = n_jobs - n_j_allocated
            n_p_remain = self.n_procs - i
            n_j_to_allocate = int(n_j_remain / n_p_remain)
            if self.rank == i:
                return range(n_j_allocated, \
                             n_j_allocated + n_j_to_allocate)
            n_j_allocated += n_j_to_allocate

    # Combine arrays from all processes assuming
    # 1) array was initially zero
    # 2) each process has edited a unique slice of the array
    def complete_array(self, target_distrib):
        if self.use_mpi:
            target = np.zeros(target_distrib.shape, \
                              dtype=target_distrib.dtype)
            self.comm.Allreduce(target_distrib, target, \
                                op=self.red_op)
        else:
            target = target_distrib
        return target

    # Bayesian optimization acquisition function
    def acquisition(self, theta):

        # Compute log_posteriors
        P = self.NDEs.weighted_log_prob((self.data - self.data_shift)/self.data_scale, conditional=(theta - self.theta_shift)/self.theta_scale)
        P_mean, P_variance = self.NDEs.variance(P)

        # Check whether prior is zero or not
        return tf.multiply(self.NDEs.weighted_log_posterior(theta), tf.sqrt(P_variance))


    # train the NDEs in the stack
    def train(self, training_data=None, f_val=0.1, epochs=300, n_batch=100, patience=20):

        # default training data
        if training_data is None:
            training_data = [self.theta_train, self.data_train]

        # train the NDEs
        val_loss, train_loss = self.NDEs.fit(data=training_data, f_val=f_val, epochs=epochs, n_batch=n_batch, patience=patience)

        # save the training/validation loss
        self.training_loss = np.vstack([self.training_loss, train_loss])
        self.validation_loss = np.vstack([self.validation_loss, val_loss])

    # Bayesian optimization training
    def bayesian_optimization_training(self, simulator, compressor, n_batch, n_populations, n_optimizations = 10, \
                                       simulator_args = None, compressor_args = None, plot = False, batch_size = 100, \
                                       validation_split = 0.1, epochs = 300, patience = 20, seed_generator = None, \
                                       save_intermediate_posteriors = False, sub_batch = 1):

        # Loop over n_populations
        for i in range(n_populations):

            # Find acquisition point...
            print('Finding optimal acquisition point...')
            A_optimal = 0
            theta_optimal = self.theta_fiducial
            for i in range(n_optimizations):
                res = optimization.basinhopping(lambda x: -self.acquisition(x).numpy(), x0=self.theta_fiducial)
                if res.fun < A_optimal:
                    A_optimal = res.fun
                    theta_optimal = res.x

            # Array of parameters to run simulations
            thetas = np.array([theta_optimal for k in range(n_batch)], dtype=np.float32)

            # Run a small batch of simulations at the acquisition point
            data_batch, theta_batch = self.run_simulation_batch(n_batch, thetas, simulator, compressor, simulator_args, compressor_args, seed_generator = seed_generator, sub_batch = sub_batch)

            # Augment the training data
            self.add_simulations(data_batch, theta_batch)

            #TC - we should add maximising the ELBO between propsal distribution and NDEs, that would be the most correct thing to do and would be really quick (not yet implemented in ndes)
            # Train the networks on these initial simulations
            self.train(training_data=[self.theta_train, self.data_train], f_val=validation_split, epochs=epochs, n_batch=max(self.n_sims//8, batch_size), patience=patience)

            self.stacked_sequential_training_loss.append(np.sum(self.NDEs.weighting * self.training_loss[-1]))
            self.stacked_sequential_validation_loss.append(np.sum(self.NDEs.weighting * self.validation_loss[-1]))
            self.sequential_nsims.append(self.n_sims)

            # Save attributes if save == True
            if self.save is True:
                self.saver()

    # Run n_batch simulations
    def run_simulation_batch(self, n_batch, thetas, simulator, compressor, simulator_args, compressor_args, seed_generator = None, sub_batch = 1):

        # Random seed generator: set to unsigned 32 bit int random numbers as default
        if seed_generator is None:
            seed_generator = lambda: np.random.randint(2147483647)

        # Dimension outputs
        data_samples = np.zeros((n_batch*sub_batch, self.D))
        parameter_samples = np.zeros((n_batch*sub_batch, self.npar))

        # Run samples assigned to each process, catching exceptions
        # (when simulator returns np.nan).
        i_prop = self.inds_prop[0]
        i_acpt = self.inds_acpt[0]
        err_msg = 'Simulator returns {:s} for parameter values: {} (rank {:d})'
        if self.progress_bar:
            pbar = tqdm(total = self.inds_acpt[-1], desc = "Simulations")
        while i_acpt <= self.inds_acpt[-1]:
            try:
                sims = simulator(thetas[i_prop,:], seed_generator(), simulator_args, sub_batch)

                # Make sure the sims are the right shape
                if sub_batch == 1 and len(sims) != 1:
                    sims = np.array([sims])
                compressed_sims = np.array([compressor(sims[k], compressor_args) for k in range(sub_batch)])
                if np.all(np.isfinite(compressed_sims.flatten())):
                    data_samples[i_acpt*sub_batch:i_acpt*sub_batch+sub_batch,:] = compressed_sims
                    parameter_samples[i_acpt*sub_batch:i_acpt*sub_batch+sub_batch,:] = thetas[i_prop,:]
                    i_acpt += 1
                    if self.progress_bar:
                        pbar.update(1)
                else:
                    print(err_msg.format('NaN/inf', thetas[i_prop,:], self.rank))
            except:
                print(err_msg.format('exception', thetas[i_prop,:], self.rank))
            i_prop += 1

        # Reduce results from all processes and return
        data_samples = self.complete_array(data_samples)
        parameter_samples = self.complete_array(parameter_samples)
        return data_samples.astype(np.float32), parameter_samples.astype(np.float32)

    # weighted log posterior
    def weighted_log_posterior(self, theta):

        lnP = self.NDEs.weighted_log_prob((self.data.astype(np.float32) - self.data_shift)/self.data_scale, conditional=(theta.astype(np.float32) - self.theta_shift)/self.theta_scale ).numpy() + self.prior.log_prob(theta.astype(np.float32)).numpy()

        if np.isnan(lnP):
            return -1e100
        else:
            return lnP

    # weighted log posterior
    def log_proposal(self, theta):

        lnP = 0.5*self.NDEs.weighted_log_prob((self.data.astype(np.float32) - self.data_shift)/self.data_scale, conditional=(theta.astype(np.float32) - self.theta_shift)/self.theta_scale ).numpy() + self.prior.log_prob(theta.astype(np.float32)).numpy()

        if np.isnan(lnP):
            return -1e100
        else:
            return lnP

    # EMCEE sampler
    def emcee_sample(self, log_target=None, x0=None, burn_in_chain=100, main_chain=1000):

        # default log target
        if log_target is None:
            log_target = self.weighted_log_posterior

        # Set up default x0
        if x0 is None:
            x0 = [self.posterior_samples[-i,:] for i in range(self.nwalkers)]

        # Set up the sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, self.npar, log_target)

        # Burn-in chain
        state = sampler.run_mcmc(x0, burn_in_chain)
        sampler.reset()

        # Main chain
        sampler.run_mcmc(state, main_chain)

        # pull out the unique samples and weights
        chain, weights = np.unique(sampler.get_chain(flat=True), axis=0, return_counts=True)

        # pull out the log probabilities
        log_prob, _ = np.unique(sampler.get_log_prob(flat=True), axis=0, return_counts=True)

        return chain, weights, log_prob

    def sequential_training(self, simulator, compressor, n_initial, n_batch, n_populations, proposal = None, \
                            simulator_args = None, compressor_args = None, safety = 5, plot = True, batch_size = 100, \
                            validation_split = 0.1, epochs = 300, patience = 20, seed_generator = None, \
                            save_intermediate_posteriors = True, sub_batch = 1):

        # Set up the initial parameter proposal density
        if proposal is None:
            if self.Finv is not None:
                scale = tf.linalg.cholesky((9. *self.Finv).astype(np.float32))
                proposal = ndes.TruncatedMultivariateNormalTriL(
                    loc=self.theta_fiducial.astype(np.float32), scale_tril=scale,
                    low=self.prior.low, high=self.prior.high,
                    validate_args=False, allow_nan_stats=True,
                    name='Proposal')
            else:
                proposal = self.prior

        # Generate initial theta values from some broad proposal on
        # master process and share with other processes. Overpropose
        # by a factor of safety to (hopefully) cope gracefully with
        # the possibility of some bad proposals. Assign indices into
        # proposal array (self.inds_prop) and accepted arrays
        # (self.inds_acpt) to allow for easy MPI communication.
        if self.rank == 0:
            theta_batch = proposal.sample(safety * n_initial).numpy()
        else:
            theta_batch = np.zeros((safety * n_initial, self.npar))
        if self.use_mpi:
            self.comm.Bcast(theta_batch, root=0)
        self.inds_prop = self.allocate_jobs(safety * n_initial)
        self.inds_acpt = self.allocate_jobs(n_initial)

        # Run simulations at those theta values
        data_batch, theta_batch = self.run_simulation_batch(n_initial, theta_batch, simulator, compressor, simulator_args, compressor_args, seed_generator = seed_generator, sub_batch = sub_batch)

        # Train on master only
        if self.rank == 0:

            # Construct the initial training-set
            self.load_simulations(data_batch, theta_batch)

            #TC - we should add maximising the ELBO between propsal distribution and NDEs, that would be the most correct thing to do and would be really quick (not yet implemented in ndes)
            # Train the networks on these initial simulations
            self.train(training_data=[self.theta_train, self.data_train], f_val=validation_split, epochs=epochs, n_batch=max(self.n_sims//8, batch_size), patience=patience)

            self.stacked_sequential_training_loss.append(np.sum(self.NDEs.weighting * self.training_loss[-1]))
            self.stacked_sequential_validation_loss.append(np.sum(self.NDEs.weighting * self.validation_loss[-1]))
            self.sequential_nsims.append(self.n_sims)

            # Generate posterior samples
            if save_intermediate_posteriors:
                print('Sampling approximate posterior...')
                #x0 = [self.posterior_samples[-i,:] for i in range(self.nwalkers)]
                x0 = self.posterior_samples[np.argpartition(self.log_posterior_values, -self.nwalkers)[-self.nwalkers:], :]
                self.posterior_samples, self.posterior_weights, self.log_posterior_values = self.emcee_sample(x0=x0, main_chain=self.posterior_chain_length)

                # Save posterior samples to file
                f = open('{}posterior_samples_0.dat'.format(self.results_dir), 'w')
                np.savetxt(f, self.posterior_samples)
                f.close()

                print('Done.')

                # If plot == True, plot the current posterior estimate
                if plot == True:
                    self.triangle_plot([self.posterior_samples], weights=[self.posterior_weights], savefig=True, \
                                    filename='{}seq_train_post_0.pdf'.format(self.results_dir))

            # Save attributes if save == True
            if self.save is True:
                self.saver()

        # Loop through a number of populations
        for i in range(n_populations):

            # Propose theta values on master process and share with
            # other processes. Again, ensure we propose more sets of
            # parameters than needed to cope with bad params.
            if self.rank == 0:

                # Current population
                print('Population {}/{}'.format(i+1, n_populations))

                # Sample the current posterior approximation
                print('Sampling proposal density...')
                #x0 = [self.proposal_samples[-j,:] for j in range(self.nwalkers)]
                x0 = self.proposal_samples[np.argpartition(self.log_proposal_values, -self.nwalkers)[-self.nwalkers:], :]

                self.proposal_samples, self.proposal_weights, self.log_proposal_values = \
                    self.emcee_sample(log_target = self.log_proposal,
                                      x0=x0,
                                      main_chain=self.proposal_chain_length)
                theta_batch = self.proposal_samples[-safety * n_batch:,:]
                print('Done.')

            else:
                theta_batch = np.zeros((safety * n_batch, self.npar))
            if self.use_mpi:
                self.comm.Bcast(theta_batch, root=0)

            # Run simulations
            self.inds_prop = self.allocate_jobs(safety * n_batch)
            self.inds_acpt = self.allocate_jobs(n_batch)
            data_batch, theta_batch = self.run_simulation_batch(n_batch, theta_batch, simulator, compressor, simulator_args, compressor_args, seed_generator = seed_generator, sub_batch = sub_batch)

            # Train on master only
            if self.rank == 0:

                # Augment the training data
                self.add_simulations(data_batch, theta_batch)

                # Train the networks on these initial simulations
                self.train(training_data=[self.theta_train, self.data_train], f_val=validation_split, epochs=epochs, n_batch=max(self.n_sims//8, batch_size), patience=patience)

                self.stacked_sequential_training_loss.append(np.sum(self.NDEs.weighting * self.training_loss[-1]))
                self.stacked_sequential_validation_loss.append(np.sum(self.NDEs.weighting * self.validation_loss[-1]))
                self.sequential_nsims.append(self.n_sims)

                # Generate posterior samples
                if save_intermediate_posteriors:
                    print('Sampling approximate posterior...')
                    x0 = [self.posterior_samples[-i,:] for i in range(self.nwalkers)]
                    self.posterior_samples, self.posterior_weights, self.log_posterior_values = \
                        self.emcee_sample(x0=x0, main_chain=self.posterior_chain_length)

                    # Save posterior samples to file
                    f = open('{}posterior_samples_{:d}.dat'.format(self.results_dir, i+1), 'w')
                    np.savetxt(f, self.posterior_samples)
                    f.close()

                    print('Done.')

                    # If plot == True
                    if plot == True:
                        # Plot the posterior
                        self.triangle_plot([self.posterior_samples], weights=[self.posterior_weights], \
                                        savefig=True, \
                                        filename='{}seq_train_post_{:d}.pdf'.format(self.results_dir, i + 1))

                # Plot training convergence
                if plot == True:
                    # Plot the training loss convergence
                    self.sequential_training_plot(savefig=True, filename='{}seq_train_loss.pdf'.format(self.results_dir))

    def load_simulations(self, data_batch, theta_batch):

        if self.input_normalization is "auto":
            if self.input_normalization is False:
                self.data_shift = np.mean(data_batch, axis = 0).astype(np.float32)
                self.data_scale = np.std(data_batch, axis = 0).astype(np.float32)
                self.theta_shift = np.mean(theta_batch, axis = 0).astype(np.float32)
                self.theta_scale = np.std(theta_batch, axis = 0).astype(np.float32)

        self.theta_realizations = np.concatenate([self.theta_realizations, theta_batch])
        self.data_realizations = np.concatenate([self.data_realizations, data_batch])
        self.theta_train = (self.theta_realizations.astype(np.float32) - self.theta_shift)/self.theta_scale
        self.data_train = (self.data_realizations.astype(np.float32) - self.data_shift)/self.data_scale
        self.n_sims += len(theta_batch)

    def add_simulations(self, data_batch, theta_batch):
        self.theta_realizations = np.concatenate([self.theta_realizations, theta_batch])
        self.data_realizations = np.concatenate([self.data_realizations, data_batch])
        self.theta_train = (self.theta_realizations.astype(np.float32) - self.theta_shift)/self.theta_scale
        self.data_train = (self.data_realizations.astype(np.float32) - self.data_shift)/self.data_scale
        self.n_sims += len(theta_batch)

    def fisher_pretraining(self, n_batch=5000, plot=True, batch_size=100, validation_split=0.1, epochs=1000, patience=20):

        # Train on master only
        if self.rank == 0:

            # Generate fisher pre-training data

            # Broader proposal
            proposal = ndes.TruncatedMultivariateNormalTriL(
                loc=self.theta_fiducial, scale_tril=tf.linalg.cholesky(9*self.Finv),
                low=self.prior.low, high=self.prior.high,
                validate_args=False, allow_nan_stats=True,
                name='AsymptoticPosterior')

            # Cholesky of inverse Fisher information matrix
            L = np.linalg.cholesky(self.Finv)

            # Sample parameters from some broad proposal
            theta_batch = np.zeros((3*n_batch, self.npar))
            theta_batch[:n_batch,:] = self.prior.sample(n_batch)
            theta_batch[n_batch:2*n_batch,:] = self.asymptotic_posterior.sample(n_batch)
            theta_batch[2*n_batch:,:] = proposal.sample(n_batch)

            # Sample data assuming a Gaussian likelihood
            data_batch = np.array([theta + np.dot(L, np.random.normal(0, 1, self.npar)) for theta in theta_batch], dtype=np.float32)

            # Construct the initial training-set
            fisher_theta_train = (theta_batch.astype(np.float32).reshape((3*n_batch, self.npar)) - self.theta_shift)/self.theta_scale
            fisher_data_train = (data_batch.astype(np.float32).reshape((3*n_batch, self.npar)) - self.data_shift)/self.data_scale

            # Train the networks on these initial simulations
            self.train(training_data=[fisher_theta_train, fisher_data_train], f_val=validation_split, epochs=epochs, n_batch=batch_size, patience=patience)

            # Generate posterior samples
            if plot==True:
                print('Sampling approximate posterior...')
                x0 = [self.posterior_samples[-i,:] for i in range(self.nwalkers)]
                self.posterior_samples, self.posterior_weights, self.log_posterior_values = \
                    self.emcee_sample(x0=x0, main_chain=self.posterior_chain_length)
                print('Done.')

                # Plot the posterior
                self.triangle_plot([self.posterior_samples], weights=[self.posterior_weights], \
                                    savefig=True, \
                                    filename='{}fisher_train_post.pdf'.format(self.results_dir))

            # save current state if save=True
            if self.save is True:
                self.saver()

    def triangle_plot(self, samples = None, weights = None, savefig = False, filename = None):
        # Set samples to the posterior samples by default
        if samples is None:
            samples = self.posterior_samples
        mc_samples = [MCSamples(samples=s, weights=weights[i], names=self.names, labels=self.labels, ranges=self.ranges) for i, s in enumerate(samples)]

        # Triangle plot
        plt.close()
        with mpl.rc_context():
            g = plots.getSubplotPlotter(width_inch = 12)
            g.settings.figure_legend_frame = False
            g.settings.alpha_filled_add=0.6
            g.settings.axes_fontsize=14
            g.settings.legend_fontsize=16
            g.settings.lab_fontsize=20
            g.triangle_plot(mc_samples, filled_compare=True, normalized=True)
            for i in range(0, len(samples[0][0,:])):
                for j in range(0, i+1):
                    ax = g.subplots[i,j]
                    xtl = ax.get_xticklabels()
                    ax.set_xticklabels(xtl, rotation=45)
            plt.subplots_adjust(hspace=0, wspace=0)

            if savefig:
                plt.savefig(filename, bbox_inches='tight')
            if self.show_plot:
                plt.show()
            else:
                plt.close()

    def sequential_training_plot(self, savefig = False, filename = None):

        plt.close()
        columnwidth = 18 # cm
        aspect = 1.67
        pts_per_inch = 72.27
        inch_per_cm = 2.54
        width = columnwidth/inch_per_cm
        with mpl.rc_context({'figure.figsize': [width, width / aspect],
                                'backend': 'pdf',
                                'font.size': 15,
                                'legend.fontsize': 15,
                                'legend.frameon': False,
                                'legend.loc': 'best',
                                'lines.markersize': 3,
                                'lines.linewidth': .5,
                                'axes.linewidth': .5,
                                'axes.edgecolor': 'black'}):

            # Trace plot of the training and validation loss as a function of the number of simulations ran
            plt.plot(self.sequential_nsims, self.stacked_sequential_training_loss, markersize=5, marker='o', lw=2, alpha=0.7, label = 'training loss')
            plt.plot(self.sequential_nsims, self.stacked_sequential_validation_loss, markersize=5, marker='o', lw=2, alpha=0.7, label = 'validation loss')

            plt.xlabel(r'number of simulations, $n_\mathrm{sims}$')
            plt.ylabel(r'negative log loss, $-\mathrm{ln}\,U$')
            plt.tight_layout()
            plt.legend()

            if savefig:
                plt.savefig(filename)
            if self.show_plot:
                plt.show()
            else:
                plt.close()
