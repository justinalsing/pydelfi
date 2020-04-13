import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import pickle
import os
import numpy as np

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util

tfd = tfp.distributions
tfb = tfp.bijectors
dtype = tf.float32

__version__ = "v0.2"
__author__ = "Justin Alsing, Tom Charnock and Stephen Feeney"

class NDE():
    def __init__(self, model, prior, optimiser=tf.keras.optimizers.Adam(lr=1e-4), optimiser_arguments=None, dtype=tf.float32, **kwargs):
        self.dtype = dtype
        if self.dtype == tf.float32:
            self.itype = tf.int32
        else:
            self.itype = tf.int64

        if type(model) is list:
            self.n_stack = len(model)
        else:
            self.n_stack = 1
            model = [model]

        self.error_stack = None
        self.set_stack()

        # model weighting
        self.weighting = tf.ones((self.n_stack,), name="weighting")
        self.model = model
        self.prior = prior

        if optimiser_arguments is not None:
            self.optimiser = optimiser(optimiser_arguments)
        else:
            self.optimiser = optimiser()
        super(NDE, self).__init__(**kwargs)

    @tf.function
    def single_train_epoch(self, dataset, stack, variables_list, stack_size, n_batch):
        loss = tf.zeros((stack_size,))
        for step, xy_batch_train in enumerate(dataset):
            # Unpack batched training data
            x_batch_train, y_batch_train = xy_batch_train
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:
                # Compute the loss for this batch.
                neg_log_prob = -tf.reduce_mean(self.log_prob(y_batch_train, conditional=x_batch_train, stack=stack), -1)
                neg_total_log_prob = tf.reduce_mean(neg_log_prob)
            # Retrieve the gradients of the trainable variables wrt the loss and
            # pass to optimizer.
            grads = tape.gradient(neg_total_log_prob, variables_list)
            self.optimiser.apply_gradients(zip(grads, variables_list))
            loss = tf.add(loss, neg_log_prob)
        return tf.divide(loss, n_batch)

    @tf.function
    def single_validate_epoch(self, dataset, stack, stack_size, n_batch):
        loss = tf.zeros((stack_size,))
        for step, xy_batch_train in enumerate(dataset):
            # Unpack batched training data
            x_batch_train, y_batch_train = xy_batch_train
            # Compute the loss for this batch.
            neg_log_prob = -tf.reduce_mean(self.log_prob(y_batch_train, conditional=x_batch_train, stack=stack), -1)
            loss = tf.add(loss, neg_log_prob)
        return tf.divide(loss, n_batch)

    def fit(self, data, f_val=0.1, epochs=1000, n_batch=100,
            patience=20, file_name=None, progress_bar=True):
        """
        Training function to be called with desired parameters.
        :param data: a tuple/list of (X,Y) with data where Y is conditioned on X.
        :param f_val: fraction of training data randomly selected to be used for validation
        :param epochs: maximum number of epochs for training.
        :param n_batch: size of each batch within an epoch.
        :param patience: number of epochs for early stopping criteria.
        :param file_name: string of name (with or without folder) where model is saved.
        :param progress_bar: display progress bar?
        """

        stack = list(range(self.n_stack))
        stack_size = self.n_stack
        variables_list = self.get_flat_variables_list(stack)

        # Parse full training data and determine size of training set
        data_X, data_Y = data
        data_X = tf.convert_to_tensor(data_X, dtype=self.dtype)
        data_Y = tf.convert_to_tensor(data_Y, dtype=self.dtype)

        n_sims = data_X.shape[0]

        #is_train = tfd.Categorical(probs=[f_val, 1. - f_val], dtype=tf.bool).sample(n_sims)
        n_val = int(n_sims * f_val)
        n_train = n_sims - n_val
        is_train = tf.random.shuffle([True] * n_train + [False] * n_val)
        #n_train = tf.reduce_sum(tf.cast(is_train, dtype=tf.int64))

        n_train_batches = n_train // n_batch
        if n_train_batches == 0:
            n_train_batches = 1
        n_train_batches = tf.cast(n_train_batches, dtype=self.dtype)

        n_val_batches = int(n_val / n_batch)
        if n_val_batches == 0:
            n_val_batches = 1
        n_val_batches = tf.cast(n_val_batches, dtype=self.dtype)

        # Create training and validation Dataset objects, shuffling and batching the training data. Note
        # the default behaviour of Dataset.shuffle() sets reshuffle_each_iteration=True, and
        # Dataset.batch() sets drop_remainder=False
        train_dataset = tf.data.Dataset.from_tensor_slices((data_X[is_train],
                                                            data_Y[is_train]))
        val_dataset = tf.data.Dataset.from_tensor_slices((data_X[~is_train],
                                                          data_Y[~is_train]))
        train_dataset = train_dataset.shuffle(n_train).batch(n_batch)
        val_dataset = val_dataset.batch(n_val)

        # Early stopping variables
        es_count = tf.zeros((self.n_stack,), dtype=tf.int64)
        temp_train_loss = tf.zeros((self.n_stack,), dtype=self.dtype)
        temp_val_loss = tf.divide(tf.ones(self.n_stack, dtype=self.dtype), tf.convert_to_tensor(0, dtype=self.dtype))

        temp_variables = [self.model[i].trainable_variables for i in self.stack]

        # Validation and training losses
        train_losses = []
        val_losses = []

        # Progress bar, if desired
        if progress_bar:
            if self.isnotebook():
                pbar = tqdm.tnrange(epochs, desc="Training")
            else:
                pbar = tqdm.trange(epochs, desc="Training")
            pbar.set_postfix(ordered_dict={"train loss":0, "val loss":0}, refresh=True)

        # Main training loop
        for epoch in range(epochs):
            # Iterate over the batches of the dataset.
            this_train_loss = self.single_train_epoch(train_dataset, stack, variables_list, stack_size, n_train_batches)
            this_val_loss = self.single_validate_epoch(val_dataset, stack, stack_size, 1)

            # early stopping
            state = this_val_loss < tf.gather(temp_val_loss, stack)

            improving = tf.where(state)
            es_count = tf.squeeze(
                tf.tensor_scatter_nd_update(
                    tf.expand_dims(es_count, 1),
                    improving,
                    tf.zeros(
                        (tf.reduce_sum(
                            tf.cast(state, dtype=tf.int64)),
                         1),
                        dtype=tf.int64)),
                1)
            improving = tf.squeeze(improving, 1)
            improving_stack = tf.gather(stack, improving)
            temp_variables = self.save_models(improving_stack.numpy(), variables=temp_variables)
            temp_train_loss = tf.tensor_scatter_nd_update(
                temp_train_loss,
                tf.expand_dims(improving_stack, 1),
                tf.gather(this_train_loss, improving))
            temp_val_loss = tf.tensor_scatter_nd_update(
                temp_val_loss,
                tf.expand_dims(improving_stack, 1),
                tf.gather(this_val_loss, improving))

            not_improving = tf.where(~state)
            es_count = tf.squeeze(
                tf.tensor_scatter_nd_add(
                    tf.expand_dims(es_count, 1),
                    not_improving,
                    tf.ones(
                        (tf.reduce_sum(
                            tf.cast(~state, dtype=tf.int64)),
                         1),
                        dtype=tf.int64)),
                1)

            ended = es_count >= patience
            if tf.reduce_any(ended):
                model_indices = tf.gather(stack, tf.squeeze(tf.where(ended), 1)).numpy()
                remaining_indices = tf.squeeze(tf.where(~ended), 1)
                es_count = tf.gather(es_count, remaining_indices)
                self.load_models(model_indices, variables=temp_variables)
                stack = self.remove_from_stack(stack, model_indices, epoch=epoch)
                stack_size = len(stack)
                variables_list = self.get_flat_variables_list(stack)
                if len(stack) == 0:
                    break

            train_losses.append(temp_train_loss)
            val_losses.append(temp_val_loss)

            # Update progress if desired.
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix(
                    ordered_dict={
                        "train loss":[float("{0:.3g}".format(this_train_loss.numpy()[i]))for i in range(len(this_train_loss.numpy()))],
                        "val loss":[float("{0:.3g}".format(this_val_loss.numpy()[i])) for i in range(len(this_train_loss.numpy()))],
                        "patience counter":es_count.numpy(),
                        "stack":stack},
                    refresh=True)
        self.weighting = tf.nn.softmax(-temp_val_loss - tf.math.reduce_max(-temp_val_loss))
        self.set_stack()
        return tf.stack(val_losses), tf.stack(train_losses)

    def set_stack(self, train=False, error=None):
        stack = list(range(self.n_stack))
        if train:
            self.stack = stack
        else:
            if error is not None:
                for i in error:
                    stack.pop(i)
                self.error_stack = stack
            self.stack = stack
            if self.error_stack is not None:
                self.stack = self.error_stack

    def get_flat_variables_list(self, stack):
        variable_list = []
        for i in stack:
            for variable in self.model[i].trainable_variables:
                variable_list.append(variable)
        return variable_list

    def save_models(self, models, variables=None, directory=None, filename=None):
        if (filename is not None) or (variables is not None):
            for model in models:
                these_variables = self.model[model].trainable_variables
                if filename is not None:
                    if not os.path.isdir(directory):
                        raise ValueError(directory + " does not exist.")
                    with open(directory + "/" + filename + "_model_" + str(model) + ".pkl", "wb") as outfile:
                        pickle.dump([variable.numpy() for variable in these_variables], outfile)
                if variables is not None:
                    variables[model] = these_variables
        if variables is not None:
            return variables

    def load_models(self, models, variables=None, directory=None, filename=None):
        if (filename is not None) or (variables is not None):
            for model in models:
                if filename is not None:
                    file = directory + "/" + filename + "_model_" + str(model) + ".pkl"
                    if not os.path.isfile(file):
                        raise ValueError(file + " does not exist.")
                    with open(file, "rb") as outfile:
                        for model_variable, temp_variable in zip(self.model[model].trainable_variables, tuple(pickle.load(outfile))):
                            model_variable.assign(temp_variable)
                if variables is not None:
                    for model_variable, temp_variable in zip(self.model[model].trainable_variables, variables[model]):
                        model_variable.assign(temp_variable)

    def remove_from_stack(self, stack, models, epoch=None):
        for model in models:
            stack.remove(model)
            if epoch is not None:
                print("Training terminated for model {:d} at epoch {:d}.".format(model, epoch + 1))
        return stack

    @tf.function
    def log_prob(self, data, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        """
        log probability, returns log density ln P(d | \theta)
        :param data: data vectors to evaluate density at
        :param parameters: (conditional) input parameters to evaluate density at
        """
        return tf.stack([
            self.model[element].log_prob(data, conditional=conditional)
            for element in stack], 0)

    @tf.function
    def weighted_log_prob(self, data, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        return tf.math.log(self.weighted_prob(data, conditional=conditional, stack=stack))

    @tf.function
    def prob(self, data, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        """
        probability, returns density P(d | \theta)
        :param data: data vectors to evaluate density at
        :param parameters: (conditional) input parameters to evaluate density at
        """
        return tf.stack([
            self.model[element].prob(data, conditional=conditional)
            for element in stack], 0)

    @tf.function
    def weighted_prob(self, data, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        return tf.reduce_sum(
            tf.multiply(self.weighting,
                        self.prob(data, conditional=conditional, stack=stack)))

    @tf.function
    def sample(self, n=None, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        if n is None:
            n = 1
        return tf.stack([
            self.model[element].sample(n, conditional=conditional)
            for element in stack], 0)

    @tf.function
    def weighted_sample(self, n=None, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        """
        sample, returns samples {d} from P(d | \theta) for some input values of \theta
        :param parameters: (conditional) input parameters to draw samples at
        :param n: number of samples to draw (for each parameter set input)
        """
        if n is None:
            n = 1
        samples = self.sample(n, conditional=None, stack=stack)
        return self.variance(samples)

    @tf.function
    def log_posterior(self, data, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        return tf.add(
            self.log_prob(data, conditional=conditional, stack=stack),
            tf.cast(self.prior.log_prob(conditional), dtype=self.dtype))

    @tf.function
    def weighted_log_posterior(self, data, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        data = tf.cast(data, dtype=self.dtype)
        conditional = tf.cast(conditional, dtype=self.dtype)
        return tf.add(self.weighted_log_prob(data, conditional=conditional, stack=stack),
                      tf.cast(self.prior.log_prob(conditional), dtype=self.dtype))

    @tf.function
    def geometric_mean(self, data, conditional=None, stack=None):
        if stack is None:
            stack = self.stack
        half = tf.cast(0.5, dtype=self.dtype)
        two = tf.cast(2., dtype=self.dtype)
        data = tf.cast(data, dtype=self.dtype)
        conditional = tf.cast(conditional, dtype=self.dtype)
        return tf.multiply(half,
                           tf.add(self.weighted_log_prob(data, conditional=conditional, stack=stack),
                                  tf.multiply(two, self.prior.log_prob(conditional))))

    @tf.function
    def variance(self, x):
        weighted_sum = tf.reduce_sum(self.weighting)
        mean = tf.divide(
            tf.einsum("i...,i->...",
                x,
                self.weighting),
            weighted_sum)
        variance = tf.divide(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        x,
                        tf.expand_dims(mean, 0))),
                0),
            weighted_sum)
        return mean, variance

    def isnotebook(self):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False

def ConditionalMaskedAutoregressiveFlow(
    n_parameters, n_data, n_mades=1, n_hidden=[50,50], input_order="random",
    activation=tf.keras.layers.LeakyReLU(0.01), all_layers=True,
    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None), bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
    bias_constraint=None):
    """
    Conditional Masked Autoregressive Flow.
    """

    # construct stack of MADEs
    bijector = tfb.Chain([
        tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=n_hidden,
                activation=activation,
                event_shape=[n_data],
                conditional=True,
                conditional_shape=[n_parameters],
                conditional_input_all_layers=True,
                input_order=input_order,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint))
        for i in range(n_mades)])
    return tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=bijector,
        event_shape=[n_data])


class MixtureDensityNetwork(tfd.Distribution):
    """
    Implements a gaussian Mixture Density Network for modeling a conditional density p(d|\theta) (d="data", \theta="parameters")
    """
    def __init__(self, n_parameters, n_data, n_components=3, n_hidden=[50,50], activation=tf.keras.layers.LeakyReLU(0.01), dtype=tf.float32, reparameterization_type=None, validate_args=False, allow_nan_stats=True, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None)):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs (ie dimensionality of distribution you're parameterizing)
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param activation: activation function for network
        :param dtype: tensorflow type
        """
        super(MixtureDensityNetwork, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats)

        # dimension of data and parameter spaces
        self.n_parameters = n_parameters
        self.n_data = n_data

        # number of mixture components and network architecture
        self.n_components = n_components

        # required size of output layer for a Gaussian mixture density network
        self.n_hidden = n_hidden
        self.activation = activation
        self.architecture = [self.n_parameters] + self.n_hidden

        self._network = self.build_network(kernel_initializer)

    def build_network(self, kernel_initializer):
        """
        Individual network constructor. Builds a single mixture of Gaussians.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                self.architecture[layer + 1],
                input_shape=(size,),
                activation=self.activation,
                kernel_initializer=kernel_initializer)
            for layer, size in enumerate(self.architecture[:-1])])
        model.add(
            tf.keras.layers.Dense(
                tfp.layers.MixtureSameFamily.params_size(
                    self.n_components,
                    component_params_size=tfp.layers.MultivariateNormalTriL.params_size(self.n_data)),
                kernel_initializer=kernel_initializer))
        model.add(
            tfp.layers.MixtureSameFamily(self.n_components, tfp.layers.MultivariateNormalTriL(self.n_data)))
        return model

    def log_prob(self, x, **kwargs):
        if len(x.shape) == 1:
            x = x[tf.newaxis, ...]
        if len(kwargs["conditional"].shape) == 1:
            kwargs["conditional"] = kwargs["conditional"][tf.newaxis, ...]
            squeeze = True
        else:
            squeeze = False
        log_prob = self._network(kwargs["conditional"]).log_prob(x)
        if squeeze:
            log_prob = tf.squeeze(log_prob, 0)
        return log_prob

    def prob(self, x, **kwargs):
        if len(x.shape) == 1:
            x = x[tf.newaxis, ...]
        if len(kwargs["conditional"].shape) == 1:
            kwargs["conditional"] = kwargs["conditional"][tf.newaxis, ...]
            squeeze = True
        else:
            squeeze = False
        prob = self._network(kwargs["conditional"]).prob(x)
        if squeeze:
            prob = tf.squeeze(prob, 0)
        return prob

    def sample(self, n, **kwargs):
        if len(kwargs["conditional"].shape) == 1:
            kwargs["conditional"] = kwargs["conditional"][tf.newaxis, ...]
            squeeze = True
        else:
            squeeze = False
        samples = self._network(kwargs["conditional"]).sample(n)
        if squeeze:
            samples = tf.squeeze(samples, 1)
        return samples


class TruncatedMultivariateNormalTriL(tfd.MultivariateNormalLinearOperator):
    """The multivariate normal distribution on `R^k`.

    The Multivariate Normal distribution is defined over `R^k` and parameterized
    by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
    `scale` matrix; `covariance = scale @ scale.T` where `@` denotes
    matrix-multiplication.

    #### Mathematical Details

    The probability density function (pdf) is,

    ```none

    pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z,
    y = inv(scale) @ (x - loc),
    Z = (2 pi)**(0.5 k) |det(scale)|,
    ```

    where:

    * `loc` is a vector in `R^k`,
    * `scale` is a matrix in `R^{k x k}`, `covariance = scale @ scale.T`,
    * `Z` denotes the normalization constant, and,
    * `||y||**2` denotes the squared Euclidean norm of `y`.
    A (non-batch) `scale` matrix is:

    ```none
    scale = scale_tril
    ```

    where `scale_tril` is lower-triangular `k x k` matrix with non-zero diagonal,
    i.e., `tf.diag_part(scale_tril) != 0`.

    Additional leading dimensions (if any) will index batches.

    Note that in the truncated multivariate is not correctly normalised (yet).

    The MultivariateNormal distribution is a member of the [location-scale
    family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
    constructed as,

    ```none
    X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
    Y = scale @ X + loc
    ```

    Trainable (batch) lower-triangular matrices can be created with
    `tfp.distributions.matrix_diag_transform()` and/or
    `tfp.distributions.fill_triangular()`

    #### Examples

    ```python
    tfd = tfp.distributions

    # Initialize a single 3-variate Gaussian.
    mu = [1., 2, 3]
    cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
    scale = tf.cholesky(cov)
    # ==> [[ 0.6,  0. ,  0. ],
    #      [ 0.2,  0.5,  0. ],
    #      [ 0.1, -0.3,  0.4]])
    mvn = tfd.TruncatedMultivariateNormalTriL(
      loc=mu,
      scale_tril=scale)

    mvn.mean().eval()
    # ==> [1., 2, 3]

    # Covariance agrees with cholesky(cov) parameterization.
    mvn.covariance().eval()
    # ==> [[ 0.36,  0.12,  0.06],
    #      [ 0.12,  0.29, -0.13],
    #      [ 0.06, -0.13,  0.26]]

    # Compute the pdf of an observation in `R^3` ; return a scalar.
    mvn.prob([-1., 0, 1]).eval()  # shape: []

    # Initialize a 2-batch of 3-variate Gaussians.
    mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
    tril = ...  # shape: [2, 3, 3], lower triangular, non-zero diagonal.
    mvn = tfd.TruncatedMultivariateNormalTriL(
      loc=mu,
      scale_tril=tril)

    # Compute the pdf of two `R^3` observations; return a length-2 vector.
    x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
    mvn.prob(x).eval()    # shape: [2]

    # Instantiate a "learnable" MVN.
    dims = 4
    mvn = tfd.TruncatedMultivariateNormalTriL(
      loc=tf.Variable(tf.zeros([dims], dtype=tf.float32), name="mu"),
      scale_tril=tfp.utils.DeferredTensor(
          tfp.bijectors.ScaleTriL().forward,
          tf.Variable(tf.zeros([dims * (dims + 1) // 2], dtype=tf.float32),
                      name="raw_scale_tril")))
    ```

    """

    def __init__(self,
               loc,
               scale_tril,
               low,
               high,
               validate_args=False,
               allow_nan_stats=True,
               dtype=tf.float32,
               name='truncatedMultivariateNormalTriL'):
        """Construct Multivariate Normal distribution on `R^k` with samples
        from a truncated boundary.

        The `batch_shape` is the broadcast shape between `loc` and `scale`
        arguments.

        The `event_shape` is given by last dimension of the matrix implied by
        `scale`. The last dimension of `loc` (if provided) must broadcast with
        this.

        Recall that `covariance = scale @ scale.T`. A (non-batch) `scale`
        matrix is:

        ```none
        scale = scale_tril
        ```

        where `scale_tril` is lower-triangular `k x k` matrix with non-zero
        diagonal, i.e., `tf.diag_part(scale_tril) != 0`.

        Additional leading dimensions (if any) will index batches.

        Args:
          loc: Floating-point `Tensor`. Must have shape `[B1, ..., Bb, k]`
            where `b >= 0` and `k` is the event size.
          scale_tril: Floating-point, lower-triangular `Tensor` with non-zero
            diagonal elements. `scale_tril` has shape `[B1, ..., Bb, k, k]`
            where `b >= 0` and `k` is the event size.
          low: Floating-point `Tensor`. Must have `[B1, ..., Bb, k]` where
            `b >= 0` and `k` is the event size. Defines the lower boundary for
            the samples.
          high: Floating-point `Tensor`. Must have `[B1, ..., Bb, k]` where
            `b >= 0` and `k` is the event size. Defines the upper boundary for
            the samples.
          validate_args: Python `bool`, default `False`. When `True`
            distribution
            parameters are checked for validity despite possibly degrading
            runtime performance. When `False` invalid inputs may silently
            render incorrect outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`,
            statistics (e.g., mean, mode, variance) use the value "`NaN`" to
            indicate the result is undefined. When `False`, an exception is
            raised if one or more of the statistic's batch members are
            undefined.
          name: Python `str` name prefixed to Ops created by this class.

        Raises:
          ValueError: if neither `loc` nor `scale_tril` are specified.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([loc, scale_tril, low, high], dtype)
            loc = tensor_util.convert_nonref_to_tensor(loc, name='loc',
                                                       dtype=dtype)
            scale_tril = tensor_util.convert_nonref_to_tensor(
            scale_tril, name='scale_tril', dtype=dtype)
            self.high = tensor_util.convert_nonref_to_tensor(high, name='high',
                                                             dtype=dtype)
            self.low = tensor_util.convert_nonref_to_tensor(low, name='low',
                                                            dtype=dtype)
            scale = tf.linalg.LinearOperatorLowerTriangular(
                scale_tril, is_non_singular=True, is_self_adjoint=False,
                is_positive_definite=False)
            self.mvn = tfd.MultivariateNormalLinearOperator(
                loc=loc, scale=scale, validate_args=validate_args,
                allow_nan_stats=allow_nan_stats, name=name)
            self.u = tfd.Blockwise(
                [tfd.Uniform(low=low[i], high=high[i])
                 for i in range(self.low.shape[0])])
            super(TruncatedMultivariateNormalTriL, self).__init__(
                loc=loc,
                scale=scale,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name)
            self._parameters = parameters

    def _log_prob(self, x, **kwargs):
        return tf.math.log(self._log_prob(x, **kwargs))

    def _prob(self, x, **kwargs):
        return tf.multiply(self.mvn.prob(x, **kwargs),
                           self.u.prob(x, **kwargs))

    def _sample_n(self, n, seed=None, **kwargs):
        samples = self.mvn.sample(n, seed=seed, **kwargs)
        too_low = samples < self.low
        too_high = samples > self.high
        rejected = tf.reduce_any(tf.logical_or(too_low, too_high), -1)
        while tf.reduce_any(rejected):
            new_n = tf.reduce_sum(tf.cast(rejected, dtype=tf.int32))
            new_samples = self.mvn.sample(new_n, seed=seed, **kwargs)
            samples = tf.tensor_scatter_nd_update(samples, tf.where(rejected),
                                                  new_samples)
            too_low = samples < self.low
            too_high = samples > self.high
            rejected = tf.reduce_any(tf.logical_or(too_low, too_high), -1)
        return samples

    @classmethod
    def _params_event_ndims(cls):
        return dict(loc=1, scale_tril=2)
