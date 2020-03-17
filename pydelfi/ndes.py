import tensorflow as tf
import tensorflow_probability as tfp
import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors

class NDE():
    def __init__(self, model, prior, optimiser=tf.keras.optimizers.Adam, optimiser_arguments=None, dtype=tf.float32, **kwargs):
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
    def single_train_epoch(self, dataset, variables_list, n_batch):
        loss = tf.zeros((self.n_stack,))
        for step, xy_batch_train in enumerate(dataset):
            # Unpack batched training data
            x_batch_train, y_batch_train = xy_batch_train
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:
                # Compute the loss for this batch.
                neg_log_prob = -tf.reduce_mean(self.log_prob(y_batch_train, x_batch_train), -1)
                neg_total_log_prob = tf.reduce_sum(neg_log_prob)
            # Retrieve the gradients of the trainable variables wrt the loss and
            # pass to optimizer.
            grads = tape.gradient(neg_total_log_prob, variables_list)
            self.optimiser.apply_gradients(zip(grads, variables_list))
            #grads = [tape.gradient(neg_array_log_prob[i], self.model[i].trainable_variables) for i in self.stack]
            #del tape
            #for i in self.stack:
            #    self.optimiser.apply_gradients(zip(grads[i], self.model[i].trainable_variables))
            loss = tf.add(loss, neg_log_prob)
        return tf.divide(loss, n_batch)

    @tf.function
    def single_validate_epoch(self, dataset, n_batch):
        loss = tf.zeros((self.n_stack,))
        for step, xy_batch_train in enumerate(dataset):
            # Unpack batched training data
            x_batch_train, y_batch_train = xy_batch_train
            # Compute the loss for this batch.
            neg_log_prob = -tf.reduce_mean(self.log_prob(y_batch_train, x_batch_train), -1)
            loss = tf.add(loss, neg_log_prob)
        return tf.divide(loss, n_batch)

    def fit(self, data, f_val=0.1, epochs=1000, n_batch=100,
            patience=20, file_name='tmp_model', progress_bar=True):
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

        self.set_stack(train=True)
        variables_list = self.get_variables_list()

        # Parse full training data and determine size of training set
        data_X, data_Y = data
        data_X = tf.convert_to_tensor(data_X, dtype=self.dtype)
        data_Y = tf.convert_to_tensor(data_Y, dtype=self.dtype)

        n_sims = data_X.shape[0]

        is_train = tfd.Categorical(probs=[f_val, 1. - f_val], dtype=tf.bool).sample(n_sims)
        n_train = tf.reduce_sum(tf.cast(is_train, dtype=tf.int64))
        n_val = n_sims - n_train

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
        #self.weighting = tf.Variable(tf.divide(tf.ones((self.n_stack,), dtype=self.dtype), 0.))
        es_count = tf.zeros((self.n_stack,), dtype=tf.int64)
        temp_params = [[] for i in self.stack]
        temp_weighting = [tf.divide(tf.ones(1, dtype=self.dtype), tf.convert_to_tensor(0, dtype=self.dtype)) for i in self.stack]

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
            train_losses.append(self.single_train_epoch(train_dataset, variables_list, n_train_batches))
            val_losses.append(self.single_validate_epoch(val_dataset, n_val_batches))

            ''' TC there is an error in the early stopping - should be a relatively easy fix
            # Use validation loss to stop training early
            improving = val_losses[-1] < tf.concat([temp_weighting[i] for i in self.stack], 0)
            for elem, improvement in enumerate(improving):
                m_ind = self.stack[elem]
                if improvement:
                    #self.model[m_ind].save_weights(file_name + "_" + str(m_ind), save_format="tf")
                    temp_params[m_ind] = self.model[m_ind].trainable_variables
                    temp_weighting[m_ind] = tf.expand_dims(val_losses[-1][elem], 0)
                    es_count = tf.tensor_scatter_nd_update(es_count, [[m_ind]], [0])
                else:
                    es_count = tf.tensor_scatter_nd_add(es_count, [[m_ind]], [1])
                    if es_count[m_ind] >= patience:
                        print("Training terminated for model {:d} at epoch {:d}.".format(m_ind, epoch))
                        #self.model[m_ind].load_weights(file_name + "_" + str(m_ind))
                        self.model[m_ind].trainable_parameters = temp_params[m_ind]
                        self.stack.pop(elem)
                        variables_list = self.get_variables_list()
                        if len(self.stack) == 0:
                            break
            '''
            # Update progress if desired.
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix(
                    ordered_dict={
                        "train loss":train_losses[-1].numpy(),
                        "val loss":val_losses[-1].numpy(),
                        "patience counter":es_count.numpy()},
                    refresh=True)
        self.weighting = tf.exp(-tf.concat(temp_weighting, 0))
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

    def get_variables_list(self):
        variable_list = []
        for i in self.stack:
            for variable in self.model[i].trainable_variables:
                variable_list.append(variable)
        return variable_list

    @tf.function
    def log_prob(self, data, conditional=None):
        """
        log probability, returns log density ln P(d | \theta)
        :param data: data vectors to evaluate density at
        :param parameters: (conditional) input parameters to evaluate density at
        """
        return tf.stack([
            self.model[element].log_prob(data, conditional=conditional)
            for element in self.stack], 0)

    @tf.function
    def weighted_log_prob(self, data, conditional=None):
        return tf.math.log(self.weighted_prob(data, conditional=conditional))

    @tf.function
    def prob(self, data, conditional=None):
        """
        probability, returns density P(d | \theta)
        :param data: data vectors to evaluate density at
        :param parameters: (conditional) input parameters to evaluate density at
        """
        return tf.stack([
            self.model[element].prob(data, conditional=conditional)
            for element in self.stack], 0)

    @tf.function
    def weighted_prob(self, data, conditional=None):
        return tf.reduce_sum(
            tf.multiply(self.weighting,
                        self.prob(data, conditional=conditional)))

    @tf.function
    def sample(self, n=None, conditional=None):
        if n is None:
            n = 1
        return tf.stack([
            self.model[element].sample(n, conditional=conditional)
            for element in self.stack], 0)

    @tf.function
    def weighted_sample(self, n=None, conditional=None):
        """
        sample, returns samples {d} from P(d | \theta) for some input values of \theta
        :param parameters: (conditional) input parameters to draw samples at
        :param n: number of samples to draw (for each parameter set input)
        """
        if n is None:
            n = 1
        samples = self.sample(n, conditional=None)
        return self.variance(samples)

    @tf.function
    def log_posterior(self, data, conditional=None):
        return tf.add(
            self.log_prob(data, conditional=conditional),
            tf.cast(self.prior.log_prob(conditional), dtype=self.dtype))

    @tf.function
    def weighted_log_posterior(self, data, conditional=None):
        data = tf.cast(data, dtype=self.dtype)
        conditional = tf.cast(conditional, dtype=self.dtype)
        return tf.add(self.weighted_log_prob(data, conditional=conditional),
                      tf.cast(self.prior.log_prob(conditional), dtype=self.dtype))

    @tf.function
    def geometric_mean(self, data, conditional=None):
        half = tf.cast(0.5, dtype=self.dtype)
        two = tf.cast(2., dtype=self.dtype)
        data = tf.cast(data, dtype=self.dtype)
        conditional = tf.cast(conditional, dtype=self.dtype)
        return tf.multiply(half,
                           tf.add(self.weighted_log_prob(data, conditional=conditional),
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

    '''
    def retile(self, data, conditional=None):
        data_shape = data.shape[0]
        conditional_shape = conditional.shape
        if data_shape != conditional_shape:
            data_ndims = len(data.shape)
            multiples = [1 for i in range(data_ndims)]
            multiples[0] = conditional_shape
            data = tf.tile(data, multiples)
            conditional = tf.repeat(conditional, data_shape, 0)
        return data, conditional, data_shape, conditional_shape

    def reshape(self, value, data_shape, conditional_shape):
        value_shape = value.shape
        return tf.reshape(value, [data_shape, conditional_shape] + value_shape)
    '''

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


def ConditionalMaskedAutoregressiveFlow(n_parameters, n_data, n_hidden=[50,50],
                                        activation=tf.keras.activations.tanh, all_layers=True):
    """
    Conditional Masked Autoregressive Flow.
    """
    return tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=n_hidden,
                activation=activation,
                event_shape=[n_data],
                conditional=True,
                conditional_shape=[n_parameters],
                conditional_input_all_layers=True)),
        event_shape=[n_data])

'''
class MixtureDensityNetwork(Train):
    """
    Implements a gaussian Mixture Density Network for modeling a conditional density p(d|\theta) (d="data", \theta="parameters")
    """
    def __init__(self, n_parameters, n_data, n_components=3, n_hidden=[50,50], activation=tf.keras.activations.tanh, dtype=tf.float32):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs (ie dimensionality of distribution you're parameterizing)
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param activation: activation function for network
        :param dtype: tensorflow type
        """
        super(MixtureDensityNetwork, self).__init__()

        # dimension of data and parameter spaces
        self.n_parameters = n_parameters
        self.n_data = n_data

        # number of mixture components and network architecture
        self.n_components = n_components
        self.n_loc = self.n_data
        self.n_scale = self.n_data * (self.n_data + 1) // 2
        self.n_output = (self.n_loc + self.n_scale + 1) * self.n_components

        # required size of output layer for a Gaussian mixture density network
        self.n_hidden = n_hidden
        self.activation = activation
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_output]

        self._network = self.build_network()

    def build_network(self):
        """
        Individual network constructor. Builds a single mixture of Gaussians.
        """
        layers = [tf.keras.layers.Input((self.n_parameters,))]
        for layer, size in enumerate(self.architecture[1:]):
            layers.append(tf.keras.layers.Dense(size)(layers[-1]))
            if layer + 1 < len(self.architecture):
                layers.append(tf.keras.layers.Activation(self.activation)(layers[-1]))
        amplitude, loc_, scale_ = tf.keras.layers.Lambda(lambda x: tf.split(x, [self.n_components,
                                                                                self.n_components * self.n_loc,
                                                                                self.n_components * self.n_scale],
                                                                          axis=-1))(layers[-1])
        loc = tf.keras.layers.Lambda(lambda x: tf.split(x, self.n_components, axis=-1))(loc_),
        scale = tf.keras.layers.Lambda(lambda x: tf.split(x, self.n_components, axis=-1))(scale_)
        model = tf.keras.models.Model(inputs=layers[0], outputs=[amplitude, loc, scale])
        return model

    def log_prob(self, x, **kwargs):
        with tf.name_scope(self.name or 'MDN_call'):
            if "conditional_input" not in kwargs.keys():
                raise ValueError('`conditional_input` must be passed to'
                                 ' bijector_kwargs.')
            conditional = kwargs["conditional_input"]
            if tensorshape_util.rank(conditional.shape) == 1:
                conditional = conditional[tf.newaxis, ...]
            amplitude, loc, scale = self._network(conditional)

            print(amplitude)
            return tfp.distributions.MixtureSameFamily(
                mixture_distribution=tfp.distributions.Categorical(
                    logits=None,
                    probs=tf.transpose(amplitude),
                    validate_args=False,
                    allow_nan_stats=True,
                    name='Categorical'),
                components_distribution=tfp.distributions.MultivariateNormalTriL(
                    loc=loc,
                    scale_tril=tfp.math.fill_triangular(scale),
                    validate_args=False,
                    allow_nan_stats=True,
                    name='MultivariateNormalTriL'),
                reparameterize=False,
                validate_args=False,
                allow_nan_stats=True,
                name='MixtureSameFamily').log_prob(x)

    def prob(self, x, **kwargs):
        with tf.name_scope(self.name or 'MDN_call'):
            if "conditional_input" not in kwargs.keys():
                raise ValueError('`conditional_input` must be passed to'
                                 ' bijector_kwargs.')
            conditional = kwargs["conditional_input"]
            if tensorshape_util.rank(conditional.shape) == 1:
                conditional = conditional[tf.newaxis, ...]
            amplitude, loc, scale = self._network(conditional)
            return tfp.distributions.MixtureSameFamily(
                mixture_distribution=tfp.distributions.Categorical(
                    logits=amplitude,
                    probs=None,
                    dtype=tf.int32,
                    validate_args=False,
                    allow_nan_stats=True,
                    name='Categorical'),
                components_distribution=tfp.distributions.MultivariateNormalTriL(
                    loc=loc,
                    scale_tril=scale,
                    validate_args=False,
                    allow_nan_stats=True,
                    name='MultivariateNormalTriL'),
                reparameterize=False,
                validate_args=False,
                allow_nan_stats=True,
                name='MixtureSameFamily').prob

    def sample(self, n, **kwargs):
            with tf.name_scope(self.name or 'MDN_call'):
                if "conditional_input" not in kwargs.keys():
                    raise ValueError('`conditional_input` must be passed to'
                                     ' bijector_kwargs.')
                conditional = kwargs["conditional_input"]
                if tensorshape_util.rank(conditional.shape) == 1:
                    conditional = conditional[tf.newaxis, ...]
                amplitude, loc, scale = self._network(conditional)
                return tfp.distributions.MixtureSameFamily(
                    mixture_distribution=tfp.distributions.Categorical(
                        logits=amplitude,
                        probs=None,
                        dtype=tf.int32,
                        validate_args=False,
                        allow_nan_stats=True,
                        name='Categorical'),
                    components_distribution=tfp.distributions.MultivariateNormalTriL(
                        loc=loc,
                        scale_tril=scale,
                        validate_args=False,
                        allow_nan_stats=True,
                        name='MultivariateNormalTriL'),
                    reparameterize=False,
                    validate_args=False,
                    allow_nan_stats=True,
                    name='MixtureSameFamily').sample(n)
'''
