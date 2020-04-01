import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import pickle
import os

tfd = tfp.distributions
tfb = tfp.bijectors

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
                        "train loss":["{0:.3g}".format(this_train_loss.numpy()[i]) for i in range(len(this_train_loss.numpy()))],
                        "val loss":["{0:.3g}".format(this_val_loss.numpy()[i]) for i in range(len(this_train_loss.numpy()))],
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


def ConditionalMADE(n_parameters, n_data, n_hidden=[50,50],
                                        activation=tf.keras.layers.LeakyReLU(0.01), all_layers=True):
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

def ConditionalMaskedAutoregressiveFlow(n_parameters, n_data, n_mades=5, n_hidden=[50,50],
                                        activation=tf.keras.layers.LeakyReLU(0.01), all_layers=True):
    """
    Conditional Masked Autoregressive Flow.
    """

    # construct stack of MADEs
    bijector = tfb.Chain([tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=n_hidden,
                activation=activation,
                event_shape=[n_data],
                conditional=True,
                conditional_shape=[n_parameters],
                conditional_input_all_layers=True)) for i in range(n_mades)])

    return tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=bijector,
        event_shape=[n_data])


class MixtureDensityNetwork(tfd.Distribution):
    """
    Implements a gaussian Mixture Density Network for modeling a conditional density p(d|\theta) (d="data", \theta="parameters")
    """
    def __init__(self, n_parameters, n_data, n_components=3, n_hidden=[50,50], activation=tf.keras.layers.LeakyReLU(0.01), dtype=tf.float32, reparameterization_type=None, validate_args=False, allow_nan_stats=True):
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

        self._network = self.build_network()

    def build_network(self):
        """
        Individual network constructor. Builds a single mixture of Gaussians.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                self.architecture[layer + 1],
                input_shape=(size,),
                activation=self.activation)
            for layer, size in enumerate(self.architecture[:-1])])
        model.add(
            tf.keras.layers.Dense(
                tfp.layers.MixtureSameFamily.params_size(
                    self.n_components, 
                    component_params_size=tfp.layers.MultivariateNormalTriL.params_size(self.n_data))))
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



class ConditionalGaussianMADEJustin(tf.keras.Model):
    """
    Implements a gaussian MADE for modeling a conditional density p(d|\theta) (d="data", \theta="parameters")
    """
    
    def __init__(self, n_parameters, n_data, n_hidden=[50,50], activations=[tf.tanh, tf.tanh], output_order='sequential', mode='sequential'):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs (ie dimensionality of distribution you're parameterizing)
        :param n_hidden: list with number of hidden units for each hidden layer
        :param activations: activation functions for each hidden layer
        """
        
        # super
        super(ConditionalGaussianMADE, self).__init__()

        # dimension of data and parameter spaces
        self.n_parameters = n_parameters
        self.n_data = n_data
        
        # network architecture
        self.n_hidden = n_hidden
        self.activations = activations
        
        # mode
        self.mode = mode
        
        # degrees
        degrees = self.create_degrees(output_order)
        
        # masks
        self.Ms, self.Mmp = self.create_masks(degrees)
        
        # weights and biases
        self.Ws = []
        self.bs = []

        self.n_units = np.concatenate(([self.n_data], self.n_hidden))
        
        self.Wx = tf.Variable(tf.random.normal([self.n_parameters, self.n_hidden[0]], 0., np.sqrt(1./(self.n_parameters + 1))), name="Wx" )

        for l, (N0, N1) in enumerate(zip(self.n_units[:-1], self.n_units[1:])):

            W = tf.Variable(tf.random.normal([N0, N1], 0., np.sqrt(1./(1+N0))), name="W"+str(l))
            b = tf.Variable(tf.zeros([1, N1]), name="b"+str(l))
            self.Ws.append(W)
            self.bs.append(b)

        self.Wm = tf.Variable(tf.random.normal([self.n_units[-1], self.n_data], 0., np.sqrt(1./(self.n_units[-1] + 1))), name="Wm" )
        self.Wp = tf.Variable(tf.random.normal([self.n_units[-1], self.n_data], 0., np.sqrt(1./(self.n_units[-1] + 1))), name="Wp" )
        self.bm = tf.Variable(tf.zeros([1, self.n_data]), name="bm")
        self.bp = tf.Variable(tf.zeros([1, self.n_data]), name="bp")
        
        # output order
        self.output_order = degrees[0]
    
    @tf.function
    def call(self, data, parameters):
        
        # pass through layers of the network:
        
        # first layer
        h = self.activations[0](tf.matmul(parameters, self.Wx) + tf.matmul(data, self.Ms[0] * self.Ws[0]) + self.bs[0])
        
        # subsequent hidden layers
        for l, (M, W, b) in enumerate(zip(self.Ms[1:], self.Ws[1:], self.bs[1:])):
            h = self.activations[l+1](tf.matmul(h, M * W) + b)
        
        # output means
        mu = tf.add(tf.matmul(h, self.Mmp * self.Wm), self.bm)

        # output log precisions
        logp = tf.add(tf.matmul(h, self.Mmp * self.Wp), self.bp)
        
        # return shift and log_scale
        return mu, logp
    
    @tf.function
    def u(self, data, parameters):
        
        # shift and log_scale
        mu, logp = self.call(data, parameters)
        
        # random numbers
        u = tf.exp(0.5 * logp) * (data - mu)
        
        return u
    
    @tf.function
    def log_prob(self, data, parameters):
        
        # shift and log_scale
        mu, logp = self.call(data, parameters)
        
        # random numbers
        u = tf.exp(0.5 * logp) * (data - mu)
        
        # log density
        return tf.multiply(-0.5, self.n_data * np.log(2 * np.pi) + tf.reduce_sum(u**2 - logp, axis=1))
    
    @tf.function
    def prob(self, data, parameters):
        
        # shift and log_scale
        mu, logp = self.call(data, parameters)
        
        # random numbers
        u = tf.exp(0.5 * logp) * (data - mu)
        
        # log density
        return tf.exp(tf.multiply(-0.5, self.n_data * np.log(2 * np.pi) + tf.reduce_sum(u**2 - logp, axis=1)))

    
    def create_degrees(self, input_order):
        """
        Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
        degree less than d.
        :param n_hidden: a list with the number of hidden units
        :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
        :return: list of degrees
        """

        degrees = []

        # create degrees for inputs
        if isinstance(input_order, str):

            if input_order == 'random':
                degrees_0 = np.arange(1, self.n_data + 1)
                rng.shuffle(degrees_0)

            elif input_order == 'sequential':
                degrees_0 = np.arange(1, self.n_data + 1)

            else:
                raise ValueError('invalid output order')

        else:
            input_order = np.array(input_order)
            assert np.all(np.sort(input_order) == np.arange(1, self.n_data + 1)), 'invalid input order'
            degrees_0 = input_order
        degrees.append(degrees_0)

        # create degrees for hiddens
        if self.mode == 'random':
            for N in self.n_hidden:
                min_prev_degree = min(np.min(degrees[-1]), self.n_data - 1)
                degrees_l = rng.randint(min_prev_degree, self.n_data, N)
                degrees.append(degrees_l)

        elif self.mode == 'sequential':
            for N in self.n_hidden:
                degrees_l = np.arange(N) % max(1, self.n_data - 1) + min(1, self.n_data - 1)
                degrees.append(degrees_l)

        else:
            raise ValueError('invalid mode')

        return degrees

    def create_masks(self, degrees):
        """
        Creates the binary masks that make the connectivity autoregressive.
        :param degrees: a list of degrees for every layer
        :return: list of all masks, as theano shared variables
        """

        Ms = []

        for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
            M = d0[:, np.newaxis] <= d1
            M = tf.constant(M, dtype=dtype, name='M' + str(l+1))
            Ms.append(M)

        Mmp = degrees[-1][:, np.newaxis] < degrees[0]
        Mmp = tf.constant(Mmp, dtype=dtype, name='Mmp')

        return Ms, Mmp


class ConditionalMaskedAutoregressiveFlowJustin(tf.keras.Model):
    """
    Conditional Masked Autoregressive Flow.
    """

    def __init__(self, n_parameters, n_data, n_hidden=[50,50], activations=[tf.tanh, tf.tanh], n_mades=5,
                 output_order='sequential', mode='sequential'):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs
        :param n_hidden: list with number of hidden units for each hidden layer
        :param activations: tensorflow activation functions
        :param n_mades: number of mades in the flow
        :param output_order: order of outputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        """
        
        # super
        super(ConditionalMaskedAutoregressiveFlow, self).__init__()

        # save input arguments
        self.n_parameters = n_parameters
        self.n_data = n_data
        self.n_hidden = n_hidden
        self.activations = activations
        self.n_mades = n_mades
        self.mode = mode

        self.mades = []

        # create some MADEs
        for i in range(n_mades):
            with tf.name_scope("made_" + str(i+1)):
                self.mades.append(ConditionalGaussianMADE(n_parameters, n_data, n_hidden, activations, output_order, mode))
            output_order = output_order if output_order is 'random' else self.mades[-1].output_order[::-1]

        self.output_order = self.mades[0].output_order
            
    @tf.function
    def call(self, data, parameters):
        
        u = [data]
        
        # loop through the MADEs
        for i in range(self.n_mades):
            # update state
            u.append(self.mades[i].u(u[-1], parameters))
        
        # transformed vector
        return u[-1]
    
    @tf.function
    def log_prob(self, data, parameters):
        
        u = [data]
        logdet_dudy = tf.zeros(data.shape[0])
        
        # loop through the MADEs
        for i in range(self.n_mades):

            # update jacobian
            _, logp = self.mades[i](u[-1], parameters)
            
            # update state
            u.append(self.mades[i].u(u[-1], parameters))
            
            logdet_dudy = logdet_dudy + 0.5 * tf.reduce_sum(logp, axis=1)
        
        # log density
        return -0.5 * self.n_data * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(u[-1] ** 2, axis=1) + logdet_dudy

    @tf.function
    def prob(self, data, parameters):
        
        u = [data]
        logdet_dudy = tf.zeros(data.shape[0])
        
        # loop through the MADEs
        for i in range(self.n_mades):

            # update jacobian
            _, logp = self.mades[i](u[-1], parameters)
            logdet_dudy = logdet_dudy + 0.5 * tf.reduce_sum(logp, axis=1)

            # update state
            u.append(self.mades[i].u(u[-1], parameters))
    
        # likelihood
        return tf.exp(-0.5 * self.n_data * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(u[-1] ** 2, axis=1) + logdet_dudy)
