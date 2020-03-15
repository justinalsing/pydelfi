#IN PROGRESS...


import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import tensorshape_util
from pydelfi.train import Train

tfd = tfp.distributions
tfb = tfp.bijectors

tf.keras.backend.set_floatx('float32')

class ConditionalMaskedAutoregressiveFlow(Train):
    """
    Conditional Masked Autoregressive Flow.
    """

    def __init__(self, n_parameters, n_data, n_hidden=[50,50], activation=tf.keras.activations.tanh,
                 all_layers=True, dtype=tf.float32, **kwargs):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs
        :param n_hidden: list with number of hidden units for each hidden layer
        :param activation: tensorflow activation function
        """
        # super
        super(ConditionalMaskedAutoregressiveFlow, self).__init__(**kwargs)

        # save input arguments
        self.n_parameters = n_parameters
        self.n_data = n_data
        self.n_hidden = n_hidden
        self.activation = activation
        self.all_layers = all_layers
        self.dtype = dtype
        
        self.model = tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=0., scale=1.),
            bijector=tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                    params=2, 
                    hidden_units=self.n_hidden, 
                    activation=self.activation,
                    event_shape=[self.n_data],
                    conditional=True,
                    conditional_shape=[self.n_parameters],
                    conditional_input_all_layers=self.all_layers)),
            event_shape=[self.n_data])
        
    def log_prob(self, data, parameters):
        parameters = tf.convert_to_tensor(parameters, dtype=self.dtype)
        return self.model.log_prob(data, bijector_kwargs={"conditional_input": parameters})
    
    def prob(self, data, parameters):
        parameters = tf.convert_to_tensor(parameters, dtype=self.dtype)
        return self.model.prob(data, bijector_kwargs={"conditional_input": parameters})
    
    def sample(self, n, parameters):
        parameters = tf.convert_to_tensor(parameters, dtype=self.dtype)
        return self.model.sample(n, bijector_kwargs={"conditional_input": parameters})
            
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
