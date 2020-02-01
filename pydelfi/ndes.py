import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

tf.keras.backend.set_floatx('float32')
dtype = tf.float32

class MixtureDensityNetwork(tf.keras.Model):
    """
    Implements a gaussian Mixture Density Network for modeling a conditional density p(d|\theta) (d="data", \theta="parameters")
    """
    
    def __init__(self, n_parameters, n_data, n_components=3, n_hidden=[50,50], activations=['tanh', 'tanh']):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs (ie dimensionality of distribution you're parameterizing)
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param activations: activation functions for each hidden layer
        """
        
        # super
        super(MixtureDensityNetwork, self).__init__()

        # dimension of data and parameter spaces
        self.n_parameters = n_parameters
        self.n_data = n_data
        
        # number of mixture components and network architecture
        self.n_components = n_components
        self.n_output = int((self.n_data + self.n_data * (self.n_data + 1) / 2 + 1)*self.n_components) # required size of output layer for a Gaussian mixture density network
        self.n_hidden = n_hidden
        self.activations = activations + [None]
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_output]
        
        # dense network layers
        self.network_layers = []
        for i in range(len(self.architecture)-1):
            self.network_layers.append(tfkl.Dense(self.architecture[i+1], input_shape=[self.architecture[i]], activation=self.activations[i]))
        
        # (output) mixture density layer
        self.network_layers.append(tfpl.MixtureSameFamily(self.n_components, tfpl.MultivariateNormalTriL(self.n_data)))
    
    def call(self, parameters):
        """
        Call. Returns callable tfp gaussian mixture model, for the conditional parameters given as inputs.
        :param parameters: (conditional) input parameters
        """
        
        # pass through layers of the network
        outputs = [parameters]
        for i in range(0, len(self.network_layers)):
            outputs.append(self.network_layers[i](outputs[i]))
        
        return outputs[-1]
    
    def log_prob(self, data, parameters):
        """
        log probability, returns log density ln P(d | \theta)
        :param data: data vectors to evaluate density at
        :param parameters: (conditional) input parameters to evaluate density at
        """
        
        return self.call(parameters).log_prob(data)
    
    def prob(self, data, parameters):
        """
        probability, returns density P(d | \theta)
        :param data: data vectors to evaluate density at
        :param parameters: (conditional) input parameters to evaluate density at
        """
        
        return self.call(parameters).prob(data)

    def sample(self, parameters, n):
        """
        sample, returns samples {d} from P(d | \theta) for some input values of \theta
        :param parameters: (conditional) input parameters to draw samples at
        :param n: number of samples to draw (for each parameter set input)
        """
        
        return self.call(parameters).sample(n)




class ConditionalGaussianMADE(tf.keras.Model):
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
    
    def u(self, parameters, data):
        
        # shift and log_scale
        mu, logp = self.call(data, parameters)
        
        # random numbers
        u = tf.exp(0.5 * logp) * (data - mu)
        
        return u
    
    def log_prob(self, data, parameters):
        
        # shift and log_scale
        mu, logp = self.call(data, parameters)
        
        # random numbers
        u = tf.exp(0.5 * logp) * (data - mu)
        
        # log density
        return tf.multiply(-0.5, self.n_data * np.log(2 * np.pi) + tf.reduce_sum(u**2 - logp, axis=1, keepdims=True))
    
    def prob(self, data, parameters):
        
        # shift and log_scale
        mu, logp = self.call(data, parameters)
        
        # random numbers
        u = tf.exp(0.5 * logp) * (data - mu)
        
        # log density
        return tf.exp(tf.multiply(-0.5, self.n_data * np.log(2 * np.pi) + tf.reduce_sum(u**2 - logp, axis=1, keepdims=True)))

    
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



class ConditionalMaskedAutoregressiveFlow:
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
            
            self.mades.append(ConditionalGaussianMADE(n_parameters, n_data, n_hidden, activations, output_order, mode))
            output_order = output_order if output_order is 'random' else self.mades[-1].output_order[::-1]

        self.output_order = self.mades[0].output_order

    def call(self, data, parameters):
        
        u = [data]
        
        # loop through the MADEs
        for i in range(self.n_mades):
            # update state
            u.append(self.mades[i].u(u[-1], parameters))
        
        # transformed vector
        return u[-1]
    
    def log_prob(self, data, parameters):
        
        u = [data]
        logdet_dudy = tf.zeros(1)
        
        # loop through the MADEs
        for i in range(self.n_mades):
            # update state
            u.append(self.mades[i].u(u[-1], parameters))
            
            # update jacobian
            _, logp = self.mades[i](u[-1], parameters)
            logdet_dudy += 0.5 * tf.reduce_sum(logp, axis=1, keepdims=True)
        
        # log density
        return tf.add(-0.5 * self.n_data * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(u[-1] ** 2, axis=1, keepdims=True), logdet_dudy)

    def prob(self, data, parameters):
        
        u = [data]
        logdet_dudy = tf.zeros(1)
        
        # loop through the MADEs
        for i in range(self.n_mades):
            # update state
            u.append(self.mades[i].u(u[-1], parameters))
            
            # update jacobian
            _, logp = self.mades[i](u[-1], parameters)
            logdet_dudy += 0.5 * tf.reduce_sum(logp, axis=1, keepdims=True)
    
        # likelihood
        return tf.exp(tf.add(-0.5 * self.n_data * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(u[-1] ** 2, axis=1, keepdims=True), logdet_dudy))
