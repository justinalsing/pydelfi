import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

tf.keras.backend.set_floatx('float32')

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
