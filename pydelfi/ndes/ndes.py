import numpy as np
import numpy.random as rng
import tensorflow as tf
import pydelfi.ndes.mades

dtype = tf.float32

class ConditionalMaskedAutoregressiveFlow:
    """
    Conditional Masked Autoregressive Flow.
    """

    def __init__(self, n_parameters, n_data, n_hiddens, act_fun, n_mades,
                 output_order='sequential', mode='sequential', input_parameters=None, input_data=None, index=1):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: tensorflow activation function
        :param n_mades: number of mades in the flow
        :param output_order: order of outputs of last made
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input_parameters: tensorflow placeholder to serve as input for the parameters part of the training data; if None, a new placeholder is created
        :param input_data: tensorflow placeholder to serve as input for data-realizations part of the training data; if None, a new placeholder is created
        :param index: index of the NDE; crucial when using ensembles of NDEs to keep their scopes separate
        """

        # save input arguments
        self.n_parameters = n_parameters
        self.n_data = n_data
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_mades = n_mades
        self.mode = mode

        self.parameters = tf.placeholder(dtype=dtype,shape=[None,n_parameters],name='parameters') if input_parameters is None else input_parameters
        self.data = tf.placeholder(dtype=dtype,shape=[None,n_data],name='data') if input_data is None else input_data
        self.parms = []

        self.mades = []
        self.bns = []
        self.u = self.data
        self.logdet_dudy = 0.0

        for i in range(n_mades):
            
            # create a new made
            with tf.variable_scope('nde_' + str(index) + '_made_' + str(i + 1)):
                made = pydelfi.ndes.mades.ConditionalGaussianMade(n_parameters, n_data, n_hiddens, act_fun,
                                                 output_order, mode, self.parameters, self.u)
            self.mades.append(made)
            self.parms += made.parms
            output_order = output_order if output_order is 'random' else made.output_order[::-1]

            # inverse autoregressive transform
            self.u = made.u
            self.logdet_dudy += 0.5 * tf.reduce_sum(made.logp, axis=1,keepdims=True)

        self.output_order = self.mades[0].output_order

        # log likelihoods
        self.L = tf.add(-0.5 * n_data * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(self.u ** 2, axis=1,keepdims=True), self.logdet_dudy,name='L')

        # train objective
        self.trn_loss = -tf.reduce_mean(self.L,name='trn_loss')

    def eval(self, xy, sess, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param sess: tensorflow session where the graph is run
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """
        
        x, y = xy
        lprob = sess.run(self.L,feed_dict={self.parameters:x,self.data:y})[0]

        return lprob if log else np.exp(lprob)

class MixtureDensityNetwork:
    """
    Implements a Mixture Density Network for modeling p(y|x)
    """

    def __init__(self, n_parameters, n_data, n_components = 3, n_hidden=[50,50], activations=[tf.tanh, tf.tanh],
                 input_parameters=None, input_data=None, logpdf=None, index=1):
        """
        Constructor.
        :param n_parameters: number of (conditional) inputs
        :param n_data: number of outputs (ie dimensionality of distribution you're parameterizing)
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param activations: tensorflow activation functions for each hidden layer
        :param input: tensorflow placeholder to serve as input; if None, a new placeholder is created
        :param output: tensorflow placeholder to serve as output; if None, a new placeholder is created
        """
        
        # save input arguments
        self.n_parameters = n_parameters
        self.n_data = n_data
        self.M = n_components
        self.N = int((self.n_data + self.n_data * (self.n_data + 1) / 2 + 1)*self.M)
        self.n_hidden = n_hidden
        self.activations = activations
        
        self.parameters = tf.placeholder(dtype=dtype,shape=[None,self.n_parameters],name='parameters') if input_parameters is None else input_parameters
        self.data = tf.placeholder(dtype=dtype,shape=[None,self.n_data],name='data') if input_data is None else input_data
        self.logpdf = tf.placeholder(dtype=dtype,shape=[None],name='logpdf') if logpdf is None else logpdf
        
        # Build the layers of the network
        self.layers = [self.parameters]
        self.weights = []
        self.biases = []
        for i in range(len(self.n_hidden)):
            with tf.variable_scope('nde_' + str(index) + '_layer_' + str(i + 1)):
                if i == 0:
                    self.weights.append(tf.get_variable("weights", [self.n_parameters, self.n_hidden[i]], initializer = tf.random_normal_initializer(0., np.sqrt(2./self.n_parameters))))
                    self.biases.append(tf.get_variable("biases", [self.n_hidden[i]], initializer = tf.constant_initializer(0.0)))
                elif i == len(self.n_hidden) - 1:
                    self.weights.append(tf.get_variable("weights", [self.n_hidden[i], self.N], initializer = tf.random_normal_initializer(0., np.sqrt(2./self.n_hidden[i]))))
                    self.biases.append(tf.get_variable("biases", [self.N], initializer = tf.constant_initializer(0.0)))
                else:
                    self.weights.append(tf.get_variable("weights", [self.n_hidden[i], self.n_hidden[i + 1]], initializer = tf.random_normal_initializer(0., np.sqrt(2/self.n_hidden[i]))))
                    self.biases.append(tf.get_variable("biases", [self.n_hidden[i + 1]], initializer = tf.constant_initializer(0.0)))
            if i < len(self.n_hidden) - 1:
                self.layers.append(self.activations[i](tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1])))
            else:
                self.layers.append(tf.add(tf.matmul(self.layers[-1], self.weights[-1]), self.biases[-1]))

        # Map the output layer to mixture model parameters
        self.mu, self.sigma, self.alpha = tf.split(self.layers[-1], [self.M * self.n_data, self.M * self.n_data * (self.n_data + 1) // 2, self.M], 1)
        self.mu = tf.reshape(self.mu, (-1, self.M, self.n_data))
        self.sigma = tf.reshape(self.sigma, (-1, self.M, self.n_data * (self.n_data + 1) // 2))
        self.alpha = tf.nn.softmax(self.alpha)
        self.Sigma = tf.contrib.distributions.fill_triangular(self.sigma)
        self.Sigma = self.Sigma - tf.linalg.diag(tf.linalg.diag_part(self.Sigma)) + tf.linalg.diag(tf.exp(tf.linalg.diag_part(self.Sigma)))
        self.det = tf.reduce_prod(tf.linalg.diag_part(self.Sigma), axis=-1)

        self.mu = tf.identity(self.mu, name = "mu")
        self.Sigma = tf.identity(self.Sigma, name = "Sigma")
        self.alpha = tf.identity(self.alpha, name = "alpha")
        self.det = tf.identity(self.det, name = "det")
        
        # Log likelihoods
        self.L = tf.log(tf.reduce_sum(tf.exp(-0.5*tf.reduce_sum(tf.square(tf.einsum("ijlk,ijk->ijl", self.Sigma, tf.subtract(tf.expand_dims(self.data, 1), self.mu))), 2) + tf.log(self.alpha) + tf.log(self.det) - self.n_data*np.log(2. * np.pi) / 2.), 1) + 1e-37, name = "L")

        # Objective loss function
        self.trn_loss = -tf.reduce_mean(self.L, name = "trn_loss")

    def eval(self, xy, sess, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param sess: tensorflow session where the graph is run
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """
        
        x, y = xy
        lprob = sess.run(self.L,feed_dict={self.parameters:x,self.data:y})

        return lprob if log else np.exp(lprob)



