import numpy as np
import numpy.random as rng
import tensorflow as tf
dtype = tf.float32

class ConditionalGaussianMade:
    """
    Implements a Made, where each conditional probability is modelled by a single gaussian component.
    """

    def __init__(self, n_parameters, n_data, n_hiddens, act_fun, output_order='sequential', mode='sequential', input_parameters=None, input_data=None, logpdf=None):
        """
        Constructor.
        :param n_inputs: number of (conditional) inputs
        :param n_outputs: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: tensorflow activation function
        :param output_order: order of outputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: tensorflow placeholder to serve as input; if None, a new placeholder is created
        :param output: tensorflow placeholder to serve as output; if None, a new placeholder is created
        """

        # save input arguments
        self.n_parameters = n_parameters
        self.n_data = n_data
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.mode = mode

        # create network's parameters
        degrees = self.create_degrees(output_order)
        Ms, Mmp = self.create_masks(degrees)
        Wx, Ws, bs, Wm, bm, Wp, bp = self.create_weights_conditional(None)
        self.parms = [Wx] + Ws + bs + [Wm, bm, Wp, bp]
        self.output_order = degrees[0]

        # activation function
        f = self.act_fun

        # input matrices
        self.parameters = tf.placeholder(dtype=dtype,shape=[None,n_parameters],name='parameters') if input_parameters is None else input_parameters
        self.data = tf.placeholder(dtype=dtype,shape=[None,n_data],name='data') if input_data is None else input_data
        self.logpdf = tf.placeholder(dtype=dtype,shape=[None,1],name='logpdf') if logpdf is None else logpdf

        # feedforward propagation
        h = f(tf.matmul(self.parameters, Wx) + tf.matmul(self.data, Ms[0] * Ws[0]) + bs[0],name='h1')
        for l, (M, W, b) in enumerate(zip(Ms[1:], Ws[1:], bs[1:])):
            h = f(tf.matmul(h, M * W) + b,name='h'+str(l + 2))

        # output means
        self.m = tf.add(tf.matmul(h, Mmp * Wm), bm, name='m')

        # output log precisions
        self.logp = tf.add(tf.matmul(h, Mmp * Wp), bp, name='logp')

        # random numbers driving made
        self.u = tf.exp(0.5 * self.logp) * (self.data - self.m)

        # log likelihoods
        self.L = tf.multiply(-0.5,self.n_data * np.log(2 * np.pi) + \
                     tf.reduce_sum(self.u ** 2 - self.logp, axis=1,keepdims=True),name='L')

        # train objective
        self.trn_loss = -tf.reduce_mean(self.L,name='trn_loss')
        self.reg_loss = tf.losses.mean_squared_error(self.L, self.logpdf)

    def create_degrees(self, input_order):
        """
        Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
        degree less than d.
        :param n_hiddens: a list with the number of hidden units
        :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
        :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
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
            for N in self.n_hiddens:
                min_prev_degree = min(np.min(degrees[-1]), self.n_data - 1)
                degrees_l = rng.randint(min_prev_degree, self.n_data, N)
                degrees.append(degrees_l)

        elif self.mode == 'sequential':
            for N in self.n_hiddens:
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
    
    def create_weights_conditional(self, n_comps):
        """
        Creates all learnable weight matrices and bias vectors.
        :param n_comps: number of gaussian components
        :return: weights and biases, as tensorflow variables
        """

        Ws = []
        bs = []

        n_units = np.concatenate(([self.n_data], self.n_hiddens))
        
        Wx = tf.get_variable("Wx", [self.n_parameters, self.n_hiddens[0]], initializer = tf.random_normal_initializer(0., np.sqrt(1./(self.n_parameters + 1))) )

        for l, (N0, N1) in enumerate(zip(n_units[:-1], n_units[1:])):

            W = tf.get_variable("W"+str(l), [N0, N1], initializer = tf.random_normal_initializer(0., np.sqrt(1./(1+N0))) )
            b = tf.get_variable("b"+str(l), [1, N1], initializer = tf.constant_initializer(0.0) )
            Ws.append(W)
            bs.append(b)

        if n_comps is None:

            Wm = tf.get_variable("Wm", [n_units[-1], self.n_data], initializer = tf.random_normal_initializer(0., np.sqrt(1./(n_units[-1] + 1))) )
            Wp = tf.get_variable("Wp", [n_units[-1], self.n_data], initializer = tf.random_normal_initializer(0., np.sqrt(1./(n_units[-1] + 1))) )
            bm = tf.get_variable("bm", [1, self.n_data], initializer = tf.constant_initializer(0.0) )
            bp = tf.get_variable("bp", [1, self.n_data], initializer = tf.constant_initializer(0.0) )

            return Wx, Ws, bs, Wm, bm, Wp, bp

        else:

            Wm = tf.get_variable("Wm", [n_units[-1], self.n_data, n_comps], initializer = tf.random_normal_initializer(0., np.sqrt(1./(n_units[-1] + 1))) )
            Wp = tf.get_variable("Wp", [n_units[-1], self.n_data, n_comps], initializer = tf.random_normal_initializer(0., np.sqrt(1./(n_units[-1] + 1))) )
            Wa = tf.get_variable("Wa", [n_units[-1], self.n_data, n_comps], initializer = tf.random_normal_initializer(0., np.sqrt(1./(n_units[-1] + 1))) )
            bm = tf.get_variable("bm", [self.n_data, n_comps], initializer = tf.random_normal_initializer() )
            bp = tf.get_variable("bp", [self.n_data, n_comps], initializer = tf.random_normal_initializer() )
            ba = tf.get_variable("ba", [self.n_data, n_comps], initializer = tf.random_normal_initializer() )

            return Wx, Ws, bs, Wm, bm, Wp, bp, Wa, ba

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


class ConditionalMaskedAutoregressiveFlow:
    """
    Conditional Masked Autoregressive Flow.
    """

    def __init__(self, n_parameters, n_data, n_hiddens, act_fun, n_mades,
                 output_order='sequential', mode='sequential', input_parameters=None, input_data=None, logpdf=None, index=1):
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
        self.logpdf = tf.placeholder(dtype=dtype,shape=[None,1],name='logpdf') if logpdf is None else logpdf
        
        self.parms = []

        self.mades = []
        self.bns = []
        self.u = self.data
        self.logdet_dudy = 0.0

        for i in range(n_mades):
            
            # create a new made
            with tf.variable_scope('nde_' + str(index) + '_made_' + str(i + 1)):
                made = ConditionalGaussianMade(n_parameters, n_data, n_hiddens, act_fun,
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
        self.reg_loss = tf.losses.mean_squared_error(self.L, self.logpdf)

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
        self.logpdf = tf.placeholder(dtype=dtype,shape=[None,1],name='logpdf') if logpdf is None else logpdf
        
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
        self.L = tf.log(tf.reduce_sum(tf.exp(-0.5*tf.reduce_sum(tf.square(tf.einsum("ijlk,ijk->ijl", self.Sigma, tf.subtract(tf.expand_dims(self.data, 1), self.mu))), 2) + tf.log(self.alpha) + tf.log(self.det) - self.n_data*np.log(2. * np.pi) / 2.), 1, keepdims=True) + 1e-37, name = "L")

        # Objective loss function
        self.trn_loss = -tf.reduce_mean(self.L, name = "trn_loss")
        self.reg_loss = tf.losses.mean_squared_error(self.L, self.logpdf)

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



