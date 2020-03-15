import tensorflow as tf

class EnsembleMethods():
    def __init__(self, model, stack=1, **kwargs):
        super(EnsembleMethods, self).__init__()

        self.n_stack = stack
        self.error_stack = None
        self.set_stack()
            
        # model weighting
        self.weighting = tf.ones((self.n_stack,), name="weighting")
        
        self.model = [model for element in self.stack]
        
        super(Train, self).__init__(**kwargs)
    
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
                
    def log_prob(self, data, parameters):
        """
        log probability, returns log density ln P(d | \theta)
        :param data: data vectors to evaluate density at
        :param parameters: (conditional) input parameters to evaluate density at
        """
        return tf.stack([
            self.model[element].log_prob(data, parameters) 
            for element in self.stack], 0)

    def prob(self, data, parameters):
        """
        probability, returns density P(d | \theta)
        :param data: data vectors to evaluate density at
        :param parameters: (conditional) input parameters to evaluate density at
        """
        return tf.stack([
            self.model[element].prob(data, parameters)
            for element in self.stack], 0)

    def sample(self, parameters, n):
        """
        sample, returns samples {d} from P(d | \theta) for some input values of \theta
        :param parameters: (conditional) input parameters to draw samples at
        :param n: number of samples to draw (for each parameter set input)
        """
        weighted_sum = tf.reduce_sum(self.weighting)
        samples = tf.stack([
            self.model[element].sample(n, paramaters) 
            for element in self.stack], 0)
        mean = tf.divide(
            tf.einsum("i...,i->...",
                samples,
                self.weighting),
            weighted_sum)
        variance = tf.divide(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        samples,
                        tf.expand_dims(mean, 0))),
                0),
            weighted_sum)
        return mean, variance