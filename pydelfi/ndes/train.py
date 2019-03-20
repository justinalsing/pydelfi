import tensorflow as tf
import numpy as np
import numpy.random as rng
import os
import tqdm

def isnotebook():
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
                    
class ConditionalTrainer():
    
    def __init__(self, model, optimizer=tf.train.AdamOptimizer, optimizer_arguments={}):
        """
            Constructor that defines the training operation.
            :param model: made/maf instance to be trained.
            :param optimizer: tensorflow optimizer class to be used during training.
            :param optimizer_arguments: dictionary of arguments for optimizer intialization.
            """
        
        self.model = model
        self.nb = isnotebook()
        self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss)

    """
    Training class for the conditional MADEs/MAFs classes using a tensorflow optimizer.
    """           
    def train(self, sess, train_data, validation_split = 0.1, epochs=1000, batch_size=100,
              patience=20, saver_name='tmp_model', progress_bar=True):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: a tuple/list of (X,Y) with training data where Y is conditioned on X.
        :param validation_split: percentage of training data randomly selected to be used for validation
        :param epochs: maximum number of epochs for training.
        :param batch_size: batch size of each batch within an epoch.
        :param early_stopping: number of epochs for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        """
        
        # Training data
        train_data_X, train_data_Y  = train_data
        train_idx = np.arange(train_data_X.shape[0])
        
        # validation data using p_val percent of the data
        rng.shuffle(train_idx)
        N = train_data_X.shape[0]
        val_data_X = train_data_X[train_idx[-int(validation_split*N):]]
        train_data_X = train_data_X[train_idx[:-int(validation_split*N)]]
        val_data_Y = train_data_Y[train_idx[-int(validation_split*N):]]
        train_data_Y = train_data_Y[train_idx[:-int(validation_split*N)]]
        train_idx = np.arange(train_data_X.shape[0])
            
        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()
        
        # Validation and training losses
        validation_losses = []
        training_losses = []
        
        # Main training loop
        if progress_bar:
            if self.nb:
                pbar = tqdm.tqdm_notebook(total = epochs, desc = "Training")
            else:
                pbar = tqdm.tqdm(total = epochs, desc = "Training")
            pbar.set_postfix(ordered_dict={"train loss":0, "val loss":0}, refresh=True)
        for epoch in range(epochs):
            # Shuffel training indices
            rng.shuffle(train_idx)
            for batch in range(len(train_idx)//batch_size):
                # Last batch will have maximum number of elements possible
                batch_idx = train_idx[batch*batch_size:np.min([(batch+1)*batch_size,len(train_idx)])]

                sess.run(self.train_op,feed_dict={self.model.parameters:train_data_X[batch_idx],
                                                      self.model.data:train_data_Y[batch_idx]})
            # Early stopping check
            val_loss = sess.run(self.model.trn_loss,feed_dict={self.model.parameters:val_data_X,
                                                                    self.model.data:val_data_Y})
            train_loss = sess.run(self.model.trn_loss,feed_dict={self.model.parameters:train_data_X,
                                                                         self.model.data:train_data_Y})
            if progress_bar:
                pbar.update()
                pbar.set_postfix(ordered_dict={"train loss":train_loss, "val loss":val_loss}, refresh=True)
            validation_losses.append(val_loss)
            training_losses.append(train_loss)
                
            if val_loss < bst_loss:
                bst_loss = val_loss
                if saver_name is not None:
                    saver.save(sess,"./"+saver_name)
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            if early_stopping_count >= patience:
                #pbar.set_postfix(str="Early stopping: terminated", refresh=True)
                break

        # Restore best model
        if saver_name is not None:
            saver.restore(sess,"./"+saver_name)

        return np.array(validation_losses), np.array(training_losses)

class ConditionalRegressionTrainer():
    
    def __init__(self, model, optimizer=tf.train.AdamOptimizer, optimizer_arguments={}):
        """
            Constructor that defines the training operation.
            :param model: made/maf instance to be trained.
            :param optimizer: tensorflow optimizer class to be used during training.
            :param optimizer_arguments: dictionary of arguments for optimizer intialization.
            """
        
        self.model = model
        
        # If the model has batch norm and it is activated, update operations on moving
        # mean and moving average have to be added to the training operation
        if hasattr(self.model,'batch_norm') and self.model.batch_norm is True:
            self.has_batch_norm = True
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss_reg)
        else:
            self.has_batch_norm = False
            self.train_op = optimizer(**optimizer_arguments).minimize(self.model.trn_loss_reg)

    """
    Training class for the conditional MADEs/MAFs classes using a tensorflow optimizer.
    """           
    def train(self, sess, train_data, validation_split = 0.1, epochs=1000, batch_size=100,
              patience=20, check_every_N=1, saver_name='tmp_model'):
        """
        Training function to be called with desired parameters within a tensorflow session.
        :param sess: tensorflow session where the graph is run.
        :param train_data: a tuple/list of (X,Y) with training data where Y is conditioned on X.
        :param validation_split: percentage of training data randomly selected to be used for validation
        :param epochs: maximum number of epochs for training.
        :param batch_size: batch size of each batch within an epoch.
        :param early_stopping: number of epochs for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param saver_name: string of name (with or without folder) where model is saved. If none is given,
            a temporal model is used to save and restore best model, and removed afterwards.
        """
        
        # Training data
        train_data_X, train_data_Y, train_data_L  = train_data
        train_idx = np.arange(train_data_X.shape[0])
        
        # validation data using p_val percent of the data
        rng.shuffle(train_idx)
        N = train_data_X.shape[0]
        val_data_X = train_data_X[train_idx[-int(validation_split*N):]]
        train_data_X = train_data_X[train_idx[:-int(validation_split*N)]]
        val_data_Y = train_data_Y[train_idx[-int(validation_split*N):]]
        train_data_Y = train_data_Y[train_idx[:-int(validation_split*N)]]
        val_data_L = train_data_L[train_idx[-int(validation_split*N):]]
        train_data_L = train_data_L[train_idx[:-int(validation_split*N)]]
        train_idx = np.arange(train_data_X.shape[0])
            
        # Early stopping variables
        bst_loss = np.infty
        early_stopping_count = 0
        saver = tf.train.Saver()
        
        # Validation and training losses
        validation_losses = []
        training_losses = []
        
        # Main training loop
        for epoch in range(epochs):
            # Shuffel training indices
            rng.shuffle(train_idx)
            for batch in range(len(train_idx)//batch_size):
                # Last batch will have maximum number of elements possible
                batch_idx = train_idx[batch*batch_size:np.min([(batch+1)*batch_size,len(train_idx)])]
                if self.has_batch_norm:
                    sess.run(self.train_op,feed_dict={self.model.input:train_data_X[batch_idx],
                                                      self.model.y:train_data_Y[batch_idx],
                                                      self.model.logpdf:train_data_L[batch_idx],
                                                      self.model.training:True})
                else:
                    sess.run(self.train_op,feed_dict={self.model.input:train_data_X[batch_idx],
                                                      self.model.y:train_data_Y[batch_idx],
                                                      self.model.logpdf:train_data_L[batch_idx]})
            # Early stopping check
            if epoch%check_every_N == 0:
                this_loss = sess.run(self.model.trn_loss_reg,feed_dict={self.model.input:val_data_X,
                                                                    self.model.y:val_data_Y,
                                                                    self.model.logpdf:val_data_L})
                train_loss = sess.run(self.model.trn_loss_reg,feed_dict={self.model.input:train_data_X,
                                                                         self.model.y:train_data_Y,
                                                                         self.model.logpdf:train_data_L})
                print("Epoch {:05d}, Train_loss: {:05.4f}, Val_loss: {:05.4f}".format(epoch,train_loss,this_loss))
                validation_losses.append(this_loss)
                training_losses.append(train_loss)
                
                if this_loss < bst_loss:
                    bst_loss = this_loss
                    saver.save(sess,"./"+saver_name)
                    early_stopping_count = 0
                else:
                    early_stopping_count += check_every_N
            if early_stopping_count >= patience:
                break
        print("Training finished")
        print("Best epoch {:05d}, Val_loss: {:05.4f}".format(epoch-check_every_N,bst_loss))

        # Restore best model
        saver.restore(sess,"./"+saver_name)

        return np.array(validation_losses), np.array(training_losses)

