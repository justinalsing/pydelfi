import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Train():
    def __init__(self, optimiser=tf.keras.optimizers.Adam, optimiser_arguments=None, **kwargs):
        # optimiser parameters
        if optimiser_arguments is not None:
            self.optimiser = optimiser(optimiser_arguments)
        else:
            self.optimiser = optimiser() 
        
    @tf.function
    def single_train_epoch(self, dataset, variables_list, n_batch):
        loss = tf.zeros((self.n_stack,), dtype=self.type)
        for step, xy_batch_train in enumerate(dataset):
            # Unpack batched training data
            x_batch_train, y_batch_train = xy_batch_train
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:
                # Compute the loss for this batch.
                neg_log_prob = -tf.reduce_mean(self.model.log_prob(y_batch_train, x_batch_train), -1)
            # Retrieve the gradients of the trainable variables wrt the loss and 
            # pass to optimizer.
            grads = tape.gradient(neg_total_log_prob, variables_list)
            self.optimiser.apply_gradients(zip(grads, variables_list))
            loss = tf.add(loss, tf.reduce_sum(neg_log_prob))
        return tf.divide(loss, n_batch)
    
    @tf.function
    def single_validate_epoch(self, dataset, n_batch):
        loss = tf.zeros((self.n_stack,), dtype=self.type)
        for step, xy_batch_train in enumerate(dataset):
            # Unpack batched training data
            x_batch_train, y_batch_train = xy_batch_train
            # Compute the loss for this batch.
            neg_log_prob = -tf.reduce_mean(self.model.log_prob(y_batch_train, x_batch_train), -1)
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
        n_sims = data_X.shape[0]
        
        # Split into training and validation data
        n_val = int(n_sims * f_val)
        n_train = n_sims - n_val
        is_val = np.random.permutation([False] * n_train + [True] * n_val)

        n_train_batches = int(n_train / n_batch)
        if n_train_batches == 0:
            n_train_batches = 1
        n_train_batches = tf.cast(n_train_batches, dtype=self.type)
        
        n_val_batches = int(n_val / n_batch)
        if n_val_batches == 0:
            n_val_batches = 1
        n_val_batches = tf.cast(n_val_batches, dtype=self.type)
        
        # Create training and validation Dataset objects, shuffling and batching the training data. Note
        # the default behaviour of Dataset.shuffle() sets reshuffle_each_iteration=True, and 
        # Dataset.batch() sets drop_remainder=False
        train_dataset = tf.data.Dataset.from_tensor_slices((data_X[~is_val],
                                                            data_Y[~is_val]))
        val_dataset = tf.data.Dataset.from_tensor_slices((data_X[is_val],
                                                          data_Y[is_val]))
        train_dataset = train_dataset.shuffle(n_train).batch(n_batch)
        val_dataset = val_dataset.batch(n_val)

        # Early stopping variables
        bst_loss = np.infty
        es_count = np.zeros((self.n_stack,), dtype=np.int)
        
        # Validation and training losses
        train_losses = []
        val_losses = []
        
        # Progress bar, if desired
        if progress_bar:
            pbar = tqdm(total = epochs, desc = "Training")
            pbar.set_postfix(ordered_dict={"train loss":0, "val loss":0}, refresh=True)
        
        # Main training loop
        for epoch in range(epochs):
            # Iterate over the batches of the dataset.
            train_losses.append(self.single_train_epoch(train_dataset, variables_list, n_train_batches).numpy())
            val_losses.append(self.single_validate_epoch(val_dataset, n_val_batches).numpy())
    
            # Use validation loss to stop training early
            ind = np.where(val_losses[-1] < bst_loss)[0]
            not_ind = np.where(val_losses[-1] >= bst_loss)[0]
            for i in ind:
                m_ind = self.stack[i]
                self.model[m_ind].save_weights(file_name + "_" + str(m_ind), save_format="tf")
                self.weighting = tf.tensor_scatter_nd_update(self.weighting, [[m_ind]], [tf.exp(-val_losses[-1][i])])
                es_count[m_ind] = 0
            for i in not_ind:
                m_ind = self.stack[i]
                es_count[m_ind] += 1
                if es_count[m_ind] >= patience:
                    print("Training terminated for model {:d} at epoch {:d}.".format(m_ind, epoch))
                    self.model[m_ind].load_weights(file_name + "_" + str(m_ind))
                    self.stack.pop(i)
                    variables_list = self.get_variables_list()
            if len(self.stack) == 0:
                break
            # Update progress if desired.
            if progress_bar:
                pbar.update(1)
                pbar.set_postfix(
                    ordered_dict={
                        "train loss":train_losses[-1],
                        "val loss":val_losses[-1]}, 
                    refresh=True)

        self.set_stack()
        return np.array(val_losses), np.array(train_losses)