import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

class ConditionalTrainer():
    
    def __init__(self, model, optimizer=tf.keras.optimizers.Adam, optimizer_arguments=None):
        """
            Constructor that defines the training operation.
            :param model: made/maf instance to be trained.
            :param optimizer: tensorflow optimizer class to be used during training.
            :param optimizer_arguments: dictionary of arguments for optimizer intialization.
        """
        
        self.model = model
        if optimizer_arguments is not None:
            self.optimizer = optimizer(optimizer_arguments)
        else:
            self.optimizer = optimizer()

    """
    Training class for the conditional MADEs/MAFs classes using a tensorflow optimizer.
    """           
    def train(self, train_data, f_val=0.1, epochs=1000, n_batch=100,
              patience=20, file_name='tmp_model', progress_bar=True, mode='samples'):
        """
        Training function to be called with desired parameters.
        :param train_data: a tuple/list of (X,Y) with training data where Y is conditioned on X.
        :param f_val: fraction of training data randomly selected to be used for validation
        :param epochs: maximum number of epochs for training.
        :param n_batch: size of each batch within an epoch.
        :param patience: number of epochs for early stopping criteria.
        :param file_name: string of name (with or without folder) where model is saved.
        :param progress_bar: display progress bar?
        :param mode: 'samples' to fit samples drawn from distribution, or 'regression' to fit 
                     PDF values at given positions.
        """
        
        # Parse full training data and determine size of training set
        if mode == 'samples':
            train_data_X, train_data_Y = train_data
        elif mode == 'regression':
            train_data_X, train_data_Y, train_data_PDF = train_data
        n_train_all = train_data_X.shape[0]
        
        # Split into training and validation data
        n_val = int(n_train_all * f_val)
        n_train = n_train_all - n_val
        is_val = np.random.permutation([False] * n_train + [True] * n_val)

        # Create training and validation Dataset objects, shuffling and batching the training data. Note
        # the default behaviour of Dataset.shuffle() sets reshuffle_each_iteration=True, and 
        # Dataset.batch() sets drop_remainder=False
        if mode == 'samples':
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data_X[~is_val], \
                                                                train_data_Y[~is_val]))
            val_dataset = tf.data.Dataset.from_tensor_slices((train_data_X[is_val], \
                                                              train_data_Y[is_val]))
        elif mode == 'regression':
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data_X[~is_val], \
                                                                train_data_Y[~is_val], \
                                                                train_data_PDF[~is_val]))
            val_dataset = tf.data.Dataset.from_tensor_slices((train_data_X[is_val], \
                                                              train_data_Y[is_val], \
                                                              train_data_PDF[is_val]))
        train_dataset = train_dataset.shuffle(n_train).batch(n_batch)
        val_dataset = val_dataset.batch(n_val)

        # Early stopping variables
        bst_loss = np.infty
        es_count = 0
        
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
            for step, xy_batch_train in enumerate(train_dataset):
                
                # Unpack batched training data
                if mode == 'samples':
                    x_batch_train, y_batch_train = xy_batch_train
                elif mode == 'regression':
                    x_batch_train, y_batch_train, pdf_batch_train = xy_batch_train

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables autodifferentiation.
                with tf.GradientTape() as tape:

                    # Compute the loss for this batch.
                    log_probs = self.model.log_prob(y_batch_train, x_batch_train)
                    if mode == 'samples':
                        train_loss = -tf.reduce_sum(log_probs)
                    elif mode == 'regression':
                        train_loss = tf.keras.losses.MSE(pdf_batch_train, log_probs)

                # Retrieve the gradients of the trainable variables wrt the loss and 
                # pass to optimizer.
                grads = tape.gradient(train_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Compute the validation loss, and store it alongside the training loss.
            if mode == 'samples':
                for x_batch_val, y_batch_val in val_dataset:
                    val_loss = -tf.reduce_sum(self.model.log_prob(y_batch_val, x_batch_val))
            elif mode == 'regression':
                for x_batch_val, y_batch_val, pdf_batch_val in val_dataset:
                    val_loss = tf.keras.losses.MSE(pdf_batch_val, self.model.log_prob(y_batch_val, x_batch_val))
            train_losses.append(train_loss.numpy())
            val_losses.append(val_loss.numpy())
    
            # Use validation loss to stop training early
            if val_losses[-1] < bst_loss:
                bst_loss = val_losses[-1]
                self.model.save_weights(file_name, save_format='tf')
                es_count = 0
            else:
                es_count += 1
                if es_count >= patience:
                    print('Training terminated early at epoch {:d}.'.format(epoch))
                    self.model.load_weights(file_name)
                    break

            # Update progress if desired.
            if progress_bar:
                pbar.update()
                pbar.set_postfix(ordered_dict={"train loss":train_losses[-1], \
                                               "val loss":val_losses[-1]}, refresh=True)

        return np.array(val_losses), np.array(train_losses)
