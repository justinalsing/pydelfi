import tensorflow as tf
from tqdm.auto import tqdm
import numpy as np

def sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2):

    # progress-bar
    pbar = tqdm(total=n_steps, desc="Sampling")  # Jupyter notebook or qtconsole
    
    # initialize current state
    current_state1 = tf.Variable(walkers1)
    current_state2 = tf.Variable(walkers2)

    # initial target log prob for the walkers (and set any nans to -inf)...
    logp_current1 = log_prob(current_state1)
    logp_current2 = log_prob(current_state2)
    logp_current1 = tf.where(tf.math.is_nan(logp_current1), tf.ones_like(logp_current1)*tf.math.log(0.), logp_current1)
    logp_current2 = tf.where(tf.math.is_nan(logp_current2), tf.ones_like(logp_current2)*tf.math.log(0.), logp_current2)

    # holder for the whole chain
    chain = [tf.concat([current_state1, current_state2], axis=0)]
    
    # MCMC loop
    for epoch in range(n_steps):

        # first set of walkers:

        # proposals
        partners1 = tf.gather(current_state2, np.random.randint(0, n_walkers, n_walkers))
        z1 = 0.5*(tf.random.uniform([n_walkers], minval=0, maxval=1)+1)**2
        proposed_state1 = partners1 + tf.transpose(z1*tf.transpose(current_state1 - partners1))

        # target log prob at proposed points
        logp_proposed1 = log_prob(proposed_state1)
        logp_proposed1 = tf.where(tf.math.is_nan(logp_proposed1), tf.ones_like(logp_proposed1)*tf.math.log(0.), logp_proposed1)

        # acceptance probability
        p_accept1 = tf.math.minimum(tf.ones(n_walkers), z1**(n_params-1)*tf.exp(logp_proposed1 - logp_current1) )

        # accept or not
        accept1_ = (tf.random.uniform([n_walkers], minval=0, maxval=1) <= p_accept1)
        accept1 = tf.cast(accept1_, tf.float32)

        # update the state
        current_state1 = tf.transpose( tf.transpose(current_state1)*(1-accept1) + tf.transpose(proposed_state1)*accept1)
        logp_current1 = tf.where(accept1_, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals
        partners2 = tf.gather(current_state1, np.random.randint(0, n_walkers, n_walkers))
        z2 = 0.5*(tf.random.uniform([n_walkers], minval=0, maxval=1)+1)**2
        proposed_state2 = partners2 + tf.transpose(z2*tf.transpose(current_state2 - partners2))

        # target log prob at proposed points
        logp_proposed2 = log_prob(proposed_state2)
        logp_proposed2 = tf.where(tf.math.is_nan(logp_proposed2), tf.ones_like(logp_proposed2)*tf.math.log(0.), logp_proposed2)

        # acceptance probability
        p_accept2 = tf.math.minimum(tf.ones(n_walkers), z2**(n_params-1)*tf.exp(logp_proposed2 - logp_current2) )

        # accept or not
        accept2_ = (tf.random.uniform([n_walkers], minval=0, maxval=1) <= p_accept2)
        accept2 = tf.cast(accept2_, tf.float32)

        # update the state
        current_state2 = tf.transpose( tf.transpose(current_state2)*(1-accept2) + tf.transpose(proposed_state2)*accept2)
        logp_current2 = tf.where(accept2_, logp_proposed2, logp_current2)

        # append to chain
        chain.append(tf.concat([current_state1, current_state2], axis=0))

        # update the progressbar
        pbar.update(1)

    # stack up the chain
    chain = tf.stack(chain, axis=0)
    
    return chain


