import numpy as np
import scipy.integrate as integrate
from .moped import *

# Distance modulus
def apparent_magnitude(theta, auxiliary_data):
    
    # Cosmological parameters
    Om = theta[0]
    w0 = theta[1]
    h = 0.7
    
    # Systematics parameters
    Mb = theta[2]
    alpha = theta[3]
    beta = theta[4]
    delta_m = theta[5]
    
    # Pull out the relevant things from the data
    z = auxiliary_data[:,0]
    x = auxiliary_data[:,1]
    c = auxiliary_data[:,2]
    v3 = auxiliary_data[:,3]
    
    # Holders
    distance_modulus = np.zeros(len(z))
    
    for i in range(len(z)):
        integral = integrate.quad(lambda zz: 1./np.sqrt( Om*(1+zz)**3 + (1-Om)*(1+zz)**(3*(1+w0)) ), 0, z[i])[0]
        distance_modulus[i] = 25 - 5*np.log10(h) + 5*np.log10(3000*(1+z[i])*integral)
    
    return Mb - alpha*x + beta*c + delta_m*v3 + distance_modulus

# Generate realisation of \mu
def simulation(theta, sim_args):
    
    # Pull out data
    auxiliary_data = sim_args[0]
    L = sim_args[1]
    
    # Signal
    mb = apparent_magnitude(theta, auxiliary_data)
        
    # Noise
    noise = np.dot(L, np.random.normal(0, 1, len(L)))
    
    # Return signal + noise
    return mb + noise

# Generate realisation of \mu
def simulation_seeded(theta, seed, sim_args):
    
    # Pull out data
    auxiliary_data = sim_args[0]
    L = sim_args[1]
    
    # Signal
    mb = apparent_magnitude(theta, auxiliary_data)
        
    # Noise
    np.random.seed(seed)
    noise = np.dot(L, np.random.normal(0, 1, len(L)))
    
    # Return signal + noise
    return mb

def compressor(data, args):

    theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Qinv, prior_mean = args
    
    return mle(theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Qinv, prior_mean, data)

def compressor_projected(data, args):
    
    theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Qinv, prior_mean, F, P1, P2 = args

    # MOPED compress the data
    d_twidle = mle(theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Qinv, prior_mean, data)
    
    # Now do the projection
    d_twidle = np.dot(F, d_twidle - theta_fiducial - np.dot(Finv, np.dot(Qinv, prior_mean - theta_fiducial)))
    d_twidle = np.dot(Finv[0:2, 0:2], np.array([d_twidle[0] - np.dot(P1, d_twidle[2:]), d_twidle[1] - np.dot(P2, d_twidle[2:])]))
    d_twidle = d_twidle + theta_fiducial[:2] + np.dot(Finv[:2,:2], np.dot(Qinv[:2,:2], prior_mean[:2] - theta_fiducial[:2]))

    return d_twidle


