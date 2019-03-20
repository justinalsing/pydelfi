import numpy as np
import numpy.linalg as linalg

# Generate covariance matrix
def covariance(theta_fiducial, simulation, nsims, ndata, sim_args):
    
    C = np.zeros((ndata,ndata))
    D = np.zeros((nsims, ndata))
    dbar = np.zeros((ndata))
    d2bar = np.zeros((ndata, ndata))
    for i in range(0, nsims):
        sim = simulation(theta_fiducial, sim_args)
        dbar += sim/nsims
        d2bar += np.outer(sim, sim)/nsims
        D[i,:] = sim
    C = d2bar - np.outer(dbar,dbar)
        
    return dbar, C, D

# Generate derivative of \mu w.r.t cosmological parameters
def dmudtheta(theta_fiducial, simulation_seeded, step, npar, ndata, sim_args):
    
    # dmdt
    dmdt = np.zeros((npar, ndata))
    
    # Fiducial data
    d_fiducial = simulation_seeded(theta_fiducial, 0, sim_args)
        
    # Loop over parameters
    for i in range(0, npar):
        
        # Step theta
        theta = np.copy(theta_fiducial)
        theta[i] += step[i]
        
        # Shifted data with same seed
        d_dash = simulation_seeded(theta, 0, sim_args)

        # One step derivative
        dmdt[i,:] = (d_dash - d_fiducial)/step[i]
        
    return dmdt


# Compute the moped compression vectors from the covariance and dmudtheta
def moped_matrix_gram_schmidt(Cinv, dmdt, npar, ndata):
    
    # moped vectors
    B = np.zeros((npar, ndata))
    
    # First vector
    B[0,:] = np.dot(Cinv, dmdt[0,:])/np.sqrt(np.dot(dmdt[0,:],np.dot(Cinv, dmdt[0,:])))
    
    # Subsequent vectors with Gram-Schmidt orthogonalization
    for i in range(1, npar):
        
        # Compute the Gram-Schmidt orthogonal component
        X = np.zeros((ndata))
        for j in range(0, i):
            X += np.dot(dmdt[i,:], B[j,:])*B[j,:]
            
        # Construce the MOPED vector
        B[i,:] = (np.dot(Cinv, dmdt[i,:]) - X)/np.sqrt(np.dot(dmdt[i,:], np.dot(Cinv, dmdt[i,:]) - X))
    
    return B

# Compute the moped compression vectors from the covariance and dmudtheta
def moped_matrix(Cinv, dmdt, npar, ndata):
    
    # moped vectors
    B = np.zeros((npar, ndata))
    
    # Subsequent vectors with Gram-Schmidt orthogonalization
    for a in range(0, npar):
        B[a,:] = np.dot(Cinv, dmdt[a,:])

    return B

# Compute the Fisher matrix
def fisher(dmdt, dCdt, Cinv, Sinv, npar):
    
    # Fisher matrix
    F = np.zeros((npar, npar))
    
    # Mean derivatives part
    for a in range(0, npar):
        for b in range(0, npar):
            F[a, b] += 0.5*(np.dot(dmdt[a,:], np.dot(Cinv, dmdt[b,:])) + np.dot(dmdt[b,:], np.dot(Cinv, dmdt[a,:])))

    # Covariance derivatives part
    for a in range(0, npar):
        for b in range(0, npar):
            F[a, b] += 0.5*np.trace( np.dot( np.dot(Cinv, dCdt[a,:,:]), np.dot(Cinv, dCdt[b,:,:]) ) )

    # Add the prior
    F = F + Sinv

    Finv = np.linalg.inv(F)
    return F, Finv

# Compute the maximum likelihood estimator assuming Gaussianity and linearity
def mle(theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, Sinv, mu_prior, data):
    
    # Number of parameters
    npar = len(theta_fiducial)
    
    # Compute the score
    dLdt = np.zeros(npar)
    
    # Add terms from mean derivatives
    for a in range(npar):
        dLdt[a] += np.dot(dmdt[a,:], np.dot(Cinv, (data - mu)))

    # Add terms from covariance derivatives
    for a in range(npar):
        dLdt[a] += -0.5*np.trace(np.dot(Cinv, dCdt[a,:,:])) + 0.5*np.dot((data - mu), np.dot( np.dot(Cinv, np.dot(dCdt[a,:,:], Cinv)), (data - mu)))

    return theta_fiducial + np.dot(Finv, np.dot(Sinv, mu_prior - theta_fiducial)) + np.dot(Finv, dLdt)

# Compute the compressed data
def compressed(data, index, compression_args):
    
    theta_fiducial = compression_args[0]
    Finv = compression_args[1]
    Cinv = compression_args[2]
    dmdt = compression_args[3]
    dCdt = compression_args[4]
    mu = compression_args[5]
    
    return mle(theta_fiducial, Finv, Cinv, dmdt, dCdt, mu, data)[index]

def compressed_gram_schmidt(data, Cinv, dmdt):

    # moped vectors
    npar = len(dmdt)
    ndata = len(data)
    B = np.zeros((npar, ndata))
    
    # First vector
    B[0,:] = np.dot(Cinv, dmdt[0,:])/np.sqrt(np.dot(dmdt[0,:],np.dot(Cinv, dmdt[0,:])))
    
    # Subsequent vectors with Gram-Schmidt orthogonalization
    for i in range(1, npar):
        
        # Compute the Gram-Schmidt orthogonal component
        X = np.zeros((ndata))
        for j in range(0, i):
            X += np.dot(dmdt[i,:], B[j,:])*B[j,:]
        
        # Construce the MOPED vector
        B[i,:] = (np.dot(Cinv, dmdt[i,:]) - X)/np.sqrt(np.dot(dmdt[i,:], np.dot(Cinv, dmdt[i,:]) - X))
    
    return np.dot(B, data)

