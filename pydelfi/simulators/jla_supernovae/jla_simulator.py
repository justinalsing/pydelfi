import numpy as np
import scipy.integrate as integrate
import pydelfi.simulators.jla_supernovae.jla_parser as jla_parser

class JLA_Model():

    def __init__(self, jla_data_path = 'simulators/jla_supernovae/jla_data/'):

        # Import data
        jla_data, jla_cmats = jla_parser.b14_parse(z_min=None, z_max=None, qual_cut=False,
                                           jla_path='simulators/jla_supernovae/jla_data/')
        self.data = jla_data['mb']
        delta_m_cut = 10
        self.auxiliary_data = np.column_stack([jla_data['zcmb'], jla_data['x1'], jla_data['color'], np.array([(jla_data['3rdvar'] > delta_m_cut)], dtype=int)[0]])

        # Om, w0, M_b, alpha, beta, delta_m
        self.npar = 6
        self.theta_fiducial = np.array([  0.20181324,  -0.74762939, -19.04253368,   0.12566322,   2.64387045, -0.05252869])

        # Covariance matrix
        self.C = jla_parser.b14_covariance(jla_data, jla_cmats, self.theta_fiducial[3], self.theta_fiducial[4])
        self.Cinv = np.linalg.inv(self.C)
        self.L = np.linalg.cholesky(self.C)

        # Derivative of the covariance matrix
        self.n_sn = len(self.C)
        self.dCdt = np.zeros((self.npar, self.n_sn, self.n_sn))

        # N data points
        self.ndata = len(jla_data['mb'])

        # Compute the mean
        self.mu = self.apparent_magnitude(self.theta_fiducial)

    # Distance modulus
    def apparent_magnitude(self, theta):
        
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
        z = self.auxiliary_data[:,0]
        x = self.auxiliary_data[:,1]
        c = self.auxiliary_data[:,2]
        v3 = self.auxiliary_data[:,3]
        
        # Holders
        distance_modulus = np.zeros(len(z))
        
        for i in range(len(z)):
            integral = integrate.quad(lambda zz: 1./np.sqrt( Om*(1+zz)**3 + (1-Om)*(1+zz)**(3*(1+w0)) ), 0, z[i])[0]
            distance_modulus[i] = 25 - 5*np.log10(h) + 5*np.log10(3000*(1+z[i])*integral)
        
        return Mb - alpha*x + beta*c + delta_m*v3 + distance_modulus

    # Generate realisation of \mu
    def simulation(self, theta, seed):
        
        # Set the seed
        np.random.seed(seed)

        # Signal
        mb = self.apparent_magnitude(theta)
        
        # Noise
        noise = np.dot(self.L, np.random.normal(0, 1, len(self.L)))
        
        # Return signal + noise
        return mb + noise

    # Generate derivative of \mu w.r.t cosmological parameters
    def dmudt(self, theta_fiducial, h):
        
        # dmdt
        dmdt = np.zeros((self.npar, self.ndata))
        
        # Fiducial data
        d_fiducial = self.apparent_magnitude(theta_fiducial)
        
        # Loop over parameters
        for i in range(0, 2):
            
            # Step theta
            theta = np.copy(self.theta_fiducial)
            theta[i] += h[i]
            
            # Shifted data with same seed
            d_dash = self.apparent_magnitude(theta)
            
            # One step derivative
            dmdt[i,:] = (d_dash - d_fiducial)/h[i]
        
        dmdt[2,:] = np.ones(self.n_sn)
        dmdt[3,:] = -self.auxiliary_data[:,1]
        dmdt[4,:] = self.auxiliary_data[:,2]
        dmdt[5,:] = self.auxiliary_data[:,3]
        
        return dmdt





