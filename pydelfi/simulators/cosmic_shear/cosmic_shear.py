import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.interpolate as interpolate
from scipy.stats import norm
from scipy.special import jv
from scipy.stats import wishart
from scipy.stats import norm as normal
from scipy.stats import multivariate_normal
import pickle
import scipy.integrate as integrate
from .cosmology import *
import scipy.constants as sc


# Cosmic shear class
class TomographicCosmicShear():

    def __init__(self, pz = None, lmin = 10, lmax = 3000, n_ell_bins = 10, Area = 15000, sigma_e = 0.3, nbar = 30):

        # Photo-z distributions
        self.pz = pz
        self.nz = len(pz)

        # ell binning
        self.npar = 5
        self.n_ell_bins = n_ell_bins
        modes_edges = np.logspace(np.log10(lmin), np.log10(lmax), self.n_ell_bins+1)
        self.modes = (modes_edges[0:-1] + modes_edges[1:])/2
        l = np.arange(lmin, lmax)
        self.nl = np.array([sum((2*l[(l >= modes_edges[i])*(l < modes_edges[i+1])] + 1)) for i in range(len(self.modes))], dtype=int)
        
        # Partial sky factor
        self.Area = Area*(np.pi/180)**2
        self.fsky = self.Area/(4*np.pi)
        self.nl = np.array(self.nl*self.fsky, dtype=int)
    
        # Noise covariance
        self.sigma_e = sigma_e
        self.nbar = (nbar/self.nz)*(60*180./np.pi)**2
        self.N = (self.sigma_e**2/self.nbar)*np.eye(self.nz)

    def simulate(self, theta, seed):
        
        # Set the seed
        np.random.seed(seed)
    
        # Compute theory power spectrum
        C = self.power_spectrum(theta)
    
        # Realize noisy power spectrum
        C_hat = np.zeros((self.n_ell_bins, self.nz, self.nz))
        for i in range(self.n_ell_bins):
            C_hat[i, :, :] = wishart.rvs(df=self.nl[i], scale=C[i, :,:])/self.nl[i]
    
        return C_hat
    
    def simulate_batch(self, theta, seed, batch):

        # Set the seed
        np.random.seed(seed)
        
        # Compute theory power spectrum
        C = self.power_spectrum(theta) - self.N
    
        # Compute the Cholseky power spectra
        L = np.array([np.linalg.cholesky(C[i,:,:]) for i in range(self.n_ell_bins)])
        L_N = np.linalg.cholesky(self.N)
    
        # Generate random field
        f = [0]*self.n_ell_bins
        for i in range(self.n_ell_bins):
            f[i] = np.array([np.dot(L[i,:,:], np.random.normal(0, 1, self.nz)) for k in range(self.nl[i])])

        # Generate batch of sims with different noise realizations
        sims = np.zeros((batch, self.n_ell_bins, self.nz, self.nz))
        for s in range(batch):
            
            # Generate noise realization
            n = [0]*self.n_ell_bins
            for i in range(self.n_ell_bins):
                n[i] = np.array([np.dot(L_N, np.random.normal(0, 1, self.nz)) for k in range(self.nl[i])])

            # Add noise
            fn = f + n

            # Compute power for each ell bin
            for i in range(self.n_ell_bins):
                for j in range(self.nl[i]):
                    sims[s,i,:,:] += np.outer(fn[i][j], fn[i][j])/self.nl[i]

        return sims
            
    # Compute the data vector
    def power_spectrum(self, theta):

        # Evaluate the required (derived) cosmological parameters
        omm = theta[0]
        sigma8 = theta[1]*np.sqrt(0.3/theta[0])
        omb = theta[2]
        h = theta[3]
        ns = theta[4]
        omde = 1.0 - omm
        omnu = 0
        omk = 0
        hubble = h*100
        w0 = -1.
        wa = 0
        
        # Initialize cosmology object
        cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=0)

        # Numerics parameters
        zmax = 2
        rmax = cosmo.a2chi(z2a(zmax))
        power_zpoints = int(np.ceil(5*zmax))
        power_kpoints = 200
        distance_zpoints = int(np.ceil(10*zmax))
        wpoints = int(np.ceil(15*zmax))
        kmax = 10
        clpoints = 2**7 + 1

        # Compute the matter power spectrum at the cosmology
        z = np.linspace(0, zmax, power_zpoints)
        logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
        logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))

        # 2D linear interpolator for P(k;z)
        logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

        # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
        zvalues = np.linspace(0, zmax, distance_zpoints)
        rvalues = np.zeros((len(zvalues)))

        # Perform integration to compute r(z) at specified points according to cosmology
        for i in range(0, len(zvalues)):
            rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i], divmax=100)

        # Generate interpolation functions to give r(z) and z(r) given cosmology
        r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
        z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

        # Set the maximum comoving distance corresponding to the maximum redshift
        rmax = rvalues[-1]

        # Compute lensing weights...

        w = []
        
        # Compute the weight function associated with each bin in turn, over r-points and then interpolate
        for i in range(0, self.nz):

            # r-points to evaluate weight at before interpolation
            rpoints = np.linspace(0, rmax, wpoints)

            # Initialize weights
            weight = np.zeros(wpoints)

            # Compute integral for the rest of the points
            for j in range(1, wpoints):
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*self.pz[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                weight[j] = integrate.romb(intvals, dx)

            # Interpolate (generate interpolation function) and add interpolation function to the array w
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
            w.append(interp)
        
        # Tensor for cls
        cls = np.zeros((self.n_ell_bins, self.nz, self.nz))

        # Pull required cosmological parameters out of cosmo
        r_hubble = sc.c/(1000*hubble)
        A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

        # Compute Cls
        for L in range(self.n_ell_bins):
            l = self.modes[L]
            rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
            dr = rs[1] - rs[0]
            for i in range(0, self.nz):
                for j in range(i, self.nz):
                    intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)*w[i](rs)*w[j](rs) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                    cls[L, i, j] = integrate.romb(intvals, dr)
                    cls[L, j, i] = cls[L, i, j]
            cls[L, :,:] = cls[L, :,:] + self.N
                    
        return cls
    
    # Compute the data vector
    def power_spectrum_noiseless(self, theta):

        # Evaluate the required (derived) cosmological parameters
        omm = theta[0]
        sigma8 = theta[1]*np.sqrt(0.3/theta[0])
        omb = theta[2]
        h = theta[3]
        ns = theta[4]
        omde = 1.0 - omm
        omnu = 0
        omk = 0
        hubble = h*100
        w0 = -1.
        wa = 0
        
        # Initialize cosmology object
        cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=0)

        # Numerics parameters
        zmax = 2
        rmax = cosmo.a2chi(z2a(zmax))
        power_zpoints = int(np.ceil(5*zmax))
        power_kpoints = 200
        distance_zpoints = int(np.ceil(10*zmax))
        wpoints = int(np.ceil(15*zmax))
        kmax = 10
        clpoints = 2**7 + 1

        # Compute the matter power spectrum at the cosmology
        z = np.linspace(0, zmax, power_zpoints)
        logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
        logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))

        # 2D linear interpolator for P(k;z)
        logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

        # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
        zvalues = np.linspace(0, zmax, distance_zpoints)
        rvalues = np.zeros((len(zvalues)))

        # Perform integration to compute r(z) at specified points according to cosmology
        for i in range(0, len(zvalues)):
            rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i], divmax=100)

        # Generate interpolation functions to give r(z) and z(r) given cosmology
        r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
        z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

        # Set the maximum comoving distance corresponding to the maximum redshift
        rmax = rvalues[-1]

        # Compute lensing weights...

        w = []
        
        # Compute the weight function associated with each bin in turn, over r-points and then interpolate
        for i in range(0, self.nz):

            # r-points to evaluate weight at before interpolation
            rpoints = np.linspace(0, rmax, wpoints)

            # Initialize weights
            weight = np.zeros(wpoints)

            # Compute integral for the rest of the points
            for j in range(1, wpoints):
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*self.pz[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                weight[j] = integrate.romb(intvals, dx)

            # Interpolate (generate interpolation function) and add interpolation function to the array w
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
            w.append(interp)
        
        # Tensor for cls
        cls = np.zeros((self.n_ell_bins, self.nz, self.nz))

        # Pull required cosmological parameters out of cosmo
        r_hubble = sc.c/(1000*hubble)
        A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

        # Compute Cls
        for L in range(self.n_ell_bins):
            l = self.modes[L]
            rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
            dr = rs[1] - rs[0]
            for i in range(0, self.nz):
                for j in range(i, self.nz):
                    intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)*w[i](rs)*w[j](rs) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                    cls[L, i, j] = integrate.romb(intvals, dr)
                    cls[L, j, i] = cls[L, i, j]
            cls[L, :,:] = cls[L, :,:]
        
        return cls


    def compute_derivatives(self, theta_fiducial, step):

        # Holder for derivatives
        dCdt = np.zeros((self.npar, self.n_ell_bins, self.nz, self.nz))

        # Derivatives wrt cosmological parameters
        for i in range(self.npar):
            theta_plus = np.copy(theta_fiducial)
            theta_plus[i] += step[i]
            theta_minus = np.copy(theta_fiducial)
            theta_minus[i] -= step[i]
            
            Cp = self.power_spectrum_noiseless(theta_plus)
            Cm = self.power_spectrum_noiseless(theta_minus)
            
            dCdt[i, :, :, :] = (Cp - Cm)/(2*step[i])

        return dCdt


# Cosmic shear class
class TomographicCosmicShearPhotoz():

    def __init__(self, pz = None, lmin = 10, lmax = 3000, n_ell_bins = 10, Area = 15000, sigma_e = 0.3, nbar = 30):

        # Photo-z distributions
        self.pz = pz
        self.nz = len(pz)

        # ell binning
        self.npar = 5 + len(pz)
        self.n_ell_bins = n_ell_bins
        modes_edges = np.logspace(np.log10(lmin), np.log10(lmax), self.n_ell_bins+1)
        self.modes = (modes_edges[0:-1] + modes_edges[1:])/2
        l = np.arange(lmin, lmax)
        self.nl = np.array([sum((2*l[(l >= modes_edges[i])*(l < modes_edges[i+1])] + 1)) for i in range(len(self.modes))], dtype=int)
        
        # Partial sky factor
        self.Area = Area*(np.pi/180)**2
        self.fsky = self.Area/(4*np.pi)
        self.nl = np.array(self.nl*self.fsky, dtype=int)

        # Noise covariance
        self.sigma_e = sigma_e
        self.nbar = (nbar/self.nz)*(60*180./np.pi)**2
        self.N = (self.sigma_e**2/self.nbar)*np.eye(self.nz)

    def simulate(self, theta, seed):
        
        # Set the seed
        np.random.seed(seed)
    
        # Compute theory power spectrum
        C = self.power_spectrum(theta)
    
        # Realize noisy power spectrum
        C_hat = np.zeros((self.n_ell_bins, self.nz, self.nz))
        for i in range(self.n_ell_bins):
            C_hat[i, :, :] = wishart.rvs(df=self.nl[i], scale=C[i, :,:])/self.nl[i]
    
        return C_hat

    # Compute the data vector
    def power_spectrum(self, theta):

        # Evaluate the required (derived) cosmological parameters
        omm = theta[0]
        sigma8 = theta[1]*np.sqrt(0.3/theta[0])
        omb = theta[2]
        h = theta[3]
        ns = theta[4]
        omde = 1.0 - omm
        omnu = 0
        omk = 0
        hubble = h*100
        w0 = -1.
        wa = 0
        
        # Initialize cosmology object
        cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=0)

        # Photo-z parameters
        z = np.linspace(0, self.pz[0].get_knots()[-1], len(self.pz[0].get_knots()))
        pz_new = [0]*self.nz
        for i in range(self.nz):
            p = self.pz[i](z+theta[5+i])
            p = p/np.trapz(p, z)
            pz_new[i] = interpolate.InterpolatedUnivariateSpline(z, p, k=3)
        
        # Numerics parameters
        zmax = 2
        rmax = cosmo.a2chi(z2a(zmax))
        power_zpoints = int(np.ceil(5*zmax))
        power_kpoints = 200
        distance_zpoints = int(np.ceil(10*zmax))
        wpoints = int(np.ceil(15*zmax))
        kmax = 10
        clpoints = 2**7 + 1

        # Compute the matter power spectrum at the cosmology
        z = np.linspace(0, zmax, power_zpoints)
        logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
        logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))

        # 2D linear interpolator for P(k;z)
        logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

        # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
        zvalues = np.linspace(0, zmax, distance_zpoints)
        rvalues = np.zeros((len(zvalues)))

        # Perform integration to compute r(z) at specified points according to cosmology
        for i in range(0, len(zvalues)):
            rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i], divmax=100)

        # Generate interpolation functions to give r(z) and z(r) given cosmology
        r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
        z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

        # Set the maximum comoving distance corresponding to the maximum redshift
        rmax = rvalues[-1]

        # Compute lensing weights...

        w = []
        
        # Compute the weight function associated with each bin in turn, over r-points and then interpolate
        for i in range(0, self.nz):

            # r-points to evaluate weight at before interpolation
            rpoints = np.linspace(0, rmax, wpoints)

            # Initialize weights
            weight = np.zeros(wpoints)

            # Compute integral for the rest of the points
            for j in range(1, wpoints):
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*pz_new[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                weight[j] = integrate.romb(intvals, dx)

            # Interpolate (generate interpolation function) and add interpolation function to the array w
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
            w.append(interp)
        
        # Tensor for cls
        cls = np.zeros((self.n_ell_bins, self.nz, self.nz))

        # Pull required cosmological parameters out of cosmo
        r_hubble = sc.c/(1000*hubble)
        A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

        # Compute Cls
        for L in range(self.n_ell_bins):
            l = self.modes[L]
            rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
            dr = rs[1] - rs[0]
            for i in range(0, self.nz):
                for j in range(i, self.nz):
                    intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)*w[i](rs)*w[j](rs) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                    cls[L, i, j] = integrate.romb(intvals, dr)
                    cls[L, j, i] = cls[L, i, j]
            cls[L, :,:] = cls[L, :,:] + self.N
                    
        return cls

    def compute_derivatives(self, theta_fiducial, h):

        # Holder for derivatives
        dCdt = np.zeros((self.npar, self.nz, self.nz, self.n_ell_bins))

        # Derivatives wrt cosmological parameters
        for i in range(self.npar - self.nz):
            theta_plus = np.copy(theta_fiducial)
            theta_plus[i] += h[i]
            theta_minus = np.copy(theta_fiducial)
            theta_minus[i] -= h[i]
            
            Cp = self.power_spectrum(theta_plus)
            Cm = self.power_spectrum(theta_minus)
            
            dCdt[i, :, :, :] = (Cp - Cm)/(2*h[i])
        
        # Derivatives wrt photo-z bias parameters
        for i in range(self.nz):
            
            pzprime = [interpolate.InterpolatedUnivariateSpline(np.linspace(0, self.pz[0].get_knots()[-1], len(self.pz[0].get_knots())), np.zeros(len(self.pz[0].get_knots()))) for X in range(self.nz)]
            pzprime[i] = self.pz[i].derivative()
            dCdt[self.npar-self.nz+i, :, :, :] = self.d_power_spectrum_db(theta_fiducial, pzprime)

        return dCdt


    # Compute the data vector
    def d_power_spectrum_db(self, theta, pzprime):

        # Evaluate the required (derived) cosmological parameters
        omm = theta[0]
        sigma8 = theta[1]*np.sqrt(0.3/theta[0])
        omb = theta[2]
        h = theta[3]
        ns = theta[4]
        omde = 1.0 - omm
        omnu = 0
        omk = 0
        hubble = h*100
        w0 = -1.
        wa = 0
        
        # Initialize cosmology object
        cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=0)

        # Numerics parameters
        zmax = 2
        rmax = cosmo.a2chi(z2a(zmax))
        power_zpoints = int(np.ceil(5*zmax))
        power_kpoints = 200
        distance_zpoints = int(np.ceil(10*zmax))
        wpoints = int(np.ceil(15*zmax))
        kmax = 10
        clpoints = 2**7 + 1

        # Compute the matter power spectrum at the cosmology
        z = np.linspace(0, zmax, power_zpoints)
        logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
        logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))

        # 2D linear interpolator for P(k;z)
        logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

        # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
        zvalues = np.linspace(0, zmax, distance_zpoints)
        rvalues = np.zeros((len(zvalues)))

        # Perform integration to compute r(z) at specified points according to cosmology
        for i in range(0, len(zvalues)):
            rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i])

        # Generate interpolation functions to give r(z) and z(r) given cosmology
        r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
        z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

        # Set the maximum comoving distance corresponding to the maximum redshift
        rmax = rvalues[-1]

        # Compute lensing weights...

        w = []
        dwdb = []

        # Compute the weight function associated with each bin in turn, over r-points and then interpolate
        for i in range(0, self.nz):

            # r-points to evaluate weight at before interpolation
            rpoints = np.linspace(0, rmax, wpoints)

            # Initialize weights
            weight = np.zeros(wpoints)
            dweightdb = np.zeros(wpoints)
            # Compute integral for the rest of the points
            for j in range(1, wpoints):
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*self.pz[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                weight[j] = integrate.romb(intvals, dx)
                
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*pzprime[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                dweightdb[j] = integrate.romb(intvals, dx)

            # Interpolate (generate interpolation function) and add interpolation function to the array w
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
            w.append(interp)
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, dweightdb, k = 3)
            dwdb.append(interp)
        
        # Tensor for cls
        cls = np.zeros((self.n_ell_bins, self.nz, self.nz))

        # Pull required cosmological parameters out of cosmo
        r_hubble = sc.c/(1000*hubble)
        A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

        # Compute Cls
        for L in range(self.n_ell_bins):
            l = self.modes[L]
            rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
            dr = rs[1] - rs[0]
            for i in range(0, self.nz):
                for j in range(i, self.nz):
                    intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)* (w[i](rs)*dwdb[j](rs) + dwdb[i](rs)*w[j](rs)) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                    cls[L, i, j] = integrate.romb(intvals, dr)
                    cls[L, j, i] = cls[L, i, j]
            cls[L,:,:] = cls[L,:,:]
                    
        return cls



# Cosmic shear class
class TomographicCosmicShearDE():

    def __init__(self, pz = None, lmin = 10, lmax = 3000, n_ell_bins = 10, Area = 15000, sigma_e = 0.3, nbar = 30):

        # Photo-z distributions
        self.pz = pz
        self.nz = len(pz)

        # ell binning
        self.npar = 6
        self.n_ell_bins = n_ell_bins
        modes_edges = np.logspace(np.log10(lmin), np.log10(lmax), self.n_ell_bins+1)
        self.modes = (modes_edges[0:-1] + modes_edges[1:])/2
        l = np.arange(lmin, lmax)
        self.nl = np.array([sum((2*l[(l >= modes_edges[i])*(l < modes_edges[i+1])] + 1)) for i in range(len(self.modes))], dtype=int)
        
        # Partial sky factor
        self.Area = Area*(np.pi/180)**2
        self.fsky = self.Area/(4*np.pi)
        self.nl = np.array(self.nl*self.fsky, dtype=int)
    
        # Noise covariance
        self.sigma_e = sigma_e
        self.nbar = (nbar/self.nz)*(60*180./np.pi)**2
        self.N = (self.sigma_e**2/self.nbar)*np.eye(self.nz)

    def simulate(self, theta, seed):
        
        # Set the seed
        np.random.seed(seed)
    
        # Compute theory power spectrum
        C = self.power_spectrum(theta)
    
        # Realize noisy power spectrum
        C_hat = np.zeros((self.n_ell_bins, self.nz, self.nz))
        for i in range(self.n_ell_bins):
            C_hat[i, :, :] = wishart.rvs(df=self.nl[i], scale=C[i, :,:])/self.nl[i]
    
        return C_hat
    
    # Compute the data vector
    def power_spectrum(self, theta):

        # Evaluate the required (derived) cosmological parameters
        omm = theta[0]
        sigma8 = theta[1]*np.sqrt(0.3/theta[0])
        omb = theta[2]/theta[3]**2
        h = theta[3]
        ns = theta[4]
        omde = 1.0 - omm
        omnu = 0
        omk = 0
        hubble = h*100
        w0 = theta[5]
        wa = 0
        
        # Initialize cosmology object
        cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=wa)

        # Numerics parameters
        zmax = 2
        rmax = cosmo.a2chi(z2a(zmax))
        power_zpoints = int(np.ceil(5*zmax))
        power_kpoints = 200
        distance_zpoints = int(np.ceil(10*zmax))
        wpoints = int(np.ceil(15*zmax))
        kmax = 10
        clpoints = 2**7 + 1

        # Compute the matter power spectrum at the cosmology
        z = np.linspace(0, zmax, power_zpoints)
        logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
        logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))

        # 2D linear interpolator for P(k;z)
        logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

        # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
        zvalues = np.linspace(0, zmax, distance_zpoints)
        rvalues = np.zeros((len(zvalues)))

        # Perform integration to compute r(z) at specified points according to cosmology
        for i in range(0, len(zvalues)):
            rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i], divmax=100)

        # Generate interpolation functions to give r(z) and z(r) given cosmology
        r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
        z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

        # Set the maximum comoving distance corresponding to the maximum redshift
        rmax = rvalues[-1]

        # Compute lensing weights...

        w = []
        
        # Compute the weight function associated with each bin in turn, over r-points and then interpolate
        for i in range(0, self.nz):

            # r-points to evaluate weight at before interpolation
            rpoints = np.linspace(0, rmax, wpoints)

            # Initialize weights
            weight = np.zeros(wpoints)

            # Compute integral for the rest of the points
            for j in range(1, wpoints):
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*self.pz[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                weight[j] = integrate.romb(intvals, dx)

            # Interpolate (generate interpolation function) and add interpolation function to the array w
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
            w.append(interp)
        
        # Tensor for cls
        cls = np.zeros((self.n_ell_bins, self.nz, self.nz))

        # Pull required cosmological parameters out of cosmo
        r_hubble = sc.c/(1000*hubble)
        A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

        # Compute Cls
        for L in range(self.n_ell_bins):
            l = self.modes[L]
            rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
            dr = rs[1] - rs[0]
            for i in range(0, self.nz):
                for j in range(i, self.nz):
                    intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)*w[i](rs)*w[j](rs) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                    cls[L, i, j] = integrate.romb(intvals, dr)
                    cls[L, j, i] = cls[L, i, j]
            cls[L, :,:] = cls[L, :,:] + self.N
        
        return cls
    
    # Compute the data vector
    def power_spectrum_noiseless(self, theta):

        # Evaluate the required (derived) cosmological parameters
        omm = theta[0]
        sigma8 = theta[1]*np.sqrt(0.3/theta[0])
        omb = theta[2]/theta[3]**2
        h = theta[3]
        ns = theta[4]
        omde = 1.0 - omm
        omnu = 0
        omk = 0
        hubble = h*100
        w0 = theta[5]
        wa = 0
        
        # Initialize cosmology object
        cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=0)

        # Numerics parameters
        zmax = 2
        rmax = cosmo.a2chi(z2a(zmax))
        power_zpoints = int(np.ceil(5*zmax))
        power_kpoints = 200
        distance_zpoints = int(np.ceil(10*zmax))
        wpoints = int(np.ceil(15*zmax))
        kmax = 10
        clpoints = 2**7 + 1

        # Compute the matter power spectrum at the cosmology
        z = np.linspace(0, zmax, power_zpoints)
        logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
        logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))

        # 2D linear interpolator for P(k;z)
        logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

        # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
        zvalues = np.linspace(0, zmax, distance_zpoints)
        rvalues = np.zeros((len(zvalues)))

        # Perform integration to compute r(z) at specified points according to cosmology
        for i in range(0, len(zvalues)):
            rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i], divmax=100)

        # Generate interpolation functions to give r(z) and z(r) given cosmology
        r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
        z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

        # Set the maximum comoving distance corresponding to the maximum redshift
        rmax = rvalues[-1]

        # Compute lensing weights...

        w = []
        
        # Compute the weight function associated with each bin in turn, over r-points and then interpolate
        for i in range(0, self.nz):

            # r-points to evaluate weight at before interpolation
            rpoints = np.linspace(0, rmax, wpoints)

            # Initialize weights
            weight = np.zeros(wpoints)

            # Compute integral for the rest of the points
            for j in range(1, wpoints):
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*self.pz[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                weight[j] = integrate.romb(intvals, dx)

            # Interpolate (generate interpolation function) and add interpolation function to the array w
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
            w.append(interp)
        
        # Tensor for cls
        cls = np.zeros((self.n_ell_bins, self.nz, self.nz))

        # Pull required cosmological parameters out of cosmo
        r_hubble = sc.c/(1000*hubble)
        A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

        # Compute Cls
        for L in range(self.n_ell_bins):
            l = self.modes[L]
            rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
            dr = rs[1] - rs[0]
            for i in range(0, self.nz):
                for j in range(i, self.nz):
                    intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)*w[i](rs)*w[j](rs) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                    cls[L, i, j] = integrate.romb(intvals, dr)
                    cls[L, j, i] = cls[L, i, j]
            cls[L, :,:] = cls[L, :,:]
        
        return cls


    def compute_derivatives(self, theta_fiducial, step):

        # Holder for derivatives
        dCdt = np.zeros((self.npar, self.nz, self.nz, self.n_ell_bins))

        # Derivatives wrt cosmological parameters
        for i in range(self.npar):
            theta_plus = np.copy(theta_fiducial)
            theta_plus[i] += step[i]
            theta_minus = np.copy(theta_fiducial)
            theta_minus[i] -= step[i]
            
            Cp = self.power_spectrum_noiseless(theta_plus)
            Cm = self.power_spectrum_noiseless(theta_minus)
            
            dCdt[i, :, :, :] = (Cp - Cm)/(2*step[i])

        return dCdt

# Cosmic shear class
class TomographicCosmicShearPhotozDE():

    def __init__(self, pz = None, lmin = 10, lmax = 3000, n_ell_bins = 10, Area = 15000, sigma_e = 0.3, nbar = 30):

        # Photo-z distributions
        self.pz = pz
        self.nz = len(pz)

        # ell binning
        self.npar = 6 + len(pz)
        self.n_ell_bins = n_ell_bins
        modes_edges = np.logspace(np.log10(lmin), np.log10(lmax), self.n_ell_bins+1)
        self.modes = (modes_edges[0:-1] + modes_edges[1:])/2
        l = np.arange(lmin, lmax)
        self.nl = np.array([sum((2*l[(l >= modes_edges[i])*(l < modes_edges[i+1])] + 1)) for i in range(len(self.modes))], dtype=int)
        
        # Partial sky factor
        self.Area = Area*(np.pi/180)**2
        self.fsky = self.Area/(4*np.pi)
        self.nl = np.array(self.nl*self.fsky, dtype=int)

        # Noise covariance
        self.sigma_e = sigma_e
        self.nbar = (nbar/self.nz)*(60*180./np.pi)**2
        self.N = (self.sigma_e**2/self.nbar)*np.eye(self.nz)

    def simulate(self, theta, seed):
        
        # Set the seed
        np.random.seed(seed)
    
        # Compute theory power spectrum
        C = self.power_spectrum(theta)
    
        # Realize noisy power spectrum
        C_hat = np.zeros((self.n_ell_bins, self.nz, self.nz))
        for i in range(self.n_ell_bins):
            C_hat[i, :, :] = wishart.rvs(df=self.nl[i], scale=C[i, :,:])/self.nl[i]
    
        return C_hat

    # Compute the data vector
    def power_spectrum(self, theta):

        # Evaluate the required (derived) cosmological parameters
        omm = theta[0]
        sigma8 = theta[1]*np.sqrt(0.3/theta[0])
        omb = theta[2]/theta[3]**2
        h = theta[3]
        ns = theta[4]
        omde = 1.0 - omm
        omnu = 0
        omk = 0
        hubble = h*100
        w0 = theta[5]
        wa = 0
        
        # Initialize cosmology object
        cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=0)

        # Photo-z parameters
        z = np.linspace(0, self.pz[0].get_knots()[-1], len(self.pz[0].get_knots()))
        pz_new = [0]*self.nz
        for i in range(self.nz):
            p = self.pz[i](z+theta[6+i])
            p = p/np.trapz(p, z)
            pz_new[i] = interpolate.InterpolatedUnivariateSpline(z, p, k=3)
        
        # Numerics parameters
        zmax = 2
        rmax = cosmo.a2chi(z2a(zmax))
        power_zpoints = int(np.ceil(5*zmax))
        power_kpoints = 200
        distance_zpoints = int(np.ceil(10*zmax))
        wpoints = int(np.ceil(15*zmax))
        kmax = 10
        clpoints = 2**7 + 1

        # Compute the matter power spectrum at the cosmology
        z = np.linspace(0, zmax, power_zpoints)
        logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
        logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))

        # 2D linear interpolator for P(k;z)
        logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

        # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
        zvalues = np.linspace(0, zmax, distance_zpoints)
        rvalues = np.zeros((len(zvalues)))

        # Perform integration to compute r(z) at specified points according to cosmology
        for i in range(0, len(zvalues)):
            rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i], divmax=100)

        # Generate interpolation functions to give r(z) and z(r) given cosmology
        r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
        z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

        # Set the maximum comoving distance corresponding to the maximum redshift
        rmax = rvalues[-1]

        # Compute lensing weights...

        w = []
        
        # Compute the weight function associated with each bin in turn, over r-points and then interpolate
        for i in range(0, self.nz):

            # r-points to evaluate weight at before interpolation
            rpoints = np.linspace(0, rmax, wpoints)

            # Initialize weights
            weight = np.zeros(wpoints)

            # Compute integral for the rest of the points
            for j in range(1, wpoints):
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*pz_new[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                weight[j] = integrate.romb(intvals, dx)

            # Interpolate (generate interpolation function) and add interpolation function to the array w
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
            w.append(interp)
        
        # Tensor for cls
        cls = np.zeros((self.n_ell_bins, self.nz, self.nz))

        # Pull required cosmological parameters out of cosmo
        r_hubble = sc.c/(1000*hubble)
        A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

        # Compute Cls
        for L in range(self.n_ell_bins):
            l = self.modes[L]
            rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
            dr = rs[1] - rs[0]
            for i in range(0, self.nz):
                for j in range(i, self.nz):
                    intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)*w[i](rs)*w[j](rs) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                    cls[L, i, j] = integrate.romb(intvals, dr)
                    cls[L, j, i] = cls[L, i, j]
            cls[L, :,:] = cls[L, :,:] + self.N
        
        return cls

    def compute_derivatives(self, theta_fiducial, h):

        # Holder for derivatives
        dCdt = np.zeros((self.npar, self.nz, self.nz, self.n_ell_bins))

        # Derivatives wrt cosmological parameters
        for i in range(self.npar - self.nz):
            theta_plus = np.copy(theta_fiducial)
            theta_plus[i] += h[i]
            theta_minus = np.copy(theta_fiducial)
            theta_minus[i] -= h[i]
            
            Cp = self.power_spectrum(theta_plus)
            Cm = self.power_spectrum(theta_minus)
            
            dCdt[i, :, :, :] = (Cp - Cm)/(2*h[i])
        
        # Derivatives wrt photo-z bias parameters
        for i in range(self.nz):
            
            pzprime = [interpolate.InterpolatedUnivariateSpline(np.linspace(0, self.pz[0].get_knots()[-1], len(self.pz[0].get_knots())), np.zeros(len(self.pz[0].get_knots()))) for X in range(self.nz)]
            pzprime[i] = self.pz[i].derivative()
            dCdt[self.npar-self.nz+i, :, :, :] = self.d_power_spectrum_db(theta_fiducial, pzprime)

        return dCdt


    # Compute the data vector
    def d_power_spectrum_db(self, theta, pzprime):

        # Evaluate the required (derived) cosmological parameters
        omm = theta[0]
        sigma8 = theta[1]*np.sqrt(0.3/theta[0])
        omb = theta[2]/theta[3]**2
        h = theta[3]
        ns = theta[4]
        omde = 1.0 - omm
        omnu = 0
        omk = 0
        hubble = h*100
        w0 = theta[5]
        wa = 0
        
        # Initialize cosmology object
        cosmo = cosmology(Omega_m=omm, Omega_de=omde, Omega_b=omb, h=h, n=ns, sigma8=sigma8, w0=w0, wa=0)

        # Numerics parameters
        zmax = 2
        rmax = cosmo.a2chi(z2a(zmax))
        power_zpoints = int(np.ceil(5*zmax))
        power_kpoints = 200
        distance_zpoints = int(np.ceil(10*zmax))
        wpoints = int(np.ceil(15*zmax))
        kmax = 10
        clpoints = 2**7 + 1

        # Compute the matter power spectrum at the cosmology
        z = np.linspace(0, zmax, power_zpoints)
        logk = np.log(np.logspace(-3, np.log10(kmax), power_kpoints))
        logpkz = np.log(cosmo.pk(np.exp(logk), z2a(z)))

        # 2D linear interpolator for P(k;z)
        logpkz = interpolate.RectBivariateSpline(logk, z, logpkz, kx=3, ky=3)

        # Generate list of z-values at which we will compute r(z), initialize array of r-values to hold computed values of r(z)
        zvalues = np.linspace(0, zmax, distance_zpoints)
        rvalues = np.zeros((len(zvalues)))

        # Perform integration to compute r(z) at specified points according to cosmology
        for i in range(0, len(zvalues)):
            rvalues[i] = integrate.romberg(lambda x: 1.0/np.sqrt(omm*(1+x)**3 + omnu*(1+x)**4+omk*(1+x)**2 + omde*np.exp(-3*wa*x/(1+x))*(1+x)**(3*(1+w0+wa))), 0, zvalues[i])

        # Generate interpolation functions to give r(z) and z(r) given cosmology
        r = interpolate.InterpolatedUnivariateSpline(zvalues, rvalues, k = 3)
        z = interpolate.InterpolatedUnivariateSpline(rvalues, zvalues, k = 3)

        # Set the maximum comoving distance corresponding to the maximum redshift
        rmax = rvalues[-1]

        # Compute lensing weights...

        w = []
        dwdb = []

        # Compute the weight function associated with each bin in turn, over r-points and then interpolate
        for i in range(0, self.nz):

            # r-points to evaluate weight at before interpolation
            rpoints = np.linspace(0, rmax, wpoints)

            # Initialize weights
            weight = np.zeros(wpoints)
            dweightdb = np.zeros(wpoints)
            # Compute integral for the rest of the points
            for j in range(1, wpoints):
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*self.pz[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                weight[j] = integrate.romb(intvals, dx)
                
                x = np.linspace(rpoints[j], rmax, 2**6 + 1)
                dx = x[1] - x[0]
                intvals = rpoints[j]*pzprime[i](z(x)) * h*cosmo.H(z2a(z(x))) * (1.0/hubble) * (x-rpoints[j])/x
                dweightdb[j] = integrate.romb(intvals, dx)

            # Interpolate (generate interpolation function) and add interpolation function to the array w
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, weight, k = 3)
            w.append(interp)
            interp = interpolate.InterpolatedUnivariateSpline(rpoints, dweightdb, k = 3)
            dwdb.append(interp)
        
        # Tensor for cls
        cls = np.zeros((self.n_ell_bins, self.nz, self.nz))

        # Pull required cosmological parameters out of cosmo
        r_hubble = sc.c/(1000*hubble)
        A = (1000/sc.c)**3*(9*omm**2*hubble**3/(4*h**3))

        # Compute Cls
        for L in range(self.n_ell_bins):
            l = self.modes[L]
            rs = np.linspace(r(l/(h*r_hubble*kmax)), rmax, clpoints)
            dr = rs[1] - rs[0]
            for i in range(0, self.nz):
                for j in range(i, self.nz):
                    intvals = ((l/(l+0.5))**4)*A*(1.0/rs**2)* (w[i](rs)*dwdb[j](rs) + dwdb[i](rs)*w[j](rs)) * (1+z(rs))**2 * np.exp(logpkz.ev(np.log((l+0.5)/(h*rs*r_hubble)), z(rs)))
                    cls[L, i, j] = integrate.romb(intvals, dr)
                    cls[L, j, i] = cls[L, i, j]
            cls[L,:,:] = cls[L,:,:]
        
        return cls
