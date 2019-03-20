import numpy as np

def b14_parse(z_min=None, z_max=None, qual_cut=False, \
        jla_path='/Users/sfeeney/Software_Packages/jla_v6/jla_likelihood_v6/data/'):

    # read lightcurve data
    #print '* reading B14 inputs'
    data = np.genfromtxt(jla_path + 'jla_lcparams.txt', \
                         dtype = None, names = True)
    n_sn_in = len(data)

    # cut if desired
    inds = (np.arange(n_sn_in),)
    if z_min is not None:
        if z_max is not None:
            inds = np.where((data['zcmb'] > z_min) & \
                            (data['zcmb'] < z_max))
        else:
            inds = np.where(z_min < data['zcmb'])
    elif z_max is not None:
        inds = np.where(data['zcmb'] < z_max)
    data = data[inds]
    n_sn = len(data)

    # read V (non-diagonal) covariance matrices
    cmats = {'v0': np.zeros((n_sn, n_sn)), \
             'va': np.zeros((n_sn, n_sn)), \
             'vb': np.zeros((n_sn, n_sn)), \
             'v0a': np.zeros((n_sn, n_sn)), \
             'v0b': np.zeros((n_sn, n_sn)), \
             'vab': np.zeros((n_sn, n_sn))}
    for cmat in cmats:
        d = np.genfromtxt(jla_path + 'jla_' + cmat + '_covmatrix.dat')
        #for i in range(n_sn):
        #    cmats[cmat][i, :] = d[i * n_sn + 1: (i + 1) * n_sn + 1]
        for i in range(n_sn):
            cmats[cmat][i, :] = d[inds[0][i] * n_sn_in + 1 + inds[0]]
        #print np.allclose(cmats[cmat], cmats[cmat].T)
#print '* B14 inputs read'

    return data, cmats

def b14_covariance(data, cmats, alpha, beta):
    n_sn = len(data)
    c_mat = cmats['v0'] + alpha ** 2 * cmats['va'] + \
            beta ** 2 * cmats['vb'] + 2 * alpha * cmats['v0a'] - \
            2 * beta * cmats['v0b'] - \
            2 * alpha * beta * cmats['vab']
    d_mat = data['dmb'] ** 2 + (alpha * data['dx1']) ** 2 + \
            (beta * data['dcolor']) ** 2 + \
            2 * alpha * data['cov_m_s'] - \
            2 * beta * data['cov_m_c'] - \
            2 * alpha * beta * data['cov_s_c']
    return c_mat + np.diag(d_mat)

def b14_covariance_derivative(data, cmats, alpha, beta):
    
    n_sn = len(data)
    
    dC = np.zeros((2, n_sn, n_sn))
    
    # Derivatives w.r.t. alpha
    c_mat = 2*alpha*cmats['va'] + 2*cmats['v0a'] - 2*beta*cmats['vab']
    d_mat = 2*alpha*data['dx1']**2 + 2*data['cov_m_s'] - 2*beta*data['cov_s_c']
    dC[0,:,:] = c_mat + np.diag(d_mat)
    
    # Derivatives w.r.t. beta
    c_mat = 2*beta*cmats['vb'] - 2*cmats['v0b'] - 2*alpha*cmats['vab']
    d_mat = 2*beta*data['dcolor']**2 - 2*data['cov_m_c'] - 2*alpha*data['cov_s_c']
    dC[1,:,:] = c_mat + np.diag(d_mat)
    
    return dC

def b14_chi_sq(pars, data, cmats, h_0 = 70.0, delta_m_cut = 10.0):
    alpha, beta, abs_mag, delta_m, om_m = pars
    mu = z2mu(data['zcmb'], om_m, h_0)
    res = data['mb'] - (abs_mag - alpha * data['x1'] + \
                        beta * data['color'] + \
                        delta_m * (data['3rdvar'] > delta_m_cut)) - mu
    cov_mat = b14_cov_mat(data, cmats, alpha, beta)
    cov_mat_chol = np.linalg.cholesky(cov_mat)
    white_res = np.linalg.solve(cov_mat_chol, res)
    chi_sq = np.dot(white_res, white_res)
    return chi_sq

def emcee_b14_ln_p(pars, data, cmats, h_0 = 70.0, delta_m_cut = 10.0):
    if 0.0 < pars[0] < 0.5 and 0.0 < pars[1] < 6.0 and -25.0 < pars[2] < -15.0 and -0.5 < pars[3] < 0.5 and 0.2 < pars[4] < 0.4:
        return -b14_chi_sq(pars, data, cmats, h_0, delta_m_cut) / 2.0
    else:
        return -np.inf
