from scipy.spatial.distance import pdist, squareform
import numpy as np
import scipy.stats as scs



def reshape_samples(X,Y):
    """ Ensure that X and Y will be 2D arrays """
    U, V = X.copy(), Y.copy()
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    if len(X.shape) == 1:
        U = X[:, None]
    if len(Y.shape) == 1:
        V = Y[:, None]
    return U, V

def AB_matrices(U, V):
    """
    Returns the matrices needed to compute the distance correlation
    Must be executed after reshape_samples
    """    
    a = squareform(pdist(U))
    b = squareform(pdist(V))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    return A, B, a, b

def dist_corr(X,Y):
    """ Computes the distance correlation function """
    U, V = reshape_samples(X, Y)
    A, B, _, _ = AB_matrices(U, V)
    n = U.shape[0]

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    
    return dcor

def dist_cov(X, Y):
    """ Computes the distance covariance function """
    U, V = reshape_samples(X, Y)
    A, B, _, _ = AB_matrices(U, V)
    n = U.shape[0]
    
    dcov2_xy = (A * B).sum()/float(n * n)
    return dcov2_xy

def test_corr(X,Y,n_samples='None',n_mc=1):
    """ n_samples taken from X and Y for each iteration (n_mc iterations) """
    if n_samples == 'None':
        n_samples = len(X)
    n = len(X)
        
    U, V = reshape_samples(X, Y)
    
    stat_list = []
    for i in range(n_mc):
        ind = np.random.choice(n, n_samples, replace=False)
        X_m, Y_m = U[ind,:], V[ind,:]
        
        A, B, a, b = AB_matrices(X_m, Y_m)
    
        dcov2_xy = (A * B).sum()/float(n_samples * n_samples)
        S2_xy = a.sum()*b.sum()/float(n_samples**4)
    
        stat = n_samples*dcov2_xy/(S2_xy)
        stat_list.append(stat)
        
    stat = np.mean(stat_list)
    p_value = 2*(1 - scs.norm.cdf(np.sqrt(stat)))
    return stat, p_value
    