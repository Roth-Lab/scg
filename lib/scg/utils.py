'''
Created on 2015-01-23

@author: Andrew Roth
'''
from __future__ import division

from scipy.special import gammaln as log_gamma, psi

import numpy as np

def compute_e_log_dirichlet(x):
    return psi(x) - psi(np.expand_dims(x.sum(axis=-1), axis=-1))

def get_indicator_matrix(states, X):
    Y = np.zeros((len(states), X.shape[0], X.shape[1]))
    
    for s in states:
        Y[s, :, :] = (X == s).astype(int)
    
    return Y

def compute_e_log_p_dirichlet(posterior, prior):
    log_p = log_gamma(prior.sum()) - \
            log_gamma(prior).sum() + \
            safe_multiply(prior - 1, compute_e_log_dirichlet(posterior)).sum()
 
    return log_p

def compute_e_log_q_dirichlet(x):
    a_0 = x.sum()
       
    K = len(x)
 
    return log_gamma(a_0) - log_gamma(x).sum() + safe_multiply(x - 1, psi(x)).sum() - safe_multiply(a_0 - K, psi(a_0))

def compute_e_log_q_discrete(log_x):
    return np.sum(safe_multiply(np.exp(log_x), log_x))

def safe_multiply(x, y):
    return np.sign(x) * np.sign(y) * np.exp(np.log(np.abs(x)) + np.log(np.abs(y)))