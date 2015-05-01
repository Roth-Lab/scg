'''
Created on 2015-01-23

@author: Andrew Roth
'''
from __future__ import division

from scipy.misc import logsumexp as log_sum_exp
from scipy.special import gammaln as log_gamma, psi

import numpy as np
import pandas as pd

def compute_e_log_dirichlet(x):
    return psi(x) - psi(np.expand_dims(x.sum(axis=-1), axis=-1))

def get_indicator_matrix(states, X):
    Y = np.zeros((len(states), X.shape[0], X.shape[1]))
    
    for i, s in enumerate(states):
        Y[i, :, :] = (X == s).astype(int)
    
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

def log_space_normalise(log_X, axis=0):
    return log_X - np.expand_dims(log_sum_exp(log_X, axis=axis), axis=axis) 
    
def init_Z(K, N, labels):
    if labels is None:
        labels = np.random.random(size=(N, K)).argmax(axis=1)
        
    log_Z = np.zeros((N, K))
    
    for i, s in enumerate(range(len(set(labels)))):
        log_Z[:, i] = (labels == s).astype(int)
    
    log_Z = np.log(log_Z + 1e-10)
    
    return log_space_normalise(log_Z, axis=1)

def load_labels(cell_ids, file_name):
    if file_name is None:
        labels = None
    
    else:
        labels = pd.read_csv(file_name, compression='gzip', index_col='cell_id', sep='\t')
        
        labels = labels.loc[cell_ids, 'cluster']
    
    return labels