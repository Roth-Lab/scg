'''
Created on 2015-01-09

@author: Andrew Roth
'''
from __future__ import division

from scipy.misc import logsumexp as log_sum_exp
from scipy.special import psi, gammaln as log_gamma

import numpy as np

class VariationalBayesGenotyper(object):
    def __init__(self, alpha_prior, gamma_prior, kappa_prior, G_prior, state_map, X):
        self.alpha_prior = alpha_prior
        
        self.gamma_prior = gamma_prior
        
        self.kappa_prior = kappa_prior
        
        self.G_prior = G_prior
        
        self.state_map = state_map
        
        self.K = len(self.kappa_prior)
        
        self.M = X.shape[1]
        
        self.N = X.shape[0]
        
        self.S = gamma_prior.shape[0]
        
        self.T = gamma_prior.shape[1]
        
        self.X = get_indicator_matrix(range(self.T), X)
        
        self._init_G()
        
        self.alpha = alpha_prior.copy()
        
        self.gamma = gamma_prior.copy()
        
        self.kappa = np.ones(self.K)
        
        self.lower_bound = [float('-inf')]
        
        self._init_inverse_state_map()
        
        self._debug_lower_bound = [float('-inf')]
        
        self.converged = False
        
    def _init_G(self):
        G = np.random.random((self.S, self.K, self.M))
        
        G = G / np.expand_dims(G.sum(axis=0), axis=0)
        
        self.log_G = np.log(G)

    def _init_inverse_state_map(self):
        self._inverse_state_map = {}
        
        for s in self.state_map:
            for (u, v) in self.state_map[s]:
                self._inverse_state_map[(u, v)] = s
    
    @property
    def e_log_d(self):
        return compute_e_log_dirichlet(self.alpha)
    
    @property
    def e_log_epsilon(self):
        return compute_e_log_dirichlet(self.gamma)
    
    @property
    def e_log_pi(self):
        return compute_e_log_dirichlet(self.kappa)

    @property
    def G(self):
        return np.exp(self.log_G)

    @property
    def Z(self):
        return np.exp(self.log_Z)
    
    @property
    def Z_0(self):
        return np.exp(self.log_Z_0)
    
    @property
    def Z_1(self):
        return np.exp(self.log_Z_1)
    
    @property
    def Z_1_k_k(self):
        return np.exp(self.log_Z_1_k_k)
    
    @property
    def log_Z_1_k_k(self):
        return np.diagonal(self.log_Z_1, axis1=1, axis2=2)

    @property
    def log_Z_0(self):
        return self.log_Z[:, :self.K]
    
    @property
    def log_Z_1(self):
        return self.log_Z[:, self.K:].reshape(self.N, self.K, self.K)
    
    @property
    def Y(self):
        return np.exp(self.log_Y)
    
    @property
    def log_Y(self):
        return np.vstack((log_sum_exp(self.log_Z_0, axis=1),
                          log_sum_exp(self.log_Z_1.reshape(self.N, self.K * self.K), axis=1)))
    
    def fit(self, convergence_tolerance=1e-6, debug=False, num_iters=100):
        for i in range(num_iters):

            self._update_Z()
            
            if debug:
                print 'a', self._diff_lower_bound()

            self._update_G()
            
            if debug:
                print 'c', self._diff_lower_bound()
            
            self._update_gamma()
            
            if debug:
                print 'd', self._diff_lower_bound()
            
            self._update_kappa()
            
            if debug:
                print 'e', self._diff_lower_bound()
            
            self._update_alpha()
            
            if debug:
                print 'f', self._diff_lower_bound()
    
                print self.alpha
                   
                print sum((self.kappa / self.kappa.sum()) > 1e-2)

            
            self.lower_bound.append(self._compute_lower_bound())
             
            diff = (self.lower_bound[-1] - self.lower_bound[-2]) / abs(self.lower_bound[-1])
             
            print i, self.lower_bound[-1], diff
             
            if abs(diff) < convergence_tolerance:
                print 'Converged'
                
                self.converged = True
                
                break
            
            elif diff < 0:
                print 'Lower bound decreased'
                
                if not debug:
                    self.converged = False
                    
                    break            
            
    def _update_G(self):
        # TxNxKxM
        singlet_term = self.Z_0[np.newaxis, :, :, np.newaxis] * self.X[:, :, np.newaxis, :]
        
        # TxKxM
        singlet_term = singlet_term.sum(axis=1)
        
        # SxTxKxM
        singlet_term = self.e_log_epsilon[:, :, np.newaxis, np.newaxis] * singlet_term[np.newaxis, :, :, :]
        
        # SxKxM
        singlet_term = singlet_term.sum(axis=1)

        # TxNxKxKxM
        doublet_diff_term_temp = self.Z_1[np.newaxis, :, :, :, np.newaxis] * self.X[:, :, np.newaxis, np.newaxis, :]

        # TxKxKxM
        doublet_diff_term_temp = doublet_diff_term_temp.sum(axis=1)

        # SxTxKxKxM
        doublet_diff_term_temp = np.exp(self.log_G[:, np.newaxis, np.newaxis, :, :] + np.log(doublet_diff_term_temp[np.newaxis, :, :, :, :] + 1e-100))
        
#         doublet_diff_term_temp = self.log_G[:, np.newaxis, np.newaxis, :, :] * doublet_diff_term_temp[np.newaxis, :, :, :, :]

        # SxTxKxM
        doublet_diff_term_temp = doublet_diff_term_temp.sum(axis=3) - np.swapaxes(np.diagonal(doublet_diff_term_temp, axis1=2, axis2=3), -2, -1)

        doublet_diff_term = np.zeros(doublet_diff_term_temp.shape)
        
        for s in self.state_map:
            for w in self.state_map:
                for (u, v) in self.state_map[w]:
                    if(u != s) and (v != s):
                        continue
                     
                    elif (u == s):
                        doublet_diff_term[s, :, :, :] += self.e_log_epsilon[w, :, np.newaxis, np.newaxis] * doublet_diff_term_temp[v, :, :, :]
                         
                    elif (v == s) :
                        doublet_diff_term[s, :, :, :] += self.e_log_epsilon[w, :, np.newaxis, np.newaxis] * doublet_diff_term_temp[u, :, :, :]
        
        # SxKxM
        doublet_diff_term = doublet_diff_term.sum(axis=1)

        # TxNxKxM
        doublet_same_term_temp = self.Z_1_k_k[np.newaxis, :, :, np.newaxis] * self.X[:, :, np.newaxis, :]
        
        # TxKxM
        doublet_same_term_temp = doublet_same_term_temp.sum(axis=1)
        
        # SxTxKxM
        doublet_same_term = np.zeros((self.S, self.T, self.K, self.M))
      
        for s in self.state_map:
            w = self._inverse_state_map[(s, s)]

            doublet_same_term[s, :, :, :] = self.e_log_epsilon[w, :, np.newaxis, np.newaxis] * doublet_same_term_temp[np.newaxis, :, :, :]
        
        # SxKxM
        doublet_same_term = doublet_same_term.sum(axis=1)        
        
        # SxKxM
        data_term = singlet_term + doublet_diff_term + doublet_same_term
        
        log_G = np.log(self.G_prior[:, np.newaxis, np.newaxis]) + data_term
        
        log_G = log_G - np.expand_dims(log_sum_exp(log_G, axis=0), axis=0)
        
        self.log_G = log_G
  
    def _update_Z(self):
        # SxTxNxKxM
        singlet_term = self.G[:, np.newaxis, np.newaxis, :, :] * self.X[np.newaxis, :, :, np.newaxis, :]
        
        # SxTxNxK
        singlet_term = singlet_term.sum(axis=-1)
        
        singlet_term = safe_multiply(self.e_log_epsilon[:, :, np.newaxis, np.newaxis], singlet_term)
        
        # NxK
        singlet_term = singlet_term.sum(axis=(0, 1))
        
        singlet_term = self.e_log_d[0] + self.e_log_pi[np.newaxis, :] + singlet_term
        
        # SxKxKxM
        doublet_diff_term = self._get_G_G_marginalised()
        
        # SxTxNxKxKxM
        doublet_diff_term = self.X[np.newaxis, :, :, np.newaxis, np.newaxis, :] * doublet_diff_term[:, np.newaxis, np.newaxis, :, :, :]
        
        # SxTxNxKxK
        doublet_diff_term = doublet_diff_term.sum(axis=-1)
        
        doublet_diff_term = safe_multiply(self.e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis], doublet_diff_term)
        
        # NxKxK
        doublet_diff_term = doublet_diff_term.sum(axis=(0, 1))
        
        doublet_diff_term = self.e_log_d[1] + self.e_log_pi[np.newaxis, :, np.newaxis] + self.e_log_pi[np.newaxis, np.newaxis, :] + doublet_diff_term
        
        # SxKxM
        doublet_same_term = self._get_G_same_marginalised()
        
        # SxTxNxKxM
        doublet_same_term = self.X[np.newaxis, :, :, np.newaxis, :] * doublet_same_term[:, np.newaxis, np.newaxis, :, :]
        
        # SxTxNxK
        doublet_same_term = doublet_same_term.sum(axis=-1)
        
        doublet_same_term = safe_multiply(self.e_log_epsilon[:, :, np.newaxis, np.newaxis], doublet_same_term)
        
        # NxK
        doublet_same_term = doublet_same_term.sum(axis=(0, 1))
        
        doublet_same_term = self.e_log_d[1] + 2 * self.e_log_pi[np.newaxis, :] + doublet_same_term
            
        log_Z_1 = singlet_term
        
        log_Z_2 = doublet_diff_term
        
        for k in range(self.K):
            log_Z_2[:, k, k] = doublet_same_term[:, k]
        
        log_Z = np.concatenate((log_Z_1, log_Z_2.reshape((self.N, self.K * self.K))), axis=1)
        
        log_Z_nc = log_sum_exp(log_Z, axis=1)
        
        self.log_Z = log_Z - log_Z_nc[:, np.newaxis]
 
    def _update_alpha(self):
        self.alpha = self.alpha_prior + self._get_alpha_data_term()
    
    def _update_kappa(self):
        self.kappa = self.kappa_prior + self._get_kappa_data_term()
    
    def _update_gamma(self):
        self.gamma = self.gamma_prior + self._get_gamma_data_term()
    
    def _compute_lower_bound(self):
        return self._compute_e_log_p() - self._compute_e_log_q()
    
    def _compute_e_log_p(self):
        alpha_prior = compute_e_log_p_dirichlet(self.alpha, self.alpha_prior)
        
        alpha_posterior = self._compute_e_log_p_alpha_posterior()

        gamma_prior = sum([compute_e_log_p_dirichlet(x, y) for x, y in zip(self.gamma, self.gamma_prior)])

        gamma_posterior = self._compute_e_log_p_gamma_posterior()
        
        kappa_prior = compute_e_log_p_dirichlet(self.kappa, self.kappa_prior)
        
        kappa_posterior = self._compute_e_log_p_kappa_posterior()
        
        G_prior = self._compute_log_p_G()
        
        return sum([alpha_prior,
                    alpha_posterior,
                    gamma_prior,
                    gamma_posterior,
                    kappa_prior,
                    kappa_posterior,
                    G_prior])
    
    def _compute_e_log_p_alpha_posterior(self):
        return np.sum(safe_multiply(self.e_log_d, self._get_alpha_data_term()))
    
    def _compute_e_log_p_kappa_posterior(self):
        return np.sum(safe_multiply(self.e_log_pi, self._get_kappa_data_term()))
    
    def _compute_e_log_p_gamma_posterior(self):
        return np.sum(safe_multiply(self.e_log_epsilon, self._get_gamma_data_term()))
    
    def _compute_log_p_G(self):
        return np.sum(self.G_prior * self.G.sum(axis=(1, 2)))
            
    def _compute_e_log_q(self):
        log_q_d = compute_e_log_q_dirichlet(self.alpha)
        
        log_q_epsilon = sum([compute_e_log_q_dirichlet(x) for x in self.gamma])
      
        log_q_pi = compute_e_log_q_dirichlet(self.kappa)

        log_q_g = compute_e_log_q_discrete(self.log_G)
        
        log_q_z = compute_e_log_q_discrete(self.log_Z)

        return np.sum([log_q_d,
                       log_q_epsilon,
                       log_q_pi,
                       log_q_g,
                       log_q_z])
 
    def _get_G_G_marginalised(self):
        g_g = np.zeros((self.S, self.K, self.K, self.M))

        for s in self.state_map:
            for (u, v) in self.state_map[s]:
                g_g[s, :, :, :] += np.exp(self.log_G[u, :, np.newaxis, :] + self.log_G[v, np.newaxis, :, :])
#                 g_g[s, :, :, :] += self.G[u, :, np.newaxis, :] * self.G[v, np.newaxis, :, :]
 
        return g_g
    
    def _get_G_same_marginalised(self):
        g = np.zeros((self.S, self.K, self.M))
        
        for s in self.state_map:
            for (u, v) in self.state_map[s]:
                if u != v:
                    continue
                
                g[s, :, :] += self.G[u, :, :]
        
        return g
    
    def _get_alpha_data_term(self):
        return np.exp(log_sum_exp(self.log_Y, axis=1))
     
    def  _get_gamma_data_term(self):
        # SxNxKxM
        singlet_term = np.exp(self.log_Z_0[np.newaxis, :, :, np.newaxis] + self.log_G[:, np.newaxis, :, :])
        
        # SxKxKxM
        doublet_diff_term = self._get_G_G_marginalised()

        # SxNxKxKxM
        doublet_diff_term = safe_multiply(self.Z_1[np.newaxis, :, :, :, np.newaxis], doublet_diff_term[:, np.newaxis, :, :, :])
        
        # SxNxKxM
        doublet_diff_term = doublet_diff_term.sum(axis=3) - np.swapaxes(np.diagonal(doublet_diff_term, axis1=2, axis2=3), -2, -1)
      
        # SxKxM
        doublet_same_term = np.zeros((self.S, self.K, self.M))
        
        for s in self.state_map:
            for (u, v) in self.state_map[s]:
                if u != v:
                    continue
                
                doublet_same_term[s, :, :] += self.G[u, :, :]
        
        doublet_same_term = safe_multiply(doublet_same_term[:, np.newaxis, :, :], self.Z_1_k_k[np.newaxis, :, :, np.newaxis])

        data_term = singlet_term + doublet_diff_term + doublet_same_term

        # SxTxNxKxM
        data_term = data_term[:, np.newaxis, :, :, :] * self.X[np.newaxis, :, :, np.newaxis, :]

        # SxT
        return data_term.sum(axis=(2, 3, 4))     
    
    def _get_kappa_data_term(self):
        singlet_term = self.Z_0.sum(axis=0)
 
        doublet_term = self.Z_1.sum(axis=(0, 1)) + self.Z_1.sum(axis=(0, 2))

        return singlet_term + doublet_term

    def _diff_lower_bound(self):
        self._debug_lower_bound.append(self._compute_lower_bound())
        
        diff = self._debug_lower_bound[-1] - self._debug_lower_bound[-2]
        
        if diff < 0:
            print 'Bound decreased',
#             raise Exception(diff)
        
        return diff


#=======================================================================================================================
# Utility functions
#=======================================================================================================================
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

if __name__ == '__main__':
    from sklearn.metrics import v_measure_score
    
    from simulate import simualte_data
    
    np.seterr(all='warn')
    
    np.random.seed(11)
    
    num_iters = 100

    K = 40
    
    M = 48
    
    N = 50
    
    K_true = 9
    
    num_iters = 10
    
    state_map = {0 : [(0, 0)],
                 1 : [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
                 2 : [(2, 2)]}
    
    inverse_state_map = {}
    
    for s in state_map:
        for (u, v) in state_map[s]:
            inverse_state_map[(u, v)] = s
    

    alpha_prior = np.array([10, 1])
    
    gamma_prior = np.array([[98, 1, 1],
#                             [1, 98, 1],
                            [25, 50, 25],
                            [1, 1, 98]])
    
    S = gamma_prior.shape[0]
    
    G_prior = np.ones(S) * 1 / S
    
    kappa_prior = np.concatenate([np.ones(K_true), np.ones(K - K_true) * 1e-6])
    
    sim = simualte_data(alpha_prior, gamma_prior, kappa_prior, G_prior, M, N, inverse_state_map)
    
    print sim['Y'].sum()
    
    X = sim['X']
    
    kappa_prior = np.ones(K) * 1e-3
    
#     alpha_prior = np.array([9, 1])
#     
#     
#     gamma_prior = np.array([[, 1, 1],
#                             [1, 1, 1],
#                             [1, 1, 1]])
         
    
    model = VariationalBayesGenotyper(alpha_prior, gamma_prior, kappa_prior, G_prior, state_map, X)
    
#     model.log_G = np.log(get_indicator_matrix(range(S), sim['G']) + 1e-10)
    
    model.fit(num_iters=10)

    Z = model.Z_0.argmax(axis=1)
    
    doublet = model.Z_1.reshape(N, K * K)[model.Y.argmax(axis=0) == 1].argmax(axis=1)
    
    print zip(*np.unravel_index(doublet, (K, K)))
    
    print model.Y[1][sim['Y'] == 1]
    
    print sim['Y'][model.Y.argmax(axis=0) == 1]

    
#     print model.Z[1].argmax(axis=(1, 2)) > model.Z[0].argmax(axis=1)
   
    print v_measure_score(sim['Z'][0][sim['Y'] == 0], Z[sim['Y'] == 0])
