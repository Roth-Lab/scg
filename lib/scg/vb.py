'''
Created on 2015-01-09

@author: Andrew Roth
'''
from __future__ import division

from scipy.misc import logsumexp as log_sum_exp
from scipy.special import psi, gammaln as log_gamma

import numpy as np

class GenotyperVariationBayes(object):
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
        
        self.G = self._init_G()
        
        self.alpha = alpha_prior.copy()
        
        self.gamma = gamma_prior.copy()
        
        self.kappa = kappa_prior.copy()
        
        self._Z_2_indices = np.triu_indices(self.K, k=1)
        
        self.lower_bound = [float('-inf')]
    
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
    def Z_2_sym(self):
        return self.Z[1] + np.swapaxes(self.Z[1], 1, 2)
        
    def fit(self, num_iters=100, convergence_tolerance=1e-3):
        for i in range(num_iters):
            self._update_Z()

            print 'b', self._diff_lower_bound()
            
            self._update_G()
            
            print 'c', self._diff_lower_bound()
            
            self._update_gamma()
            
            print 'd', self._diff_lower_bound()
            
            self._update_kappa()
            
            print 'e', self._diff_lower_bound()
            
            self._update_alpha()
            
            print 'f', self._diff_lower_bound()
            
#             self.lower_bound.append(self._compute_lower_bound())
            
#             diff = self.lower_bound[-1] - self.lower_bound[-2]
            
#             if diff < 0:
#                 raise Exception('Lower bound decreased')
#             
#             elif diff < convergence_tolerance:
#                 break
#             
#             else:
#                 print i, self.lower_bound[-1]
    
    def _diff_lower_bound(self):
        self.lower_bound.append(self._compute_lower_bound())
        
        return self.lower_bound[-1] - self.lower_bound[-2]
    
    def _init_G(self):
        G = np.random.random((self.S, self.K, self.M))
        
        G = G / np.expand_dims(G.sum(axis=0), axis=0)
        
        return G

    def _update_Z(self):
        Z = [self._get_Z_1(),
             self._get_Z_2()]
        
        for n in range(self.N):
            Z[1][n] = np.triu(Z[1][n], k=1)
    
            log_norm_const = log_sum_exp(np.concatenate((Z[0][n],
                                                         Z[1][n][self._Z_2_indices].flatten())))
    
            Z[0][n] = np.exp(Z[0][n] - log_norm_const)
    
            Z[1][n] = np.exp(Z[1][n] - log_norm_const)
            
            Z[1][n] = np.triu(Z[1][n], k=1)
 
        self.Z = Z
    
    def _get_Z_1(self):
        X_e = self._get_X_cross_epsilon()

        G_X_e = np.expand_dims(self.G, axis=1) * np.expand_dims(X_e, axis=2)

        return self.e_log_d[1] + self.e_log_pi + np.sum(G_X_e, axis=(0, -1))
        
    def _get_Z_2(self):
        
        e_log_pi_aug = np.add.outer(self.e_log_pi, self.e_log_pi)
        
        G_G = self._get_G_G()
        
        X_e = self._get_X_cross_epsilon()
               
        G_G_X_e = np.zeros((self.S, self.N, self.K, self.K, self.M))
        
        print X_e.shape
        
        for s in range(self.S):
            for (u, v) in self.state_map[s]:
                G_G_X_e[s, :, :, :] += G_G[u, v, np.newaxis, :, :, :] * X_e[s, :, np.newaxis, np.newaxis, :]
        
        G_G_X_e = G_G_X_e.sum(axis=(0, -1))
        
        return self.e_log_d[1] + e_log_pi_aug + G_G_X_e

    def _update_G(self):
        G_1 = self._get_G_1()
        
        G_2 = self._get_G_2()
        
        prior = np.log(self.G_prior)[:, np.newaxis, np.newaxis]
        
        log_G =  prior + G_1 + G_2
        
        log_G = log_G - log_sum_exp(log_G, axis=0)
        
        self.G = np.exp(log_G)
        
    def _get_G_1(self):
        # TxNxKxM
        s_1 = self.X[:, :, np.newaxis, :] * self.Z[0][np.newaxis, :, :, np.newaxis]
        
        # TxKxM
        s_2 = s_1.sum(axis=1)
        
        # SxTxKxM
        s_3 = s_2[np.newaxis, :, :, :] * self.e_log_epsilon[:, :, np.newaxis, np.newaxis]
        
        # SxKxM
        return s_3.sum(axis=1)
    
    def _get_G_2(self):
        s_1 = np.zeros((self.S, self.S, self.K, self.M))
    
        for u in range(self.S):

            for k in range(self.K):

                for l in range(self.K):
                    
                    if k == l:
                        continue
                    
                    for s in range(self.S):                        
                
                        for (w, v) in self.state_map[s]:
                        
                            if u != w:
                                continue
            
                            s_1[u, s, k, :] += self.G[v, l, :]
        
        # SxSxTxKxM
        s_2 = self.e_log_epsilon[np.newaxis, :, :, np.newaxis, np.newaxis] * s_1[:, :, np.newaxis, :, :]
        
        # SxTxKxM
        s_3 = s_2.sum(axis=1)
        
        # SxTxNxKxM
        s_4 = self.X[np.newaxis, :, :, np.newaxis, :] * s_3[:, :, np.newaxis, :, :]
        
        # SxNxKxM
        s_5 = s_4.sum(axis=1)
        
        # SxNxKxKxM
        s_6 = self.Z_2_sym[np.newaxis, :, :, :, np.newaxis] * s_5[:, :, np.newaxis, :, :]
        
        # SxKxM
        return s_6.sum(axis=(1, 2))
    
    def _update_gamma(self):
        # TxNxKxM
        s_1 = self.Z[0][np.newaxis, :, :, np.newaxis] * self.X[:, :, np.newaxis, :]
        
        # TxKxM
        s_2 = s_1.sum(axis=1)
        
        # SxTxKxM
        s_3 = self.G[:, np.newaxis, :, :] * s_2[np.newaxis, :, :, :]
        
        # SxT
        s_4 = s_3.sum(axis=(2, 3))
        
        # TxNxKxKxM
        t_1 = self.X[:, :, np.newaxis, np.newaxis, :] * self.Z[1][np.newaxis, :, :, :, np.newaxis]
        
        # TxKxKxM
        X_Z = t_1.sum(axis=1)
        
        # SxSxKxKxM
        G_G = self._get_G_G()
        
        # SxTxKxKxM
        G_G_X_Z = np.zeros((self.S, self.T, self.K, self.K, self.M))
        
        for s in range(self.S):
            for (u, v) in self.state_map[s]:
                G_G_X_Z[s, :, :, :] += G_G[u, v, np.newaxis, :, :, :] * X_Z
    
        # SxT
        t_5 = G_G_X_Z.sum(axis=(2, 3, 4))
    
        self.gamma = self.gamma_prior + s_4 + t_5
    
    def _update_kappa(self):
        self.kappa = self.kappa_prior + self.Z[0].sum(axis=0) + self.Z_2_sym.sum(axis=(0, 2))
    
    def _update_alpha(self):
        self.alpha = np.array([self.alpha_prior[0] + self.Z[0].sum(), self.alpha_prior[1] + self.Z[1].sum()])    
    
    def _compute_lower_bound(self):
        return self._compute_e_log_p() - self._compute_e_log_q()
    
    def _compute_e_log_p(self):      
        return sum([self._compute_e_log_p_x_1(),
                    self._compute_e_log_p_x_2(),
                    compute_log_p_dirichlet(self.alpha, self.alpha_prior),
                    compute_log_p_dirichlet(self.gamma, self.gamma_prior),
                    compute_log_p_dirichlet(self.kappa, self.kappa_prior),
                    self._compute_log_p_G()])
            
    def _compute_e_log_q(self):
        log_q_d = compute_log_q_dirichlet(self.alpha)
        
        log_q_epsilon = sum([compute_log_q_dirichlet(x) for x in self.gamma])
      
        log_q_pi = compute_log_q_dirichlet(self.kappa)
        
        log_q_g = np.sum(self.G * np.log(self.G + 1e-10))
        
        log_q_z_1 = np.sum(self.Z[0] * np.log(self.Z[0] + 1e-10))
        
        Z_2 = self.Z[1][:][self._Z_2_indices]
        
        log_q_z_2 = np.sum(Z_2 * np.log(Z_2 + 1e-10))
        
        return np.sum([log_q_d,
                       log_q_epsilon,
                       log_q_pi,
                       log_q_g,
                       log_q_z_1,
                       log_q_z_2])

    def _compute_e_log_p_x_1(self):
        # TxNxKxM
        s_1 = self.X[:, :, np.newaxis, :] * self.Z[0][np.newaxis, :, :, np.newaxis]
        
        # TxKxM
        s_2 = s_1.sum(axis=1)
        
        # SxTxKxM
        s_3 = self.G[:, np.newaxis, :, :] * s_2[np.newaxis, :, :, :]
        
        # SxT
        s_4 = s_3.sum(axis=(2, 3))
        
        # SxT
        s_5 = self.e_log_epsilon * s_4
        
        return self.Z[0].sum() * self.e_log_d[0] + np.sum(self.e_log_pi * self.Z[0].sum(axis=0)) + s_5.sum()

    def _compute_e_log_p_x_2_a(self):
        # TxNxKxKxM
        s_1 = self.Z[1][np.newaxis, :, :, :, np.newaxis] * self.X[:, :, np.newaxis, np.newaxis, :]
         
        # TxKxKxM
        s_2 = s_1.sum(axis=1)
         
        s_3 = self._get_G_G()
         
        # SxTxKxKxM
        s_4 = np.zeros((self.S, self.T, self.K, self.K, self.M))
         
        for s in range(self.S):
            for (u, v) in self.state_map[s]:
                s_4[s, :, :, :] += s_3[u, v, np.newaxis, :, :, :] * s_2
         
        # SxT
        s_5 = s_4.sum(axis=(2, 3, 4))
         
        # SxT
        s_6 = self.e_log_epsilon * s_5
 
        return self.e_log_d[1] * self.Z[1].sum() + \
               np.sum(self.Z[1] * np.add.outer(self.e_log_pi, self.e_log_pi)[np.newaxis, :, :]) + \
               s_6.sum()
# #                np.sum(self.e_log_pi * self.Z_2_sym.sum(axis=(0, 2))) + \
    def _compute_e_log_p_x_2(self):
        # SxNxM
        X_e = self._get_X_cross_epsilon()
    
        G_G = self._get_G_G()
        
        G_G_X_e = np.zeros((self.S, self.N, self.K, self.K, self.M))
        
        for s in range(self.S):
            for (u, v) in self.state_map[s]:
                G_G_X_e[s, :, :, :] += G_G[u, v, np.newaxis, :, :, :] * X_e[s, :, np.newaxis, np.newaxis, :]

        G_G_X_e = G_G_X_e.sum(axis=(0, -1))
            
        Z_G_G_X_e = self.Z[1] * G_G_X_e
    
        return self.e_log_d[1] * self.Z[1].sum() + np.sum(self.Z[1] * np.add.outer(self.e_log_pi, self.e_log_pi)[np.newaxis, :, :]) + Z_G_G_X_e.sum()
    
    def _compute_log_p_G(self):
        return np.sum(self.G_prior * self.G.sum(axis=(1, 2)))
        
    def _get_G_G(self):
        g_g = np.zeros((self.S, self.S, self.K, self.K, self.M))
        
        for k in range(self.K):
            for l in range(self.K):
                for u in range(self.S):
                    for v in range(self.S):
                        g_g[u, v, k, l, :] = self.G[u, k, :] * self.G[v, l, :]
   
        return g_g
    
    def _get_X_cross_epsilon(self):
        X_e = self.X[np.newaxis, :, :, :] * self.e_log_epsilon[:, :, np.newaxis, np.newaxis]
        
        return np.sum(X_e, axis=1)
    
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

def compute_log_p_dirichlet(posterior, prior):
    posterior = np.atleast_2d(posterior)
    
    prior = np.atleast_2d(prior)
    
    log_p = log_gamma(prior.sum(axis=1)) - \
            np.sum(log_gamma(prior), axis=1) + \
            np.sum((prior - 1) * compute_e_log_dirichlet(posterior), axis=1)
    
    return log_p.sum()

def compute_log_q_dirichlet(x):
    a_0 = x.sum()
       
    K = len(x)
       
    return log_gamma(x).sum() - log_gamma(x.sum()) + (a_0 - K) * psi(a_0) - ((x - 1) * psi(x)).sum()

if __name__ == '__main__':
    from scg.simulate import simualte_data
    
    np.random.seed(1)
    
    num_iters = 100

    K = 40
    
    M = 48
    
    N = 50
    
    K_true = 8
    
    num_iters = 10
    
    state_map = {0 : [(0, 0)],
                 1 : [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
                 2 : [(2, 2)]}
    
    inverse_state_map = {}
    
    for s in state_map:
        for (u, v) in state_map[s]:
            inverse_state_map[(u, v)] = s
    

    alpha_prior = np.array([1, 5])
    
    gamma_prior = np.array([[98, 1, 1],
                            [25, 50, 25],
                            [1, 1, 98]])
    
    S = gamma_prior.shape[0]
    
    G_prior = np.ones(S) * 1 / S
    
    
    
    kappa_prior = np.concatenate([np.ones(K_true), np.ones(K - K_true) * 1e-6])
    
    sim = simualte_data(alpha_prior, gamma_prior, kappa_prior, G_prior, M, N, inverse_state_map)
    
    X = sim['X']
    
    model = GenotyperVariationBayes(alpha_prior, gamma_prior, kappa_prior, G_prior, state_map, X)
    
    model.fit()
