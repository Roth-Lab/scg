'''
Created on 2015-01-09

@author: Andrew Roth
'''
from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np

from scg.utils import compute_e_log_dirichlet, compute_e_log_p_dirichlet, compute_e_log_q_dirichlet, \
                      compute_e_log_q_discrete, safe_multiply, get_indicator_matrix

class VariationalBayesDoubletGenotyper(object):
    def __init__(self, alpha_prior, gamma_prior, kappa_prior, G_prior, state_map, X):    
        self.K = len(kappa_prior)
        
        self.alpha_prior = alpha_prior
        
        self.gamma_prior = gamma_prior
        
        self.kappa_prior = kappa_prior
        
        self.alpha = alpha_prior.copy()
            
        self.kappa = np.ones(self.K)
        
        self.G_prior = G_prior
        
        self.state_map = state_map
        
        self.data_types = X.keys()
    
        self.N = X[X.keys()[0]].shape[0]
        
        self.M = {}
        
        self.S = {}
        
        self.T = {}
        
        self.X = {}
        
        self.gamma = {}
        
        self._inverse_state_map = {}
        
        self.log_G = {}
        
        for data_type in self.data_types:
            if X[data_type].shape[0] != self.N:
                raise Exception('All data types must have the same number of rows (cells).')
                    
            self.M[data_type] = X[data_type].shape[1]
            
            self.S[data_type] = gamma_prior[data_type].shape[0]
            
            self.T[data_type] = gamma_prior[data_type].shape[1]
        
            self.X[data_type] = get_indicator_matrix(range(self.T[data_type]), X[data_type])
            
            self.gamma[data_type] = gamma_prior[data_type].copy()
        
            self._init_G(data_type)
            
            self._init_inverse_state_map(data_type)
        
        self.lower_bound = [float('-inf')]

        self._debug_lower_bound = [float('-inf')]
        
        self.converged = False
        
    def _init_G(self, data_type):
        G = np.random.random((self.S[data_type], self.K, self.M[data_type]))
        
        G = G / np.expand_dims(G.sum(axis=0), axis=0)
        
        self.log_G[data_type] = np.log(G)

    def _init_inverse_state_map(self, data_type):
        self._inverse_state_map[data_type] = {}
        
        for s in self.state_map[data_type]:
            for (u, v) in self.state_map[data_type][s]:
                self._inverse_state_map[data_type][(u, v)] = s

    
    def get_e_log_epsilon(self, data_type):
        return compute_e_log_dirichlet(self.gamma[data_type])


    def get_G(self, data_type):
        return np.exp(self.log_G[data_type])
    
    @property
    def e_log_d(self):
        return compute_e_log_dirichlet(self.alpha)
        
    @property
    def e_log_pi(self):
        return compute_e_log_dirichlet(self.kappa)
    
    @property
    def G(self):
        G = {}
        
        for data_type in self.data_types:
            G[data_type] = self.get_G(data_type)
        
        return G
    
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
    
    def fit(self, convergence_tolerance=1e-4, debug=False, num_iters=100):
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
             
            diff = (self.lower_bound[-1] - self.lower_bound[-2]) / np.abs(self.lower_bound[-1])
             
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
        for data_type in self.data_types:
            self._update_G_d(data_type)
        
    def _update_G_d(self, data_type):
        X = self.X[data_type]
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)

        G = self.get_G(data_type)
        
        state_map = self.state_map[data_type]
        
        inverse_state_map = self._inverse_state_map[data_type]
        
        G_prior = self.G_prior[data_type]
        
        M = self.M[data_type]
        
        S = self.S[data_type]
        
        T = self.T[data_type]
        
        # SxKxM
        singlet_term = np.einsum('stnkm, stnkm, stnkm -> skm',
                                 self.Z_0[np.newaxis, np.newaxis, :, :, np.newaxis],
                                 X[np.newaxis, :, :, np.newaxis, :],
                                 e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis])
        # SxTxKxM
        doublet_diff_term_temp = np.einsum('stnklm, stnklm, stnklm -> stklm',
                                           G[:, np.newaxis, np.newaxis, np.newaxis, :, :],
                                           self.Z_1[np.newaxis, np.newaxis, :, :, :, np.newaxis],
                                           X[np.newaxis, :, :, np.newaxis, np.newaxis, :])
 
        # SxTxKxM
        doublet_diff_term_temp = doublet_diff_term_temp.sum(axis=3) - np.swapaxes(np.diagonal(doublet_diff_term_temp, axis1=2, axis2=3), -2, -1)
 
        doublet_diff_term = np.zeros(doublet_diff_term_temp.shape)
         
        for s in state_map:
            for w in state_map:
                for (u, v) in state_map[w]:
                    if(u != s) and (v != s):
                        continue
                      
                    elif (u == s):
                        doublet_diff_term[s, :, :, :] += safe_multiply(e_log_epsilon[w, :, np.newaxis, np.newaxis],
                                                                       doublet_diff_term_temp[v, :, :, :])
                          
                    elif (v == s) :
                        doublet_diff_term[s, :, :, :] += safe_multiply(e_log_epsilon[w, :, np.newaxis, np.newaxis],
                                                                       doublet_diff_term_temp[u, :, :, :])
         
        # SxKxM
        doublet_diff_term = doublet_diff_term.sum(axis=1)
        
        # TxKxM
        doublet_same_term_temp = np.einsum('tnkm, tnkm -> tkm',
                                           self.Z_1_k_k[np.newaxis, :, :, np.newaxis],
                                           X[:, :, np.newaxis, :])
   
        # SxTxKxM
        doublet_same_term = np.zeros((S, T, self.K, M))
      
        for s in state_map:
            w = inverse_state_map[(s, s)]

            doublet_same_term[s, :, :, :] = e_log_epsilon[w, :, np.newaxis, np.newaxis] * doublet_same_term_temp[np.newaxis, :, :, :]
        
        # SxKxM
        doublet_same_term = doublet_same_term.sum(axis=1)        
        
        # SxKxM
        data_term = singlet_term + doublet_diff_term + doublet_same_term
        
        log_G = np.log(G_prior[:, np.newaxis, np.newaxis]) + data_term
        
        log_G = log_G - np.expand_dims(log_sum_exp(log_G, axis=0), axis=0)
        
        self.log_G[data_type] = log_G
  
    def _update_Z(self):
        singlet_term = self.e_log_d[0] + self.e_log_pi[np.newaxis, :]
        
        doublet_diff_term = self.e_log_d[1] + self.e_log_pi[np.newaxis, :, np.newaxis] + self.e_log_pi[np.newaxis, np.newaxis, :]
        
        doublet_same_term = self.e_log_d[1] + 2 * self.e_log_pi[np.newaxis, :]
        
        for data_type in self.data_types:
            singlet_term = singlet_term + self._get_Z_singlet_term(data_type)
            
            doublet_diff_term = doublet_diff_term + self._get_Z_doublet_diff_term(data_type)
            
            doublet_same_term = doublet_same_term + self._get_Z_doublet_same_term(data_type)
        
        log_Z_1 = singlet_term
        
        log_Z_2 = doublet_diff_term
        
        for k in range(self.K):
            log_Z_2[:, k, k] = doublet_same_term[:, k]
        
        log_Z = np.concatenate((log_Z_1, log_Z_2.reshape((self.N, self.K * self.K))), axis=1)
        
        log_Z_nc = log_sum_exp(log_Z, axis=1)
        
        self.log_Z = log_Z - log_Z_nc[:, np.newaxis]
    
    def _get_Z_singlet_term(self, data_type):
        G = self.get_G(data_type)
        
        X = self.X[data_type]
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        return np.einsum('stnkm, stnkm, stnkm -> nk',
                         G[:, np.newaxis, np.newaxis, :, :],
                         X[np.newaxis, :, :, np.newaxis, :],
                         e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis])

    def _get_Z_doublet_diff_term(self, data_type):
        X = self.X[data_type]
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        G_G = self._get_G_G_marginalised(data_type)
        
        return np.einsum('stnklm, stnklm, stnklm -> nkl',
                         X[np.newaxis, :, :, np.newaxis, np.newaxis, :],
                         G_G[:, np.newaxis, np.newaxis, :, :, :],
                         e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
    
    def _get_Z_doublet_same_term(self, data_type):
        X = self.X[data_type]
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        G_G = self._get_G_same_marginalised(data_type)
        
        return np.einsum('stnkm, stnkm, stnkm -> nk',
                         X[np.newaxis, :, :, np.newaxis, :],
                         G_G[:, np.newaxis, np.newaxis, :, :],
                         e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis])

    def _update_alpha(self):
        self.alpha = self.alpha_prior + self._get_alpha_data_term()
    
    def _update_kappa(self):
        self.kappa = self.kappa_prior + self._get_kappa_data_term()
    
    def _update_gamma(self):
        for data_type in self.data_types:
            self.gamma[data_type] = self.gamma_prior[data_type] + self._get_gamma_data_term(data_type)
    
    def _compute_lower_bound(self):
        return self._compute_e_log_p() - self._compute_e_log_q()
    
    def _compute_e_log_p(self):
        alpha_prior = compute_e_log_p_dirichlet(self.alpha, self.alpha_prior)
        
        alpha_posterior = self._compute_e_log_p_alpha_posterior()
        
        gamma_prior = 0
        
        gamma_posterior = 0
        
        for data_type in self.data_types:
            gamma_prior += sum([compute_e_log_p_dirichlet(x, y) for x, y in zip(self.gamma[data_type], self.gamma_prior[data_type])])
    
            gamma_posterior += self._compute_e_log_p_gamma_posterior(data_type)
        
        kappa_prior = compute_e_log_p_dirichlet(self.kappa, self.kappa_prior)
        
        kappa_posterior = self._compute_e_log_p_kappa_posterior()
        
        G_prior = 0
        
        for data_type in self.data_types:
            G_prior += self._compute_log_p_G(data_type)
        
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
    
    def _compute_e_log_p_gamma_posterior(self, data_type):
        return np.sum(safe_multiply(self.get_e_log_epsilon(data_type), self._get_gamma_data_term(data_type)))
    
    def _compute_log_p_G(self, data_type):
        return np.sum(self.G_prior[data_type] * self.get_G(data_type).sum(axis=(1, 2)))
            
    def _compute_e_log_q(self):
        log_q_d = compute_e_log_q_dirichlet(self.alpha)
        
        log_q_epsilon = 0
        
        for data_type in self.data_types:
            log_q_epsilon += sum([compute_e_log_q_dirichlet(x) for x in self.gamma[data_type]])
      
        log_q_pi = compute_e_log_q_dirichlet(self.kappa)
        
        log_q_g = 0
        
        for data_type in self.data_types:
            log_q_g += compute_e_log_q_discrete(self.log_G[data_type])
        
        log_q_z = compute_e_log_q_discrete(self.log_Z)

        return np.sum([log_q_d,
                       log_q_epsilon,
                       log_q_pi,
                       log_q_g,
                       log_q_z])
 
    def _get_G_G_marginalised(self, data_type):
        log_G = self.log_G[data_type]
        
        M = self.M[data_type]
        
        S = self.S[data_type]
        
        state_map = self.state_map[data_type]
        
        g_g = np.zeros((S, self.K, self.K, M))

        for s in state_map:
            for (u, v) in state_map[s]:
                g_g[s, :, :, :] = g_g[s, :, :, :] + np.exp(log_G[u, :, np.newaxis, :] + log_G[v, np.newaxis, :, :])
        
        return g_g
    
    def _get_G_same_marginalised(self, data_type):
        G = self.get_G(data_type)
        
        M = self.M[data_type] 
        
        S = self.S[data_type]
        
        state_map = self.state_map[data_type]
        
        g_marginal = np.zeros((S, self.K, M))
        
        for s in state_map:
            for (u, v) in state_map[s]:
                if u != v:
                    continue
                
                g_marginal[s, :, :] += G[u, :, :]
        
        return g_marginal
    
    def _get_alpha_data_term(self):
        return np.exp(log_sum_exp(self.log_Y, axis=1))
     
    def  _get_gamma_data_term(self, data_type):
        log_G = self.log_G[data_type]
        
        G = self.get_G(data_type)
        
        M = self.M[data_type]
        
        S = self.S[data_type]
        
        X = self.X[data_type]
        
        state_map = self.state_map[data_type]
        
        # SxNxKxM
        singlet_term = np.exp(self.log_Z_0[np.newaxis, :, :, np.newaxis] + log_G[:, np.newaxis, :, :])
        
        singlet_term = np.einsum('stnkm, stnkm -> st',
                                 singlet_term[:, np.newaxis, :, :, :],
                                 X[np.newaxis, :, :, np.newaxis, :])
        
        # TxKxKxM
        doublet_diff_term_temp = np.einsum('tnklm, tnklm -> tklm',
                                           self.Z_1[np.newaxis, :, :, :, np.newaxis],
                                           X[:, :, np.newaxis, np.newaxis, :])
        
        # SxKxKxM
        G_G = self._get_G_G_marginalised(data_type)
        
        doublet_diff_term = np.einsum('stklm, stklm -> st',
                                      doublet_diff_term_temp[np.newaxis, :, :, :],
                                      G_G[:, np.newaxis, :, :, :])
        
        doublet_diff_term_correction = np.einsum('stkm, stkm -> st',
                                                 np.diagonal(doublet_diff_term_temp, axis1=1, axis2=2)[np.newaxis, :, :, :],
                                                 np.diagonal(G_G, axis1=1, axis2=2)[:, np.newaxis, :, :])
        
        doublet_diff_term = doublet_diff_term - doublet_diff_term_correction
     
        # SxKxM
        doublet_same_term = np.zeros((S, self.K, M))
        
        for s in state_map:
            for (u, v) in state_map[s]:
                if u != v:
                    continue
                
                doublet_same_term[s, :, :] += G[u, :, :]
        
        # SxNxKxM
        doublet_same_term = safe_multiply(doublet_same_term[:, np.newaxis, :, :], self.Z_1_k_k[np.newaxis, :, :, np.newaxis])
        
        # SxT
        doublet_same_term = np.einsum('stnkm, stnkm -> st',
                                      doublet_same_term[:, np.newaxis, :, :, :],
                                      X[np.newaxis, :, :, np.newaxis, :])
        

        # SxT
        return singlet_term + doublet_same_term + doublet_diff_term
    
    def _get_kappa_data_term(self):
        singlet_term = self.Z_0.sum(axis=0)
 
        doublet_term = self.Z_1.sum(axis=(0, 1)) + self.Z_1.sum(axis=(0, 2))

        return singlet_term + doublet_term

    def _diff_lower_bound(self):
        self._debug_lower_bound.append(self._compute_lower_bound())
        
        diff = (self._debug_lower_bound[-1] - self._debug_lower_bound[-2]) / np.abs(self._debug_lower_bound[-1])
        
        if diff < 0:
            print 'Bound decreased',
#             raise Exception(diff)
        
        return diff


if __name__ == '__main__':
    from sklearn.metrics import v_measure_score
    
    from simulate import get_default_dirichlet_mixture_sim, get_default_genotyper_sim
    
    np.seterr(all='warn')

    state_map = {
                 'snv' : {0 : [(0, 0)],
                          1 : [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
                          2 : [(2, 2)]},
                 'breakpoint' : {0 : [(0, 0)],
                                 1 : [(0, 1), (1, 0), (1, 1)]}
                 }

    alpha_prior = np.array([10, 1])
    
    gamma_prior = {
                   'snv' : np.array([[98, 1, 1],
                                     [25, 50, 25],
                                     [1, 1, 98]]),
                   'breakpoint' : np.array([[90, 10],
                                            [1, 99]])}
    
    kappa_prior = np.ones(20)
    
    G_prior = {}
    
    for data_type in state_map:
        S = gamma_prior[data_type].shape[0]
        
        G_prior[data_type] = np.ones(S) * 1 / S
    
    np.random.seed(0)
    
    sim = get_default_dirichlet_mixture_sim()

    v_dmm = []
     
    for i in range(10):
        np.random.seed(i)
        
        model = VariationalBayesDoubletGenotyper(alpha_prior, gamma_prior, kappa_prior, G_prior, state_map, sim['X'])
     
        model.fit(num_iters=100)
     
        Z = model.Z.argmax(axis=1)
        
        v_dmm.append(v_measure_score(sim['Z'], Z))    
    
    np.random.seed(0)
    
    sim = get_default_genotyper_sim()

    v_gen = []
     
    for i in range(10):
        np.random.seed(i)
        
        model = VariationalBayesDoubletGenotyper(alpha_prior, gamma_prior, kappa_prior, G_prior, state_map, sim['X'])
     
        model.fit(num_iters=100)
     
        Z = model.Z.argmax(axis=1)
        
        v_gen.append(v_measure_score(sim['Z'][0][sim['Y'] == 0], Z[sim['Y'] == 0]))
         
    print max(v_dmm), max(v_gen)
