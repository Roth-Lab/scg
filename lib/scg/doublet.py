'''
Created on 2015-01-09

@author: Andrew Roth
'''
from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np
import pandas as pd

from scg.utils import compute_e_log_dirichlet, compute_e_log_p_dirichlet, compute_e_log_q_dirichlet, \
                      compute_e_log_q_discrete, get_indicator_matrix, safe_multiply

class VariationalBayesDoubletGenotyper(object):
    def __init__(self, 
                 alpha_prior, 
                 gamma_prior, 
                 kappa_prior, 
                 G_prior, 
                 state_map, 
                 X, 
                 init_labels=None, 
                 samples=None, 
                 use_position_specific_gamma=False):   
        
        self.K = len(kappa_prior)
        
        self.N = X[X.keys()[0]].shape[0]
        
        if samples is None:
            self.samples = pd.Series([0] * self.N)
            
        else: 
            self.samples = samples
        
        self.use_position_specific_gamma = use_position_specific_gamma
        
        self.alpha_prior = alpha_prior
        
        self.gamma_prior = gamma_prior
        
        self.kappa_prior = kappa_prior
        
        self.alpha = alpha_prior.copy()
            
        self._init_kappa()
        
        self.G_prior = G_prior
        
        self.state_map = state_map
        
        self.data_types = X.keys()
    
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
            
            self._init_gamma(data_type)
            
            self._init_G(data_type)
            
            self._init_inverse_state_map(data_type)
        
        self.lower_bound = [float('-inf')]

        self._debug_lower_bound = [float('-inf')]
        
        self.converged = False
    
    def _init_gamma(self, data_type):
        if self.use_position_specific_gamma:
            self.gamma[data_type] = np.repeat(self.gamma_prior[data_type], self.M[data_type])
            
            self.gamma[data_type] = self.gamma[data_type].reshape((self.S[data_type],
                                                                   self.T[data_type],
                                                                   self.M[data_type]))
            
        else:
            self.gamma[data_type] = self.gamma_prior[data_type].copy()
    
    def _init_kappa(self):
        self.kappa = {}
        
        for sample in self.samples:
            self.kappa[sample] = np.ones(self.K)
    
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
        if self.use_position_specific_gamma:
            e_log_epsilon = np.zeros((self.gamma[data_type].shape))
        
            for m in range(self.M[data_type]):
                e_log_epsilon[:, :, m] = compute_e_log_dirichlet(self.gamma[data_type][:, :, m])
        
        else:
            e_log_epsilon = compute_e_log_dirichlet(self.gamma[data_type])
        
        return e_log_epsilon
    
    def get_e_log_pi(self, sample):
        return compute_e_log_dirichlet(self.kappa[sample])
    
    def get_G(self, data_type):
        return np.exp(self.log_G[data_type])
    
    @property
    def e_log_d(self):
        return compute_e_log_dirichlet(self.alpha)
    
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
                self._check_lower_bound_increase('Z')
            
            g_old = self._compute_lower_bound()
            
            self._update_G()
            
            g_new = self._compute_lower_bound()
            
            g_diff = g_new - g_old
            
            if debug:
                self._check_lower_bound_increase('G')
            
            self._update_gamma()
            
            if debug:
                self._check_lower_bound_increase('gamma')
            
            self._update_kappa()
            
            if debug:
                self._check_lower_bound_increase('kappa')
            
            self._update_alpha()
            
            if debug:
                self._check_lower_bound_increase('alpha')         
            
            self.lower_bound.append(self._compute_lower_bound())
             
            diff = (self.lower_bound[-1] - self.lower_bound[-2]) / np.abs(self.lower_bound[-1])
             
            print i, self.lower_bound[-1], diff
            
            if (diff < 0) and (g_diff < 0):   
                diff = diff + abs(g_diff)
                
                if diff > 0:
                    print 'Update of G decreased lower bound. Ignoring this.'
             
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

        G = self.get_G(data_type)
        
        state_map = self.state_map[data_type]
        
        inverse_state_map = self._inverse_state_map[data_type]
        
        G_prior = self.G_prior[data_type]
        
        M = self.M[data_type]
        
        S = self.S[data_type]
        
        T = self.T[data_type]
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        # SxTxNxKxM
        if self.use_position_specific_gamma:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, :]
        
        else:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis]
        
        # SxKxM
        singlet_term = np.einsum('stnkm, stnkm, stnkm -> skm',
                                 self.Z_0[np.newaxis, np.newaxis, :, :, np.newaxis],
                                 X[np.newaxis, :, :, np.newaxis, :],
                                 e_log_epsilon)
        # SxTxKxM
        doublet_diff_term_temp = np.einsum('stnklm, stnklm, stnklm -> stklm',
                                           G[:, np.newaxis, np.newaxis, np.newaxis, :, :],
                                           self.Z_1[np.newaxis, np.newaxis, :, :, :, np.newaxis],
                                           X[np.newaxis, :, :, np.newaxis, np.newaxis, :])
 
        # SxTxKxM
        doublet_diff_term_temp = doublet_diff_term_temp.sum(axis=3) - np.swapaxes(np.diagonal(doublet_diff_term_temp, axis1=2, axis2=3), -2, -1)
 
        doublet_diff_term = np.zeros(doublet_diff_term_temp.shape)
        
        # SxTxKxM
        e_log_epsilon = np.squeeze(e_log_epsilon, axis=2)
         
        for s in state_map:
            for w in state_map:
                for (u, v) in state_map[w]:

                    if(u != s) and (v != s):
                        continue
                      
                    elif (u == s):
                        doublet_diff_term[s, :, :, :] += safe_multiply(e_log_epsilon[w],
                                                                       doublet_diff_term_temp[v])
                          
                    elif (v == s) :
                        doublet_diff_term[s, :, :, :] += safe_multiply(e_log_epsilon[w],
                                                                       doublet_diff_term_temp[u])
         
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

            doublet_same_term[s, :, :, :] = e_log_epsilon[w] * np.expand_dims(doublet_same_term_temp, axis=0)
        
        # SxKxM
        doublet_same_term = doublet_same_term.sum(axis=1)        
        
        # SxKxM
        data_term = singlet_term + doublet_diff_term + doublet_same_term
        
        log_G = np.log(G_prior[:, np.newaxis, np.newaxis]) + data_term
        
        log_G = log_G - np.expand_dims(log_sum_exp(log_G, axis=0), axis=0)
        
        self.log_G[data_type] = log_G
  
    def _update_Z(self):
        singlet_term = self.e_log_d[0]
        
        doublet_diff_term = self.e_log_d[1]
        
        doublet_same_term = self.e_log_d[1]
                
        for data_type in self.data_types:
            singlet_term = singlet_term + self._get_Z_singlet_term(data_type)
            
            doublet_diff_term = doublet_diff_term + self._get_Z_doublet_diff_term(data_type)
            
            doublet_same_term = doublet_same_term + self._get_Z_doublet_same_term(data_type)
        
        for sample in self.samples.unique():
            e_log_pi = self.get_e_log_pi(sample)
            
            indices = np.where(self.samples == sample)
            
            singlet_term[indices] = e_log_pi[np.newaxis, :] + singlet_term[indices]
            
            doublet_diff_term[indices] = e_log_pi[np.newaxis, :, np.newaxis] + e_log_pi[np.newaxis, np.newaxis, :] + doublet_diff_term[indices]
            
            doublet_same_term[indices] = 2 * e_log_pi[np.newaxis, :] + doublet_same_term[indices]
        
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
                        
        # SxTxNxKxM
        if self.use_position_specific_gamma:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, :]
        
        else:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis]
        
        return np.einsum('stnkm, stnkm, stnkm -> nk',
                         G[:, np.newaxis, np.newaxis, :, :],
                         X[np.newaxis, :, :, np.newaxis, :],
                         e_log_epsilon)

    def _get_Z_doublet_diff_term(self, data_type):
        G_G = self._get_G_G_marginalised(data_type)
        
        X = self.X[data_type]
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        # SxTxNxKxKxM
        if self.use_position_specific_gamma:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis, :]
        
        else:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
                
        return np.einsum('stnklm, stnklm, stnklm -> nkl',
                         X[np.newaxis, :, :, np.newaxis, np.newaxis, :],
                         G_G[:, np.newaxis, np.newaxis, :, :, :],
                         e_log_epsilon)
    
    def _get_Z_doublet_same_term(self, data_type):
        G_G = self._get_G_same_marginalised(data_type)
        
        X = self.X[data_type]
                
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        # SxTxNxKxM
        if self.use_position_specific_gamma:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, :]
        
        else:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis]
        
        
        return np.einsum('stnkm, stnkm, stnkm -> nk',
                         X[np.newaxis, :, :, np.newaxis, :],
                         G_G[:, np.newaxis, np.newaxis, :, :],
                         e_log_epsilon)

    def _update_alpha(self):
        self.alpha = self.alpha_prior + self._get_alpha_data_term()
    
    def _update_kappa(self):
        for sample in self.samples:
            self.kappa[sample] = self.kappa_prior + self._get_kappa_data_term(sample)
    
    def _update_gamma(self):
        for data_type in self.data_types:
            prior = self.gamma_prior[data_type]
        
            if self.use_position_specific_gamma:
                prior = prior[:, :, np.newaxis]
            
            self.gamma[data_type] = prior + self._get_gamma_data_term(data_type)
    
    def _compute_lower_bound(self):
        return self._compute_e_log_p() - self._compute_e_log_q()
    
    def _compute_e_log_p(self):
        log_p_alpha = self._compute_log_p_alpha()
        
        log_p_gamma = self._compute_log_p_gamma()
        
        log_p_kappa = self._compute_log_p_kappa()

        log_p_G_prior = 0
        
        for data_type in self.data_types:
            log_p_G_prior += self._compute_log_p_G(data_type)
        
        return sum([log_p_alpha,
                    log_p_gamma,
                    log_p_kappa,
                    log_p_G_prior])
    
    def _compute_log_p_gamma(self):
        log_p_prior = 0
        
        log_p_posterior = 0
        
        for data_type in self.data_types:
            if self.use_position_specific_gamma:
                for m in range(self.M[data_type]):
                    log_p_prior += sum([compute_e_log_p_dirichlet(x, y) for x, y in zip(self.gamma[data_type][:, :, m], self.gamma_prior[data_type])])
            
            else:
                log_p_prior += sum([compute_e_log_p_dirichlet(x, y) for x, y in zip(self.gamma[data_type], self.gamma_prior[data_type])])
    
            log_p_posterior += self._compute_e_log_p_gamma_posterior(data_type)
        
        return log_p_prior + log_p_posterior
    
    def _compute_log_p_alpha(self):
        log_p_prior = compute_e_log_p_dirichlet(self.alpha, self.alpha_prior)
        
        log_p_posterior = self._compute_e_log_p_alpha_posterior()
        
        return log_p_prior + log_p_posterior
    
    def _compute_log_p_kappa(self):
        log_p_prior = sum([compute_e_log_p_dirichlet(x, self.kappa_prior) for x in self.kappa.values()])
        
        log_p_posterior = self._compute_e_log_p_kappa_posterior()
        
        return log_p_prior + log_p_posterior
    
    def _compute_e_log_p_alpha_posterior(self):
        return np.sum(safe_multiply(self.e_log_d, self._get_alpha_data_term()))
     
    def _compute_e_log_p_gamma_posterior(self, data_type):
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        data_term = self._get_gamma_data_term(data_type)
        
        if self.use_position_specific_gamma:
            result = 0
            
            for m in range(self.M[data_type]):    
                result += np.sum(safe_multiply(e_log_epsilon[:, :, m], data_term[:, :, m]))
        
        else:
            result = np.sum(safe_multiply(e_log_epsilon, data_term))
            
        return result
       
    def _compute_e_log_p_kappa_posterior(self):
        e_log_p = 0
        
        for sample in self.kappa:
            e_log_p += np.sum(safe_multiply(self.get_e_log_pi(sample), self._get_kappa_data_term(sample)))
        
        return e_log_p
    
    def _compute_log_p_G(self, data_type):
        return np.sum(self.G_prior[data_type] * self.get_G(data_type).sum(axis=(1, 2)))
            
    def _compute_e_log_q(self):
        log_q_d = compute_e_log_q_dirichlet(self.alpha)
        
        log_q_epsilon = 0
        
        for data_type in self.data_types:
            if self.use_position_specific_gamma:
                for m in range(self.M[data_type]):
                    log_q_epsilon += sum([compute_e_log_q_dirichlet(x)  for x in self.gamma[data_type][:, :, m]])
        
            else:
                log_q_epsilon += sum([compute_e_log_q_dirichlet(x) for x in self.gamma[data_type]])
      
        log_q_pi = sum([compute_e_log_q_dirichlet(self.kappa[sample]) for sample in self.samples.unique()])
        
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
        
        if self.use_position_specific_gamma:
            out_dim = 'stm'
        
        else:
            out_dim = 'st'
        
        # SxNxKxM
        singlet_term = np.exp(self.log_Z_0[np.newaxis, :, :, np.newaxis] + log_G[:, np.newaxis, :, :])
        
        singlet_term = np.einsum('stnkm, stnkm -> {0}'.format(out_dim),
                                 singlet_term[:, np.newaxis, :, :, :],
                                 X[np.newaxis, :, :, np.newaxis, :])
        
        # TxKxKxM
        doublet_diff_term_temp = np.einsum('tnklm, tnklm -> tklm',
                                           self.Z_1[np.newaxis, :, :, :, np.newaxis],
                                           X[:, :, np.newaxis, np.newaxis, :])
        
        # SxKxKxM
        G_G = self._get_G_G_marginalised(data_type)
        
        doublet_diff_term = np.einsum('stklm, stklm -> {0}'.format(out_dim),
                                      doublet_diff_term_temp[np.newaxis, :, :, :],
                                      G_G[:, np.newaxis, :, :, :])
        
        doublet_diff_term_correction = np.einsum('stmk, stmk -> {0}'.format(out_dim),
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
        doublet_same_term = np.einsum('stnkm, stnkm -> {0}'.format(out_dim),
                                      doublet_same_term[:, np.newaxis, :, :, :],
                                      X[np.newaxis, :, :, np.newaxis, :])
        

        # SxT
        return singlet_term + doublet_same_term + doublet_diff_term
    
    def _get_kappa_data_term(self, sample):
        indices = np.where(self.samples == sample)

        singlet_term = self.Z_0[indices].sum(axis=0)
 
        doublet_term = self.Z_1[indices].sum(axis=(0, 1)) + self.Z_1[indices].sum(axis=(0, 2))

        return singlet_term + doublet_term

    def _check_lower_bound_increase(self, variable):
        self._debug_lower_bound.append(self._compute_lower_bound())
        
        diff = (self._debug_lower_bound[-1] - self._debug_lower_bound[-2]) / np.abs(self._debug_lower_bound[-1])
        
        if diff < 0:                
            print 'Bound decreased by {0} when updating {1}.'.format(diff, variable)


if __name__ == '__main__':
    def test_run(samples, use_position_specific_gamma):
        np.random.seed(0)
    
        model = VariationalBayesDoubletGenotyper(alpha_prior, 
                                                 gamma_prior, 
                                                 kappa_prior, 
                                                 G_prior, 
                                                 state_map, 
                                                 sim['X'],
                                                 samples=samples,
                                                 use_position_specific_gamma=use_position_specific_gamma)
 
        model.fit(debug=False, num_iters=100)
        
        print model.gamma['snv'].shape, model.kappa.keys()
    
    from simulate import get_default_dirichlet_mixture_sim
    
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
    
    samples = pd.Series(['a' for _ in range(30)] + ['b' for _ in range(70)])
    
    print 'Standard'
    
    test_run(None, False)
    
    print 'Position specific'
    
    test_run(None, True)
    
    print 'Sample specific'
    
    test_run(samples, False)
    
    print 'Sample position specific'
    
    test_run(samples, True)