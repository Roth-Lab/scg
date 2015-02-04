from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np

from scg.utils import compute_e_log_dirichlet, compute_e_log_q_dirichlet, compute_e_log_p_dirichlet, \
                      compute_e_log_q_discrete, get_indicator_matrix, safe_multiply

class VariationalBayesSingletGenotyper(object):
    def __init__(self, gamma_prior, kappa_prior, G_prior, X):    
        self.K = len(kappa_prior)
        
        self.gamma_prior = gamma_prior
        
        self.kappa_prior = kappa_prior
            
        self.kappa = np.ones(self.K)
        
        self.G_prior = G_prior
        
        self.data_types = X.keys()
    
        self.N = X[X.keys()[0]].shape[0]
        
        self.M = {}
        
        self.S = {}
        
        self.T = {}
        
        self.X = {}
        
        self.gamma = {}
        
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
        
        self.lower_bound = [float('-inf')]

        self._debug_lower_bound = [float('-inf')]
        
        self.converged = False
    
    def _init_G(self, data_type):
        G = np.random.random((self.S[data_type], self.K, self.M[data_type]))
        
        G = G / np.expand_dims(G.sum(axis=0), axis=0)
        
        self.log_G[data_type] = np.log(G)

    def get_e_log_epsilon(self, data_type):
        return compute_e_log_dirichlet(self.gamma[data_type])

    def get_G(self, data_type):
        return np.exp(self.log_G[data_type])

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
    
    def fit(self, convergence_tolerance=1e-4, debug=False, num_iters=100):
        for i in range(num_iters):

            self._update_Z()
            
            if debug:
                print 'Z', self._diff_lower_bound()

            self._update_G()
            
            if debug:
                print 'G', self._diff_lower_bound()
            
            self._update_gamma()
            
            if debug:
                print 'gamma', self._diff_lower_bound()
            
            self._update_kappa()
            
            if debug:
                print 'kappa', self._diff_lower_bound()
            
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
        
        Z = self.Z
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        G_prior = self.G_prior[data_type]
        
        # TxNxKxM
        log_G = X[:, :, np.newaxis, :] * Z[np.newaxis, :, :, np.newaxis]
        
        # TxKxM
        log_G = log_G.sum(axis=1)
        
        # SxTxKxM
        log_G = e_log_epsilon[:, :, np.newaxis, np.newaxis] * log_G[np.newaxis, :, :, :]
        
        # SxKxM
        log_G = log_G.sum(axis=1)
        
        # SxKxM
        log_G = np.log(G_prior)[:, np.newaxis, np.newaxis] + log_G
        
        # KxM
        log_G_norm = log_sum_exp(log_G, axis=0)
        
        # SxKxM
        self.log_G[data_type] = log_G - log_G_norm[np.newaxis, :, :]
        
    def _update_Z(self):
        log_Z = self.e_log_pi[np.newaxis, :]
        
        for data_type in self.data_types:
            log_Z = log_Z + self._get_log_Z_d(data_type)
    
        log_Z_norm = np.logaddexp.reduce(log_Z, axis=1)
        
        log_Z = log_Z - log_Z_norm[:, np.newaxis]
        
        self.log_Z = log_Z
    
    def _get_log_Z_d(self, data_type):
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        G = self.get_G(data_type)
        
        X = self.X[data_type]
        
        # SxTxNxKxM
        log_Z = G[:, np.newaxis, np.newaxis, :, :] * X[np.newaxis, :, :, np.newaxis, :]
        
        # SxTxNxK
        log_Z = log_Z.sum(axis=4)
        
        # SxTxNxK
        log_Z = e_log_epsilon[:, :, np.newaxis, np.newaxis] * log_Z
        
        # NxK
        log_Z = log_Z.sum(axis=(0, 1))
        
        return log_Z
    
    def _update_gamma(self):
        for data_type in self.data_types:
            self.gamma[data_type] = self.gamma_prior[data_type] + self._get_gamma_data_term(data_type)
    
    def _update_kappa(self):
        self.kappa = self.kappa_prior + self._get_kappa_data_term()

    def _compute_lower_bound(self):
        return self._compute_e_log_p() - self._compute_e_log_q()
    
    def _compute_e_log_p(self):
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
        
        return sum([gamma_prior,
                    gamma_posterior,
                    kappa_prior,
                    kappa_posterior,
                    G_prior])
    
    def _compute_e_log_p_kappa_posterior(self):
        return np.sum(safe_multiply(self.e_log_pi, self._get_kappa_data_term()))
    
    def _compute_e_log_p_gamma_posterior(self, data_type):
        return np.sum(safe_multiply(self.get_e_log_epsilon(data_type), self._get_gamma_data_term(data_type)))
    
    def _compute_log_p_G(self, data_type):
        return np.sum(self.G_prior[data_type] * self.get_G(data_type).sum(axis=(1, 2)))
            
    def _compute_e_log_q(self):
        log_q_epsilon = 0
        
        for data_type in self.data_types:
            log_q_epsilon += sum([compute_e_log_q_dirichlet(x) for x in self.gamma[data_type]])
      
        log_q_pi = compute_e_log_q_dirichlet(self.kappa)
        
        log_q_g = 0
        
        for data_type in self.data_types:
            log_q_g += compute_e_log_q_discrete(self.log_G[data_type])
        
        log_q_z = compute_e_log_q_discrete(self.log_Z)

        return np.sum([log_q_epsilon,
                       log_q_pi,
                       log_q_g,
                       log_q_z])
    
    def _get_gamma_data_term(self, data_type):
        log_G = self.log_G[data_type]
        
        X = self.X[data_type]
        
        # SxNxKxM
        data_term = np.exp(self.log_Z[np.newaxis, :, :, np.newaxis] + log_G[:, np.newaxis, :, :])
        
        # SxTxNxKxM
        data_term = data_term[:, np.newaxis, :, :, :] * X[np.newaxis, :, :, np.newaxis, :]

        # SxT
        return data_term.sum(axis=(2, 3, 4))
    
    def _get_kappa_data_term(self):
        return self.Z.sum(axis=0)
    
    def _diff_lower_bound(self):
        self._debug_lower_bound.append(self._compute_lower_bound())
        
        diff = (self._debug_lower_bound[-1] - self._debug_lower_bound[-2]) / np.abs(self._debug_lower_bound[-1])
        
        if diff < 0:
            print 'Bound decreased',
        
        return diff
    
if __name__ == '__main__':
    from sklearn.metrics import v_measure_score
    
    from simulate import get_default_dirichlet_mixture_sim, get_default_genotyper_sim
    
    np.seterr(all='warn')
    
    gamma_prior = {
                   'snv' : np.array([[98, 1, 1],
                                     [25, 50, 25],
                                     [1, 1, 98]]),
                   'breakpoint' : np.array([[90, 10],
                                            [1, 99]])}
    
    kappa_prior = np.ones(20)
    
    G_prior = {}
    
    for data_type in gamma_prior:
        S = gamma_prior[data_type].shape[0]
        
        G_prior[data_type] = np.ones(S) * 1 / S
    
    np.random.seed(0)
    
    sim = get_default_dirichlet_mixture_sim()

    v_dmm = []
     
    for i in range(100):
        np.random.seed(i)
        
        model = VariationalBayesSingletGenotyper(gamma_prior, kappa_prior, G_prior, sim['X'])
     
        model.fit(num_iters=100)
     
        Z = model.Z.argmax(axis=1)
        
        v_dmm.append(v_measure_score(sim['Z'], Z))    
    
    np.random.seed(0)
    
    sim = get_default_genotyper_sim()

    v_gen = []
     
    for i in range(10):
        np.random.seed(i)
        
        model = VariationalBayesSingletGenotyper(gamma_prior, kappa_prior, G_prior, sim['X'])
     
        model.fit(num_iters=100)
     
        Z = model.Z.argmax(axis=1)
        
        v_gen.append(v_measure_score(sim['Z'][0][sim['Y'] == 0], Z[sim['Y'] == 0]))
         
    print max(v_dmm), max(v_gen)