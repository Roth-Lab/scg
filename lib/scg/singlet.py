from __future__ import division

import numpy as np
import pandas as pd

from scg.utils import compute_e_log_dirichlet, compute_e_log_q_dirichlet, compute_e_log_p_dirichlet, \
                      compute_e_log_q_discrete, get_indicator_matrix, init_Z, log_space_normalise, safe_multiply

class VariationalBayesSingletGenotyper(object):
    def __init__(self,
                 gamma_prior,
                 kappa_prior,
                 G_prior,
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

        self.gamma_prior = gamma_prior
        
        self.kappa_prior = kappa_prior
        
        self._init_kappa()    
        
        self.G_prior = G_prior
        
        self.data_types = X.keys()
    
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
            
            self._init_gamma(data_type)
        
        self.log_Z = init_Z(self.K, self.N, init_labels)
        
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

            self._update_G()
            
            if debug:
                print 'G', self._diff_lower_bound()
            
            self._update_gamma()
            
            if debug:
                print 'gamma', self._diff_lower_bound()
            
            self._update_kappa()
            
            if debug:
                print 'kappa', self._diff_lower_bound()
            
            self._update_Z()
            
            if debug:
                print 'Z', self._diff_lower_bound()
            
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
        G_prior = self.G_prior[data_type]
        
        X = self.X[data_type]
        
        Z = self.Z
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        if self.use_position_specific_gamma:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, :]
            
        else:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis]
        
        log_G = np.einsum('stnkm, stnkm, stnkm -> skm',
                          X[np.newaxis, :, :, np.newaxis, :],
                          Z[np.newaxis, np.newaxis, :, :, np.newaxis],
                          e_log_epsilon)

        # SxKxM
        log_G = np.log(G_prior)[:, np.newaxis, np.newaxis] + log_G
        
        # SxKxM
        self.log_G[data_type] = log_space_normalise(log_G, axis=0)
        
    def _update_Z(self):
        log_Z = np.zeros((self.N, self.K))
        
        for data_type in self.data_types:
            log_Z = log_Z + self._get_log_Z_d(data_type)
            
        for sample in self.samples.unique():
            e_log_pi = self.get_e_log_pi(sample)
            
            indices = np.where(self.samples == sample)
            
            log_Z[indices] = e_log_pi[np.newaxis, :] + log_Z[indices]
    
        self.log_Z = log_space_normalise(log_Z, axis=1)
    
    def _get_log_Z_d(self, data_type):
        G = self.get_G(data_type)
        
        X = self.X[data_type]
                  
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        if self.use_position_specific_gamma:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, :]
            
        else:
            e_log_epsilon = e_log_epsilon[:, :, np.newaxis, np.newaxis, np.newaxis]
                
        log_Z = np.einsum('stnkm, stnkm, stnkm -> nk',
                          G[:, np.newaxis, np.newaxis, :, :],
                          X[np.newaxis, :, :, np.newaxis, :],
                          e_log_epsilon)
        
        return log_Z
    
    def _update_gamma(self):
        for data_type in self.data_types:
            if self.use_position_specific_gamma:
                prior = self.gamma_prior[data_type][:, :, np.newaxis]
            
            else:
                prior = self.gamma_prior[data_type]
            
            self.gamma[data_type] = prior + self._get_gamma_data_term(data_type)
    
    def _update_kappa(self):
        for sample in self.samples:
            self.kappa[sample] = self.kappa_prior + self._get_kappa_data_term(sample)
    
    def _compute_lower_bound(self):
        return self._compute_e_log_p() - self._compute_e_log_q()
    
    def _compute_e_log_p(self):
        log_p_gamma = self._compute_log_p_gamma()
        
        log_p_kappa = self._compute_log_p_kappa()

        log_p_G_prior = 0
        
        for data_type in self.data_types:
            log_p_G_prior += self._compute_log_p_G(data_type)
        
        return sum([log_p_gamma,
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

    def _compute_log_p_kappa(self):
        log_p_prior = sum([compute_e_log_p_dirichlet(x, self.kappa_prior) for x in self.kappa.values()])
        
        log_p_posterior = self._compute_e_log_p_kappa_posterior()
        
        return log_p_prior + log_p_posterior
     
    def _compute_e_log_p_gamma_posterior(self, data_type):
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        data_term = self._get_gamma_data_term(data_type)
        
        if self.use_position_specific_gamma:
            log_p = 0
            
            for m in range(self.M[data_type]):    
                log_p += np.sum(safe_multiply(e_log_epsilon[:, :, m], data_term[:, :, m]))
        
        else:
            log_p = np.sum(safe_multiply(data_term, e_log_epsilon))
        
        return log_p
       
    def _compute_e_log_p_kappa_posterior(self):
        e_log_p = 0
        
        for sample in self.kappa:
            e_log_p += np.sum(safe_multiply(self.get_e_log_pi(sample), self._get_kappa_data_term(sample)))
        
        return e_log_p    

    def _compute_log_p_G(self, data_type):
        return np.sum(self.G_prior[data_type] * self.get_G(data_type).sum(axis=(1, 2)))
            
    def _compute_e_log_q(self):
        log_q_epsilon = self._compute_log_q_epsilon()
        
        log_q_pi = sum([compute_e_log_q_dirichlet(self.kappa[sample]) for sample in self.samples.unique()])
        
        log_q_G = 0
        
        for data_type in self.data_types:
            log_q_G += compute_e_log_q_discrete(self.log_G[data_type])
        
        log_q_Z = compute_e_log_q_discrete(self.log_Z)

        return np.sum([log_q_epsilon,
                       log_q_pi,
                       log_q_G,
                       log_q_Z])
    
    def _compute_log_q_epsilon(self):
        log_q = 0
        
        for data_type in self.data_types:
            if self.use_position_specific_gamma:
                for m in range(self.M[data_type]):
                    log_q += sum([compute_e_log_q_dirichlet(x)  for x in self.gamma[data_type][:, :, m]])
            
            else:
                log_q += sum([compute_e_log_q_dirichlet(x) for x in self.gamma[data_type]])
        
        return log_q
        
    def _get_gamma_data_term(self, data_type):
        log_G = self.log_G[data_type]
        
        X = self.X[data_type]
        
        # SxNxKxM
        data_term = np.exp(self.log_Z[np.newaxis, :, :, np.newaxis] + log_G[:, np.newaxis, :, :])

        if self.use_position_specific_gamma:
            out_dim = 'stm'
        
        else:
            out_dim = 'st'
        
        return np.einsum('stnkm, stnkm -> {0}'.format(out_dim),
                         data_term[:, np.newaxis, :, :, :],
                         X[np.newaxis, :, :, np.newaxis, :])
    
    def _get_kappa_data_term(self, sample):
        indices = np.where(self.samples == sample)

        return self.Z[indices].sum(axis=0)
    
    def _diff_lower_bound(self):
        self._debug_lower_bound.append(self._compute_lower_bound())
        
        diff = (self._debug_lower_bound[-1] - self._debug_lower_bound[-2]) / np.abs(self._debug_lower_bound[-1])
        
        if diff < 0:
            print 'Bound decreased',
        
        return diff
    
if __name__ == '__main__':
    def test_run(samples, use_position_specific_gamma):
        np.random.seed(0)
    
        model = VariationalBayesSingletGenotyper(gamma_prior, 
                                                 kappa_prior, 
                                                 G_prior, 
                                                 sim['X'],
                                                 samples=samples,
                                                 use_position_specific_gamma=use_position_specific_gamma)
 
        model.fit(num_iters=100)
        
        print model.gamma['snv'].shape, model.kappa.keys()
    
    from simulate import get_default_genotyper_sim
    
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
    
    sim = get_default_genotyper_sim()
    
    samples = pd.Series(['a' for _ in range(30)] + ['b' for _ in range(70)])
    
    print 'Standard'
    
    test_run(None, False)
    
    print 'Position specific'
    
    test_run(None, True)
    
    print 'Sample specific'
    
    test_run(samples, False)
    
    print 'Sample position specific'
    
    test_run(samples, True)