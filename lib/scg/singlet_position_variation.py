from __future__ import division

from scipy.misc import logsumexp as log_sum_exp

import numpy as np

from scg.singlet import VariationalBayesSingletGenotyper
from scg.utils import compute_e_log_dirichlet, compute_e_log_q_dirichlet, compute_e_log_p_dirichlet, safe_multiply

class VariationalBayesSingletGenotyperPositionSpecific(VariationalBayesSingletGenotyper):
    def _init_gamma(self, data_type):
            self.gamma[data_type] = np.repeat(self.gamma_prior[data_type], self.M[data_type])
            
            self.gamma[data_type] = self.gamma[data_type].reshape((self.S[data_type],
                                                                   self.T[data_type],
                                                                   self.M[data_type]))
        
    def get_e_log_epsilon(self, data_type):
        e_log_epsilon = np.zeros((self.gamma[data_type].shape))
        
        for m in range(self.M[data_type]):
            e_log_epsilon[:, :, m] = compute_e_log_dirichlet(self.gamma[data_type][:, :, m])
        
        return e_log_epsilon

    def _update_G_d(self, data_type):
        X = self.X[data_type]
        
        Z = self.Z
        
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        G_prior = self.G_prior[data_type]
        
        log_G = np.einsum('stnkm, stnkm, stnkm -> skm',
                          X[np.newaxis, :, :, np.newaxis, :],
                          Z[np.newaxis, np.newaxis, :, :, np.newaxis],
                          e_log_epsilon[:, :, np.newaxis, np.newaxis, :])

        # SxKxM
        log_G = np.log(G_prior)[:, np.newaxis, np.newaxis] + log_G
        
        # KxM
        log_G_norm = log_sum_exp(log_G, axis=0)
        
        # SxKxM
        self.log_G[data_type] = log_G - log_G_norm[np.newaxis, :, :]
  
    def _get_log_Z_d(self, data_type):
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        G = self.get_G(data_type)
        
        X = self.X[data_type]
        
        log_Z = np.einsum('stnkm, stnkm, stnkm -> nk',
                          G[:, np.newaxis, np.newaxis, :, :],
                          X[np.newaxis, :, :, np.newaxis, :],
                          e_log_epsilon[:, :, np.newaxis, np.newaxis, :])
        
        return log_Z
    
    def _update_gamma(self):
        for data_type in self.data_types:
            self.gamma[data_type] = self.gamma_prior[data_type][:, :, np.newaxis] + self._get_gamma_data_term(data_type)
    
    def _compute_log_p_gamma(self):
        log_p_prior = 0
        
        log_p_posterior = 0
        
        for data_type in self.data_types:
            for m in range(self.M[data_type]):
                log_p_prior += sum([compute_e_log_p_dirichlet(x, y) for x, y in zip(self.gamma[data_type][:, :, m], self.gamma_prior[data_type])])
    
            log_p_posterior += self._compute_e_log_p_gamma_posterior(data_type)
        
        return log_p_prior + log_p_posterior
        
    def _compute_e_log_p_gamma_posterior(self, data_type):
        e_log_epsilon = self.get_e_log_epsilon(data_type)
        
        data_term = self._get_gamma_data_term(data_type)
        
        result = 0
        
        for m in range(self.M[data_type]):    
            result += np.sum(safe_multiply(e_log_epsilon[:, :, m], data_term[:, :, m]))
        
        return result
  
    def _compute_log_q_epsilon(self):
        log_q = 0
        
        for data_type in self.data_types:
            for m in range(self.M[data_type]):
                log_q += sum([compute_e_log_q_dirichlet(x)  for x in self.gamma[data_type][:, :, m]])
        
        return log_q        
        
    def _get_gamma_data_term(self, data_type):
        log_G = self.log_G[data_type]
        
        X = self.X[data_type]
        
        # SxNxKxM
        data_term = np.exp(self.log_Z[np.newaxis, :, :, np.newaxis] + log_G[:, np.newaxis, :, :])

        # SxTxM
        return np.einsum('stnkm, stnkm -> stm', 
                         data_term[:, np.newaxis, :, :, :], 
                         X[np.newaxis, :, :, np.newaxis, :])

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
     
    for i in range(10):
        np.random.seed(i)
        
        model = VariationalBayesSingletGenotyperPositionSpecific(gamma_prior, kappa_prior, G_prior, sim['X'])
     
        model.fit(num_iters=100)
     
        Z = model.Z.argmax(axis=1)
        
        v_dmm.append(v_measure_score(sim['Z'], Z))    
    
    np.random.seed(0)
    
    sim = get_default_genotyper_sim()

    v_gen = []
     
    for i in range(10):
        np.random.seed(i)
        
        model = VariationalBayesSingletGenotyperPositionSpecific(gamma_prior, kappa_prior, G_prior, sim['X'])
     
        model.fit(num_iters=100)
     
        Z = model.Z.argmax(axis=1)
        
        v_gen.append(v_measure_score(sim['Z'][0][sim['Y'] == 0], Z[sim['Y'] == 0]))
         
    print max(v_dmm), max(v_gen)
