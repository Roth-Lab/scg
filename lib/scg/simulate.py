'''
Created on 2015-01-10

@author: Andrew Roth
'''
import numpy as np

def simulate_dirichlet_mixture_model_data(gamma_true, kappa_true, M, N):
    data_types = M.keys()
    
    K = len(kappa_true)
    
    pi_true = np.random.dirichlet(kappa_true)
    
    mu_true = {}
    
    for data_type in data_types:
        mu_d = []
        
        for _ in range(K):
            mu_d.append([np.random.dirichlet(gamma_true[data_type]) for m in range(M[data_type])])
        
        mu_d = np.array(mu_d)
        
        mu_true[data_type] = np.swapaxes(mu_d.T, axis1=1, axis2=2)
        
    print mu_true['snv'].shape, mu_true['breakpoint'].shape

    Z_true = np.random.multinomial(1, pi_true, size=N).argmax(axis=1)

    data = {}
    
    for data_type in data_types:
        X = []
        
        for n in range(N):
            z = Z_true[n]
            
            x_n = [np.random.multinomial(1, mu_true[data_type][:, z, m]).argmax() for m in range(M[data_type])]

            X.append(x_n)
    
        data[data_type] = np.array(X)
    
    return {'mu' : mu_true, 'pi' : pi_true, 'X' : data, 'Z' : Z_true}

def simualte_genotyper_data(alpha_true, gamma_true, kappa_true, G_prior, M, N, inverse_state_map):
    data_types = G_prior.keys()
    
    K = len(kappa_true)
    
    G_true = {}
    
    for data_type in data_types:
        G_true[data_type] = np.random.multinomial(1, G_prior[data_type], size=(K, M[data_type])).argmax(axis=2)

    d_true = np.random.dirichlet(alpha_true)
    
    e_true = {}
    
    for data_type in data_types:
        e_true[data_type] = np.array([np.random.dirichlet(x) for x in gamma_true[data_type]])

    pi_true = np.random.dirichlet(kappa_true)

    Y_true = np.random.multinomial(1, d_true, size=N).argmax(axis=1)

    Z_true = [np.random.multinomial(1, pi_true, size=N).argmax(axis=1),
              zip(np.random.multinomial(1, pi_true, size=N).argmax(axis=1), np.random.multinomial(1, pi_true, size=N).argmax(axis=1))]

    data = {}
    
    for data_type in data_types:
        X = []
        
        for n in range(N):
            z = Z_true[Y_true[n]][n]
    
            if np.isscalar(z):
                x_n = [np.random.multinomial(1, e_m).argmax() for e_m in e_true[data_type][G_true[data_type][z]]]
    
                X.append(x_n)
    
            else:
                x = [[np.random.multinomial(1, e_m).argmax() for e_m in e_true[data_type][G_true[data_type][z[0]]]],
                     [np.random.multinomial(1, e_m).argmax() for e_m in e_true[data_type][G_true[data_type][z[1]]]]]
    
                x_n = []
    
                for u, v in zip(x[0], x[1]):
                    x_n.append(inverse_state_map[data_type][(u,v)])
    
                X.append(x_n)
    
        data[data_type] = np.array(X)
    
    return {'d' : d_true, 'e' : e_true, 'pi' : pi_true, 'G' : G_true, 'X' : data, 'Y' : Y_true, 'Z' : Z_true}

def get_default_dirichlet_mixture_sim():
    N = 100
    
    K_true = 4
    
    M = {'snv' : 48, 'breakpoint' : 8}
    
    gamma_prior = {'snv' : np.ones(3), 'breakpoint' : np.ones(2)}
    
    kappa_prior = np.ones(K_true)
    
    return simulate_dirichlet_mixture_model_data(gamma_prior, kappa_prior, M, N)

def get_default_genotyper_sim():
    N = 100
    
    K_true = 4
    
    M = {'snv' : 48, 'breakpoint' : 8}
    
    state_map = {
                 'snv' : {0 : [(0, 0)],
                          1 : [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
                          2 : [(2, 2)]},
                 'breakpoint' : {0 : [(0, 0)],
                                 1 : [(0, 1), (1, 0), (1, 1)]}
                 }
    
    inverse_state_map = {}
    
    for data_type in state_map:
        inverse_state_map[data_type] = {}
        
        for s in state_map[data_type]:
            for (u, v) in state_map[data_type][s]:
                inverse_state_map[data_type][(u, v)] = s
    

    alpha_prior = np.array([10, 1])
    
    gamma_prior = {
                   'snv' : np.array([[98, 1, 1],
                                     [25, 50, 25],
                                     [1, 1, 98]]),
                   'breakpoint' : np.array([[90, 10],
                                            [1, 99]])}
    
    kappa_prior = np.ones(K_true)
    
    G_prior = {}
    
    for data_type in state_map:
        S = gamma_prior[data_type].shape[0]
        
        G_prior[data_type] = np.ones(S) * 1 / S
        
        
    return simualte_genotyper_data(alpha_prior,
                                   gamma_prior,
                                   kappa_prior,
                                   G_prior,
                                   M,
                                   N,
                                   inverse_state_map)