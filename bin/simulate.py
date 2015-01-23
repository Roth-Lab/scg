'''
Created on 2015-01-10

@author: Andrew Roth
'''
import numpy as np

def simualte_data(alpha_true, gamma_true, kappa_true, G_prior, M, N, inverse_state_map):
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
