'''
Created on 2015-01-10

@author: Andrew Roth
'''
import numpy as np

def simualte_data(alpha_true, gamma_true, kappa_true, G_prior, M, N, inverse_state_map):
    K = len(kappa_true)
    
    G_true = np.random.multinomial(1, G_prior, size=(K, M)).argmax(axis=2)

    d_true = np.random.beta(alpha_true[0], alpha_true[1])

    e_true = np.array([np.random.dirichlet(gamma_true[0]),
                       np.random.dirichlet(gamma_true[1]),
                       np.random.dirichlet(gamma_true[2])])

    pi_true = np.random.dirichlet(kappa_true)

    Y_true = np.random.multinomial(1, [1 - d_true, d_true], size=N).argmax(axis=1)

    Z_true = [np.random.multinomial(1, pi_true, size=N).argmax(axis=1),
              zip(np.random.multinomial(1, pi_true, size=N).argmax(axis=1), np.random.multinomial(1, pi_true, size=N).argmax(axis=1))]

    X = []

    for n in range(N):
        z = Z_true[Y_true[n]][n]

        if np.isscalar(z):
            x_n = [np.random.multinomial(1, e_m).argmax() for e_m in e_true[G_true[z]]]

            X.append(x_n)

        else:
            x = [[np.random.multinomial(1, e_m).argmax() for e_m in e_true[G_true[z[0]]]],
                 [np.random.multinomial(1, e_m).argmax() for e_m in e_true[G_true[z[1]]]]]

            x_n = []

            for u, v in zip(x[0], x[1]):
                x_n.append(inverse_state_map[(u,v)])

            X.append(x_n)

    X = np.array(X)
    
    return {'d' : d_true, 'e' : e_true, 'pi' : pi_true, 'G' : G_true, 'X' : X, 'Y' : Y_true, 'Z' : Z_true}
