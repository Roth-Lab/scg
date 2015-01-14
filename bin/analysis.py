from __future__ import division

from vb import VariationalBayesGenotyper

import gzip
import numpy as np
import os
import pandas as pd
import yaml

def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
       
    data, cell_ids, event_ids = load_data(args.in_file)
    
    print 'Number of cells: {0}'.format(len(cell_ids))
    print 'Number of events {0}'.format(len(event_ids))
    
    priors, state_map = load_config(args.priors_file)
    
#     print data.head()
    
#     print priors
    
    model = VariationalBayesGenotyper(priors['alpha'], 
                                      priors['gamma'], 
                                      priors['kappa'], 
                                      priors['G'], 
                                      state_map, 
                                      data) 
    
    model.fit(convergence_tolerance=args.convergence_tolerance, num_iters=args.max_num_iters)
    
#     print model.alpha
    
#     print model.kappa
    
#     print model.gamma
    
    write_cluster_posteriors(cell_ids, model, args.out_dir)
    
    write_params(model, args.out_dir)

    write_genotype_posteriors(data, model, args.out_dir)

def load_data(file_name):
    data = _load_data_frame(file_name)
    
    cell_ids = data.index

    event_ids = list(data.columns) 
    
    return data, cell_ids, event_ids

def _load_data_frame(file_name):
    with gzip.GzipFile(file_name) as fh:
        df = pd.read_csv(fh, index_col='cell_id', sep='\t')
    
    return df

def load_config(file_name):
    if file_name is None:
        cwd = os.path.dirname(os.path.realpath(__file__))
        
        file_name = os.path.join(cwd, 'config.yaml')
    
    with open(file_name) as fh:
        config = yaml.load(fh)
    
    state_map = config['state_map']
    
    priors = {}
    
    priors['alpha'] = np.array(config['priors']['alpha'])
    
    priors['gamma'] = np.array(config['priors']['gamma'])
    
    priors['kappa'] = config['priors']['kappa']
    
    S = len(state_map)

    priors['G'] = np.ones(S) * 1 / S
    
    if np.isscalar(priors['kappa']):
        priors['kappa'] = np.ones(config['num_clusters']) * priors['kappa']
    
    else:
        print 'Number of clusters is being ignored because a vector was specified for kappa prior.'
        
        priors['kappa'] = np.array(priors['kappa'])
        
    return priors, state_map

def write_cluster_posteriors(cell_ids, model, out_dir):
    file_name = os.path.join(out_dir, 'Y.tsv.gz')
    
    df = pd.DataFrame(model.Y.T, index=cell_ids)
    
    with gzip.GzipFile(file_name, 'w') as fh:
        df.to_csv(fh, index_label='cell_id', sep='\t')
    
    file_name = os.path.join(out_dir, 'Z_0.tsv.gz')
    
    df = pd.DataFrame(model.Z_0, index=cell_ids)
    
    with gzip.GzipFile(file_name, 'w') as fh:
        df.to_csv(fh, index_label='cell_id', sep='\t')
    
    file_name = os.path.join(out_dir, 'Z_1.tsv.gz')
    
    df = pd.DataFrame(model.Z_1.reshape(model.N, model.K * model.K), index=cell_ids)
    
    with gzip.GzipFile(file_name, 'w') as fh:
        df.to_csv(fh, index_label='cell_id', sep='\t')
    
def write_genotype_posteriors(data, model, out_dir):
    file_name = os.path.join(out_dir, 'G.tsv.gz')
    
    G = []
    
    for s, df in enumerate(model.G):
        df = pd.DataFrame(df, columns=data.columns)
        
        df = df.stack().reset_index()
        
        df.columns = 'cluster_id', 'event_id', 'probability'
        
        df.insert(1, 'state', s)
        
        G.append(df)
    
    G = pd.concat(G)
  
    with gzip.GzipFile(file_name, 'w') as fh:
        G.to_csv(fh, index=False, sep='\t')

def write_params(model, out_dir):
    file_name = os.path.join(out_dir, 'params.yaml')
    
    params = {
              'alpha' : [float(x) for x in model.alpha],
              'gamma' : [[float(x) for x in row] for row in model.gamma],
              'kappa' : [float(x) for x in model.kappa],
              'lower_bound' : float(model.lower_bound[-1]),
              'converged' : model.converged
              }

    with open(file_name, 'w') as fh:
        yaml.dump(params, fh, default_flow_style=False)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--in_file', required=True)
    
    parser.add_argument('--out_dir', required=True)
    
    # Optional args
    parser.add_argument('--concentration', default=1, type=float)
    
    parser.add_argument('--convergence_tolerance', default=1e-6, type=float)
    
    parser.add_argument('--max_num_iters', default=1000, type=int)
    
    parser.add_argument('--seed', type=int, default=None)
    
    parser.add_argument('--priors_file', default=None)
    
    args = parser.parse_args()
    
    main(args)
