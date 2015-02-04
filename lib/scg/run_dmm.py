'''
Created on 2015-01-23

@author: Andrew Roth
'''
from __future__ import division

import gzip
import numpy as np
import os
import pandas as pd
import yaml

from scg.dirichlet import VariationalBayesDirichletMixtureModel

def run_dirichlet_mixture_model_analysis(args):
    if args.seed is not None:
        np.random.seed(args.seed)
       
    cell_ids, data, event_ids, priors = load_data(args.config_file)
    
    print 'Number of cells: {0}'.format(len(cell_ids))
    print 'Number of events {0}'.format(len(event_ids))

    model = VariationalBayesDirichletMixtureModel(priors['gamma'], priors['kappa'], data)
        
    model.fit(convergence_tolerance=args.convergence_tolerance, num_iters=args.max_num_iters)

    write_cluster_posteriors(cell_ids, model.Z, args.out_dir)
    
    write_genotype_posteriors(event_ids, model.gamma, args.out_dir)
        
    write_params(model, args.out_dir)
    
def load_data(file_name):
    with open(file_name) as fh:
        config = yaml.load(fh)
    
    cell_ids = []
    
    data = {}
    
    event_ids = {}
    
    priors = {'gamma' : {}, 'G' : {}}

    for data_type in config['data']:
        data[data_type] = _load_data_frame(config['data'][data_type]['file'])
        
        priors['gamma'][data_type] = np.array(config['data'][data_type]['gamma_prior'])
        
        cell_ids.append(data[data_type].index)
        
        event_ids[data_type] = list(data[data_type].columns)

    cell_ids = sorted(set.intersection(*[set(x) for x in cell_ids]))
    
    for data_type in data:
        data[data_type] = data[data_type].loc[cell_ids]
    
    priors['kappa'] = np.ones(config['num_clusters']) * config['kappa_prior']
    
    return  cell_ids, data, event_ids, priors

def _load_data_frame(file_name):
    print 'Loading {0}.'.format(file_name)
    
    df = pd.read_csv(file_name, compression='gzip', index_col='cell_id', sep='\t')
    
    return df

def write_cluster_posteriors(cell_ids, Z, out_dir):
    file_name = os.path.join(out_dir, 'cluster_posteriors.tsv.gz')

    df = pd.DataFrame(Z, index=cell_ids)
    
    with gzip.GzipFile(file_name, 'w') as fh:
        df.to_csv(fh, index_label='cell_id', sep='\t')

def write_genotype_posteriors(event_ids, gamma, out_dir):
    def get_probability(df):
        df['probability'] = df['gamma_parameter'] / df['gamma_parameter'].sum()
    
        return df

    file_name = os.path.join(out_dir, 'genotype_posteriors.tsv.gz')
    
    gamma_out = []
    
    for data_type in event_ids:
        for t in range(gamma[data_type].shape[0]):
            df = gamma[data_type][t, :, :]
            
            df = pd.DataFrame(df, columns=event_ids[data_type])
            
            df = df.stack().reset_index()
            
            df.columns = 'cluster_id', 'event_id', 'gamma_parameter'
            
            df.insert(1, 'event_type', data_type)
            
            df.insert(3, 'event_value', t)
            
            df = df.groupby(['cluster_id', 'event_id']).apply(get_probability)
             
            gamma_out.append(df)
    
    gamma_out = pd.concat(gamma_out)
  
    with gzip.GzipFile(file_name, 'w') as fh:
        gamma_out.to_csv(fh, index=False, sep='\t')

def write_params(model, out_dir):
    file_name = os.path.join(out_dir, 'params.yaml')
    
    params = {
              'kappa' : [float(x) for x in model.kappa],
              'lower_bound' : float(model.lower_bound[-1]),
              'converged' : model.converged
              }
    
    with open(file_name, 'w') as fh:
        yaml.dump(params, fh, default_flow_style=False)