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

from scg.doublet import VariationalBayesDoubletGenotyper
from scg.singlet import VariationalBayesSingletGenotyper
from scg.utils import load_labels

def run_doublet_analysis(args):
    if args.seed is not None:
        np.random.seed(args.seed)
       
    cell_ids, data, event_ids, priors = load_data(args.config_file)
    
    labels = load_labels(cell_ids, args.labels_file)
    
    samples = load_samples(cell_ids, args.samples_file)

    with open(args.state_map_file) as fh:
        state_map = yaml.load(fh)
    
    model = VariationalBayesDoubletGenotyper(priors['alpha'],
                                             priors['gamma'],
                                             priors['kappa'],
                                             priors['G'],
                                             state_map,
                                             data,
                                             init_labels=labels,
                                             samples=samples,
                                             use_position_specific_gamma=args.use_position_specific_error_rate) 
    
    model.fit(convergence_tolerance=args.convergence_tolerance, num_iters=args.max_num_iters)
    
    if args.lower_bound_file is not None:
        write_lower_bound(model, args.lower_bound_file)
            
    if args.out_dir is not None:
        write_cluster_posteriors(cell_ids, model.Z_0, args.out_dir)
        
        write_double_cluster_posteriors(cell_ids, model, args.out_dir)
        
        write_doublet_posteriors(cell_ids, model, args.out_dir)
    
        write_genotype_posteriors(event_ids, model.G, args.out_dir)
            
        write_params(model, args.out_dir, event_ids, position_specific_error=args.use_position_specific_error_rate)

def run_singlet_analysis(args):
    if args.seed is not None:
        np.random.seed(args.seed)
       
    cell_ids, data, event_ids, priors = load_data(args.config_file)

    labels = load_labels(cell_ids, args.labels_file)

    samples = load_samples(cell_ids, args.samples_file)
    
    model = VariationalBayesSingletGenotyper(priors['gamma'],
                                             priors['kappa'],
                                             priors['G'],
                                             data,
                                             init_labels=labels,
                                             samples=samples,
                                             use_position_specific_gamma=args.use_position_specific_error_rate)
        
    model.fit(convergence_tolerance=args.convergence_tolerance, num_iters=args.max_num_iters)
    
    if args.lower_bound_file is not None:
        write_lower_bound(model, args.lower_bound_file)

    if args.out_dir is not None:
        write_cluster_posteriors(cell_ids, model.Z, args.out_dir)
        
        write_genotype_posteriors(event_ids, model.G, args.out_dir)
            
        write_params(model, args.out_dir, event_ids, position_specific_error=args.use_position_specific_error_rate)
    
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
        
        priors['G'][data_type] = np.array(config['data'][data_type]['state_prior'])
        
        priors['G'][data_type] = priors['G'][data_type] / priors['G'][data_type].sum()
        
        cell_ids.append(data[data_type].index)
        
        event_ids[data_type] = list(data[data_type].columns)

    cell_ids = sorted(set.intersection(*[set(x) for x in cell_ids]))
    
    for data_type in data:
        data[data_type] = data[data_type].loc[cell_ids]
    
    priors['kappa'] = np.ones(config['num_clusters']) * config['kappa_prior']
    
    if 'alpha_prior' in config:
        priors['alpha'] = np.array(config['alpha_prior'])
    
    print 'Number of cells: {0}'.format(len(cell_ids))
    
    print 'Number of data types: {0}'.format(len(event_ids))
    
    for data_type in event_ids:
        print 'Number of {0} events: {1}'.format(data_type, len(event_ids[data_type]))
    
    return  cell_ids, data, event_ids, priors

def _load_data_frame(file_name):
    print 'Loading {0}.'.format(file_name)
    df = pd.read_csv(file_name, compression='gzip', index_col='cell_id', sep='\t')
    
    return df

def load_samples(cell_ids, file_name):
    if file_name is not None:
        samples = pd.read_csv(file_name, compression='gzip', index_col='cell_id', sep='\t')
        
        if not set(samples.index) == set(cell_ids):
            raise Exception('Samples file must contain all entries from the data files.')
        
        samples = samples.loc[cell_ids, 'sample']
        
        print 'Number of samples: {0}'.format(samples.nunique())
    
    else:
        samples = None
        
        print 'No samples file supplied. All cells assumed to come from same sample.'
    
    return samples

def write_cluster_posteriors(cell_ids, Z, out_dir):
    file_name = os.path.join(out_dir, 'cluster_posteriors.tsv.gz')

    df = pd.DataFrame(Z, index=cell_ids)
    
    with gzip.GzipFile(file_name, 'w') as fh:
        df.to_csv(fh, index_label='cell_id', sep='\t')

def write_doublet_posteriors(cell_ids, model, out_dir):
    file_name = os.path.join(out_dir, 'doublet_posteriors.tsv.gz')
    
    df = pd.DataFrame(model.Y.T, index=cell_ids)
    
    with gzip.GzipFile(file_name, 'w') as fh:
        df.to_csv(fh, index_label='cell_id', sep='\t')

def write_double_cluster_posteriors(cell_ids, model, out_dir):
    file_name = os.path.join(out_dir, 'double_cluster_posteriors.tsv.gz')
    
    df = pd.DataFrame(model.Z_1.reshape(model.N, model.K * model.K), index=cell_ids)
    
    with gzip.GzipFile(file_name, 'w') as fh:
        df.to_csv(fh, index_label='cell_id', sep='\t')    

def write_genotype_posteriors(event_ids, G, out_dir):
    file_name = os.path.join(out_dir, 'genotype_posteriors.tsv.gz')
    
    G_out = []
    
    for data_type in event_ids:
        for i, df in enumerate(G[data_type]):
            df = pd.DataFrame(df, columns=event_ids[data_type])
            
            df = df.stack().reset_index()
            
            df.columns = 'cluster_id', 'event_id', 'probability'
            
            df.insert(1, 'event_type', data_type)
            
            df.insert(3, 'event_value', i)
            
            G_out.append(df)
    
    G_out = pd.concat(G_out)
  
    with gzip.GzipFile(file_name, 'w') as fh:
        G_out.to_csv(fh, index=False, sep='\t')

def write_params(model, out_dir, event_ids, position_specific_error):
    file_name = os.path.join(out_dir, 'params.yaml')
    
    params = {
              'lower_bound' : float(model.lower_bound[-1]),
              'converged' : model.converged
              }
    
    params['kappa'] = {}
    
    for sample in model.kappa:
        params['kappa'][str(sample)] = [float(x) for x in model.kappa[sample]]
    
    params['gamma'] = {}
    
    for data_type in model.gamma:
        if position_specific_error:
            params['gamma'][data_type] = {}
            
            for m, e in enumerate(event_ids[data_type]):
                params['gamma'][data_type][e] = [[float(x) for x in row] for row in model.gamma[data_type][:, :, m]]
        
        else:
            params['gamma'][data_type] = [[float(x) for x in row] for row in model.gamma[data_type]]
    
    if hasattr(model, 'alpha'):
        params['alpha'] = [float(x) for x in model.alpha]

    with open(file_name, 'w') as fh:
        yaml.dump(params, fh, default_flow_style=False)
        
def write_lower_bound(model, out_file):
    with open(out_file, 'w') as fh:
        result = {'lower_bound' : float(model.lower_bound[-1]), 'converged' : model.converged}
        
        yaml.dump(result, fh, default_flow_style=False)
