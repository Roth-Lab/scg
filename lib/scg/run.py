"""
Created on 2015-01-23

@author: Andrew Roth
"""

import h5py
import numpy as np
import os
import pandas as pd
import yaml

from scg.doublet import VariationalBayesDoubletGenotyper
from scg.singlet import VariationalBayesSingletGenotyper

import scg.post_process


def fit(
    in_file,
    out_file,
    convergence_threshold=1e-6,
    max_iters=int(1e4),
    seed=None):
    
    if seed is not None:
        np.random.seed(seed)
        
    with open(in_file, "r") as fh:
        config = FitConfig(
            yaml.load(fh, Loader=yaml.SafeLoader)
        )
    
    if config.model == "doublet":
        if config.state_map is None:
            raise Exception("State map must be specified when using doublet model.")
        
        if "alpha" not in config.priors:
            raise Exception("Alpha prior parameters must be specified when using the doublet model.")

        model = VariationalBayesDoubletGenotyper(
            config.priors["alpha"],
            config.priors["gamma"],
            config.priors["kappa"],
            config.priors["G"],
            config.state_map,
            config.data,
            samples=config.samples,
            use_position_specific_gamma=config.use_position_specific_errors
        )
        
    elif config.model == "singlet":
        model = VariationalBayesSingletGenotyper(
            config.priors["gamma"],
            config.priors["kappa"],
            config.priors["G"],
            config.data,
            samples=config.samples,
            use_position_specific_gamma=config.use_position_specific_errors
        )
    
    model.fit(convergence_threshold=convergence_threshold, num_iters=max_iters)

    with h5py.File(out_file, "w") as fh:
        meta = fh.create_group("meta")
        
#         meta.attrs["data_types"] = config.data_types
        
        meta.attrs["model"] = config.model
        
        meta.attrs["K"] = model.K
        
        meta.attrs["N"] = model.N
        
        # Save data info
        fh.create_dataset(
            "/data/cell_ids",
            data=np.array(config.cell_ids, dtype=h5py.string_dtype(encoding="utf-8"))
        )
        
        for data_type in config.data_types:
            fh.create_dataset(
                "/data/event_ids/{}".format(data_type),
                data=np.array(config.event_ids[data_type], dtype=h5py.string_dtype(encoding="utf-8"))
            )
        
        if config.samples is not None:
            fh.create_dataset(
                "/data/samples",
                data=np.array(config.samples, dtype=h5py.string_dtype(encoding="utf-8"))
            )
            
        fh.create_dataset("/stats/elbo", data=np.array(model.lower_bound))
        
        # Save priors
        for data_type in config.data_types:
            fh.create_dataset("/priors/G/{}".format(data_type), data=config.priors["G"][data_type])

            fh.create_dataset("/priors/gamma/{}".format(data_type), data=config.priors["gamma"][data_type])
        
        fh.create_dataset("/priors/kappa", data=config.priors["kappa"])
        
        if config.model == "doublet":
            fh.create_dataset("/priors/alpha", data=config.priors["alpha"])
        
        # Variational params
        fh.create_dataset("/var_params/Z", data=model.Z)
        
        for sample in model.kappa:
            fh.create_dataset("/var_params/kappa/{}".format(sample), data=model.kappa[sample])
        
        for data_type in config.data_types:
            fh.create_dataset("/var_params/G/{}".format(data_type), data=model.G[data_type])
            
            if config.use_position_specific_errors:
                for m, e in enumerate(config.event_ids[data_type]):
                    fh.create_dataset(
                        "/var_params/gamma/{0}/{1}".format(data_type, e),
                        data=model.gamma[data_type][:,:, m]
                    )
                 
            else:
                fh.create_dataset(
                    "/var_params/gamma/{}".format(data_type),
                    data=model.gamma[data_type]
                )
                
        if config.model == "doublet":
            fh.create_dataset("/var_params/alpha", data=model.alpha)
            
            fh.create_dataset("/var_params/Y", data=model.Y)


class FitConfig(object):

    def __init__(self, config):
        self.model = config["model"]
        
        self._load_data(config)
        
        self._load_priors(config)
        
        self._load_samples(config)
        
        self.state_map = config.get("state_map", None)
        
        self.use_position_specific_errors = config.get("use_position_specific_errors", False)
   
        print("Number of cells: {0}".format(len(self.cell_ids)))
        
        print("Number of data types: {0}".format(len(self.event_ids)))
        
        for data_type in self.event_ids:
            print("Number of {0} events: {1}".format(data_type, len(self.event_ids[data_type])))
    
    @property
    def data_types(self):
        return self.data.keys()
            
    def _load_data(self, config):
        self.cell_ids = []

        self.data = {}
        
        self.event_ids = {}

        for data_type in config["data"]:
            self._load_data_file(config, data_type)
            
            self.cell_ids.append(self.data[data_type].index)
            
            self.event_ids[data_type] = list(self.data[data_type].columns)
    
        self.cell_ids = sorted(set.intersection(*[set(x) for x in self.cell_ids]))
                               
        for data_type in self.data:
            self.data[data_type] = self.data[data_type].loc[self.cell_ids]

    def _load_data_file(self, config, data_type):
        file_name = config["data"][data_type]["file"]
        
        print("Loading {0}.".format(file_name))
        
        self.data[data_type] = pd.read_csv(file_name, compression="gzip", index_col="cell_id", sep="\t")
       
    def _load_priors(self, config):
        self.priors = {"gamma": {}, "G": {}}
    
        for data_type in config["data"]:
            self.priors["gamma"][data_type] = np.array(config["data"][data_type]["gamma_prior"])
            
            G = np.array(config["data"][data_type]["state_prior"])
            
            self.priors["G"][data_type] = G / G.sum()
        
        self.priors["kappa"] = np.ones(config["num_clusters"]) * config.get("kappa_prior", 1)
        
        if "alpha_prior" in config:
            self.priors["alpha"] = np.array(config["alpha_prior"])
            
    def _load_samples(self, config):
        if "samples_file" in config:
            samples = pd.read_csv(config["samples_file"], compression="gzip", index_col="cell_id", sep="\t")
            
            if not set(samples.index) == set(self.cell_ids):
                raise Exception("Samples file must contain all entries from the data files.")
            
            samples = samples.loc[self.cell_ids, "sample"]
            
            print("Number of samples: {0}".format(samples.nunique()))
        
        else:
            samples = None
            
            print("No samples file supplied. All cells assumed to come from same sample.")
            
        self.samples = samples


def write_results_tsv(in_file, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cell_df, cluster_df = scg.post_process.load_results_dfs(in_file)

    cell_df.to_csv(os.path.join(out_dir, "cells.tsv.gz"), compression='gzip', float_format='%.4f', index=False, sep='\t')
    
    cluster_df.to_csv(os.path.join(out_dir, "clusters.tsv.gz"), compression='gzip', float_format='%.4f', index=False, sep='\t')

