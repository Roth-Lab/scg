import h5py
import numpy as np
import pandas as pd


def load_results_dfs(file_name):
    cell_df = load_cell_df(file_name)
    
    cluster_df = load_cluster_df(file_name)
    
    cluster_map = dict(zip(
        cell_df['cluster_id'].unique(), np.arange(cell_df['cluster_id'].nunique())
    ))
    
    cluster_df = cluster_df[cluster_df["cluster_id"].isin(cluster_map.keys())]
    
    cell_df["cluster_id"] = cell_df["cluster_id"].map(cluster_map)
    
    cell_df = cell_df.sort_values(by="cluster_id")
    
    cluster_df["cluster_id"] = cluster_df["cluster_id"].map(cluster_map)
    
    cluster_df = cluster_df.sort_values(by=["cluster_id", "event_id"])

    return cell_df, cluster_df


def load_cell_df(file_name):
    with h5py.File(file_name, 'r') as fh:
        model = fh["meta"].attrs["model"]
        
        cell_ids = fh["/data/cell_ids"].asstr()[()]
        
        if model == "doublet":
            df = _load_doublet_cell_df(fh)
            
        else:
            df = _load_singlet_cell_df(fh)
        
        df.insert(0, "cell_id", cell_ids)
    
    return df


def _load_doublet_cell_df(fh):
    K = fh["meta"].attrs["K"]
    
    Y = fh["/var_params/Y"][()]
    
    Z = fh["/var_params/Z"][()]
    
    # Keep only the columns for single clusters
    Z = Z[:,:K]

    Z = Z / Z.sum(axis=1)[:, np.newaxis]
    
    return pd.DataFrame({
        "cluster_id": Z.argmax(axis=1),
        "cluster_prob": Z.max(axis=1),
        "doublet_prob": Y[1]
    })


def _load_singlet_cell_df(fh):
    Z = fh["/var_params/Z"][()]
    
    return pd.DataFrame({
        "cluster_id": Z.argmax(axis=1),
        "cluster_prob": Z.max(axis=1)
    })


def load_cluster_df(file_name):
    with h5py.File(file_name, 'r') as fh:
        data_types = list(fh["/data/event_ids"].keys())
        
        df = []
        
        for dt in data_types: 
            event_ids = fh["/data/event_ids/{}".format(dt)].asstr()[()]
            
            G = fh["/var_params/G/{}".format(dt)][()]
            
            # MAP assignments
            G_idx = G.argmax(axis=0)
            
            G_idx = pd.DataFrame(G_idx, columns=event_ids)
            
            G_idx = G_idx.stack().reset_index()
            
            G_idx.columns = "cluster_id", "event_id", "genotype"
            
            # Probs
            G_prob = G.max(axis=0)

            G_prob = pd.DataFrame(G_prob, columns=event_ids)
            
            G_prob = G_prob.stack().reset_index()
            
            G_prob.columns = "cluster_id", "event_id", "genotype_prob"
            
            # Merge
            dt_df = pd.merge(G_idx, G_prob, on=["cluster_id", "event_id"])
            
            dt_df["data_type"] = dt
            
            df.append(dt_df)
        
        return pd.concat(df)

