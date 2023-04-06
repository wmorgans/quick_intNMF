"""Module containing useful functions for intNMF module."""

import numpy as np
from nmf_models_mod_updates import intNMF
from typing import Optional, Union, Mapping  # Special
import pandas as pd
import muon as mu
import anndata as ad
import os

def get_top_features(nmf_model:intNMF,
                     topics: Optional[list] = None,
                     n_features: Optional[int] = 1,
                     modality: Optional[str] = 'rna',
                     mode: Optional[str] = 'abs'):
    """Get the highest ranked features from selected topics.

    Parameters
    ----------
    nmf_model : intNMF
        intNMF model with loadings to plot (.fit has been run)
    topics : list[int], default None (all topics)
        list of topics to get top features for
    n_features : int, default 1
        Number of features
    modality : str, 'rna' | 'atac', default 'rna'
        modality fro which to return features
    mode : str, 'abs' | 'diff', default 'abs'
        select features based on absolute value or diff with other topics average
    Returns
    -------
    list[str] of features from selected modality
    """



    if modality == 'rna':
        phi = nmf_model.phi_rna
        col_names = nmf_model.rna_features
    elif modality == 'atac':
        phi = nmf_model.phi_atac
        col_names = nmf_model.atac_features
    else:
        print('select modality from rna or atac')
        #return

    if col_names is None:
        print('Missing feature labels')
        #return
    phi_df = pd.DataFrame(phi, columns=col_names, index=['Factor {}'.format(i) for i in np.arange(nmf_model.k)])

    if topics is None:
        topics = np.arange(nmf_model.k)
    top_features = []
    if mode == 'abs':
        for topic in topics:
            top_features.append(list(phi_df.iloc[topic, :].sort_values(ascending=False)[0:n_features].index))

    elif mode == 'diff':
        for topic in topics:
            top_features.append(list((phi_df.iloc[topic, :] - phi_df.drop(phi_df.index[topic], axis=0).mean(axis=0)).sort_values(ascending=False)[0:n_features].index))

    return [item for topic_features in top_features for item in topic_features]

def load_multiome(file: str, labels: Optional[str] = None):
    """Function to load multiome data from .h5, .h5ad or .h5mu file types

    Parameters
    ----------
    File : str
        Location of multiome dataset. SHould be .h5 (output of cell ranger), h5ad or h5mu
    labels : str, default None
        Location of cell labels. Should be a tsv file format.

    Returns
    -------
    muon.MuData object multimodal data contanainer built on anndata
    """

    _, extension = os.path.splitext(file)
    if extension == '.h5':
        mu_data = mu.read_10x_h5(file)

    elif extension == '.h5ad':
        h5ad = ad.read_h5ad(file)
        rna = h5ad[:, h5ad.var['feature_types'] == 'GEX']
        atac = h5ad[:, h5ad.var['feature_types'] == 'ATAC']
        mu_data = mu.MuData({'rna': rna, 'atac': atac})
        mu_data.update()
        mu.pp.intersect_obs(mu_data)
    elif extension == '.h5mu':
        mu_data = mu.read(file)

    # If there are labels for the dataset load the labels and remove cells without a label.
    if labels is None:
        print('no labels')
    else:
        meta = pd.read_csv(labels, sep="\t", header=0, index_col=0)
        mu.pp.filter_obs(mu_data, meta.index.values)
        mu_data.obs = meta

    return mu_data
