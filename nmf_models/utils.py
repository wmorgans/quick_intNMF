import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from nmf_models_mod_updates import intNMF
from typing import Optional, Union, Mapping  # Special
import pandas as pd
import muon as mu
import anndata as ad
import os
#"https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot/47664533#47664533"
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

def get_top_features(nmf_model:intNMF,
                     topics: Optional[list] = None,
                     n_features: Optional[int] = 1,
                     modality: Optional[str] = 'rna'):
    '''Get the highest ranked features from selected topics.

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

    Returns
    -------
    list[str] of features from selected modality
    '''

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
    for topic in topics:
        top_features.append(list(phi_df.iloc[topic, :].sort_values(ascending=False)[0:n_features].index))
    
    return [item for topic_features in top_features for item in topic_features]

def load_multiome(file: str, labels: Optional[str] = None):
    '''Function to load multiome data from .h5, .h5ad or .h5mu file types
    
    Parameters
    ----------
    File : str
        Location of multiome dataset. SHould be .h5 (output of cell ranger), h5ad or h5mu
    labels : str, default None
        Location of cell labels. Should be a tsv file format.

    Returns
    -------
    muon.MuData object multimodal data contanainer built on anndata
    '''

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
