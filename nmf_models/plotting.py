"""Plotting functions for intNMF.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from nmf_models_mod_updates import intNMF
from typing import Optional, Union, Mapping  # Special
import seaborn as sns
import matplotlib as mpl
from utils import SeabornFig2Grid as sfg
import sys

sys.setrecursionlimit(100000)

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
mpl.rc('image', cmap='cividis')
sns.set_palette(plt.cm.cividis(np.linspace(0, 1, 8))[1:-1])


def _view_mat(mat, xlab, ax, ylab=None):
    """Base matrix plotter. Saturates at 1
    """
    im_ratio = mat.shape[1]/mat.shape[0]
    print(im_ratio)
    im = ax.imshow(np.where(mat > 1, 1, mat), aspect=im_ratio,
                   interpolation=None)
    ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)

    return im, ax


def view_embed(nmf_model: intNMF,
               clustered: Optional[bool] = False):
    """View the joint embeding matrix.
    """

    if not clustered:
        fig, ax = plt.subplots()
        im, ax = _view_mat(nmf_model.theta, 'topics', ax, 'cells')
        fig.colorbar(im)
        fig.tight_layout()
    else:
        ax = sns.clustermap(nmf_model.theta, method="complete", vmin=0, vmax=1,
                            yticklabels='None')

    return ax


def view_loadings(nmf_model: intNMF,
                  modality: str,
                  clustered: Optional[bool] = False):
    """View loadings matrix.
    """

    if modality == "both":
        if not clustered:
            fig = plt.figure(1, (8, 4))
            grid = ImageGrid(fig, 111, nrows_ncols=(1, 2),
                             axes_pad=0.1, cbar_mode='single')

            im1, ax1 = _view_mat(nmf_model.phi_rna, 'genes', grid[0], 'topics')
            im2, ax2 = _view_mat(nmf_model.phi_atac, 'regions', grid[1])

            cax = grid.cbar_axes[0]

            cb = fig.colorbar(im2, cax=cax)
            ax = [ax1, ax2]
        else:

            g0 = sns.clustermap(nmf_model.phi_rna, vmin=0, vmax=1,
                                xticklabels='None')
            # create new gridspec for the right part
            g1 = sns.clustermap(nmf_model.phi_atac, vmin=0, vmax=1,
                                xticklabels='None')

            fig = plt.figure(figsize=(15, 8))
            gs = mpl.gridspec.GridSpec(1, 1)

            mg0 = sfg.SeabornFig2Grid(g0, fig, gs[0])
            mg1 = sfg.SeabornFig2Grid(g1, fig, gs[1])

            gs.tight_layout(fig)

    elif modality == "atac":
        fig, ax = plt.subplots()
        im, ax = _view_mat(nmf_model.phi_atac, 'regions', ax, 'topics')
        fig.colorbar(im, ax=ax)

    elif modality == "rna":
        fig, ax = plt.subplots()
        im, ax = _view_mat(nmf_model.phi_rna, 'genes', ax, 'topics')
        fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return ax


def loss(nmf_model: intNMF,
         modality: Optional[str] = 'both'):
    """View loss."""
    if modality == "both":
        fig, ax = plt.subplots()
        ax.plot(nmf_model.loss, label='total')
        ax.plot(nmf_model.loss_atac, '--', label='atac')
        ax.plot(nmf_model.loss_rna, '--', label='rna')
        ax.legend()
    if modality == "atac":
        fig, ax = plt.subplots()
        ax.plot(nmf_model.loss_atac)
    if modality == "rna":
        fig, ax = plt.subplots()

        ax.plot(nmf_model.loss_rna)

    return ax

# TODO add correlation plots from phi and theta matrices.


def correlation_plot(cor_mat):
    """view matrix correlation"""
    sns.heatmap(cor_mat, cmap='RdBu', annot=True,
                annot_kws={"size": 7}, vmin=-1, vmax=1)

#def correlation_embed(nmf)
