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
import pandas as pd

sys.setrecursionlimit(100000)

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
mpl.rc('image', cmap='cividis')
sns.set_palette(plt.cm.cividis(np.linspace(0, 1, 8))[1:-1])


def _view_mat(mat, xlab, ax, ylab=None):
    """Base matrix plotter. Saturates at 1

        Parameters
        ----------
        mat : array-like
            Matrix to plot
        xlab : str
            x axis label
        ax : matplotlib axis
            axis for plot
        ylab : str
            y axis label

        Returns
        -------
        AxesImage
            imshow of matric provided
        matplotlib axes
            axes for figure

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
    """View the joint embeding matrix of an intNMF object.

        Parameters
        ----------
        nmf_model : intNMF
            joint NMF object. With joint embed to plot
        clustered : boolean
            cluster cells and topics using seaborn clustermap

        Returns
        -------
        matplotlib axes
            axes for figure
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

        Parameters
        ----------
        nmf_model : intNMF
            joint NMF object. With loadings to plot
        modality : str, "both" | "rna" | "atac"
            choose modality to plot
        clustered : boolean
            cluster topics and features using seaborn clustermap

        Returns
        -------
        matplotlib axes
            axes for figure. if bodality == 'both' list of axes is returned
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
        if not clustered:
            fig, ax = plt.subplots()
            im, ax = _view_mat(nmf_model.phi_atac, 'regions', ax, 'topics')
            fig.colorbar(im, ax=ax)
        else:
            ax = sns.clustermap(nmf_model.phi_atac, method="complete", vmin=0, vmax=1,
                            yticklabels='None')

    elif modality == "rna":
        if not clustered:
            fig, ax = plt.subplots()
            im, ax = _view_mat(nmf_model.phi_rna, 'genes', ax, 'topics')
            fig.colorbar(im, ax=ax)
        else:
            ax = sns.clustermap(nmf_model.theta, method="complete", vmin=0, vmax=1,
                            yticklabels='None')
    fig.tight_layout()

    return ax


def loss(nmf_model: intNMF,
         modality: Optional[str] = 'both'):
    """View loss.
    
        Parameters
        ---------
        nmf_model: intNMF
            intNMF model with loss to plot
        modality: str, 'both' | 'rna' | 'atac'
            plot combined or modality specific loss. Default combined
        
        Returns
        ------
        matplotlib ax
    """
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
    """view matrix correlation
    
        Parameters
        ----------
        cor_mat : array-lile
            correlation matrix to plot
        
        Returns
        -------
        matplotlib ax
    """
    ax = sns.heatmap(cor_mat, cmap='RdBu', annot=True,
                annot_kws={"size": 7}, vmin=-1, vmax=1)
    return ax
#def correlation_embed(nmf)


def embed_plot(nmf_model: intNMF,
               TX: Optional[int] = 0,
               TY: Optional[int] = 1,
               labs: Optional[list] = None,
               scale_x: Optional[float] = 1,
               scale_y: Optional[float] = 1,
               title: Optional[str] = None,
               y_lab: Optional[str] = None,
               x_lab: Optional[str] = None):
    """Plot nmf embedding.

        Parameters
        ----------
        nmf_model : intNMF
            intNMF model with embed to view (.fit has been run)
        TX : int
            factor/topic for x-axis
        TY : int
            factor/topic for y-axis
        labs : list
            cell labels used to colour plot. discrete or continuos determined by fraction of unique elements
        scale_x : float
            value to divide x-axis values by
        scale_y : float
            value to divide y-axis values by
        title : str
            plot title, default None
        y_lab : str
            y-axis label, default None
        x_lab : str
            x-axis label, default None

        Returns
        -------
        matplotlib ax
    """

    fig, ax = plt.subplots()
    fig.tight_layout()
    print(len(np.unique(labs))/len(labs) > 0.05)
    if labs is not None:
        # treat labs as continous
        if len(np.unique(labs))/len(labs) > 0.05:
            im = ax.scatter(nmf_model.theta[:, TX]*scale_x, nmf_model.theta[:, TY]*scale_y, c=labs)
            fig.colorbar(im, ax=ax)
        # treat labs as discrete
        else:
            lab_cmap = dict((val, ["{}".format(val), mpl.colormaps['rainbow'](i/len(np.unique(labs)))]) for i, val in enumerate(np.unique(labs)))
            u_lab, u_col = map(list, zip(*lab_cmap.values()))
            label, colour =  map(list, zip(*[lab_cmap[el] for el in labs]))

            ax.scatter(nmf_model.theta[:, TX]*scale_x, nmf_model.theta[:, TY]*scale_y,
                       c=colour, label=label)
            # produce a legend with the unique colors from the scatter
            handles = [plt.scatter([0,0],[0,0], color=col, marker='o') for col in u_col]
            ax.legend(handles, u_lab, ncol=2)

    else:
        ax.scatter(nmf_model.theta[:, TX]*scale_x, nmf_model.theta[:, TY]*scale_y)

    if x_lab is None:
        ax.set_xlabel("Factor {}".format(TX))
    else:
        ax.set_xlabel(x_lab)

    if y_lab is None:
        ax.set_ylabel("Factor {}".format(TY))
    else:
        ax.set_ylabel(y_lab)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def factor_biplot(nmf_model: intNMF,
                  TX: Optional[int] = 0,
                  TY: Optional[int] = 1,
                  n_labs: Optional[int] = 5,
                  modality: Optional[str] = 'rna',
                  labs: Optional[list] = None,
                  select: Optional[str] = None,
                  title: Optional[str] = None,
                  x_lab: Optional[str] = None,
                  y_lab: Optional[str] = None):
    """Plot nmf embedding. With features projected on. Plot is scaled so that the projected features don't exceed 1.

        Parameters
        ----------
        nmf_model : intNMF
            intNMF model with embed to view (.fit has been run)
        TX : int
            factor/topic for x-axis
        TY : int
            factor/topic for y-axis
        n_labs: int
            number of features to plot per topic (i.e. 2)
        modality: str, 'rna' | 'atac'
            select modality from which features are projected from.
        labs : list
            cell labels used to colour plot. Discrete or continuos determined by fraction of unique elements
        title : str
            plot title, default None
        y_lab : str
            y-axis label, default None
        x_lab : str
            x-axis label, default None

        Returns
        -------
        matplotlib ax
    """

    scale_x_embed = 1.0/nmf_model.theta[:, TX].max()
    scale_y_embed = 1.0/nmf_model.theta[:, TY].max()

    ax = embed_plot(nmf_model, TX=TX, TY=TY,
                    labs=labs, scale_x=scale_x_embed, scale_y=scale_y_embed,
                    title=title, x_lab=x_lab, y_lab=y_lab)

    if modality == 'rna':
        phi = nmf_model.phi_rna
        col_names = nmf_model.rna_features
    elif modality == 'atac':
        phi = nmf_model.phi_atac
        col_names = nmf_model.atac_features
    else:
        print('select modality from rna or atac')
        return

    if col_names is None:
        print('Missing feature labels continuing anyway')

    phi_df = pd.DataFrame(phi, columns=col_names, index=['Topic' + str(i) for i in np.arange(nmf_model.k)])

    top_features_TX = phi_df.iloc[TX, :].sort_values(ascending=False)[0:n_labs]
    top_features_TY = phi_df.iloc[TY, :].sort_values(ascending=False)[0:n_labs]

    features_to_add = np.append(top_features_TX.index.values, top_features_TY.index.values)

    if select is not None and set(select).issubset(phi_df.columns):
        features_to_add = np.append(features_to_add, select)

    scale_x_loading = 1/(phi_df.loc[phi_df.index[TX], features_to_add].max())
    scale_y_loading = 1/(phi_df.loc[phi_df.index[TY], features_to_add].max())

    for feature in features_to_add:

        x_coord = phi_df.loc[phi_df.index[TX], feature]*scale_x_loading
        y_coord = phi_df.loc[phi_df.index[TY], feature]*scale_y_loading

        ax.plot([0, x_coord], [0, y_coord], color='tab:orange')
        ax.text(x_coord - 0.05, y_coord * 1.05, feature)

    plt.tight_layout()

    return ax


def plot_weights_rank(nmf_model: intNMF,
                      factor: int,
                      n_labs: Optional[int] = 5,
                      modality: Optional[str] = 'rna',
                      select: Optional[list] = None,
                      title: Optional[str] = None):
    """Rank plot of the feature weights.

        Parameters
        ----------
        nmf_model : intNMF
            intNMF model with loadings to rank (.fit has been run)
        factor : int
            factor/topic to plot feature weights for        
        n_labs: int
            number of features to label
        modality: str, 'rna' | 'atac'
            select modality from which features are ranked.
        select : list[str]
            list of additional features to label
        title : str
            plot title, default None

        Returns
        -------
        matplotlib ax
    """

    if modality == 'rna':
        phi = nmf_model.phi_rna
        col_names = nmf_model.rna_features
    elif modality == 'atac':
        phi = nmf_model.phi_atac
        col_names = nmf_model.atac_features
    else:
        print('select modality from rna or atac')
        return

    if col_names is None:
        print('Missing feature labels continuing anyway')

    phi_df = pd.DataFrame(phi, columns=col_names, index=['Factor {}'.format(i) for i in np.arange(nmf_model.k)])

    # Plot ranks
    ranks = phi_df.iloc[factor, :].sort_values(ascending=False)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(ranks)), ranks)

    # Plot annotations of top ranked genes and named ranks
    ranks_df = ranks.to_frame()
    ranks_df['rank'] = np.arange(len(ranks_df))

    top_N = ranks_df.iloc[0:n_labs, :]

    if select is not None and set(select).issubset(phi_df.columns):
        genes_to_add = ranks_df.loc[select, :]
        top_N = pd.concat([top_N, genes_to_add])

    jitter = np.linspace(0.1, -1, num=len(top_N))
    for i, (index, row) in enumerate(top_N.iterrows()):
        rank = int(row[1])
        value = row[0]
        ax.annotate(index, xy=(rank, value), xytext=(rank + (len(ranks)*0.1), np.abs(value + jitter[i])),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=1))
    ax.set_xlabel('Feature rank')
    ax.set_ylabel('Factor loading')
    if title is not None:
        ax.set_title(title)
    return ax


def plot_weights_scatter(nmf_model: intNMF,
                         factors: tuple,
                         n_labs: Optional[int] = 5,
                         modality: Optional[str] = 'rna',
                         select: Optional[list] = None,
                         title: Optional[str] = None):
    """Rank plot of the feature weights.

        Parameters
        ----------
        nmf_model : intNMF
            intNMF model with loadings to plot (.fit has been run)
        factors : tuple(int, int)
            factor/topics to plot feature weights for        
        n_labs: int
            number of features to label per topic/factor
        modality: str, 'rna' | 'atac'
            select modality from which features are plotted.
        select : list[str]
            list of additional features to label
        title : str
            plot title, default None

        Returns
        -------
        matplotlib ax
    """
    TX = factors[0]
    TY = factors[1]

    if modality == 'rna':
        phi = nmf_model.phi_rna
        col_names = nmf_model.rna_features
    elif modality == 'atac':
        phi = nmf_model.phi_atac
        col_names = nmf_model.atac_features
    else:
        print('select modality from rna or atac')
        return

    if col_names is None:
        print('Missing feature labels continuing anyway')

    phi_df = pd.DataFrame(phi, columns=col_names, index=['Factor {}'.format(i) for i in np.arange(nmf_model.k)])

    fig, ax = plt.subplots()
    ax.scatter(phi_df.loc[phi_df.index[TX], :], phi_df.loc[phi_df.index[TY], :])

    top_features_TX = phi_df.iloc[TX, :].sort_values(ascending=False)[0:n_labs].sort_values(ascending=True)
    top_features_TY = phi_df.iloc[TY, :].sort_values(ascending=False)[0:n_labs]

    features_to_add = np.append(top_features_TY.index.values, top_features_TX.index.values)

    if select is not None and set(select).issubset(phi_df.columns):
        features_to_add = np.append(features_to_add, select)

    jitter = np.linspace(0, np.pi, num=len(features_to_add))

    for i, feature in enumerate(features_to_add):
        x_coord = phi_df.loc[phi_df.index[TX], feature]
        y_coord = phi_df.loc[phi_df.index[TY], feature]
        ax.annotate(feature, xy=(x_coord, y_coord),
                    xytext=(x_coord + 0.2*np.sin(jitter[i]), y_coord + 0.2*np.cos(jitter[i])),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=1))

    ax.set_xlabel(phi_df.index[TX])
    ax.set_ylabel(phi_df.index[TY])

    return ax

def plot_weights_bar(nmf_model: intNMF,
                     genes: list,
                     factors: Optional[list] = None,
                     modality: Optional[str] = 'rna',
                     title: Optional[str] = None):
    """Plot stacked bar chart. With features weights plotted per feature and coloured the factor.

        Parameters
        ----------
        nmf_model : intNMF
            intNMF model with loadings to plot (.fit has been run)
        genes : list[str]
            list of features whose weights are plooted. (used to determine the number of bars)
        factors : list[int]
            factor/topics to plot feature weights. (used to color sections of the bar)        
        modality: str, 'rna' | 'atac'
            select modality from which features are plotted.
        title : str
            plot title, default None

        Returns
        -------
        matplotlib ax
    """

    if modality == 'rna':
        phi = nmf_model.phi_rna
        col_names = nmf_model.rna_features
    elif modality == 'atac':
        phi = nmf_model.phi_atac
        col_names = nmf_model.atac_features
    else:
        print('select modality from rna or atac')
        return

    if col_names is None:
        print('Missing feature labels continuing anyway')

    phi_df = pd.DataFrame(phi, columns=col_names, index=['Factor {}'.format(i) for i in np.arange(nmf_model.k)])


    if not set(genes).issubset(phi_df.columns):
        print('features given not in features')
        return
        
    if factors is None:
        factors = np.arange(nmf_model.k)
        
    plot_df = phi_df.loc[phi_df.index[factors], genes]
    if plot_df.shape[0] > 10:
        lab_cmap = dict((val, mpl.colormaps['rainbow'](i/plot_df.shape[0])) for i, val in enumerate(plot_df.index))
    else:
        lab_cmap = dict((val, plt.get_cmap('tab10')(i)) for i, val in enumerate(plot_df.index))
    fig, ax = plt.subplots()

    bottom  = np.zeros(len(genes))

    for index, row in plot_df.iterrows():
        ax.bar(row.index, row.values, 0.5, label=index, bottom=bottom, color=lab_cmap[index])
        bottom+=row.values
    ax.legend(ncol=2)
    ax.tick_params(axis='x', labelrotation=45)
    return ax