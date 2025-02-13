"""Module containing the inNMF class and tf-idf function."""

import numpy as np
import sklearn
from  sklearn.utils.extmath import randomized_svd, squared_norm, safe_sparse_dot
from sklearn.utils import check_random_state, check_array
import time
import scipy
from scipy import sparse
import sys
import logging
from typing import Optional, Union, Mapping, List  # Special
import pandas as pd


class intNMF():
    """Class to run int NMF on multiome data

    Attributes
    ----------
    theta : array-like
        cell x topic matrix (joint low dimensional embedding)
    phi_rna : array-like
        topic x gene matrix. Gives the loading matrix to define topics.
    phi_atac : array-like
        topic x region matrix. Gives the loading matrix to define topics.
    loss : float
        l2 norm of the reconstruction error i.e. ||X_rna - WH_rna||_2 + ||X_atac - WH_atac||_2
    loss_atac : float
        l2 norm for the reconstruction error of just the atac matrix i.e. ||X_atac - WH_atac||_2
    loss_rna : float
        l2 norm for the reconstruction error of just the rna matrix i.e. ||X_rna - WH_rna||_2

    Parameters
    ----------
    n_topics : int
        The number of latent topics
    epochs : int
        Number of interations during optimisation.  Defaults to 15.
    init : string
        Method of initialising W and H. Defaults to random.
    mod1_skew : float
        Relative weighting of two modalities between 0-1. Defaults to 0.5.
    reg : string
        Include l1 or l2 regularisation or not. (This is TODO). Default None
    seed : int
        Random seed to use. Defaults to None ,i.e., no control of random seed (Useful for reproducability when using random initilisation)

    """

    def __init__(self, n_topics, epochs=15, init='random', mod1_skew=0.5,
                 reg=None, l1_weight=1e-4, seed=None):

        if (mod1_skew > 1) or (mod1_skew < 0):
            raise ValueError

        self.k = n_topics
        self.epochs = epochs
        self.init = init
        self.mod1_skew = mod1_skew
        self.loss = []
        self.loss_atac = []
        self.loss_rna = []
        self.epoch_times = []
        self.epoch_iter = []
        self.reg = reg
        self.rand = seed
        if self.reg:
            self.l1_weight = l1_weight
        self.rand = seed
        self.rna_features = None
        self.atac_features = None
        self.theta_df = None
        self.phi_rna_df = None
        self.phi_atac_df = None

    def _set_theta_df(self, barcodes):
         self.theta_df = pd.DataFrame(self.theta,
                                      columns=['Topic' + str(i) for i in np.arange(self.k)],
                                      index=barcodes)

    def _set_phi_rna_df(self):
        self.phi_rna_df =  pd.DataFrame(self.phi_rna,
                                        columns=self.rna_features,
                                        index=['Topic' + str(i) for i in np.arange(self.k)])
        
    def _set_phi_atac_df(self):
        self.phi_rna_df =  pd.DataFrame(self.phi_atac,
                                        columns=self.atac_features,
                                        index=['Topic' + str(i) for i in np.arange(self.k)])

    def _add_feature_names(self, rna_names: List[str], atac_names: Optional[List[str]] = None):
        """
        Add feature names to the nmf model. This is useful for plotting functions

        Parameters
        ----------
        rna_names: list of gene names must be same length as columns in rna_mat
        atac_names: Optional list. Must be the same length as columns in atac_mat
        """
        if len(rna_names) != self.phi_rna.shape[1]:
            print('rna dims dont match. Features not added')
            return
        self.rna_features = rna_names

        if atac_names is not None:
            if len(atac_names) != self.phi_atac.shape[1]:
                print('atac features not added. Dims dont match')
            self.atac_features = atac_names

    def fit(self, rna_mat: Union[np.array,  sparse.csr_matrix], atac_mat: Union[np.array,  sparse.csr_matrix],
            rna_names: Optional[List[str]] = None, atac_names = None):
        """
        Optimise NMF.

        Uses accelerated Hierarchical alternating least squares algorithm proposed here, but modified to
        joint factorise two matrices. https://arxiv.org/pdf/1107.5194.pdf. Only required arguments are the matrices to use for factorisation.
        GEX and ATAC matrices are expected in cell by feature format. Matrices should be scipy sparse matrices.
        min ||X_rna - (theta . phi_rna)||_2 and min ||X_atac - (theta . phi_atac)||_2 s.t. theta, phi_rna, phi_atac > 0. So that theta hold the latent topic scores for a cell. And phi
        the allows recontruction of X

        Parameters
        ----------
        rna_mat: scipy sparse matrix (or coercible) of single cell gene expression
        atac_mat: scipy sparse matrix (or coercible) of single cell gene expression
        rna_names: Optional list of gene names must be same length as columns in rna_mat
        atac_names: Optional list. Must be the same length as columns in atac_mat

        Returns
        --------
        self
            Access low dim embed with self.theta or the loadings with self.phi_rna or self.phi_atac
        """
        start = time.perf_counter()

        cells = atac_mat.shape[0]
        regions = atac_mat.shape[1]
        genes = rna_mat.shape[1]

        RNA_mat = rna_mat
        ATAC_mat = atac_mat

        nM_rna = sparse.linalg.norm(RNA_mat, ord='fro')**2
        nM_atac = sparse.linalg.norm(ATAC_mat, ord='fro')**2

        # intialise matrices. Default is random. Dense numpy arrays.
        theta, phi_rna, phi_atac = self._initialize_nmf(RNA_mat, ATAC_mat,
                                                        self.k, init=self.init)

        early_stopper = 0
        counter = 0
        interval = round(self.epochs/10)
        progress = 0

        self.loss = []
        # perform the optimisation. A, B, theta and phi matrices are modified
        # to fit the update function
        for i in range(self.epochs):
            epoch_start = time.perf_counter()

            # update theta/W
            rnaMHt = safe_sparse_dot(RNA_mat, phi_rna.T)
            rnaHHt = phi_rna.dot(phi_rna.T)
            atacMHt = safe_sparse_dot(ATAC_mat, phi_atac.T)
            atacHHt = phi_atac.dot(phi_atac.T)

            if i == 0:
                scale = ((np.sum(rnaMHt*theta)/np.sum(rnaHHt*(theta.T.dot(theta)))) +
                         np.sum(atacMHt*theta)/np.sum(atacHHt*(theta.T.dot(theta))))/2
                theta = theta*scale

            theta, theta_it = self._HALS_W(theta, rnaHHt, rnaMHt,
                                           atacHHt, atacMHt)

            # update phi_rna/H
            A_rna = safe_sparse_dot(theta.T, RNA_mat)
            B_rna = (theta.T).dot(theta)

            phi_rna, phi_rna_it = self._HALS(phi_rna, B_rna, A_rna)

            # update phi_atac/H

            A_atac = safe_sparse_dot(theta.T, ATAC_mat)
            B_atac = (theta.T).dot(theta)

            phi_atac, phi_atac_it = self._HALS(phi_atac, B_atac, A_atac)

            error_rna = np.sqrt(nM_rna - 2*np.sum(phi_rna*A_rna) +
                                np.sum(B_rna*(phi_rna.dot(phi_rna.T))))
            error_atac = np.sqrt(nM_atac - 2*np.sum(phi_atac*A_atac) +
                                 np.sum(B_atac*(phi_atac.dot(phi_atac.T))))

            epoch_end = time.perf_counter()

            epoch_duration = epoch_end - epoch_start
            logging.info('''epoch duration: {}
                            loss: {}'''.format(epoch_duration,
                                               error_rna+error_atac))
            logging.info('''theta iter: {}
                            phi rna iter: {}
                            phi atac iter: {}'''.format(theta_it,
                                                        phi_rna_it,
                                                        phi_atac_it))
            self.epoch_times.append(epoch_duration)
            self.loss.append(error_rna+error_atac)
            self.loss_atac.append(error_atac)
            self.loss_rna.append(error_rna)
            self.epoch_iter.append(theta_it + phi_rna_it + phi_atac_it)

            # early stopping condition requires 50 consecutive iterations
            # with no change.
            try:
                if self.loss[-2] < self.loss[-1]:
                    early_stopper += 1
                elif early_stopper > 0:
                    early_stopper = 0
                if early_stopper > 50:
                    break
            except IndexError:
                continue

            counter += 1

        self.theta = theta
        self.phi_rna = phi_rna
        self.phi_atac = phi_atac

        self.theta[self.theta < 1e-10] = 0
        self.phi_rna[self.phi_rna < 1e-10] = 0
        self.phi_atac[self.phi_atac < 1e-10] = 0

        end = time.perf_counter()

        if rna_names is not None:
            self._add_feature_names(rna_names, atac_names)

        self.total_time = end - start
        logging.info(self.total_time)
        del RNA_mat
        del ATAC_mat

    def _HALS(self, H, WtW, WtM, delta=0.1):
        """Optimizing min_{V >= 0} ||M-UV||_F^2 with an exact block-coordinate descent.

        UtU and UtM are exprensive to compute so multiple updates are calcuated.
        stop condition
        (epoch run time ) < (precompute time + current iteration time)/2
        AND
        sum of squares of updates to V at current epoch > 0.01 * sum of squares of updates to V at first epoch

        Parameters:
        -----------
        H: Array-like
            mat to update
        WtW : Array-like
            precomputed dense array
        WtM : Array-like
            precomputed dense array
        delta : float
            control loss based stop criteria

        Returns
        ---------------------
        Array-like
            H (loading matrix)
        int
            number of iterations (usually 6/7, max 10)

        """
        r, n = H.shape
        cnt = 1
        eps = 1
        eps0 = 1
        n_it = 0

        while (cnt == 1 or
               ((n_it < 10) and
                (eps >= (delta**2)*eps0))):

            nodelta = 0

            for k in range(r):
                if self.reg:
                    deltaH = np.maximum((WtM[k, :] - WtW[k, :].dot(H) -
                                         self.l1_weight * np.ones(n)) /
                                        WtW[k, k], -H[k, :])
                else:
                    deltaH = np.maximum((WtM[k, :] - WtW[k, :].dot(H)) /
                                        WtW[k, k], -H[k, :])

                H[k, :] = H[k, :] + deltaH
                nodelta = nodelta + deltaH.dot(deltaH.T)
                H[k, H[k, :] == 0] = 1e-16*np.max(H)

            if cnt == 1:
                eps0 = nodelta

            eps = nodelta
            cnt = 0
            n_it += 1

        return H, n_it

    def _HALS_W(self, W, rnaHHt, rnaMHt, atacHHt, atacMHt, delta=0.1):
        """Optimizing min_{W >= 0} ||X1-WH1||_F^2 + ||X2-WH2||_F^2.

        An exact block-coordinate descent scheme is applied to update W. HHt
        and MHt (X) are exprensive to compute so multiple updates are
        pre-calcuated.

        stop condition

        (epoch run time ) < (precompute time + current iteration time)/2
        AND
        sum of squares of updates to V at current epoch > 0.01 * sum of squares of updates to V at first epoch

        Parameters:
        -----------
        W : array-like
            mat to update
        atacHHt : array-like
            precomputed dense array
        atacMHt : array-like
            precomputed dense array
        rnaHHt : array-like
            precomputed dense array
        rnaMHt : array-like
            precomputed dense array
        delta : float
            control loss based stop criteria

        Returns
        ---------------------
        array-like
            W optimised for current H_atac and H_rna values
        int
            number of iterations (usually 6/7)
        """
        n, K = W.shape
        cnt = 1
        eps = 1
        eps0 = 1
        n_it = 0
        mod1_skew = self.mod1_skew

        while (cnt == 1 or
               ((n_it < 10) and
                (eps >= (delta**2)*eps0))):

            nodelta = 0

            for k in range(K):
                #print(W.shape, atacHHt.shape)rng
                #print(W.dot(atacHHt[:,k]).shape)
                if self.reg:
                    deltaW = np.maximum((((mod1_skew*(rnaMHt[:, k] -
                                           W.dot(rnaHHt[:, k]))) +
                                          ((1-mod1_skew)*(atacMHt[:, k] -
                                           W.dot(atacHHt[:, k]))) -
                                          (self.l1_weight*np.ones(n))) /
                                        ((mod1_skew*rnaHHt[k, k]) +
                                         ((1-mod1_skew)*atacHHt[k, k]))),
                                        -W[:, k])
                else:
                    deltaW = np.maximum((((mod1_skew*(rnaMHt[:, k] -
                                           W.dot(rnaHHt[:, k]))) +
                                          ((1-mod1_skew)*(atacMHt[:, k] -
                                           W.dot(atacHHt[:, k])))) /
                                        ((mod1_skew*rnaHHt[k, k]) +
                                         ((1-mod1_skew)*atacHHt[k, k]))),
                                        -W[:, k])

                W[:, k] = W[:, k] + deltaW
                nodelta = nodelta + deltaW.dot(deltaW.T)
                W[W[:, k] == 0, k] = 1e-16*np.max(W)

            if cnt == 1:
                eps0 = nodelta

            eps = nodelta
            cnt = 0

            n_it += 1

        return W, n_it

    ### PLAN OT CHANGE THIS TO GILLIS METHOD WITH AUTOMATIC TOPIC DETECTION
    #https://github.com/CostaLab/scopen/blob/6be56fac6470e5b6ecdc5a2def25eb60ed6a1bcc/scopen/MF.py#L696
    def _initialize_nmf(self, X, ATAC_mat, n_components, eps=1e-6,
                        init='random'):
        """ Algorithms for NMF initialization.
        Computes an initial guess for the non-negative
        rank k matrix approximation for X: X = WH
        ParametersATAC_mat_tfidf_log
        ----------
        X : array-like, shape (n_samples, n_features)
            The data matrix to be decomposed.
        n_components : integer
            The number of components desired in the approximation.
        init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
            Method used to initialize the procedure.
            Default: None.
            Valid options:
            - None: 'nndsvd' if n_components <= min(n_samples, n_features),
                otherwise 'random'.
            - 'random': non-negative random matrices, scaled with:
                sqrt(X.mean() / n_components)
            - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
        eps : float
            Truncate all values less then this in output to zero.
        random_state : int, RandomState instance or None, optional, default: None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`. Used when ``random`` == 'nndsvdar' or 'random'.
        Returns
        -------
        W : array-like, shape (n_samples, n_components)
            Initial guesses for solving X ~= WH
        H : array-like, shape (n_components, n_features)
            Initial guesses for solving X ~= WH
        H_atac : array-like, shape (n_components, n_features)
            Initial guesses for solving X_atac ~= WH_atac
        References
        ----------
        C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start
        for nonnegative matrix factorization - Pattern Recognition, 2008,
        http://tinyurl.com/nndsvd"""


        n_samples, n_features = X.shape

        if not sparse.issparse(X) and np.any(np.isnan(X)):
            raise ValueError("NMF initializations with NNDSVD are not ",
                             "available with missing values (np.nan).")

        if init == 'random':
            X_mean = X.mean()
            avg = np.sqrt(X_mean / n_components)
            avg_atac = np.sqrt(ATAC_mat.mean() / n_components)

            rng = check_random_state(self.rand)
            H = avg * rng.randn(n_components, n_features)
            W = avg * rng.randn(n_samples, n_components)
            H_atac = avg_atac * rng.randn(n_components, ATAC_mat.shape[1])

            # we do not write np.abs(H, out=H) to stay compatible with
            # numpy 1.5 and earlier where the 'out' keyword is not
            # supported as a kwarg on ufuncs
            np.abs(H, H)
            np.abs(W, W)
            np.abs(H_atac, H_atac)
            return W, H, H_atac

        # NNDSVD initialization
        U, S, V = randomized_svd(X, n_components, random_state=self.rand)
        W, H = np.zeros(U.shape), np.zeros(V.shape)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        for j in range(1, n_components):
            x, y = U[:, j], V[j, :]

            # extract positive and negative parts of column vectors
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

            # and their norms
            x_p_nrm, y_p_nrm = np.sqrt(squared_norm(x_p)), np.sqrt(squared_norm(y_p))
            x_n_nrm, y_n_nrm = np.sqrt(squared_norm(x_n)), np.sqrt(squared_norm(y_n))

            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p

            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n

            lbd = np.sqrt(S[j] * sigma)
            W[:, j] = lbd * u
            H[j, :] = lbd * v

        W[W < eps] = 0
        H[H < eps] = 0

        H_atac = safe_sparse_dot(np.linalg.pinv(W), ATAC_mat)
        H_atac[H_atac < eps] = 0

        return W, H, H_atac


def log_tf_idf(mat_in, scale = 10000):
    """
    Return a TF-IDF transformed matrix.

    Parameters
    ----------

    mat_in :  `csr matrix type <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_
        Positional (required) matrix to be TF-IDF transformed. Should be cell x feature.
    scale : int
        scale values by this number (optional)

    Raises
    ------
    nmf_models.InvalidMatType: If the mat type is invalid

    Returns
    -------
    csr matrix
        TF-IDF transformed matrix
    """
    mat = sparse.csr_matrix(mat_in, copy=True)

    cell_counts = mat.sum(axis = 1)
    for row in range(len(cell_counts)):
        mat.data[mat.indptr[row]:mat.indptr[row+1]] = (mat.data[mat.indptr[row]:mat.indptr[row+1]]*scale)/cell_counts[row]

    mat = sparse.csc_matrix(mat)
    [rows, cols] = mat.nonzero()
    feature_count = np.zeros(mat.shape[1])
    unique, counts = np.unique(cols, return_counts=True)

    feature_count[unique] = counts
    for col in range(len(feature_count)):
        mat.data[mat.indptr[col]:mat.indptr[col+1]] = mat.data[mat.indptr[col]:mat.indptr[col+1]]*(mat.shape[0]/(feature_count[col]+1))  #idf Number of cells/Number of cells region is open in + 1
    mat = mat.log1p()
    return mat

class InvalidMatType(Exception):
    """Raised if the mat is invalid."""
    pass
