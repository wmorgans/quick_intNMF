import anndata as ad
import numpy as np
#import scanpy as sc
import sklearn
from  sklearn.utils.extmath import randomized_svd, squared_norm,safe_sparse_dot
from sklearn.utils import check_random_state, check_array
import time
import scipy
from scipy import sparse
import sys
import logging

class intNMF():
    """
    intNMF
    ======

    Class to run int NMF on multiome data

    Attributes
    ---------------


    Methods
    ---------------

    """

    def __init__(self, n_topics, epochs=200, init='random', mod1_skew=1,
                 reg=None, l1_weight=1e-4, seed=None):
        """
        Parameters
        ----------
        n_topics (k): is the number of latent topics
        epochs: Number of interations during optimisation
        init: initialisation  method. random | svd
        Lists to store training metrics.
        """

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
        self.rand = seed

    def fit(self, rna_mat, atac_mat):
        """optimise NMF. Uses accelerated Hierarchical alternating least squares algorithm proposeed here, but modified to
        joint factorise two matrices. https://arxiv.org/pdf/1107.5194.pdf. Only required arguments are the matrices to use for factorisation.
        GEX and ATAC matrices are expected in cell by feature format. Matrices should be scipy sparse matrices.
        min ||X_rna - (theta . phi_rna)||_2 and min ||X_atac - (theta . phi_atac)||_2 s.t. theta, phi_rna, phi_atac > 0. So that theta hold the latent topic scores for a cell. And phi
        the allows recontruction of X
        Parameters
        ----------
        rna_mat: scipy sparse matrix of single cell gene expression
        atac_mat: scipy sparse matrix of single cell gene expression"""

        start =time.perf_counter()

        cells = atac_mat.shape[0]
        regions = atac_mat.shape[1]
        genes = rna_mat.shape[1]

        RNA_mat = rna_mat
        ATAC_mat = atac_mat

        nM_rna = sparse.linalg.norm(RNA_mat, ord='fro')**2
        nM_atac = sparse.linalg.norm(ATAC_mat, ord='fro')**2

        #intialise matrices. Default is random. Dense numpy arrays.
        theta, phi_rna, phi_atac = self._initialize_nmf(RNA_mat, ATAC_mat, self.k, init=self.init)

        early_stopper = 0
        counter = 0
        interval = round(self.epochs/10)
        progress = 0

        self.loss = []
        #perform the optimisation. A, B, theta and phi matrices are modified to fit the update function
        for i in range(self.epochs):

            epoch_start =time.perf_counter()


            #update theta/W
            eit1 = time.perf_counter()
            rnaMHt = safe_sparse_dot(RNA_mat, phi_rna.T)
            rnaHHt = phi_rna.dot(phi_rna.T)
            atacMHt = safe_sparse_dot(ATAC_mat, phi_atac.T)
            atacHHt = phi_atac.dot(phi_atac.T)
            eit1 = time.perf_counter() - eit1

            if i == 0:
                scale = ((np.sum(rnaMHt*theta)/np.sum(rnaHHt*(theta.T.dot(theta)))) +
                        np.sum(atacMHt*theta)/np.sum(atacHHt*(theta.T.dot(theta))))/2
                theta=theta*scale


            theta, theta_it = self._HALS_W(theta, rnaHHt, rnaMHt, atacHHt, atacMHt, eit1)

            #update phi_rna/H

            eit1 = time.perf_counter()
            A_rna = safe_sparse_dot(theta.T, RNA_mat)
            B_rna = (theta.T).dot(theta)
            eit1 = time.perf_counter() - eit1

            phi_rna, phi_rna_it = self._HALS(phi_rna, B_rna, A_rna, eit1)

            #update phi_atac/H

            eit1 = time.perf_counter()
            A_atac = safe_sparse_dot(theta.T, ATAC_mat)
            B_atac = (theta.T).dot(theta)
            eit1 = time.perf_counter() - eit1

            phi_atac, phi_atac_it = self._HALS(phi_atac, B_atac, A_atac, eit1)

            error_rna = np.sqrt(nM_rna - 2*np.sum(phi_rna*A_rna) + np.sum(B_rna*(phi_rna.dot(phi_rna.T))))
            error_atac =  np.sqrt(nM_atac - 2*np.sum(phi_atac*A_atac) + np.sum(B_atac*(phi_atac.dot(phi_atac.T))))

            epoch_end =time.perf_counter()

            epoch_duration = epoch_end - epoch_start
            logging.info('epoch duration: {}\nloss: {}'.format(epoch_duration, error_rna+error_atac))
            logging.info('theta iter: {}\nphi rna iter: {}\nphi atac iter: {}\n'.format(theta_it, phi_rna_it, phi_atac_it))
            self.epoch_times.append(epoch_duration)
            self.loss.append(error_rna+error_atac)
            self.loss_atac.append(error_atac)
            self.loss_rna.append(error_rna)
            self.epoch_iter.append(theta_it + phi_rna_it  + phi_atac_it)
            #print('total number of iterations in epoch {} is {}'.format(i, theta_rna_it, phi_rna_it + theta_atac_it + phi_atac_it))

            #early stopping condition requires 50 consecutive iterations with no change.
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

        end = time.perf_counter()

        self.total_time = end - start
        logging.info(self.total_time)
        del RNA_mat
        del ATAC_mat



    def _HALS(self, H, WtW, WtM, eit1, alpha=0.5, delta=0.1):
        """Optimizing min_{V >= 0} ||M-UV||_F^2 with an exact block-coordinate descent schemeUpdate V.
        UtU and UtM are exprensive to compute so multiple updates are calcuated.
        Parameters:
        -----------
        V: Array like mat to update
        UtU: precomputed dense array
        UtM: precomputed dense array
        eit1: precompute time
        alpha: control time based stop criteria
        delta: control loss based stop criteria
        Returns
        ---------------------
        V: optimised array
        n_it: number of iterations (usually 6/7)

        stop condition

        (epoch run time ) < (precompute time + current iteration time)/2
        AND
        sum of squares of updates to V at current epoch > 0.01 * sum of squares of updates to V at first epoch
        """
        r, n = H.shape
        eit2 = time.perf_counter()  # start time
        cnt = 1
        eps = 1
        eps0 = 1
        eit3 = 0  # iteration time
        n_it = 0

        while cnt == 1 or ((time.perf_counter()-eit2 < (eit1+eit3)*alpha) and (eps >= (delta**2)*eps0)):
            nodelta = 0
            if cnt == 1:
                eit3 = time.perf_counter()

            for k in range(r):
                deltaH = np.maximum((WtM[k, :]-WtW[k, :].dot(H))/WtW[k, k], -H[k, :])

                H[k, :] = H[k, :] + deltaH
                nodelta = nodelta + deltaH.dot(deltaH.T)
                H[k, H[k, :] == 0] = 1e-16*np.max(H)

            if cnt == 1:
                eps0 = nodelta
                eit3 = time.perf_counter() - eit3

            eps = nodelta
            cnt = 0
            n_it += 1

        return H, n_it

    def _HALS_W(self, W, rnaHHt, rnaMHt, atacHHt, atacMHt, eit1,
                alpha=0.5, delta=0.1):
        """Optimizing min_{W >= 0} ||X1-WH1||_F^2 + ||X2-WH2||_F^2.

        An exact block-coordinate descent scheme is applied to update W. HHt
        and HtM (X) are exprensive to compute so multiple updates are
        pre-calcuated.

        Parameters:
        -----------
        W: Array like mat to update
        atacHHt: precomputed dense array
        atacMHt: precomputed dense array
        rnaHHt: precomputed dense array
        rnaMHt: precomputed dense array
        eit1: precompute time
        alpha: control time based stop criteria
        delta: control loss based stop criteria
        Returns
        ---------------------
        V: optimised array
        n_it: number of iterations (usually 6/7)

        stop condition

        (epoch run time ) < (precompute time + current iteration time)/2
        AND
        sum of squares of updates to V at current epoch > 0.01 * sum of squares of updates to V at first epoch
        """
        n, K = W.shape
        eit2 = time.perf_counter()  # start time
        cnt = 1
        eps = 1
        eps0 = 1
        eit3 = 0  # iteration time
        n_it = 0
        mod1_skew = self.mod1_skew

        while cnt == 1 or ((time.perf_counter()-eit2 < (eit1+eit3)*alpha) and (eps >= (delta**2)*eps0)):
            nodelta=0
            if cnt == 1:
                eit3 = time.perf_counter()


            for k in range(K):
                #print(W.shape, atacHHt.shape)
                #print(W.dot(atacHHt[:,k]).shape)
                deltaW = np.maximum(((mod1_skew*(rnaMHt[:,k] -W.dot(rnaHHt[:,k]) + (1-mod1_skew)*(atacMHt[:,k]-W.dot(atacHHt[:,k]))) )/
                         ((mod1_skew*rnaHHt[k, k]) + ((1-mod1_skew)*atacHHt[k,k]))), -W[:, k])

                W[:,k] = W[:,k] + deltaW
                nodelta = nodelta + deltaW.dot(deltaW.T)
                W[W[:,k] == 0, k] =   1e-16*np.max(W)


            if cnt == 1:
                eps0 = nodelta
                eit3 = time.perf_counter() - eit3

            eps = nodelta
            cnt = 0

            n_it += 1

        return W, n_it

    ### PLAN OT CHANGE THIS TO GILLIS METHOD WITH AUTOMATIC TOPIC DETECTION
    #https://github.com/CostaLab/scopen/blob/6be56fac6470e5b6ecdc5a2def25eb60ed6a1bcc/scopen/MF.py#L696
    def _initialize_nmf(self, X, ATAC_mat, n_components, eps=1e-6,
                        init='random'):

            """Algorithms for NMF initialization.
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
            C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
            nonnegative matrix factorization - Pattern Recognition, 2008,
            http://tinyurl.com/nndsvd"""

            n_samples, n_features = X.shape

            if not sparse.issparse(X) and np.any(np.isnan(X)):
                raise ValueError("NMF initializations with NNDSVD are not available "
                                 "with missing values (np.nan).")

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

            # The leading singular triplreget is non-negative
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

   :param mat_in: Positional (required) matrix to be TF-IDF transformed. Should be cell x feature.
   :type mat_in: `csr matrix type <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_, or other sparse/dense matrices which can be coerced to csr format.
   :raise nmf_models.InvalidMatType: If the mat type is invalid
   :param scale: Optional number to scale values by.
   :type scale: int
   :return: TF-IDF transformed matrix
   :rtype: csr matrix
    """
    mat = sparse.csr_matrix(mat_in)

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
