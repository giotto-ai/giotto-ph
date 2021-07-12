"""
MIT License
Copyright (c) 2018 Christopher Tralie and Nathaniel Saul
Copyright (c) 2021 Julian Burella Pérez and Umberto Lupo
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import gc
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.sparse import issparse, csr_matrix, coo_matrix
from scipy.spatial.distance import squareform
from sklearn.exceptions import EfficiencyWarning
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import column_or_1d

from ..modules import gph_ripser, gph_ripser_coeff, gph_collapser


def _compute_ph_vr_dense(DParam, maxHomDim, thresh=-1, coeff=2, n_threads=1):
    if coeff == 2:
        ret = gph_ripser.rips_dm(DParam, DParam.shape[0], coeff,
                                 maxHomDim, thresh, n_threads)
    else:
        ret = gph_ripser_coeff.rips_dm(DParam, DParam.shape[0], coeff,
                                       maxHomDim, thresh, n_threads)
    return ret


def _compute_ph_vr_sparse(I, J, V, N, maxHomDim, thresh=-1, coeff=2,
                          n_threads=1):
    if coeff == 2:
        ret = gph_ripser.rips_dm_sparse(I, J, V, I.size, N, coeff,
                                        maxHomDim, thresh, n_threads)
    else:
        ret = gph_ripser_coeff.rips_dm_sparse(I, J, V, I.size, N, coeff,
                                              maxHomDim, thresh, n_threads)
    return ret


def _resolve_symmetry_conflicts(coo):
    """Given a sparse matrix in COO format, filter out any entry at location
    (i, j) strictly below the diagonal if the entry at (j, i) is also
    stored. Return row, column and data information for an upper diagonal
    COO matrix."""
    _row, _col, _data = coo.row, coo.col, coo.data

    below_diag = _col < _row
    # Check if there is anything below the main diagonal
    if below_diag.any():
        # Initialize filtered COO data with information in the upper triangle
        in_upper_triangle = np.logical_not(below_diag)
        row = _row[in_upper_triangle]
        col = _col[in_upper_triangle]
        data = _data[in_upper_triangle]

        # Filter out entries below the diagonal for which entries at
        # transposed positions are already available
        upper_triangle_indices = set(zip(row, col))
        additions = tuple(
            zip(*((j, i, x) for (i, j, x) in zip(_row[below_diag],
                                                 _col[below_diag],
                                                 _data[below_diag])
                  if (j, i) not in upper_triangle_indices))
            )
        # Add surviving entries below the diagonal to final COO data
        if additions:
            row_add, col_add, data_add = additions
            row = np.concatenate([row, row_add])
            col = np.concatenate([col, col_add])
            data = np.concatenate([data, data_add])

        return row, col, data
    else:
        return _row, _col, _data


def _collapse_coo(row, col, data, thresh):
    """Run edge collapser on off-diagonal data and then reinsert diagonal
    data."""
    diag = row == col
    row_diag, col_diag, data_diag = row[diag], col[diag], data[diag]
    row, col, data = gph_collapser. \
        flag_complex_collapse_edges_coo(row, col, data.astype(np.float32),
                                        thresh)
    return (np.hstack([row_diag, row]),
            np.hstack([col_diag, col]),
            np.hstack([data_diag, data]))


def _compute_dtm_weights(dm, n_neighbors, weights_r):
    with catch_warnings():
        simplefilter("ignore", category=EfficiencyWarning)
        knn = kneighbors_graph(dm, n_neighbors=n_neighbors,
                               metric="precomputed", mode="distance",
                               include_self=False)

    weights = np.squeeze(np.asarray(knn.power(weights_r).sum(axis=1)))
    weights /= n_neighbors + 1
    weights **= (1 / weights_r)
    weights *= 2

    return weights


def _weight_filtration(dist, weights_x, weights_y, p):
    """Create a weighted distance matrix. For dense data, `weights_x` is a
    column vector, `weights_y` is a 1D array, `dist` is the original distance
    matrix, and the computations exploit array broadcasting. For sparse data,
    all three are 1D arrays. `p` can only be ``numpy.inf``, ``1``, or ``2``."""
    if p == np.inf:
        return np.maximum(dist, np.maximum(weights_x, weights_y))
    elif p == 1:
        return np.where(dist <= np.abs(weights_x - weights_y) / 2,
                        np.maximum(weights_x, weights_y),
                        dist + (weights_x + weights_y) / 2)
    elif p == 2:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(
                dist <= np.abs(weights_x**2 - weights_y**2)**.5 / 2,
                np.maximum(weights_x, weights_y),
                np.sqrt((dist**2 + ((weights_x + weights_y) / 2)**2) *
                        (dist**2 + ((weights_x - weights_y) / 2)**2)) / dist
                )
    else:
        raise NotImplementedError(f"Weighting not supported for p = {p}")


def _weight_filtration_sparse(row, col, data, weights, p):
    weights_x = weights[row]
    weights_y = weights[col]

    return _weight_filtration(data, weights_x, weights_y, p)


def _weight_filtration_dense(dm, weights, p):
    weights_2d = weights[:, None]

    return _weight_filtration(dm, weights_2d, weights, p)


def _check_weights(weights, n_points):
    weights = column_or_1d(weights)
    if len(weights) != n_points:
        raise ValueError(
            f"Input distance/adjacency matrix implies {n_points} "
            f"vertices but {len(weights)} weights were passed."
        )
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative."
                         "Negative weights passed.")

    return weights


def _compute_weights(dm, weights, weight_params, n_points,
                     sparse_kwargs={}):
    """TODO: Add documentation"""

    # If one sparse argument is provided, then we compute weights
    # for sparse
    is_sparse = len(sparse_kwargs) != 0

    if (is_sparse and (dm < 0).nnz) or (not is_sparse and (dm < 0).any()):
        raise ValueError("Distance matrix has negative entries. "
                         "Weighted Rips filtration unavailable.")

    if is_sparse:
        row = sparse_kwargs['row']
        col = sparse_kwargs['col']
        data = sparse_kwargs['data']

    weight_params = {} if weight_params is None else weight_params
    weights_p = weight_params.get("p", 1)

    if isinstance(weights, str) and (weights == "DTM"):
        n_neighbors = weight_params.get("n_neighbors", 3)
        weights_r = weight_params.get("r", 2)

        if is_sparse:

            # Restrict to off-diagonal entries for weights computation since
            # diagonal ones are given by `weights`. Explicitly set the diagonal
            # to 0 -- this is also important for DTM since otherwise
            # kneighbors_graph with include_self=False skips the first true
            # neighbor.
            off_diag = row != col
            row, col, data = (np.hstack([row[off_diag], np.arange(n_points)]),
                              np.hstack([col[off_diag], np.arange(n_points)]),
                              np.hstack([data[off_diag], np.zeros(n_points)]))
            # CSR matrix must be symmetric for kneighbors_graph to give
            # correct results
            dm = csr_matrix((np.hstack([data, data[:-n_points]]),
                             (np.hstack([row, col[:-n_points]]),
                              np.hstack([col, row[:-n_points]]))))
        else:
            if not np.array_equal(dm, dm.T):
                dm = np.triu(dm, k=1)
                dm += dm.T

        weights = _compute_dtm_weights(dm, n_neighbors, weights_r)
    elif isinstance(weights, str):
        raise ValueError("'{}' passed for `weights` but the "
                         "only allowed string is 'DTM'".format(weights))
    else:
        weights = _check_weights(weights, n_points)

    if is_sparse:
        data = _weight_filtration_sparse(row, col, data, weights,
                                         weights_p)
        return row, col, data
    else:
        dm = _weight_filtration_dense(dm, weights, weights_p)
        np.fill_diagonal(dm, weights)
        return dm


def _ideal_thresh(dm, thresh):
    """This fonction computes enclosing radius. Enclosing radius
    indicates that all distances above this threshold will produce
    trivial homology

    Enclosing radius is only computed if input matrix is square
    """

    # Check that matrix is square
    if dm.shape[0] != dm.shape[1]:
        return thresh

    # Compute an array containing the maximal value by column
    if dm.shape[1] != 1:
        max_by_array = np.apply_along_axis(np.amax, axis=1, arr=dm)
    else:
        dm_ = dm.T[0]
        max_by_array = []
        for i in range(dm.shape[0]):
            max_by_array.append(np.amax(dm_[i:]))

    enclosing_radius = np.amin(max_by_array)

    return np.amin([enclosing_radius, thresh])


def ripser_parallel(X, maxdim=1, thresh=np.inf, coeff=2, metric="euclidean",
                    metric_params={}, weights=None, weight_params=None,
                    collapse_edges=False, n_threads=1):
    """Compute persistence diagrams for X data array using Ripser [1]_.

    If X is a point cloud, it will be converted to a distance matrix
    using the chosen metric.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        A numpy array of either data or distance matrix. Can also be a sparse
        distance matrix of type scipy.sparse

    maxdim : int, optional, default: ``1``
        Maximum homology dimension computed. Will compute all dimensions lower
        than or equal to this value. For 1, both H_0 and H_1 will be computed.

    thresh : float, optional, default: ``numpy.inf``
        Maximum distances considered when constructing filtration. If
        ``numpy.inf``, compute the entire filtration.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field Z/pZ for p=coeff.

    metric : string or callable, optional, default: ``'euclidean'``
        The metric to use when calculating distance between instances in a
        feature array. If set to ``'precomputed'``, input data is interpreted
        as a distance matrix or of adjacency matrices of a weighted undirected
        graph. If a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        or a metric listed in
        :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including
        ``'euclidean'``, ``'manhattan'`` or ``'cosine'``. If a callable, it
        should take pairs of vectors (1D arrays) as input and, for each two
        vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    metric_params : dict, optional, default: ``{}``
        Additional parameters to be passed to the distance function.

    weights : ``"DTM"``, ndarray or None, optional, default: ``None``
        If not ``None``, the persistence of a weighted Vietoris-Rips filtration
        is computed as described in [3]_, and this parameter determines the
        vertex weights in the modified adjacency matrix. ``"DTM"`` denotes the
        empirical distance-to-measure function defined, following [3]_, by

        .. math:: w(x) = 2\\left(\\frac{1}{n+1} \\sum_{k=1}^n
           \\mathrm{dist}(x, x_k)^r \\right)^{1/r}.

        Here, :math:`\\mathrm{dist}` is the distance metric used, :math:`x_k`
        is the :math:`k`-th :math:`\\mathrm{dist}`-nearest neighbour of
        :math:`x` (:math:`x` is not considered a neighbour of itself),
        :math:`n` is the number of nearest neighbors to include, and :math:`r`
        is a parameter (see `weight_params`). If an ndarray is passed, it is
        interpreted as a user-defined list of vertex weights for the modified
        adjacency matrix. In either case, the edge weights
        :math:`\\{w_{ij}\\}_{i, j}` for the modified adjacency matrix are
        computed from the original distances and the new vertex weights
        :math:`\\{w_i\\}_i` as follows:

        .. math:: w_{ij} = \\begin{cases} \\max\\{ w_i, w_j \\}
           &\\text{if } 2\\mathrm{dist}_{ij} \\leq
           |w_i^p - w_j^p|^{\\frac{1}{p}} \\\\
           t &\\text{otherwise} \\end{cases}

        where :math:`t` is the only positive root of

        .. math:: 2 \\mathrm{dist}_{ij} = (t^p - w_i^p)^\\frac{1}{p} +
           (t^p - w_j^p)^\\frac{1}{p}

        and :math:`p` is a parameter specified in `metric_params`.

    weight_params : dict or None, optional, default: ``None``
        Parameters to be used in the case of weighted filtrations, see
        `weights`. In this case, the key ``"p"`` determines the power to be
        used in computing edge weights from vertex weights. It can be one of
        ``1``, ``2`` or ``np.inf`` and defaults to ``1``. If `weights` is
        ``"DTM"``, the additional keys ``"r"`` (default: ``2``) and
        ``"n_neighbors"`` (default: ``3``) are available (see `weights`,
        where the latter corresponds to :math:`n`).


    collapse_edges : bool, optional, default: ``False``
        Whether to use the edge collapse algorithm as described in [2]_ prior
        to calling ``ripser_parallel``.

    n_threads : int, optional, default: ``1``
        Maximum number of threads available to use during persistent
        homology computation. When passing ``-1``, it will try to use the
        maximal number of threads available on the host machine.

    Returns
    -------
    A dictionary holding all of the results of the computation
    {
        'dgms': list (size maxdim) of ndarray (n_pairs, 2)
            A list of persistence diagrams, one for each dimension less
            than maxdim. Each diagram is an ndarray of size (n_pairs, 2)
            with the first column representing the birth time and the
            second column representing the death time of each pair.
    }

    Notes
    -----
    `Ripser <https://github.com/Ripser/ripser>`_ is used as a C++ backend
    for computing Vietoris–Rips persistent homology. Python bindings were
    modified for performance from the `ripser.py
    <https://github.com/scikit-tda/ripser.py>`_ package.

    Ripser supports two memory representations, dense and sparse. The sparse
    representation is used in the following cases:
        - Input is sparse of type scipy.sparse
        - Collapser is enable
        - The user provide a threshold
    The dense representation will be used in the following cases:
        - Input is a point-cloud or a distance matrix
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for the edge collapse algorithm described in [2]_.

    References
    ----------
    .. [1] U. Bauer, "Ripser: efficient computation of Vietoris–Rips
           persistence barcodes", 2021; `arXiv:1908.02518
           <https://arxiv.org/abs/1908.02518>`_.

    .. [2] J.-D. Boissonnat and S. Pritam, "Edge Collapse and Persistence of
           Flag Complexes"; in *36th International Symposium on Computational
           Geometry (SoCG 2020)*, pp. 19:1–19:15, Schloss
           Dagstuhl-Leibniz–Zentrum für Informatik, 2020;
           `DOI: 10.4230/LIPIcs.SoCG.2020.19
           <https://doi.org/10.4230/LIPIcs.SoCG.2020.19>`_.

    .. [3] H. Anai et al, "DTM-Based Filtrations"; in *Topological Data
           Analysis* (Abel Symposia, vol 15), Springer, 2020;
           `DOI: 10.1007/978-3-030-43408-3_2
           <https://doi.org/10.1007/978-3-030-43408-3_2>`_.

    """

    if metric == 'precomputed':
        dm = X
    else:
        dm = pairwise_distances(X, metric=metric, **metric_params)

    n_points = max(dm.shape)

    use_sparse_computer = True
    if issparse(dm):
        row, col, data = _resolve_symmetry_conflicts(dm.tocoo())  # Upper diag

        if weights is not None:
            sparse_kwargs = {
                'row': row,
                'col': col,
                'data': data
            }
            row, col, data = _compute_weights(dm, weights, weight_params,
                                              n_points,
                                              sparse_kwargs=sparse_kwargs)

        if collapse_edges:
            row, col, data = _collapse_coo(row, col, data, thresh)

    else:
        if weights is not None:
            dm = _compute_weights(dm, weights, weight_params, n_points)

        compute_enclosing_radius = False
        if not (dm.diagonal() != 0).any():
            # Compute ideal threshold only when a distance matrix is passed
            # as input without specifying any threshold
            # We check if any element and if no entries is present in the
            # diagonal. This allow to have the enclosing radius before
            # calling collapser if computed
            if thresh == np.inf:
                thresh = _ideal_thresh(dm, thresh)
                compute_enclosing_radius = True

        if (dm.diagonal() != 0).any():
            # Convert to sparse format, because currently that's the only
            # one handling nonzero births
            (row, col) = np.triu_indices_from(dm)
            data = dm[(row, col)]
            if collapse_edges:
                row, col, data = _collapse_coo(row, col, data, thresh)
        elif collapse_edges:
            row, col, data = gph_collapser.\
                flag_complex_collapse_edges_dense(dm.astype(np.float32),
                                                  thresh)
        elif not compute_enclosing_radius:
            # If the user specifies a threshold, we use a sparse
            # representation like Ripser does
            dm[dm > thresh] = 0.0
            row, col, data = _resolve_symmetry_conflicts(coo_matrix(dm))
        else:
            use_sparse_computer = False

    if use_sparse_computer:
        res = _compute_ph_vr_sparse(
            np.asarray(row, dtype=np.int32, order="C"),
            np.asarray(col, dtype=np.int32, order="C"),
            np.asarray(data, dtype=np.float32, order="C"),
            n_points,
            maxdim,
            thresh,
            coeff,
            n_threads)
    else:
        # Only consider strict upper diagonal
        DParam = squareform(dm, checks=False).astype(np.float32)
        # Run garbage collector to free up memory taken by `dm`
        del dm
        gc.collect()
        res = _compute_ph_vr_dense(DParam, maxdim, thresh, coeff, n_threads)

    # Unwrap persistence diagrams
    dgms = res.births_and_deaths_by_dim
    for dim in range(len(dgms)):
        N = int(len(dgms[dim]) / 2)
        dgms[dim] = np.reshape(np.array(dgms[dim]), [N, 2])

    ret = {"dgms": dgms}

    return ret
