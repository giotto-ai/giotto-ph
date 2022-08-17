# MIT License
# Copyright (c) 2018 Christopher Tralie and Nathaniel Saul
# Copyright (c) 2021 Julian Burella Pérez and Umberto Lupo
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy.sparse import issparse, csr_matrix, triu
from scipy.spatial.distance import squareform
from sklearn.exceptions import EfficiencyWarning
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.validation import column_or_1d

from ..modules import gph_ripser, gph_ripser_coeff, gph_collapser


MAX_COEFF_SUPPORTED = gph_ripser.get_max_coefficient_field_supported()


def _compute_ph_vr_dense(DParam, diagonal, maxHomDim, thresh=-1, coeff=2,
                         n_threads=1, return_generators=False):
    if coeff == 2:
        ret = gph_ripser.rips_dm(DParam, diagonal, coeff, maxHomDim, thresh,
                                 n_threads, return_generators)
    else:
        ret = gph_ripser_coeff.rips_dm(DParam, diagonal, coeff, maxHomDim,
                                       thresh, n_threads, return_generators)
    return ret


def _compute_ph_vr_sparse(I, J, V, N, maxHomDim, thresh=-1, coeff=2,
                          n_threads=1, return_generators=False):
    if coeff == 2:
        ret = gph_ripser.rips_dm_sparse(I, J, V, I.size, N, coeff,
                                        maxHomDim, thresh, n_threads,
                                        return_generators)
    else:
        ret = gph_ripser_coeff.rips_dm_sparse(I, J, V, I.size, N, coeff,
                                              maxHomDim, thresh, n_threads,
                                              return_generators)
    return ret


def _sanitize_coo(row, col, data, only_extract_upper=False):
    """Given a sparse matrix in COO format, either return its upper triangular
    portion directly, or filter out any entry at location (i, j) strictly below
    the diagonal if the entry at (j, i) is also stored."""

    row_orig, col_orig, data_orig = row, col, data

    # Initialize filtered COO data with information in the upper triangle
    in_upper_triangle = row_orig <= col_orig
    row = row_orig[in_upper_triangle]
    col = col_orig[in_upper_triangle]
    data = data_orig[in_upper_triangle]
    if only_extract_upper:
        return row, col, data

    below_diag_idxs = np.flatnonzero(np.logical_not(in_upper_triangle))
    # Check if there is anything below the main diagonal
    if len(below_diag_idxs):
        # Only keep entries below the diagonal for which entries at transposed
        # positions are not available
        upper_triangle_row_col = set(zip(row, col))
        additions_idxs = [
            i for i in below_diag_idxs
            if (col_orig[i], row_orig[i]) not in upper_triangle_row_col
            ]
        # Add surviving entries below the diagonal to final COO data
        if len(additions_idxs):
            row = np.concatenate([row, col_orig[additions_idxs]])
            col = np.concatenate([col, row_orig[additions_idxs]])
            data = np.concatenate([data, data_orig[additions_idxs]])

    return row, col, data


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


def _collapse_dense(dm, thresh):
    """Run edge collapser on off-diagonal data and then reinsert diagonal
    data if any non-zero value is present."""

    # Use 32-bit float precision here so when diagonal is extracted,
    # it is still 32-bit in the entire function operations.
    dm = dm.astype(np.float32)

    row, col, data = \
        gph_collapser.flag_complex_collapse_edges_dense(dm, thresh)

    data_diag = dm.diagonal()
    if (data_diag != 0).any():
        indices = np.arange(data_diag.shape[0])
        row = np.hstack([indices, row])
        col = np.hstack([indices, col])
        data = np.hstack([data_diag, data])

    return row, col, data


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
    """Compute the enclosing radius of an input distance matrix.

    Under a Vietoris–Rips filtration, all homology groups are trivial above
    this value because the complex becomes a cone.

    The enclosing radius is only computed if the input matrix is square."""

    # Check that matrix is square
    if dm.shape[0] != dm.shape[1]:
        return thresh

    # Compute enclosing radius
    enclosing_radius = np.min(np.max(dm, axis=1))

    return min([enclosing_radius, thresh])


def _pc_to_sparse_dm_with_threshold(X, thresh, nearest_neighbors_params,
                                    metric, metric_params, n_threads):
    """Compute a sparse matrix of pairwise distances between points in a point
    cloud, removing all distances larger than a threshold.

    Return the output as an upper triangular sparse matrix in COO format."""

    neigh = NearestNeighbors(radius=thresh,
                             metric=metric,
                             metric_params=metric_params,
                             n_jobs=n_threads,
                             **nearest_neighbors_params).fit(X)
    # Upper triangular COO output
    dm = triu(neigh.radius_neighbors_graph(mode="distance"),
              format="coo")

    return dm


def _is_prime_and_larger_than_2(x, N):
    """Test whether 2 < x <= N is prime. Returns False when x is 2."""
    if not x % 2 or x > N:
        return False

    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-
    # primes-below-n-in-python/3035188#3035188
    sieve = [True] * (x + 1)
    for i in range(3, int(math.sqrt(x)) + 1, 2):
        if sieve[i]:
            sieve[i * i::2 * i] = \
                [False] * ((x - i * i) // (2 * i) + 1)

    return sieve[x]


def ripser_parallel(X, maxdim=1, thresh=np.inf, coeff=2, metric="euclidean",
                    metric_params={}, nearest_neighbors_params={},
                    weights=None, weight_params=None, collapse_edges=False,
                    n_threads=1, return_generators=False):
    """Compute persistence diagrams from an input dense array or sparse matrix.

    If `X` represents a point cloud, a distance matrix will be internally
    created using the chosen metric and its Vietoris–Rips persistent homology
    will be computed. Computations in homology dimensions 1 and above can be
    parallelized, see `n_threads`.

    Parameters
    ----------
    X : ndarray or sparse matrix
        If `metric` is not set to ``"precomputed"``, input data of shape
        ``(n_samples, n_features)`` representing a point cloud. Otherwise,
        dense or sparse input data of shape ``(n_samples, n_samples)``
        representing a distance matrix or adjacency matrix of a weighted
        undirected graph, with the following conventions:
            - Diagonal entries indicate vertex weights, i.e. the filtration
              parameters at which vertices appear.
            - If `X` is dense, only its upper diagonal portion (including the
              diagonal) is considered.
            - If `X` is sparse, it does not need to be upper diagonal or
              symmetric. If only one of entry (i, j) and (j, i) is stored, its
              value is taken as the weight of the undirected edge {i, j}. If
              both are stored, the value in the upper diagonal is taken.
              Off-diagonal entries which are not explicitly stored are treated
              as infinite, indicating absent edges.
            - Entries of `X` should be compatible with a filtration, i.e. the
              value at index (i, j) should be no smaller than the values at
              diagonal indices (i, i) and (j, j).

    maxdim : int, optional, default: ``1``
        Maximum homology dimension computed. Will compute all dimensions lower
        than or equal to this value.

    thresh : float, optional, default: ``numpy.inf``
        Maximum value of the Vietoris–Rips filtration parameter. Points whose
        distance is greater than this value will never be connected by an edge.
        If ``numpy.inf``, compute the entire filtration. Otherwise, and if
        `metric` is not ``"precomputed"``, see `nearest_neighbors_params`.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field Z/pZ for p=coeff.

    metric : string or callable, optional, default: ``'euclidean'``
        The metric to use when calculating distance between instances in a
        feature array. If set to ``'precomputed'``, input data is interpreted
        as a distance matrix or of adjacency matrices of a weighted undirected
        graph. If a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including ``'euclidean'``, ``'manhattan'`` or ``'cosine'``. If a
        callable, it should take pairs of vectors (1D arrays) as input and, for
        each two vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    metric_params : dict, optional, default: ``{}``
        Additional parameters to be passed to the distance function.

    nearest_neighbors_params : dict, optional, default: ``{}``
        Additional parameters that can be passed when `thresh` is finite and
        `metric` is not ``"precomputed"``. Allowed keys and values are as
        follows:

            - ``"algorithm"``: ``"auto"`` | ``"ball_tree"`` | ``"kd_tree"`` |
              ``"brute"`` (default when not passed: ``"auto"``)
            - ``"leaf_size"``: int (default when not passed: ``30``)

        These are passed as keyword arguments to an instance of
        :class:`sklearn.neighbors.NearestNeighbors` to compute the thresholded
        distance matrix in a sparse format. See the relevant
        `scikit-learn User Guide
        <https://scikit-learn.org/stable/modules/neighbors.html>`_.

    weights : ``"DTM"``, ndarray or None, optional, default: ``None``
        If not ``None``, the persistence of a weighted Vietoris-Rips filtration
        is computed as described in [6]_, and this parameter determines the
        vertex weights in the modified adjacency matrix. ``"DTM"`` denotes the
        empirical distance-to-measure function defined, following [6]_, by

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
        Whether to use the edge collapse algorithm as described in [5]_ prior
        to computing persistence. Cannot be ``True`` if `return_generators` is
        also ``True``.

    n_threads : int, optional, default: ``1``
        Maximum number of threads to be used during the computation in homology
        dimensions 1 and above. ``-1`` means that the maximum number of threads
        available on the host machine will be used if possible.

    return_generators : bool, optional, default: ``False``
        Whether to return information on the simplex pairs and essential
        simplices corresponding to the finite and infinite bars (respectively)
        in the persistence barcode. If ``True``, this information is stored in
        the return dictionary under the key `gens`. Cannot be ``True`` if
        `collapse_edges` is also ``True``.

    Returns
    -------
    A dictionary holding the results of the computation. Keys and values are as
    follows:

        'dgms': list (length maxdim + 1) of ndarray (n_pairs, 2)
            A list of persistence diagrams, one for each dimension less than or
            equal to maxdim. Each diagram is an ndarray of size (n_pairs, 2)
            with the first column representing the birth time and the second
            column representing the death time of each pair.

        'gens': tuple (length 4) of ndarray or list of ndarray
            Information on the simplex pairs and essential simplices generating
            the points in 'dgms'. Each simplex of dimension 1 or above is
            replaced with the vertices of the edges that gave it its filtration
            value. The 4 entries of this tuple are as follows:

            index 0: int ndarray with 3 columns
                Simplex pairs corresponding to finite bars in dimension 0, with
                one vertex for birth and two vertices for death.
            index 1: list (length maxdim) of int ndarray with 4 columns
                Simplex pairs corresponding to finite bars in dimensions 1 to
                maxdim, with two vertices (one edge) for birth and two for
                death.
            index 2: 1D int ndarray
                Essential simplices corresponding to infinite bars in dimension
                0, with one vertex for each birth.
            index 3: list (length maxdim) of int ndarray with 2 columns
                Essential simplices corresponding to infinite bars in
                dimensions 1 to maxdim, with 2 vertices (edge) for each birth.

    Notes
    -----
    The C++ backend and Python API for the computation of Vietoris–Rips
    persistent homology are developments of the ones in the
    `ripser.py <https://github.com/scikit-tda/ripser.py>`_ project [1]_, with
    added optimizations from `Ripser <https://github.com/Ripser/ripser>`_ [2]_,
    lock-free reduction from [3]_, and additional performance improvements. See
    [4]_ for details.

    Ripser supports two memory representations, dense and sparse. The sparse
    representation is used in the following cases:
        - input is sparse of type scipy.sparse;
        - collapser is enabled;
        - a threshold is provided.
    The dense representation will be used in the following cases:
        - input is a point cloud or a distance matrix.

    The implementation of the edge collapse algorithm [5]_ is a modification of
    `GUDHI's <https://github.com/GUDHI/gudhi-devel>`_ C++ implementation.

    References
    ----------
    .. [1] C. Tralie et al, "Ripser.py: A Lean Persistent Homology Library for
           Python", *Journal of Open Source Software*, **3**(29), 2021;
           `DOI: 10.21105/joss.00925
           <https://doi.org/10.21105/joss.00925>`_.
   
    .. [2] U. Bauer, "Ripser: efficient computation of Vietoris–Rips
           persistence barcodes", *J Appl. and Comput. Topology*, **5**, pp.
           391–423, 2021; `DOI: 10.1007/s41468-021-00071-5
           <https://doi.org/10.1007/s41468-021-00071-5>`_.

    .. [3] D. Morozov and A. Nigmetov, "Towards Lockfree Persistent Homology";
           in *SPAA '20: Proceedings of the 32nd ACM Symposium on Parallelism
           in Algorithms and Architectures*, pp. 555–557, 2020;
           `DOI: 10.1145/3350755.3400244
           <https://doi.org/10.1145/3350755.3400244>`_.

    .. [4] J. Burella Pérez et al, "giotto-ph: A Python Library for
           High-Performance Computation of Persistent Homology of Vietoris–Rips
           Filtrations", 2021; `arXiv:2107.05412
           <https://arxiv.org/abs/2107.05412>`_.

    .. [5] J.-D. Boissonnat and S. Pritam, "Edge Collapse and Persistence of
           Flag Complexes"; in *36th International Symposium on Computational
           Geometry (SoCG 2020)*, pp. 19:1–19:15, Schloss
           Dagstuhl-Leibniz–Zentrum für Informatik, 2020;
           `DOI: 10.4230/LIPIcs.SoCG.2020.19
           <https://doi.org/10.4230/LIPIcs.SoCG.2020.19>`_.

    .. [6] H. Anai et al, "DTM-Based Filtrations"; in *Topological Data
           Analysis* (Abel Symposia, vol 15), Springer, 2020;
           `DOI: 10.1007/978-3-030-43408-3_2
           <https://doi.org/10.1007/978-3-030-43408-3_2>`_.

    """

    if collapse_edges and return_generators:
        raise NotImplementedError(
            "`collapse_edges` and `return_generators`cannot both be True."
        )

    if coeff != 2 and \
            not _is_prime_and_larger_than_2(coeff, MAX_COEFF_SUPPORTED):
        raise ValueError("coeff value not supported, coeff value must be prime"
                         " and lower than {}".format(MAX_COEFF_SUPPORTED))

    use_sparse_computer = True
    is_dm_sparse_and_upper_triangular = False
    if metric == 'precomputed':
        dm = X
    elif thresh != np.inf:
        dm = _pc_to_sparse_dm_with_threshold(
            X, thresh, nearest_neighbors_params, metric, metric_params,
            n_threads
            )
        is_dm_sparse_and_upper_triangular = True
    else:
        dm = pairwise_distances(X, metric=metric, **metric_params)

    n_points = max(dm.shape)

    if issparse(dm):
        coo = dm.tocoo()
        row, col, data = coo.row, coo.col, coo.data
        if not is_dm_sparse_and_upper_triangular:
            row, col, data = _sanitize_coo(row, col, data)

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

        apply_user_threshold = thresh != np.inf
        if not apply_user_threshold:
            # Compute ideal threshold only when a distance matrix is passed
            # as input without specifying any threshold
            # We check if any element and if no entries is present in the
            # diagonal. This allows to have the enclosing radius before
            # calling collapser if computed
            thresh = _ideal_thresh(dm, thresh)

        if collapse_edges:
            row, col, data = _collapse_dense(dm, thresh)

        elif apply_user_threshold:
            # If the user specifies a threshold, we use a sparse representation
            # like Ripser does
            row, col = np.nonzero(dm <= thresh)
            data = dm[(row, col)]
            row, col, data = _sanitize_coo(row, col, data,
                                           only_extract_upper=True)
        else:
            use_sparse_computer = False

    if use_sparse_computer:
        res = _compute_ph_vr_sparse(
            np.asarray(row, dtype=np.int64, order="C"),
            np.asarray(col, dtype=np.int64, order="C"),
            np.asarray(data, dtype=np.float32, order="C"),
            n_points,
            maxdim,
            thresh,
            coeff,
            n_threads,
            return_generators
            )
    else:
        # Only consider upper diagonal
        diagonal = np.diagonal(dm).astype(np.float32)
        DParam = squareform(dm, checks=False).astype(np.float32)
        res = _compute_ph_vr_dense(DParam, diagonal, maxdim, thresh,
                                   coeff, n_threads, return_generators)

    # Unwrap persistence diagrams
    # Barcodes must match the inner type of C++ core filtration value.
    # We call a method from the bindings that returns the barcodes as
    # numpy arrays with np.float32 type
    dgms = res.births_and_deaths_by_dim()
    for dim in range(len(dgms)):
        N = int(len(dgms[dim]) / 2)
        dgms[dim] = np.reshape(np.array(dgms[dim]), [N, 2])

    ret = {"dgms": dgms}

    if return_generators:
        finite_0 = np.array(res.flag_persistence_generators_by_dim.finite_0,
                            dtype=np.int64).reshape(-1, 3)
        finite_higher = [
            np.array(x, dtype=np.int64).reshape(-1, 4)
            for x in res.flag_persistence_generators_by_dim.finite_higher
            ]
        essential_0 = \
            np.array(res.flag_persistence_generators_by_dim.essential_0,
                     dtype=np.int64)
        essential_higher = [
            np.array(x, dtype=np.int64).reshape(-1, 2)
            for x in res.flag_persistence_generators_by_dim.essential_higher
            ]
        ret['gens'] = (finite_0, finite_higher, essential_0, essential_higher)

    return ret
