import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, composite
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform

from gph import ripser_parallel as ripser


def make_dm_symmetric(dm):
    # Extract strict upper diagonal and make symmetric
    dm = np.triu(dm.astype(np.float32), k=1)
    return dm + dm.T


@composite
def get_dense_distance_matrices(draw):
    """Generate 2d dense square arrays of floats, with zero along the
    diagonal."""
    shapes = draw(integers(min_value=2, max_value=30))
    dm = draw(arrays(dtype=float,
                     elements=floats(allow_nan=False,
                                     allow_infinity=True,
                                     min_value=0,
                                     exclude_min=True,
                                     width=32),
                     shape=(shapes, shapes), unique=False))
    return make_dm_symmetric(dm)


@composite
def get_sparse_distance_matrices(draw):
    """Generate 2d upper triangular sparse matrices of floats, with zero along
    the diagonal."""
    shapes = draw(integers(min_value=2, max_value=40))
    dm = draw(arrays(dtype=float,
                     elements=floats(allow_nan=False,
                                     allow_infinity=True,
                                     min_value=0,
                                     exclude_min=True,
                                     width=32),
                     shape=(shapes, shapes), unique=False))
    dm = np.triu(dm.astype(np.float32), k=1)
    dm = coo_matrix(dm)
    row, col, data = dm.row, dm.col, dm.data
    not_inf_idx = data != np.inf
    row = row[not_inf_idx]
    col = col[not_inf_idx]
    data = data[not_inf_idx]
    shape_kwargs = {} if data.size else {"shape": (0, 0)}
    dm = coo_matrix((data, (row, col)), **shape_kwargs)
    return dm


@settings(deadline=500)
@given(dm=get_sparse_distance_matrices())
def test_coo_below_diagonal_and_mixed_same_as_above(dm):
    """Test that if we feed sparse matrices representing the same undirected
    weighted graph we obtain the same results regardless of whether all entries
    are above the diagonal, all are below the diagonal, or neither.
    Furthermore, test that conflicts between stored data in the upper and lower
    triangle are resolved in favour of the upper triangle."""
    ripser_kwargs = {"maxdim": 2, "metric": "precomputed"}

    pd_above = ripser(dm, **ripser_kwargs)['dgms']

    pd_below = ripser(dm.T, **ripser_kwargs)['dgms']

    _row, _col, _data = dm.row, dm.col, dm.data
    coo_shape_kwargs = {} if _data.size else {"shape": (0, 0)}
    to_transpose_mask = np.full(len(_row), False)
    to_transpose_mask[np.random.choice(np.arange(len(_row)),
                                       size=len(_row) // 2,
                                       replace=False)] = True
    row = np.concatenate([_col[to_transpose_mask], _row[~to_transpose_mask]])
    col = np.concatenate([_row[to_transpose_mask], _col[~to_transpose_mask]])
    dm_mixed = coo_matrix((_data, (row, col)), **coo_shape_kwargs)
    pd_mixed = ripser(dm_mixed, **ripser_kwargs)['dgms']

    row = np.concatenate([row, _row[to_transpose_mask]])
    col = np.concatenate([col, _col[to_transpose_mask]])
    data = np.random.random(len(row))
    data[:len(_row)] = _data
    dm_conflicts = coo_matrix((data, (row, col)), **coo_shape_kwargs)
    pd_conflicts = ripser(dm_conflicts, **ripser_kwargs)['dgms']

    for i in range(ripser_kwargs["maxdim"] + 1):
        pd_above[i] = np.sort(pd_above[i], axis=0)
        pd_below[i] = np.sort(pd_below[i], axis=0)
        pd_mixed[i] = np.sort(pd_mixed[i], axis=0)
        pd_conflicts[i] = np.sort(pd_conflicts[i], axis=0)
        assert_almost_equal(pd_above[i], pd_below[i])


@pytest.mark.parametrize('thresh', [False, True])
@pytest.mark.parametrize('coeff', [2, 7])
@settings(deadline=500)
@given(dm=get_dense_distance_matrices())
def test_collapse_consistent_with_no_collapse_dense(thresh, coeff, dm):
    thresh = np.max(dm) / 2 if thresh else np.inf
    maxdim = 3
    pd_collapse = ripser(dm, thresh=thresh, maxdim=maxdim, coeff=coeff,
                         metric='precomputed', collapse_edges=True)['dgms']
    pd_no_collapse = ripser(dm, thresh=thresh, maxdim=maxdim, coeff=coeff,
                            metric='precomputed', collapse_edges=False)['dgms']
    for i in range(maxdim + 1):
        pd_collapse[i] = np.sort(pd_collapse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse[i], pd_no_collapse[i])


@pytest.mark.parametrize('thresh', [False, True])
@pytest.mark.parametrize('coeff', [2, 7])
@settings(deadline=500)
@given(dm=get_sparse_distance_matrices())
def test_collapse_consistent_with_no_collapse_coo(thresh, coeff, dm):
    if thresh and dm.nnz:
        thresh = np.max(dm) / 2
    else:
        thresh = np.inf
    maxdim = 3
    pd_collapse = ripser(dm, thresh=thresh, maxdim=maxdim, coeff=coeff,
                         metric='precomputed', collapse_edges=True)['dgms']
    pd_no_collapse = ripser(dm, thresh=thresh, maxdim=maxdim, coeff=coeff,
                            metric='precomputed', collapse_edges=False)['dgms']
    for i in range(maxdim + 1):
        pd_collapse[i] = np.sort(pd_collapse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse[i], pd_no_collapse[i])


def test_collapser_with_negative_weights():
    """Test that collapser works as expected when some of the vertex and edge
    weights are negative."""
    n_points = 20
    dm = make_dm_symmetric(np.random.random((n_points, n_points)))
    np.fill_diagonal(dm, -np.random.random(n_points))
    dm -= 0.2
    dm_sparse = coo_matrix(dm)

    maxdim = 2
    pd_collapse_dense = ripser(dm, metric='precomputed', maxdim=maxdim,
                               collapse_edges=True)['dgms']
    pd_collapse_sparse = ripser(dm_sparse, metric='precomputed',
                                maxdim=maxdim, collapse_edges=True)['dgms']
    pd_no_collapse = ripser(dm, metric='precomputed', maxdim=maxdim,
                            collapse_edges=False)['dgms']

    for i in range(maxdim + 1):
        pd_collapse_dense[i] = np.sort(pd_collapse_dense[i], axis=0)
        pd_collapse_sparse[i] = np.sort(pd_collapse_sparse[i], axis=0)
        pd_no_collapse[i] = np.sort(pd_no_collapse[i], axis=0)
        assert_almost_equal(pd_collapse_dense[i], pd_no_collapse[i])
        assert_almost_equal(pd_collapse_sparse[i], pd_no_collapse[i])


def test_coo_results_independent_of_order():
    """Regression test for PR #465"""
    data = np.array([6., 8., 2., 4., 5., 9., 10., 3., 1., 1.])
    row = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    col = np.array([4, 1, 3, 2, 4, 3, 2, 3, 4, 4])
    dm = coo_matrix((data, (row, col)))
    diagrams = ripser(dm, metric="precomputed")['dgms']
    diagrams_csr = ripser(dm.tocsr(), metric="precomputed")['dgms']
    expected = [np.array([[0., 1.],
                          [0., 1.],
                          [0., 2.],
                          [0., 5.],
                          [0., np.inf]]),
                np.array([], dtype=float).reshape(0, 2)]
    for i in range(2):
        assert np.array_equal(diagrams[i], expected[i])
        assert np.array_equal(diagrams_csr[i], expected[i])


@settings(deadline=500)
@given(dm=get_dense_distance_matrices())
def test_multithread_consistent(dm):
    """Test that varying the number of threads produces consistent results"""
    maxdim = 3
    nb_threads_to_test = [1, 2, 3, 4, 5]
    res = []
    for nb_threads in nb_threads_to_test:
        res.append(ripser(dm, maxdim=maxdim, metric='precomputed',
                          n_threads=nb_threads)['dgms'])
    for dim in range(maxdim + 1):
        res[0][dim] = np.sort(res[0][dim], axis=0)
        for i in range(1, len(res)):
            res[i][dim] = np.sort(res[i][dim], axis=0)
            assert_almost_equal(res[0][dim], res[i][dim])


def test_gens_edge_in_dm_and_sorted():
    """This test verifies that the representative simplicies, the index of the
    vertices matches the corresponding edge in the distance matrix and that
    the gens are aligned with the correponding barcodes"""
    X = squareform(pdist(np.random.random((100, 3))))

    ret = ripser(X, metric='precomputed', maxdim=3, thresh=np.inf,
                 collapse_edges=False, return_generators=True)

    barcodes = ret['dgms']
    gens = ret['gens']

    for dim, bar_in_dim in enumerate(barcodes):
        idx_essential = 0
        idx_finite = 0
        for barcode in bar_in_dim:
            if dim == 0 and barcode[1] != np.inf:
                # Verifies gens in dim 0, discards essential ones
                gens_of_barcode = gens[0][idx_finite]
                assert np.isclose(barcode[1],
                                  X[gens_of_barcode[1]][gens_of_barcode[2]])
                idx_finite = idx_finite + 1
            if dim > 0:
                # Verifies gens in dim > 0, expects first all non essential
                # barcodes and then the finite ones
                if barcode[1] != np.inf:
                    gens_of_barcode = gens[1][dim-1][idx_finite]
                    assert np.isclose(barcode[0],
                                      X[gens_of_barcode[0]][gens_of_barcode[1]]
                                      )
                    assert np.isclose(barcode[1],
                                      X[gens_of_barcode[2]][gens_of_barcode[3]]
                                      )
                    idx_finite = idx_finite + 1
                else:
                    gens_of_barcode = gens[3][dim-1][idx_essential]
                    assert np.isclose(barcode[0],
                                      X[gens_of_barcode[0]][gens_of_barcode[1]]
                                      )
                    idx_essential = idx_essential + 1


def test_gens_with_collapser():
    """This test ensures that you cannot use collapser and
    retrieve generators. This is a temporary behavior."""
    X = squareform(pdist(np.random.random((10, 3))))

    with pytest.raises(NotImplementedError):
        ripser(X, metric='precomputed', collapse_edges=True,
               return_generators=True)


@settings(deadline=500)
@given(dm=get_dense_distance_matrices())
@pytest.mark.parametrize('format', ['dense', 'sparse'])
def test_gens_non_0_diagonal_dim0(dm, format):
    np.fill_diagonal(dm, np.random.uniform(low=0, high=np.amin(dm),
                                           size=(dm.shape[0])))
    x_len = dm.shape[0]
    if format == 'dense':
        X = dm
    else:
        X = coo_matrix(dm).tocsr()

    ret = ripser(X, metric='precomputed', return_generators=True)

    dgms_0 = ret['dgms'][0]
    gens_fin_0 = ret['gens'][0]
    gens_ess_0 = ret['gens'][2]

    # Verifies that the number of essential and finite generators
    # combined is equal to the number of points in the point cloud
    assert (x_len - len(gens_ess_0)) == len(gens_fin_0)

    # Verifies that the birth indices for essential and finite
    # representatives are unique
    assert len(np.unique(np.sort(gens_fin_0[:, 0]))) == len(gens_fin_0[:, 0])
    assert len(np.unique(np.sort(gens_ess_0))) == len(gens_ess_0)
    # Verifies that there are no duplicates between finite and essential
    # And also the other way around
    assert len([x for x in gens_fin_0[:, 0] if x in gens_ess_0]) == 0
    assert len([x for x in gens_ess_0 if x in gens_fin_0[:, 0]]) == 0

    for barcode, rp in zip(dgms_0, gens_fin_0):
        # Verify birth of dim-0 finite representative
        # The birth is located on the diagonal
        assert np.isclose(barcode[0], X[rp[0], rp[0]])

        # Verify death of dim-0 finite representative
        assert np.isclose(barcode[1], X[rp[1], rp[2]])

    for barcode, rp in zip(dgms_0[len(dgms_0):], gens_ess_0):
        # Verify birth of dim-0 essential representative
        # The birth is located on the diagonal
        assert np.isclose(barcode[0], X[rp, rp])


def test_gens_order_vertices_higher_dimension():
    """This test verifies that function `get_youngest_edge_simplex` returns the
    correct vertices. It should return the edge with the largest diameter in
    the simplex and, if several are present with the same diameter, the oldest
    one in the reverse colexicographic order used to build the simplexwise
    refinement of the Vietoris-Rips filtration."""
    diamond = np.array(
        [[0,      1,    100,      1,      1,      1],
         [0,      0,      1,    100,      1,      1],
         [0,      0,      0,      1,      1,      1],
         [0,      0,      0,      0,      1,      1],
         [0,      0,      0,      0,      0,    100],
         [0,      0,      0,      0,      0,      0]],
        dtype=np.float64)

    diamond += diamond.T

    gens = ripser(diamond, maxdim=2, return_generators=True)['gens']
    gens_fin_dim2 = gens[1][1]

    assert len(gens_fin_dim2) == 1
    assert np.array_equal(gens_fin_dim2[0], np.array([1, 0, 5, 4]))


def test_ph_maxdim_0():
    """Regression test for issue #39, an issue was found when only computing
    up to dimension 0. The test also compares the results of dimension 0
    when maxdim=1 and maxdim=0"""
    X = np.array([[1., 2], [3, 4], [5, 0]])
    res_maxdim_0 = ripser(X, maxdim=0)['dgms'][0]
    res_maxdim_1 = ripser(X, maxdim=1)['dgms'][0]

    # Verifies that the two computations lead to the same barcodes
    assert_array_equal(res_maxdim_0, res_maxdim_1)


@settings(deadline=500)
@given(dm=get_dense_distance_matrices())
def test_equivariance(dm):
    """Test that if we shift all entries (diagonal or not) in the input matrix
    by a constant then the barcodes are also shifted by that constant. Doubles
    also as a regression test for issue #31."""
    maxdim = 1
    kwargs = {"metric": "precomputed", "maxdim": maxdim}

    # - Median may or may not make some edges zero, and will make some edges
    #   negative and all vertex weights negative
    # - Min is guaranteed to make at least one edge exactly zero, and all
    #   vertex weights negative
    # - Max is guaranteed to make all edge and vertex weights non-positive (or
    #   infinite)
    dm_flat = squareform(dm, checks=False)  # Get strict upper diagonal
    dm_finite_nonzero_flat_sorted = np.sort(
        dm_flat[np.logical_and(dm_flat > 0, dm_flat < np.inf)]
        )  # Filter out zero values and infinite values, and sort
    # This test will fail if val - offset = -offset numerically for some entry
    # val in dm, because this will make some 0-dimensional classes disappear
    # (they are born dead)
    offsets = [0, 0, 0]  # Fallback if no edge weights are finite and positive
    if len(dm_finite_nonzero_flat_sorted):
        offset_cand = [np.float32(np.median(dm_finite_nonzero_flat_sorted)),
                       np.min(dm_finite_nonzero_flat_sorted),
                       np.max(dm_finite_nonzero_flat_sorted)]
        for i, offset in enumerate(offset_cand):
            if not np.any(dm - offset == -offset):
                offsets[i] = offset_cand

    dgms_orig = ripser(dm, **kwargs)["dgms"]

    for offset in offsets:
        dgms_offset = ripser(dm - offset, **kwargs)["dgms"]
        for dim in range(maxdim + 1):
            assert_array_equal(dgms_offset[dim], dgms_orig[dim] - offset)


def test_equivariance_regression():
    """Make sure that, if `hypothesis` did not pick up a distance matrix like
    the one in issue #31, then we test it by hand anyway, to avoid regressions.
    """
    maxdim = 1
    kwargs = {"metric": "precomputed", "maxdim": maxdim}
    dm = np.array([[0, 1, 3, 5, 4],
                   [1, 0, 6, 7, 2],
                   [3, 6, 0, 8, 9],
                   [5, 7, 8, 0, 10],
                   [4, 2, 9, 10, 0]], dtype=np.float32)

    dm_flat = squareform(dm)
    offsets = [np.median(dm_flat),
               np.min(dm_flat),
               np.max(dm_flat)]

    dgms_orig = ripser(dm, **kwargs)["dgms"]

    for offset in offsets:
        dgms_offset = ripser(dm - offset, **kwargs)["dgms"]
        for dim in range(maxdim + 1):
            assert_array_equal(dgms_offset[dim], dgms_orig[dim] - offset)


def test_optimized_distance_matrix():
    """Ensure that using the optimized distance matrix computation when using
    threshold produces the same results than using one with a threshold who
    correspond to the enclosing radius.
    """
    X = np.random.default_rng(0).random((100, 3))
    maxdim = 2
    enclosing_radius = 0.8884447324918705

    dgms = ripser(X, maxdim=maxdim)["dgms"]
    dgms_thresh = ripser(X, maxdim=maxdim, thresh=enclosing_radius)["dgms"]

    for dim, dgm in enumerate(dgms):
        assert_array_equal(dgm, dgms_thresh[dim])


def test_dense_finite_thresh_zero_edges():
    """Check that if a dense distance matrix and a finite threshold are passed,
    and some edges are zero, these edges are not treated as absent. Serves as
    a regression test for issue #55."""
    dm = np.array([[0., 0.],
                   [0., 0.]])
    dgm_0 = ripser(dm)["dgms"][0]
    dgm_0_finite_thresh = ripser(dm, thresh=1.)["dgms"][0]
    dgm_0_exp = np.array([[0., np.inf]])
    assert_array_equal(dgm_0, dgm_0_exp)
    assert_array_equal(dgm_0_finite_thresh, dgm_0_exp)


def test_unsupported_coefficient():
    from gph.modules import gph_ripser

    X = squareform(pdist(np.random.random((10, 3))))

    # Verifies that an exception is thrown if the coefficient value passed
    # is not a prime number
    with pytest.raises(ValueError):
        ripser(X, metric='precomputed', coeff=4)

    # Verifies that an exception is thrown if the coefficient value passed
    # is bigger that the maximal value supported
    with pytest.raises(ValueError):
        ripser(X, metric='precomputed',
               coeff=gph_ripser.get_max_coefficient_field_supported()+1)


@settings(deadline=500)
@given(dm_dense=get_dense_distance_matrices())
def test_non_0_diagonal_internal_representation(dm_dense):
    """Checks that, when passing a full distance matrix with non-zero values in
    the diagonal, the result is the same regardless of whether the input is in
    dense or sparse format."""
    diagonal = np.random.random(dm_dense.shape[0])

    # Ensure that all entries are bigger than the diagonal
    dm_dense = dm_dense + 1
    np.fill_diagonal(dm_dense, diagonal)

    dgms1 = ripser(dm_dense, maxdim=2, metric='precomputed')['dgms']
    dgms2 = ripser(coo_matrix(dm_dense), maxdim=2,
                   metric='precomputed')['dgms']

    for bars1, bars2 in zip(dgms1, dgms2):
        assert_array_equal(bars1, bars2)


def test_infinite_deaths_always_essential():
    """Regression test for issue #37"""
    diamond_dm = np.array(
        [[0,      1,      np.inf, 1,      1,      1],
         [0,      0,      1,      np.inf, 1,      1],
         [0,      0,      0,      1,      1,      1],
         [0,      0,      0,      0,      1,      1],
         [0,      0,      0,      0,      0,      np.inf],
         [0,      0,      0,      0,      0,      0]],
        dtype=np.float64
    )
    diamond_dm += diamond_dm.T

    gens = ripser(diamond_dm, metric="precomputed", maxdim=2,
                  return_generators=True)["gens"]

    gens_fin_dim1 = gens[1][1]

    # With this example no finite generators in dimension 1 shall be found
    assert len(gens_fin_dim1) == 0
