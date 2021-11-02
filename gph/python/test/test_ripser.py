import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, composite
from numpy.testing import assert_almost_equal
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform

from gph import ripser_parallel as ripser


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
    # mirror along the diagonal the values of the
    # distance matrix
    dm = np.triu(dm.astype(np.float32), k=0)
    dm = dm + dm.T
    np.fill_diagonal(dm, 0)
    return dm


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
    dm = np.random.random((n_points, n_points))
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
    retrieve representative simplicies. This is a temporary behavior."""
    X = squareform(pdist(np.random.random((100, 3))))

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
        [[0,      1,      100, 1,      1,      1],
         [0,      0,      1,      100, 1,      1],
         [0,      0,      0,      1,      1,      1],
         [0,      0,      0,      0,      1,      1],
         [0,      0,      0,      0,      0,      100],
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

    # Verifies that the number of barcodes is the same
    assert res_maxdim_0.shape[0] == res_maxdim_1.shape[0]

    # Verifies if both computation have the same barcodes
    for barcode in res_maxdim_0:
        assert any(np.equal(barcode, res_maxdim_1).all(1))
