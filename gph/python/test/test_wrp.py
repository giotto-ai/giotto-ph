import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix

from gph import ripser_parallel as ripser


X_pc = np.array([[2., 2.47942554],
                 [2.47942554, 2.84147098],
                 [2.98935825, 2.79848711],
                 [2.79848711, 2.41211849],
                 [2.41211849, 1.92484888]])

X_dist = squareform(pdist(X_pc))

X_pc_sparse = csr_matrix(X_pc)
X_dist_sparse = csr_matrix(X_dist)

X_dist_disconnected = np.array([[0, np.inf], [np.inf, 0]])

X_vrp_exp = [
    np.array([[0., 0.43094373],
              [0., 0.5117411],
              [0., 0.60077095],
              [0., 0.62186205],
              [0., np.inf]]),
    np.array([[0.69093919, 0.80131882]])
    ]


def test_wrp_notimplemented_string_weights():
    with pytest.raises(ValueError, match="'foo' passed for `weights` but the "
                                         "only allowed string is 'DTM'"):
        ripser(X_pc, weights="foo")


def test_wrp_notimplemented_p():
    with pytest.raises(NotImplementedError):
        ripser(X_pc, weights="DTM", weight_params={"p": 1.2})


@pytest.mark.parametrize("X, metric", [(X_pc, "euclidean"),
                                       (X_pc_sparse, "euclidean"),
                                       (X_dist, "precomputed"),
                                       (X_dist_sparse, "precomputed")])
@pytest.mark.parametrize("weight_params", [{"p": 1}, {"p": 2}, {"p": np.inf}])
@pytest.mark.parametrize("collapse_edges", [True, False])
@pytest.mark.parametrize("thresh", [np.inf, 0.81])
def test_wrp_same_as_vrp_when_zero_weights(X, metric, weight_params,
                                           collapse_edges, thresh):
    def weights(x): return np.zeros(x.shape[0])
    weights = weights(X)
    X_wrp = ripser(X, weights=weights,
                   weight_params=weight_params,
                   metric=metric,
                   collapse_edges=collapse_edges,
                   thresh=thresh)['dgms']

    for i in range(2):
        assert_almost_equal(X_wrp[i], X_vrp_exp[i])


X_dtm_exp = {1: [np.array([[0.95338798, 1.474913],
                           [1.23621261, 1.51234496],
                           [1.21673107, 1.68583047],
                           [1.30722439, 1.73876917],
                           [0.92658985, np.inf]]),
                 np.empty((0, 2))],
             2: [np.array([[0.95338798, 1.08187652],
                           [1.23621261, 1.2369417],
                           [1.21673107, 1.26971364],
                           [1.30722439, 1.33688354],
                           [0.92658985, np.inf]]),
                 np.empty((0, 2))],
             np.inf: [np.array([[0.9265898, np.inf]]),
                      np.empty((0, 2))]}


@pytest.mark.parametrize("X, metric", [(X_pc, "euclidean"),
                                       (X_pc_sparse, "euclidean"),
                                       (X_dist, "precomputed"),
                                       (X_dist_sparse, "precomputed")])
@pytest.mark.parametrize("weight_params", [{"p": 1}, {"p": 2}, {"p": np.inf}])
@pytest.mark.parametrize("collapse_edges", [True, False])
def test_dtm(X, metric, weight_params, collapse_edges):
    X_dtm = ripser(X, weights="DTM", weight_params=weight_params,
                   metric=metric, collapse_edges=collapse_edges)['dgms']

    for i in range(2):
        assert_almost_equal(X_dtm[i], X_dtm_exp[weight_params["p"]][i])
