import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix

from giotto_ph.python import ripser


def replace_infinity_values(subdiagram, infinity_values):
    np.nan_to_num(subdiagram, posinf=infinity_values, copy=False)
    return subdiagram[subdiagram[:, 0] < subdiagram[:, 1]]


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
              [0., 0.62186205]]),
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
@pytest.mark.parametrize("thresh", [np.inf, 0.80131882])
def test_wrp_same_as_vrp_when_zero_weights(X, metric, weight_params,
                                           collapse_edges, thresh):
    def weights(x): return np.zeros(x.shape[0])
    weights = weights(X)
    X_wrp = ripser(X, weights=weights,
                   weight_params=weight_params,
                   metric=metric,
                   collapse_edges=collapse_edges,
                   thresh=thresh)['dgms']

    # Replace infinity barcodes values with threshold value
    X_wrp = [replace_infinity_values(diagram, thresh) for diagram in X_wrp]
    # Drop the last barcode output from ripser which is always a [birth, inf)
    X_wrp[0] = X_wrp[0][0:-1]

    for i in range(2):
        assert_almost_equal(X_wrp[i], X_vrp_exp[i])


X_dtm_exp = {1: [np.array([[0.95338798, 1.474913],
                           [1.23621261, 1.51234496],
                           [1.21673107, 1.68583047],
                           [1.30722439, 1.73876917]]),
                 np.array([[]])],
             2: [np.array([[0.95338798, 1.08187652],
                           [1.23621261, 1.2369417],
                           [1.21673107, 1.26971364],
                           [1.30722439, 1.33688354]]),
                 np.array([[]])],
             np.inf: [np.array([[]]),
                      np.array([[]])]}


@pytest.mark.parametrize("X, metric", [(X_pc, "euclidean"),
                                       (X_pc_sparse, "euclidean"),
                                       (X_dist, "precomputed"),
                                       (X_dist_sparse, "precomputed")])
@pytest.mark.parametrize("weight_params", [{"p": 1}, {"p": 2}, {"p": np.inf}])
@pytest.mark.parametrize("collapse_edges", [True, False])
def test_dtm(X, metric, weight_params, collapse_edges):
    X_dtm = ripser(X, weights="DTM", weight_params=weight_params,
                   metric=metric, collapse_edges=collapse_edges)['dgms']

    # Replace infinity barcodes values with threshold value
    X_dtm = [replace_infinity_values(diagram, np.inf) for diagram in X_dtm]
    # Drop the last barcode output from ripser which is always a [birth, inf)
    X_dtm[0] = X_dtm[0][0:-1]

    for i in range(2):
        # FIXME
        # This check is added because two empty arrays with Numpy can have
        # differents shapes ... really weird
        if X_dtm[i].size > 0 or \
                X_dtm_exp[weight_params["p"]][i].size > 0:
            assert_almost_equal(X_dtm[i], X_dtm_exp[weight_params["p"]][i])
