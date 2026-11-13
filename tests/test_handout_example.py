"""Checks against the worked example in section 5 of the handout, plus edge cases."""
import numpy as np
import pytest
from fec import FluxEquilibriumClustering

X_EX = np.array([
    [2.67, 2.95], [1.29, 4.31], [1.57, 2.33], [3.40, 2.00],
    [-3.85, -3.73], [-2.56, -1.17], [-2.97, -3.89], [-2.57, -3.48],
])


def same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return all(
        len(set(b[a == x])) == 1 for x in np.unique(a)
    ) and len(np.unique(a)) == len(np.unique(b))


@pytest.fixture
def example():
    fec = FluxEquilibriumClustering(k=2, T=5, alpha=0.7, epsilon=1e-5, beta=0)
    fec.fit(X_EX)
    return fec


def test_example_weights(example):
    expected = np.array([
        [0.4472, 0.4090], [0.1219, 0.1062], [0.4090, 0.1439], [0.4472, 0.1439],
        [0.6386, 0.3853], [0.0502, 0.0144], [0.8320, 0.6386], [0.8320, 0.3853],
    ])
    assert example.w_ij.shape == (8, 2)
    assert np.allclose(example.w_ij, expected, atol=1e-3)
    assert np.isclose(example.bandwidth_**2, 1.7836, atol=1e-3)


def test_example_first_two_iterations():
    f1 = [1.2036, 0.3000, 1.8308, 0.6656, 0.3000, 1.0000, 0.7366, 1.9634]
    f2 = [0.8258, 0.0900, 2.4446, 0.6397, 0.0900, 1.0000, 0.3519, 2.5581]
    for T, expected in ((1, f1), (2, f2)):
        fec = FluxEquilibriumClustering(k=2, T=T, alpha=0.7, epsilon=1e-5, beta=0)
        fec.fit(X_EX)
        assert np.allclose(fec.flux, expected, atol=1e-3)


def test_example_sinks_and_labels(example):
    assert example.sinks.shape == (8,)
    assert list(np.flatnonzero(example.sinks)) == [2, 5, 7]
    assert same_partition(example.labels, [3, 3, 3, 3, 2, 2, 2, 2])


def test_example_final_outflow():
    # Handout section 5.4: in round T = 5, nodes 1 and 4 each send 0.0057.
    # Bracketing epsilon around that value checks the outflow fit() uses.
    def sink_ids(eps):
        fec = FluxEquilibriumClustering(k=2, T=5, alpha=0.7, epsilon=eps, beta=0)
        return np.flatnonzero(fec.fit(X_EX).sinks).tolist()

    assert sink_ids(0.0056) == [2, 5, 7]
    assert sink_ids(0.0058) == [1, 2, 4, 5, 7]


def test_example_pre_merge_labels():
    steepest = np.array([3, 0, 0, 0, 6, 5, 7, 7])
    sinks = np.zeros(8, dtype=bool)
    sinks[[2, 5, 7]] = True
    labels = FluxEquilibriumClustering._trace_to_sinks(sinks, steepest)
    assert labels.tolist() == [3, 3, 0, 3, 2, 1, 2, 2]


def test_flux_is_conserved(example):
    assert np.isclose(example.flux.sum(), 8.0)


def test_single_point():
    fec = FluxEquilibriumClustering().fit(np.array([[1.0, 2.0]]))
    assert fec.labels.tolist() == [0]
    assert fec.w_ij.shape == (1, 0)


def test_k_larger_than_n():
    fec = FluxEquilibriumClustering(k=50).fit(X_EX)
    assert fec.w_ij.shape == (8, 7)
    assert fec.labels.shape == (8,)


def test_duplicate_points():
    X = np.zeros((5, 2))
    fec = FluxEquilibriumClustering(k=3).fit(X)
    assert fec.labels.shape == (5,)
    assert np.all(np.isfinite(fec.flux))


@pytest.mark.parametrize("beta", [None, 0, 1.0, 2.0, -1.0])
def test_beta_variants_run(beta):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(4, 0.5, (20, 2)), rng.normal(-4, 0.5, (20, 2))])
    fec = FluxEquilibriumClustering(k=8, beta=beta).fit(X)
    assert np.all(np.isfinite(fec.flux))
    assert np.isclose(fec.flux.sum(), 40.0)
    assert fec.labels.min() >= 0


def test_cycle_reached_through_a_tail():
    # 0 -> 1 -> 2 -> 1 ; node 3 is a sink
    steepest = np.array([1, 2, 1, 3])
    sinks = np.array([False, False, False, True])
    labels = FluxEquilibriumClustering._trace_to_sinks(sinks, steepest)
    assert labels[3] == 0
    assert labels[0] == labels[1] == labels[2] == 1


def test_fallback_sink_is_closest_to_centroid():
    # A huge sigma makes every node send flux, so no sink passes the threshold
    # and the point nearest the centroid (x = 1, centroid 4/3) is used.
    X = np.array([[0.0], [1.0], [3.0]])
    fec = FluxEquilibriumClustering(k=2, sigma=100.0).fit(X)
    assert np.flatnonzero(fec.sinks).tolist() == [1]


def test_explicit_sigma():
    inferred = FluxEquilibriumClustering(k=2, T=5, alpha=0.7, beta=0).fit(X_EX)
    given = FluxEquilibriumClustering(
        k=2, T=5, alpha=0.7, beta=0, sigma=inferred.bandwidth_
    ).fit(X_EX)
    assert np.allclose(given.w_ij, inferred.w_ij)
    other = FluxEquilibriumClustering(k=2, sigma=5.0).fit(X_EX)
    assert other.bandwidth_ == 5.0
    assert not np.allclose(other.w_ij, inferred.w_ij)


@pytest.mark.parametrize("bad", [0, -1.0])
def test_non_positive_sigma_rejected(bad):
    with pytest.raises(ValueError):
        FluxEquilibriumClustering(sigma=bad).fit(X_EX)


def test_no_large_cluster_leaves_labels():
    # T = 0: nothing moves, every node is a sink, all clusters are singletons
    fec = FluxEquilibriumClustering(k=2, T=0).fit(X_EX)
    assert fec.sinks.all()
    assert len(np.unique(fec.labels)) == 8


def test_one_dimensional_input():
    # Only checks that a 1-D array is treated as n points in one dimension.
    X = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    fec = FluxEquilibriumClustering(k=2).fit(X)
    assert fec.labels.shape == (6,)
    assert fec.w_ij.shape == (6, 2)


def test_zero_epsilon_keeps_silent_nodes_as_sinks():
    fec = FluxEquilibriumClustering(k=2, T=5, alpha=0.7, epsilon=0, beta=0).fit(X_EX)
    assert fec.sinks[[2, 5, 7]].all()


def test_small_cluster_threshold_uses_floor():
    size = FluxEquilibriumClustering._smallest_allowed
    assert size(8) == 2
    assert size(50) == 2   # ceil would give 3
    assert size(59) == 2
    assert size(60) == 3


def test_fit_keeps_two_point_cluster_at_n_50():
    # End to end: with n = 50, s_min = 2, so the far 2-point group survives.
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(0, 0.3, (24, 2)),
        rng.normal([10, 0], 0.3, (24, 2)),
        [[5, 40], [5.2, 40]],
    ])
    labels = FluxEquilibriumClustering(k=5).fit(X).labels
    assert 2 in np.bincount(labels)


def test_high_dimensional_distances_match():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 40))
    sq = FluxEquilibriumClustering._squared_gaps(X)
    ref = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    assert np.allclose(sq, ref)
    assert np.all(np.diag(sq) == 0)


def test_high_dimensional_distances_far_from_origin():
    # Gram form on offset data must still match the per-coordinate sum.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 50)) + 1e6
    sq = FluxEquilibriumClustering._squared_gaps(X)
    ref = np.zeros((40, 40))
    for col in X.T:
        ref += (col[:, None] - col[None, :]) ** 2
    assert np.allclose(sq, ref, rtol=1e-9, atol=1e-9)


def test_distances_exact_on_binary_high_dimensional_data():
    # Integer data must give exact squared distances so neighbour ties stay ties.
    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, (60, 40)).astype(float)
    sq = FluxEquilibriumClustering._squared_gaps(X)
    ref = (X[:, None, :] != X[None, :, :]).sum(-1).astype(float)
    assert np.array_equal(sq, ref)


def test_distances_chunked_matches_full():
    # Large enough that the row-block loop runs several times.
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 60))
    sq = FluxEquilibriumClustering._squared_gaps(X)
    d = X[:, None, :] - X[None, :, :]
    assert np.array_equal(sq, np.einsum("ijk,ijk->ij", d, d))
