from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC).

    Each point begins with one unit of flux. Flux flows along a k-nearest-
    neighbour graph toward the global centroid for T rounds. Points that end up
    sending (almost) nothing become sinks. Every other point keeps moving to the
    downhill neighbour it sent the most flux to in the last round until it lands
    on a sink, and takes that sink's cluster. Finally, clusters with fewer than
    max(2, floor(n / 20)) points are folded into the nearest larger cluster.

    Parameters
    ----------
    k : int, default=10
        Neighbours per point in the k-NN graph (capped at n - 1).
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth. None means 0.2 times the median pairwise
        distance. A value passed in must be positive.
    alpha : float, default=0.5
        Share of its stored flux that a node sends out in each round.
    T : int, default=25
        Number of transport rounds.
    epsilon : float, default=1e-4
        Sink threshold on a node's outgoing flux in the last round. A node
        that sends nothing at all is always a sink, which only makes a
        difference when epsilon <= 0.
    beta : Optional[float], default=None
        Edge reinforcement used from round 1 onward:

        * None: phi(f) = log(1 + f)
        * 0: phi(f) = 1, the original FEC
        * anything else: phi(f) = f ** beta

    Attributes set by fit
    ---------------------
    w_ij : ndarray (n, k_eff)
        Kernel weight from each point to its neighbours, nearest first.
    flux : ndarray (n,)
        Stored flux after the last round.
    sinks : bool ndarray (n,)
        Sinks found by the epsilon rule (handout Step 4). A node promoted while
        breaking a cycle in Step 5 gets its own label but is not flagged here.
    labels : int ndarray (n,)
        Cluster id per point. The ids themselves carry no meaning.
    bandwidth_ : float
        The sigma that was actually used.
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        # storing every parameter exactly as given
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for X (shape (n, m)), following handout Steps 1 to 6.

        1. Distance d(i) from each point to the centroid, the bandwidth sigma,
           tau = 0.1 * sigma, and k-NN weights w_ij = exp(-||x_i - x_j||^2 / sigma^2).
        2. One unit of flux on every node.
        3. T rounds. With w~_ij = w_ij * phi(f_j(t)) and phi = 1 at t = 0,
           node i sends delta_ij = alpha * w~_ij * f_i / (sum of w~_im over
           downhill m) to every neighbour j with d(j) <= d(i) + tau.
        4. Sinks: last-round outflow below epsilon, or exactly zero. If there
           are none, the point closest to the centroid is the sink.
        5. Each non-sink follows steepest(i) = argmax_j delta_ij(T) until it
           reaches a labelled node. A closed loop gets a new label.
        6. Clusters smaller than s_min = max(2, floor(n / 20)) join the
           nearest cluster, by centroid, that has at least s_min points.
        """
        # checking the input shape and the bandwidth argument
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        if n == 0:
            raise ValueError("FEC needs at least one point")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError("sigma must be positive")

        # ---- Step 1 ----
        # measuring how far each point sits from the global centroid
        centre = X.mean(axis=0)
        centre_dist = np.linalg.norm(X - centre, axis=1)
        gap_sq = self._squared_gaps(X)

        # picking the bandwidth (inferring it when none was given)
        if self.sigma is not None:
            bandwidth = float(self.sigma)
        else:
            upper = np.triu_indices(n, k=1)
            bandwidth = 0.2 * float(np.median(np.sqrt(gap_sq[upper]))) if n > 1 else 0.0
            if bandwidth == 0.0:
                # falling back to 1.0 when the median gap is 0 (one point or
                # mostly duplicates), so the kernel stays defined
                bandwidth = 1.0
        self.bandwidth_ = bandwidth
        tolerance = 0.1 * bandwidth

        # finding each point's k nearest neighbours; every per-edge array from
        # here on is (n, k_eff), where column r of row i means neighbour
        # neigh_idx[i, r]. The stable sort is keeping ties in index order.
        k_eff = max(0, min(int(self.k), n - 1))
        gap_sq_no_self = gap_sq.copy()
        np.fill_diagonal(gap_sq_no_self, np.inf)
        neigh_idx = np.argsort(gap_sq_no_self, axis=1, kind="stable")[:, :k_eff]
        neigh_gap_sq = np.take_along_axis(gap_sq, neigh_idx, axis=1)
        w_ij = np.exp(-neigh_gap_sq / bandwidth**2)

        # marking which edges point downhill (fixed, since d never changes)
        flows_ok = centre_dist[neigh_idx] <= centre_dist[:, None] + tolerance

        # ---- Step 2 ----
        # giving every node one unit of flux
        flux = np.ones(n)
        # starting with no transfers, so T == 0 leaves every node a sink
        sent = np.zeros((n, k_eff))

        # ---- Step 3 ----
        for t in range(int(self.T)):
            # reweighting edges by how much flux each target is holding
            boost = self._reinforce(flux, t)
            boosted_w = np.where(flows_ok, w_ij * boost[neigh_idx], 0.0)
            weight_total = boosted_w.sum(axis=1)
            # sharing alpha * f_i across downhill edges (0 when there are none)
            per_weight = np.divide(
                self.alpha * flux, weight_total, out=np.zeros(n), where=weight_total > 0
            )
            sent = boosted_w * per_weight[:, None]
            # collecting what each node receives, scatter-adding by target index
            received = np.bincount(neigh_idx.ravel(), weights=sent.ravel(), minlength=n)
            flux = flux - sent.sum(axis=1) + received

        # ---- Step 4 ----
        # reading the last round's outflow (the handout's delta(T))
        last_out = sent.sum(axis=1)
        sinks = (last_out < self.epsilon) | (last_out <= 0)
        if not sinks.any():
            # choosing the point nearest the centroid when no sink was found
            sinks[int(np.argmin(centre_dist))] = True

        # ---- Step 5 ----
        # choosing each node's next hop: the downhill neighbour it sent the
        # most to (argmax keeps the nearest neighbour on a tie)
        if k_eff > 0:
            best_col = np.argmax(np.where(flows_ok, sent, -np.inf), axis=1)
            next_hop = neigh_idx[np.arange(n), best_col]
        else:
            next_hop = np.arange(n)
        labels = self._trace_to_sinks(sinks, next_hop)

        # ---- Step 6 ----
        # absorbing clusters that are too small
        labels = self._absorb_tiny_clusters(X, labels, self._smallest_allowed(n))

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    @staticmethod
    def _squared_gaps(X: np.ndarray) -> np.ndarray:
        """Squared distance between every pair of points.

        Uses direct coordinate differences, a block of rows at a time, so
        memory stays near n * n floats. This stays exact on integer data and
        does not lose precision when points sit far from the origin.
        """
        n, m = X.shape
        out = np.empty((n, n))
        block = max(1, (1 << 22) // max(1, n * m))
        # filling the matrix one block of rows at a time
        for lo in range(0, n, block):
            diff = X[lo:lo + block, None, :] - X[None, :, :]
            out[lo:lo + block] = np.einsum("ijk,ijk->ij", diff, diff)
        return out

    def _reinforce(self, flux: np.ndarray, t: int) -> np.ndarray:
        """phi(f_j(t)) for every node j."""
        # leaving weights untouched in round 0 or when beta is 0
        if t == 0 or self.beta == 0:
            return np.ones_like(flux)
        if self.beta is None:
            return np.log1p(flux)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            boost = np.power(flux, float(self.beta))
        # closing edges whose boost blew up (negative beta on an empty node)
        return np.where(np.isfinite(boost), boost, 0.0)

    @staticmethod
    def _smallest_allowed(n: int) -> int:
        """s_min = max(2, floor(n / 20)) from handout Step 6."""
        return max(2, n // 20)

    @staticmethod
    def _trace_to_sinks(sinks: np.ndarray, next_hop: np.ndarray) -> np.ndarray:
        """Number the sinks 0, 1, ... and walk every other node to one.

        When a walk returns to a node already on it, the last node before the
        repeat gets a new label, as in handout section 5.6 where 0 -> 3 -> 0
        makes node 3 the new sink. Every node on the walk takes the label the
        walk ends on.
        """
        n = sinks.shape[0]
        cluster_of = np.full(n, -1, dtype=int)
        sink_nodes = np.flatnonzero(sinks)
        cluster_of[sink_nodes] = np.arange(sink_nodes.size)
        fresh = sink_nodes.size

        for origin in range(n):
            if cluster_of[origin] != -1:
                continue
            trail = []
            seen = set()
            here = origin
            # walking forward until hitting a labelled node or a loop
            while cluster_of[here] == -1:
                if here in seen:
                    # breaking the loop at the node that closed it
                    cluster_of[trail[-1]] = fresh
                    fresh += 1
                    here = trail[-1]
                    break
                seen.add(here)
                trail.append(here)
                here = int(next_hop[here])
            # stamping the whole trail with the label it reached
            cluster_of[trail] = cluster_of[here]
        return cluster_of

    @staticmethod
    def _absorb_tiny_clusters(X: np.ndarray, labels: np.ndarray, s_min: int) -> np.ndarray:
        """Fold every cluster under s_min points into the nearest cluster, by
        centroid, that has at least s_min points.

        Target centroids come from before any merge, so order does not matter.
        With no cluster of size s_min there is nowhere to merge, and labels
        come back unchanged. Ids are kept, so they may not be contiguous.
        """
        result = labels.copy()
        ids, sizes = np.unique(result, return_counts=True)
        large = ids[sizes >= s_min]
        tiny = ids[sizes < s_min]
        if large.size == 0 or tiny.size == 0:
            return result

        # computing the centroid of each large cluster once
        large_centres = np.array([X[result == c].mean(axis=0) for c in large])
        for c in tiny:
            # moving this small cluster into its closest large neighbour
            members = result == c
            gaps = np.linalg.norm(large_centres - X[members].mean(axis=0), axis=1)
            result[members] = large[np.argmin(gaps)]
        return result

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"
