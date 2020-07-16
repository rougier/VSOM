# -----------------------------------------------------------------------------
# SOM (Self Organizing Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import tqdm
import numpy as np
import scipy.spatial
import networkx as nx
from spatial import distribution


class SOM:
    """ Self Organizing Map """

    def __init__(self, size=1024, topology="regular", neighbours=2):
        """
        Initialize SOM

        type: string
          "regular" or "random"

        size : int
          Number of neurons

        neighbours : int
          Number of neighbours for random topology
        """

        self.size = size
        self.topology = topology
        self.neighbours = neighbours

        if self.topology == "regular":
            P, V = distribution(size, "regular", jitter=0)
            D = scipy.spatial.distance.cdist(P, P, metric="cityblock")
            n = int(np.sqrt(len(P)))
            self.edges = np.zeros((n*(n-1)*2, 2), dtype=int)
            for index, (i, j) in enumerate([(i, j) for i in range(n)
                                            for j in range(n-1)]):
                self.edges[2*index] = i*n+j, i*n+j+1
                self.edges[2*index+1] = j*n+i, (j+1)*n+i
        else:
            P, V = distribution(size, "random")
            D = scipy.spatial.distance.cdist(P, P)
            sources = np.repeat(np.arange(len(P)), neighbours)
            sources = sources.reshape(len(P), neighbours)
            targets = np.argsort(D, axis=1)[:, 1:neighbours+1]
            self.edges = np.c_[sources.ravel(), targets.ravel()]
            C = np.zeros(D.shape, dtype=int)
            C[sources, targets] = 1
            G = nx.Graph(C)
            L = nx.all_pairs_shortest_path_length(G)
            for src, lengths in L:
                D[src, list(lengths.keys())] = list(lengths.values())
                # K = np.fromiter(lengths.keys(), dtype=int)
                # V = np.fromiter(lengths.values(), dtype=int)
                # D[src, K] = V
            self.graph = G

        self.size = len(P)
        self.positions = P
        self.voronoi = V
        self.distances = D / D.max()

    def fit(self, X, Y=None, epochs=10000,
            sigma=(0.50, 0.01), lrate=(0.50, 0.01)):
        """
        Compute codebook

        X : array-like (n_samples, n_features)
          Input data

        Y : array-like (n_samples, n_targets) or None
          Output data
        """

        n_samples = len(X)
        n_features = np.prod(X.shape[1:])
        n_targets = np.prod(Y.shape[1:]) if Y is not None else 0

        codebook = np.zeros(self.size, dtype=[("X", float, (n_features,)),
                                              ("Y", float, (n_targets,))])
        codebook["X"] = np.random.uniform(0, 1, codebook["X"].shape)

        t = np.linspace(0, 1, epochs)
        lrate = lrate[0]*(lrate[1]/lrate[0])**t
        sigma = sigma[0]*(sigma[1]/sigma[0])**t

        I = np.random.randint(0, len(X), epochs)
        X, Y = X[I], Y[I] if Y is not None else None
        for i in tqdm.trange(epochs):
            # Get index of nearest node (minimum distance)
            winner = np.argmin(((codebook['X'] - X[i])**2).sum(axis=-1))

            # Gaussian centered on winner
            G = np.exp(-self.distances[winner]**2/sigma[i]**2)

            # Move nodes towards sample
            codebook['X'] -= lrate[i]*G[..., np.newaxis]*(codebook['X'] - X[i])

            # Move output towards label
            if Y is not None:
                codebook['Y'] -= lrate[i] * \
                    G[..., np.newaxis]*(codebook['Y'] - Y[i])

        self.codebook = codebook

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        """

        codebook = self.codebook['X']
        C = np.zeros(len(X), dtype=int)
        for i in tqdm.trange(len(X)):
            C[i] = np.argmin(((codebook - X[i])**2).sum(axis=-1))
        return C
