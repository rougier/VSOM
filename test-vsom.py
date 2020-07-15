# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import plot
import numpy as np
from som import SOM
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    seed = 123
    topology = "random"
    n_unit = 1024
    n_samples = 25000
    n_neighbor = 3
    n_epochs = 25000
    sigma = 0.25, 0.01
    lrate = 0.50, 0.01
    if seed is None:
        seed = np.random.randint(0, 1000)
    np.random.seed(seed)

    print("Building network (might take some time)... ", end="")
    sys.stdout.flush()
    som = SOM(n_unit, topology, n_neighbor)
    print("done!")
    print("Random seed: {0}".format(seed))
    print("Number of units: {0}".format(som.size))
    if type == "random":
        print("Number of neighbors: {0}".format(n_neighbor))

    # X = np.random.uniform(0, 1, (50000,3))
    # samples = np.random.uniform(0, 1, (50000,3))
    # samples = mnist("training")
    T = np.random.uniform(0.0, 2.0*np.pi, n_samples)
    R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, n_samples))
    X = 0.5 + np.c_[R*np.cos(T), R*np.sin(T)]/2
    Y = None

    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    # Figure 1
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(1, 2, 1, aspect=1)
    plot.network(ax, som)
    plot.letter(ax, "A")
    ax = plt.subplot(1, 2, 2, aspect=1)
    # plot.weights_1D(ax, som, "magma")
    plot.weights_2D(ax, som, X)
    # plot.weights_3D(ax, som)
    plot.letter(ax, "B")
    plt.tight_layout()
    plt.show()

    # Figure 2
    # X = [ (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0),
    #      (1.0, 0.0, 0.0), (0.0, 1.0 ,0.0), (0.0, 0.0, 1.0) ]
    X = X[np.random.randint(0, len(X), 6)]
    fig = plt.figure(figsize=(12, 8))
    for i, x in enumerate(X):
        ax = plt.subplot(2, 3, i+1, aspect=1)
        plot.activation(ax, som, np.array(x))
        plot.letter(ax, chr(ord("C")+i))
    plt.tight_layout()
    plt.show()
