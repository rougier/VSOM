# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import som
import mnist
import plot
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    seed = 1
    topology = "random"
    n_unit = 1024
    n_neighbor = 3
    n_epochs = 25000
    sigma = 0.21568, 0.0085
    lrate = 3.0234, 0.02355
    # sigma = 0.25, 0.01
    # lrate = 0.50, 0.01
    if seed is None:
        seed = np.random.randint(0, 1000)
    np.random.seed(seed)

    print("Building network (might take some time)... ", end="")
    sys.stdout.flush()
    som = som.SOM(n_unit, topology, n_neighbor)
    print("done!")
    print("Random seed: {0}".format(seed))
    print("Number of units: {0}".format(som.size))
    if type == "random":
        print("Number of neighbors: {0}".format(n_neighbor))

    X, Y = mnist.read("training")
    xshape, yshape = X.shape[1:], Y.shape[1:]
    X, Y = X.reshape(len(X), -1), Y.reshape(len(Y), -1)
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    figsize = 2.5*np.array([6, 7])
    fig = plt.figure(figsize=figsize, dpi=50)

    ax = plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=3, aspect=1)
    plot.network(ax, som)
    plot.letter(ax, "A")
    ax = plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=3, aspect=1)
    plot.weights_img(ax, som, xshape)
    plot.letter(ax, "B")

    X = X[np.random.randint(0, len(X), 6)]
    for i, x in enumerate(X):
        ax = plt.subplot2grid((7, 6), (3+2*(i//3), 2*(i % 3)),
                              colspan=2, rowspan=2, aspect=1)
        plot.activation(ax, som, np.array(x))
        plot.letter(ax, chr(ord("C")+i))
    plt.tight_layout()
    np.save("./data/experiment-4-random", som.codebook['X'])
    # plt.savefig("experiment-4.pdf", dpi=300)
    plt.show()
