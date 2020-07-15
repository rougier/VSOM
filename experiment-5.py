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
    n_samples = 50000
    n_epochs = 25000
    sigma = 0.25, 0.01
    lrate = 0.50, 0.01
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

    import imageio
    image = imageio.imread('mucha.png') / 255
    xshape = 8, 8
    X, Y = np.zeros((n_samples, xshape[0]*xshape[1])), None
    for i in range(len(X)):
        x = np.random.randint(0, image.shape[1]-xshape[1])
        y = np.random.randint(0, image.shape[0]-xshape[0])
        X[i] = image[y:y+xshape[0], x:x+xshape[1]].ravel()
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    figsize = 2.5*np.array([6, 7])
    fig = plt.figure(figsize=figsize, dpi=50)

    ax = plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=3, aspect=1)
    plot.network(ax, som)
    plot.letter(ax, "A")
    ax = plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=3, aspect=1)
    plot.weights_img(ax, som, xshape, inverse=True, zoom=1.5)
    plot.letter(ax, "B")

    X = X[np.random.randint(0, len(X), 6)]
    for i, x in enumerate(X):
        ax = plt.subplot2grid((7, 6), (3+2*(i//3), 2*(i % 3)),
                              colspan=2, rowspan=2, aspect=1)
        plot.activation(ax, som, np.array(x))
        plot.letter(ax, chr(ord("C")+i))
    plt.tight_layout()
    plt.savefig("experiment-5.pdf", dpi=300)
    plt.show()
