# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import som, plot
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    seed       = 1
    topology   = "random"
    n_unit     = 1024
    n_samples  = 25000
    n_neighbor = 3
    n_epochs   = 25000
    sigma      = 0.25, 0.01
    lrate      = 0.50, 0.01
    if seed is None:
        seed = np.random.randint(0,1000)
    np.random.seed(seed)
    
    print("Building network (might take some time)... ", end="")
    sys.stdout.flush()
    som = som.SOM(n_unit, topology, n_neighbor)
    print("done!")
    print("Random seed: {0}".format(seed))
    print("Number of units: {0}".format(som.size))
    if type == "random":
        print("Number of neighbors: {0}".format(n_neighbor))

    X,Y = np.random.uniform(0, 1, (50000,1)), None
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    # Collect minimal/maximal response from the map across all stimuli
    # vmin, vmax = None, None
    # for x in X:
    #     D = -np.sqrt(((som.codebook["X"] - x.ravel())**2).sum(axis=-1))
    #     vmin = D.min() if vmin is None else min(D.min(), vmin)
    #     vmax = D.max() if vmax is None else max(D.max(), vmax)

    
    # Figure 1
    figsize = 2.5*np.array([6,7])
    fig = plt.figure(figsize=figsize, dpi=50)
    
    ax = plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=3, aspect=1)
    plot.network(ax, som)
    plot.letter(ax, "A")
    ax = plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=3, aspect=1)
    plot.weights_1D(ax, som, "gray")
    plot.letter(ax, "B")
    
    # Figure 2
    X = np.linspace(0,1,6).reshape(6,1)
    for i,x in enumerate(X):
        ax = plt.subplot2grid((7, 6), (3+2*(i//3), 2*(i%3)),
                              colspan=2, rowspan=2, aspect=1)
        plot.activation(ax, som, np.array(x))
        plot.letter(ax, chr(ord("C")+i))
    plt.tight_layout()
    plt.savefig("experiment-1D-uniform.pdf")
    plt.show()

