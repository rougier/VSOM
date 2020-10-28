# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import som, plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    seed       = 3
    topology   = "regular"
    n_unit     = 4096 
    n_samples  = 50000
    n_neighbor = 2
    n_epochs   = 25000
    sigma      = 0.50, 0.01
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

    X,Y = np.random.uniform(0, 1, (50000,3)), None
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)
    
    figsize = 2.5*np.array([6,7])
    fig = plt.figure(figsize=figsize, dpi=50)
    
    ax = plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=3, aspect=1)
    plot.network(ax, som)
    plot.letter(ax, "A")
    ax = plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=3, aspect=1)
    plot.weights_3D(ax, som)
    plot.letter(ax, "B")

    # Collect minimal/maximal response from the map across all stimuli
    # vmin, vmax = None, None
    # for x in X:
    #     D = -np.sqrt(((som.codebook["X"] - x.ravel())**2).sum(axis=-1))
    #     vmin = D.min() if vmin is None else min(D.min(), vmin)
    #     vmax = D.max() if vmax is None else max(D.max(), vmax)

    X = np.array([ [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0], [0.0, 1.0 ,0.0], [0.0, 0.0, 1.0] ])

    winners = []
    for i,x in enumerate(X):
        D = -np.sqrt(((som.codebook["X"] - x)**2).sum(axis=-1))
        winners.append(np.argmax(D))
    P = som.positions[winners]
    ax.scatter(P[:,0], P[:,1], color="black", zorder=100) 
    for i,p in enumerate(P):
        text = ax.text(p[0]+.01, p[1]+.01, chr(ord("C")+i), zorder=200,
                       fontsize=16, fontweight="bold", transform=ax.transAxes)
        text.set_path_effects([path_effects.Stroke(linewidth=2,
                                                   foreground='white'),
                               path_effects.Normal()])
    
    for i,x in enumerate(X):
        ax = plt.subplot2grid((7, 6), (3+2*(i//3), 2*(i%3)),
                              colspan=2, rowspan=2, aspect=1)
        plot.activation(ax, som, np.array(x))
        plot.letter(ax, chr(ord("C")+i))
    plt.tight_layout()
    # np.save("experiment-3D-uniform-"+topology, som.codebook['X'])
    plt.savefig("experiment-3D-uniform.pdf", dpi=300)
    plt.show()

