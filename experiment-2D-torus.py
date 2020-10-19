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

    T = np.random.uniform(0.0, 2.0*np.pi, n_samples)
    R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, n_samples))
    X = 0.5 + np.c_[R*np.cos(T), R*np.sin(T)]/2
    Y = None
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    
    figsize = 2.5*np.array([6,7])
    fig = plt.figure(figsize=figsize, dpi=50)
    
    ax = plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=3, aspect=1)
    plot.network(ax, som)
    plot.letter(ax, "A")
    ax = plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=3, aspect=1)
    plot.weights_2D(ax, som, X)
    plot.letter(ax, "B")

    
    X = X[np.random.randint(0,len(X),6)]
    X[4] = 0.5,0.5
  
    ax.scatter(X[:,0], X[:,1], color="black", zorder=100) 
    for i,x in enumerate(X):
        text = ax.text(x[0]+.01, x[1]+.01, chr(ord("C")+i), zorder=200,
                       fontsize=24, fontweight="bold", transform=ax.transAxes)
        text.set_path_effects([path_effects.Stroke(linewidth=2,
                                                   foreground='white'),
                               path_effects.Normal()])

    
    for i,x in enumerate(X):
        ax = plt.subplot2grid((7, 6), (3+2*(i//3), 2*(i%3)),
                              colspan=2, rowspan=2, aspect=1)
        plot.activation(ax, som, np.array(x))
        plot.letter(ax, chr(ord("C")+i))
    plt.tight_layout()
    plt.savefig("experiment-2D-torus.pdf")
    plt.show()

