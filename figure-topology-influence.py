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
    n_samples  = 50000
    n_epochs   = 25000
    sigma      = 0.50, 0.01
    lrate      = 0.50, 0.01
    if seed is None:
        seed = np.random.randint(0,1000)
    np.random.seed(seed)
    
    print("Building networks (might take some time)... ", end="")
    sys.stdout.flush()
    som_2 = som.SOM(n_unit, topology, 2)
    som_3 = som.SOM(n_unit, topology, 3)
    som_4 = som.SOM(n_unit, topology, 4)
    
    print("done!")
    print("Random seed: {0}".format(seed))
    print("Number of units: {0}, {1}, {2}".format(som_2.size, som_3.size, som_4.size))

    np.random.seed(seed)
    X,Y = np.random.uniform(0, 1, (n_samples,3)), None
    som_2.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)
   
    np.random.seed(seed)
    X,Y = np.random.uniform(0, 1, (n_samples,3)), None
    som_3.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    np.random.seed(seed)
    X,Y = np.random.uniform(0, 1, (n_samples,3)), None
    som_4.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    
    figsize = 2.5*np.array([9,6])
    fig = plt.figure(figsize=figsize, dpi=50)

    for i,som in enumerate([som_2, som_3, som_4]):
        ax = plt.subplot(2, 3, 1+i, aspect=1)
        plot.network(ax, som)
        plot.letter(ax, chr(ord("A")+i))
    
        ax = plt.subplot(2, 3, 4+i, aspect=1)
        plot.weights_3D(ax, som)
        plot.letter(ax, chr(ord("D")+i))
    
    plt.tight_layout()
    plt.savefig("figure-topology-influence.pdf", dpi=300)
    plt.show()

