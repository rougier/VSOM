# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import tqdm
import som, plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    seed       = 3
    topology   = "regular"
    topology   = "random"
    n_unit     = 4096
    n_samples  = 50000
    n_neighbor = 2
    n_epochs   = 25000
    sigma      = 0.50, 0.01
    lrate      = 0.50, 0.01
    if seed is None:
        seed = np.random.randint(0,1000)
    np.random.seed(seed)

    X = np.random.uniform(0,1,(n_samples,3))
    Y = None

    print("Building network (might take some time)... ", end="")
    sys.stdout.flush()
    som_regular = som.SOM(n_unit, "regular", n_neighbor)
    som_random  = som.SOM(n_unit, "random", n_neighbor)
    print("done!")

    np.random.seed(seed)
    som_regular.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)
    
    np.random.seed(seed)
    som_random.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    n = n_epochs
    D_regular = np.zeros(n)
    D_random = np.zeros(n)
    
    np.random.seed(seed)
    I = np.random.randint(0, len(X), n)
    for i,x in enumerate(X[I]):

        codebook = som_regular.codebook
        winner = np.argmin(((codebook['X'] - x)**2).sum(axis=-1))
        D_regular[i] = np.sqrt(((codebook[winner]['X']-x)**2).sum(axis=-1))

        codebook = som_random.codebook
        winner = np.argmin(((codebook['X'] - x)**2).sum(axis=-1))
        D_random[i] = np.sqrt(((codebook[winner]['X']-x)**2).sum(axis=-1))

    plt.figure(figsize=(10,4))
    plt.hist(D_regular, bins=50, color="C0", alpha=.5, label="SOM")
    plt.hist(D_random, bins=50, color="C1", alpha=.5, label="RSOM")
    plt.legend()
    plt.show()
    

    

