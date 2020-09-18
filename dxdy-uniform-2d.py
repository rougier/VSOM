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
    # topology   = "regular"
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

    X = np.random.uniform(0, 1, (n_samples, 2))
    # T = np.random.uniform(0.0, 2.0*np.pi, n_samples)
    # R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, n_samples))
    # X = 0.5 + np.c_[R*np.cos(T), R*np.sin(T)]/2

    Y = None
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

   
    I = np.random.randint(0, len(som.codebook), (10000,2))
    X = som.positions
    dX = np.sqrt(((X[I,0] - X[I,1])**2).sum(axis=-1))
    Y = som.codebook['X']
    dY = np.sqrt(((Y[I,0] - Y[I,1])**2).sum(axis=-1))

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1,aspect=1)
    ax.scatter(dX, dY, s=5, facecolor="black", edgecolor="None", alpha=0.25)
    plt.show()
