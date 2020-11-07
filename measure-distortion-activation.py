# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import tqdm
import som, mnist, plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    seed       = 1
    n_unit     = 1024
    n_samples  = 50000
    n_epochs   = 50000
    sigma      = 0.50, 0.01
    lrate      = 0.50, 0.01
    if seed is None:
        seed = np.random.randint(0,1000)
    np.random.seed(seed)

    # X = np.random.uniform(0,1,(n_samples, 2))
    # Y = None
    # dataset = "2D uniform dataset"
    # filename = "experiment-2D-uniform-activation-distortion.pdf"
    # n_neighbor = 3
    
    # X = np.random.uniform(0,1,n_samples)
    # Y = np.random.uniform(0,1,n_samples)
    # holes = 64
    # dataset = "2D uniform dataset with holes"
    # filename = "experiment-2D-holes-activation-distortion.pdf"
    # n_neighbor = 2
    # for i in range(holes):
    #     x,y = np.random.uniform(0.1,0.9, 2)
    #     r = 0.1 * np.random.uniform(0,1)
    #     I =  ((X-x)**2 + (Y-y)**2) > r*r
    #     X, Y = X[I], Y[I]
    # X = np.c_[X, Y]
    # Y = None

    # X = np.random.uniform(0,1,(n_samples, 3))
    # Y = None
    # dataset = "3D uniform dataset"
    # filename = "experiment-3D-uniform-activation-distortion.pdf"
    # n_neighbor = 2
        
    X, Y = mnist.read("training")
    X, Y = X.reshape(len(X),-1), Y.reshape(len(Y),-1)   
    dataset = "MNIST dataset"
    filename = "experiment-MNIST-activation-distortion.pdf"
    n_neighbor = 3
    

    
    print("Building network (might take some time)... ", end="")
    sys.stdout.flush()
    som_regular = som.SOM(n_unit, "regular", n_neighbor)
    som_random  = som.SOM(n_unit, "random", n_neighbor)
    print("done!")

    
    np.random.seed(seed)
    som_regular.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)
    
    np.random.seed(seed)
    som_random.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    n = 10_000 
    A_regular = np.zeros(som_regular.size)
    A_random = np.zeros(som_random.size)
    D_regular = np.zeros(n)
    D_random = np.zeros(n)
    
    np.random.seed(seed)
    I = np.random.randint(0, len(X), n)
    for i,x in enumerate(X[I]):

        codebook = som_regular.codebook
        winner = np.argmin(((codebook['X'] - x)**2).sum(axis=-1))
        A_regular[winner] += 1
        D_regular[i] = np.sqrt(((codebook[winner]['X']-x)**2).sum(axis=-1))

        codebook = som_random.codebook
        winner = np.argmin(((codebook['X'] - x)**2).sum(axis=-1))
        A_random[winner] += 1
        D_random[i] = np.sqrt(((codebook[winner]['X']-x)**2).sum(axis=-1))


    # D_regular = np.sort(D_regular)
    # D_random = np.sort(D_random)

    xi = np.linspace(0, 1, 128)
    yi = np.linspace(0, 1, 128)
    from scipy.interpolate import griddata

    plt.figure(figsize=(10,7.5))
    gridsize = (6, 8)

    ax = plt.subplot2grid(gridsize, (0, 0), colspan=4, rowspan=4)
    ax.set_title("SOM / %s" % dataset, weight="bold")
    Z = griddata((som_regular.positions[:,0], som_regular.positions[:,1]),
                 A_regular, (xi[None,:], yi[:,None]), method='cubic')
    ax.imshow(Z, interpolation="nearest", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot2grid(gridsize, (4, 0), colspan=4, rowspan=1)
    ax.plot(np.sort(A_regular), color="black")
    ax.text(0.01, 0.95, "Ordered activation count\n(%d samples)" % n,
            ha="left", va="top", transform = ax.transAxes)

    ax = plt.subplot2grid(gridsize, (5, 0), colspan=4, rowspan=1)
    ax.hist(D_regular, bins=50, color="C0", alpha=.5)
    ax.text(0.99, 0.95, "Distortion histogram\n(%d samples)" % n,
            ha="right", va="top", transform = ax.transAxes)
    ax = plt.subplot2grid(gridsize, (0, 4), colspan=4, rowspan=4)
    ax.set_title("RSOM / %s" % dataset, weight="bold")
    Z = griddata((som_random.positions[:,0], som_random.positions[:,1]),
                 A_random, (xi[None,:], yi[:,None]), method='cubic')
    ax.imshow(Z, interpolation="nearest", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])


    ax = plt.subplot2grid(gridsize, (4, 4), colspan=4, rowspan=1)
    ax.plot(np.sort(A_random), color="black")
    ax.text(0.01, 0.95, "Ordered activation count\n(%d samples)" % n,
            ha="left", va="top", transform = ax.transAxes)
    ax = plt.subplot2grid(gridsize, (5, 4), colspan=4, rowspan=1)
    ax.hist(D_random, bins=50, color="C0", alpha=.5)
    ax.text(0.99, 0.95, "Distortion histogram\n(%d samples)" % n,
            ha="right", va="top", transform = ax.transAxes)


    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()
    

    

