# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import som, mnist, plot
import numpy as np
import matplotlib.pyplot as plt

def gaussian(shape=(16,16), center=(0,0), sigma=(1,1), theta=0):
    A = 1
    x0, y0 = center
    sigma_x, sigma_y = sigma
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    X,Y = np.meshgrid(np.arange(-5,+5,10./shape[0]),np.arange(-5,+5,10./shape[1]))
    return A*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))


if __name__ == '__main__':

    seed       = 1
    topology   = "random"
    n_unit     = 512
    n_neighbor = 3
    n_samples  = 25000
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


    rows, cols = 16,16
    xshape = rows, cols
    X = np.zeros((n_samples,rows*cols))
    Y = None
    T = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=n_samples)
    S = np.random.uniform(low=0.5, high=2.0, size=n_samples)
    
    for i in range(n_samples):
        X[i] = gaussian(shape=(rows,cols),
                        sigma=(S[i],2), theta=T[i]).ravel()
        
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)
    
    
    figsize = 2.5*np.array([6,7])
    fig = plt.figure(figsize=figsize, dpi=50)
    
    ax = plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=3, aspect=1)
    plot.network(ax, som)
    plot.letter(ax, "A")
    ax = plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=3, aspect=1)
    plot.weights_img(ax, som, xshape, zoom=1.0)
    plot.letter(ax, "B")

    # Collect minimal/maximal response from the map across all stimuli
    # vmin, vmax = None, None
    # for x in X:
    #     D = -np.sqrt(((som.codebook["X"] - x.ravel())**2).sum(axis=-1))
    #     vmin = D.min() if vmin is None else min(D.min(), vmin)
    #     vmax = D.max() if vmax is None else max(D.max(), vmax)

    X = X[np.random.randint(0,len(X),6)]
    for i,x in enumerate(X):
        ax = plt.subplot2grid((7, 6), (3+2*(i//3), 2*(i%3)),
                              colspan=2, rowspan=2, aspect=1)
        plot.activation(ax, som, np.array(x).reshape(xshape), zoom=2)
        plot.letter(ax, chr(ord("C")+i))
    plt.tight_layout()
    plt.savefig("experiment-Gaussians.pdf", dpi=300)
    plt.show()

