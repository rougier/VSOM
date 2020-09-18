# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import tqdm
import som, mnist, plot
import numpy as np
import matplotlib.pyplot as plt

# See https://doi.org/10.1016/j.patrec.2015.02.001 for reference

# Semi-automatic ground truth generation using unsupervised clustering and
# imited manual labeling: Application to handwritten character recognition
# Szilárd Vajda, Yves Rangoni, Hubert Cecotti
# Pattern Recognition Letters, Volume 58, 1 June 2015, Pages 23-28

if __name__ == '__main__':

    seed       = 1
    topology   = "random"
    # topology   = "regular"
    n_unit     = 256 # 400 #1024
    n_neighbor = 3
    n_epochs   = 50000
    sigma      = 0.10, 0.01
    lrate      = 0.25, 0.01
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

    X, Y = mnist.read("training")
    xshape, yshape = X.shape[1:], Y.shape[1:]
    X, Y = X.reshape(len(X),-1), Y.reshape(len(Y),-1)    
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)


    X, Y = mnist.read("testing")
    xshape, yshape = X.shape[1:], Y.shape[1:]
    X, Y = X.reshape(len(X),-1), Y.reshape(len(Y),-1)    
    O = np.zeros(len(X))
    
    for i in tqdm.trange(len(X)):
        # Get index of nearest node (minimum distance)
        winner = np.argmin(((som.codebook['X'] - X[i])**2).sum(axis=-1))
        label = np.argmax(som.codebook['Y'][winner])
        O[i] = (label == np.argmax(Y[i]))

    print("Recognition rate: {0:.3f} ± {1:.3f}".format(O.mean(), O.std()))
