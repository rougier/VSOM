# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import som, mnist, plot
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    seed       = None # 1
    topology   = "random"
    topology   = "regular"
    n_unit     = 1024
    n_samples  = 25000
    n_neighbor = 2
    n_epochs   = 50000
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

    X, Y = mnist.read("training")
    xshape, yshape = X.shape[1:], Y.shape[1:]
    X, Y = X.reshape(len(X),-1), Y.reshape(len(Y),-1)    
    som.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)
    
    X, Y = mnist.read("testing")
    xshape, yshape = X.shape[1:], Y.shape[1:]
    X, Y = X.reshape(len(X),-1), Y.reshape(len(Y),-1)    
    
    I = som.predict(X)
    Y_ = som.codebook['Y'][I]

    print( (np.argmax(Y,axis=-1) == np.argmax(Y_,axis=-1)).sum())
   
    # print((np.abs(Y-Y_)<.1).sum())
    
    # print( (np.sqrt((Y-Y_)**2)).sum() / len(X))

    
