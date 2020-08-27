# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier
# Released under a BSD two-clauses license
#
# References: Kohonen, Teuvo. Self-Organization and Associative Memory.
#             Springer, Berlin, 1984.
# -----------------------------------------------------------------------------
import numpy as np
from vsom import VSOM2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 32
    som = VSOM2("regular", n*n)
    
    #T = np.random.uniform(0.0, 2.0*np.pi, 25000)
    #R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, len(T)))
    #samples = np.c_[R*np.cos(T), R*np.sin(T)]

    samples = np.random.uniform(0,1,(25000,3))
    som.learn(samples, 25000, sigma=(0.50, 0.01), lrate=(0.50, 0.01))

    plt.imshow(som.codebook.reshape(32,32,3))
    plt.show()
    

    # Draw result
    fig = plt.figure(figsize=(8,8))
    axes = fig.add_subplot(1,1,1)

    # Draw samples
    x,y = samples[:,0], samples[:,1]
    plt.scatter(x, y, s=1.0, color='b', alpha=0.1, zorder=1)
    
    # Draw network
    x,y = som.codebook[:,0].reshape(n,n), som.codebook[:,1].reshape(n,n)
    for i in range(n):
        plt.plot (x[i,:], y[i,:], 'k', alpha=0.85, lw=1.5, zorder=2)
        plt.plot (x[:,i], y[:,i], 'k', alpha=0.85, lw=1.5, zorder=2)
    plt.scatter (x, y, s=50, c='w', edgecolors='k', zorder=3)
    
    plt.axis([-1,1,-1,1])
    plt.xticks([]), plt.yticks([])
    plt.show()
