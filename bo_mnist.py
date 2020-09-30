# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import tqdm
import som
import mnist
import numpy as np
import matplotlib.pylab as plt

from GPyOpt.methods import BayesianOptimization

# See https://doi.org/10.1016/j.patrec.2015.02.001 for reference

# Semi-automatic ground truth generation using unsupervised clustering and
# imited manual labeling: Application to handwritten character recognition
# Szilárd Vajda, Yves Rangoni, Hubert Cecotti
# Pattern Recognition Letters, Volume 58, 1 June 2015, Pages 23-28


def mnist_cost(x):
    lrate0 = x[:, 0]
    lratef = x[:, 1]
    sigma0 = x[:, 2]
    sigmaf = x[:, 3]

    seed = 123
    topology = "random"
    topology = "regular"
    n_unit = 256  # 400 #1024
    n_neighbor = 3
    n_epochs = 50000
    # sigma = 0.10, 0.01
    # lrate = 0.75, 0.01
    sigma = sigma0, sigmaf
    lrate = lrate0, lratef
    if seed is None:
        seed = np.random.randint(0, 1000)
    np.random.seed(seed)

    print("Building network (might take some time)... ", end="")
    sys.stdout.flush()
    net = som.SOM(n_unit, topology, n_neighbor)
    print("done!")
    print("Random seed: {0}".format(seed))
    print("Number of units: {0}".format(net.size))
    if type == "random":
        print("Number of neighbors: {0}".format(n_neighbor))

    X, Y = mnist.read("training")
    X, Y = X.reshape(len(X), -1), Y.reshape(len(Y), -1)
    net.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    X, Y = mnist.read("testing")
    X, Y = X.reshape(len(X), -1), Y.reshape(len(Y), -1)
    Out = np.zeros(len(X))

    for i in tqdm.trange(len(X)):
        # Get index of nearest node (minimum distance)
        winner = np.argmin(((net.codebook['X'] - X[i])**2).sum(axis=-1))
        label = np.argmax(net.codebook['Y'][winner])
        Out[i] = (label == np.argmax(Y[i]))
    # print("Rate: {0:.3f} ± {1:.3f}".format(Out.mean(), Out.std()))
    print(Out.mean())
    return -Out.mean()


if __name__ == '__main__':
    # optimal for 1024 units [0.02658881 0.00560278 0.11340053 0.05577199]
    # accuracy = 81.1%
    domain = [{'name': 'lrate0', 'type': 'continuous', 'domain': [10.0, 0.05]},
              {'name': 'lratef', 'type': 'continuous', 'domain': [0.05, 1e-4]},
              {'name': 'sigma0', 'type': 'continuous', 'domain': [0.7, 0.05]},
              {'name': 'sigmaf', 'type': 'continuous', 'domain': [0.05, 0.005]}]

    optimizer = BayesianOptimization(f=mnist_cost, domain=domain)
    optimizer.run_optimization(max_iter=15)
    optimizer.plot_acquisition()
    optimizer.plot_convergence()
    print(optimizer.x_opt)
    print(optimizer.fx_opt)
    plt.show()
