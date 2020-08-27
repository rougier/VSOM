# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import tqdm
import numpy as np
import scipy.spatial
import os, struct
import networkx as nx
from vsom import blue_noise


# Read the MNIST dataset (training or testing)
def mnist(dataset="training", path="MNIST"):
    if dataset == "training":
        filename_img = os.path.join(path, 'train-images-idx3-ubyte')
        filename_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        filename_img = os.path.join(path, 't10k-images-idx3-ubyte')
        filename_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(filename_lbl, 'rb') as file:
        magic, count = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)
        Y = np.zeros((len(labels), 10))
        for i in range(len(labels)):
            Y[i,labels[i]] = 1
        labels = Y

    with open(filename_img, 'rb') as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8)
        images = images.reshape(count, rows, cols)
        images = (images-images.min())/(images.max()-images.min())

    return images, labels


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    seed       = 12345
    sigma      = 0.25, 0.01
    lrate      = 0.50, 0.01

    if seed is None: seed = np.random.randint(0,1000)
    np.random.seed(seed)

    n = 16

    # Regular
    # X, Y = np.meshgrid(np.linspace(0, 1, n, endpoint=True),
    #                    np.linspace(0, 1, n, endpoint=True))
    # P = np.c_[X.ravel(), Y.ravel()]
    # D = scipy.spatial.distance.cdist(P,P)
    # D /= D.max()

    # Random
    n_neighbour = 3
    radius = np.sqrt(2/(n*n*np.pi))
    P = blue_noise((1,1), radius=radius)
    D = scipy.spatial.distance.cdist(P,P)
    sources = np.repeat(np.arange(len(P)), n_neighbour)
    sources = sources.reshape(len(P), n_neighbour)
    targets = np.argsort(D,axis=1)[:, 1:n_neighbour+1]
    C = np.zeros(D.shape, dtype=int)
    C[sources,targets] = 1
    lengths = nx.floyd_warshall_numpy(nx.Graph(C))
    D = np.array(lengths).astype(int)
    D = D/D.max()
    
    X, Y = mnist("training")
    X = X.reshape(len(X), -1)
    codebook = np.zeros((len(P), X.shape[-1]))
    labels   = np.zeros((len(P), 10))

    # Train
    n = 50000
    t = np.linspace(0, 1, n)
    lrate = lrate[0]*(lrate[1]/lrate[0])**t
    sigma = sigma[0]*(sigma[1]/sigma[0])**t
        
    I = np.random.randint(0, len(X), n)
    X, Y = X[I], Y[I]

    for i in tqdm.trange(n):
        winner = np.argmin(((codebook - X[i])**2).sum(axis=-1))
        G = np.exp(-D[winner]**2/sigma[i]**2)
        codebook -= lrate[i]*G[...,np.newaxis]*(codebook - X[i])
        labels -= lrate[i]*G[...,np.newaxis]*(labels-Y[i])
        # labels[winner] = y

    # Test
    X, Y = mnist("testing")
    

    # X = X.reshape(len(X), -1)
    # s = 0
    # for i in tqdm.trange(len(X)):
    #     # Get index of nearest node (minimum distance)
    #     winner = np.argmin(((codebook - X[i])**2).sum(axis=-1))
    #     label = np.argmax(labels[winner])
    #     s += (label == np.argmax(Y[i]))
    # print("Recognition rate: {:.1f}%".format(100*s/len(X)))

        



