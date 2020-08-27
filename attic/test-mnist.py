# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import os, sys, struct
import numpy as np
from vsom import VSOM2
import matplotlib.pyplot as plt

# Read the MNIST dataset (training or testing)
def mnist(dataset="training", path="."):
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

    with open(filename_img, 'rb') as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8)
        images = images.reshape(count, rows, cols)
        images = (images-images.min())/(images.max()-images.min())

    I = np.argsort(labels)
    return  images[I], labels[I]


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    seed       = 1
    type       = "random"
    n_unit     = 1024
    n_samples  = 50000
    n_neighbor = 3
    n_epochs   = 50000
    sigma      = 0.50, 0.001
    lrate      = 0.50, 0.001

    name             = "experiment-1"
    filename_figure1 = name + "-result-1.pdf"
    filename_figure2 = name + "-result-2.pdf"
    
    if seed is None: seed = np.random.randint(0,1000)
    np.random.seed(seed)

    
    print("Building network (might take some time)... ", end="")
    sys.stdout.flush()
    som = VSOM2(type, n_unit, n_neighbor)
    print("done!")
    print("Random seed: {0}".format(seed))
    print("Number of units: {0}".format(len(som)))
    if type == "random":
        print("Number of neighbors: {0}".format(n_neighbor))


    # samples = np.random.uniform(0, 1, (50000,1))
    # samples = np.random.uniform(0, 1, (50000,3))
    samples, labels = mnist("training")
    # T = np.random.uniform(0.0, 2.0*np.pi, n_samples)
    # R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, n_samples))
    # samples = np.c_[R*np.cos(T), R*np.sin(T)]
    
    som.learn(samples, n_epochs, sigma, lrate, labels)
    samples, labels = mnist("testing")
    som.test(samples, labels)
    
    
    # Figure 1
    fig = plt.figure(figsize=(16,8))
    ax = plt.subplot(1, 2, 1, aspect=1)
    som.plot_network(ax)
    som.plot_letter(ax, "A")
    ax = plt.subplot(1, 2, 2, aspect=1)
    som.plot_weights(ax, "gray", samples)
    som.plot_letter(ax, "B")
    plt.tight_layout()
    plt.show()

    
    # Figure 2
    # samples = [ (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0),
    #             (1.0, 0.0, 0.0), (0.0, 1.0 ,0.0), (0.0, 0.0, 1.0) ]
    # samples = samples[np.random.randint(0, len(samples), 6)]
    # fig = plt.figure(figsize=(12,8))
    # for i,sample in enumerate(samples):
    #     ax = plt.subplot(2, 3, i+1, aspect=1)
    #     som.plot_activation(ax, np.array(sample))
    #     som.plot_letter(ax, chr(ord("C")+i))
    # plt.tight_layout()
    # plt.show()

