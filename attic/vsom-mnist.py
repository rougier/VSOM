# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import os
import struct
import numpy as np
import scipy.spatial
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection, PolyCollection
from vsom import VSOM, blue_noise, voronoi, centroid

# Read the MNIST dataset (training or testing)
def read(dataset="training", path="."):
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
    images = images[I]
    labels = labels[I]

    return labels, images


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parameters
    # ----------
    seed        = 1
    radius      = 0.025 # number of neurons ~ 2/(pi*radius**2)
    n_neighbour = 4
    n_epochs    = 25000
    sigma       = 0.50, 0.01
    lrate       = 0.50, 0.01

    
    # Initialization
    # --------------
    if seed is None:
        seed = np.random.randin(0,1000)
    np.random.seed(seed)
    print("Random seed: {0}".format(seed))
        
    
    # Nice uniform random distribution (blue noise)
    # ---------------------------------------------
    P = blue_noise((1,1), radius=radius)
    print("Number of neurons: {0}".format(len(P)))

    
    # Centroidal Voronoi Tesselation (10 iterations)
    # ----------------------------------------------
    for i in range(10):
        V = voronoi(P, bbox=[0,1,0,1])
        C = []
        for region in V.filtered_regions:
            vertices = V.vertices[region + [region[0]], :]
            C.append(centroid(vertices))
        P = np.array(C)


    # Connectivity matrix (C) and distance matrix (D)
    # -----------------------------------------------
    D = scipy.spatial.distance.cdist(P,P)
    sources = np.repeat(np.arange(len(P)),n_neighbour).reshape(len(P),n_neighbour)
    targets = np.argsort(D,axis=1)[:,1:n_neighbour+1]
    edges = np.c_[sources.ravel(), targets.ravel()]
    C = np.zeros(D.shape, dtype=int)
    C[sources,targets] = 1
    lengths = nx.shortest_path_length(nx.Graph(C))
    distance = np.zeros(D.shape, dtype=int)
    for i in range(len(P)):
        for j in range(len(P)):
            distance[i,j] = lengths[i][j]

    
    # Train SOM
    # ---------

    labels, images = list(read("training"))
    labels = labels / 9
    rows, cols = images.shape[-2:]
    samples = images.reshape(len(images), rows*cols)
    som = VSOM((len(P),rows*cols), distance)
    som.learn(samples, n_epochs, sigma=sigma, lrate=lrate, labels=labels)


    # Display activation for 6 random points
    # --------------------------------------
    indices = np.random.randint(0,len(samples),6)
    fig = plt.figure(figsize=(12,8))
    for i in range(len(indices)):
        ax = plt.subplot(2, 3, i+1, aspect=1)
        data = samples[indices[i]]
        D = -np.sqrt(((som.codebook - data)**2).sum(axis=-1)) 
        cmap = matplotlib.cm.get_cmap('plasma')
        norm = matplotlib.colors.Normalize(vmin=D.min(), vmax=D.max())
        segments = []
        for region in V.filtered_regions:
            segments.append(V.vertices[region + [region[0]], :])
        collection = PolyCollection(segments, linewidth=1.0,
                                    edgecolors=cmap(norm(D)),
                                    facecolors=cmap(norm(D)))
        ax.add_collection(collection)
        text = ax.text(0.05, 0.05, chr(ord("C")+i),
                       fontsize=24, fontweight="bold", transform=ax.transAxes)
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                               path_effects.Normal()])
        ax.set_xlim(0,1), ax.set_ylim(0,1)
        ax.set_xticks([]), ax.set_yticks([])

        from scipy.interpolate import griddata
        X = np.linspace(0, 1, 512)
        Y = np.linspace(0, 1, 512)
        Z = griddata(P, D, (X[None,:], Y[:,None]), method='nearest')
        ax.contour(X, Y, Z, 8, linewidths=0.5, colors='k', alpha=0.75)

        image = np.zeros((rows,cols,4))
        image[:,:,0] = image[:,:,1] = image[:,:,2] = 0
        image[:,:,3] = data.reshape(rows,cols)
        image = OffsetImage(image, zoom=1.1, zorder=20, interpolation="nearest")
        # image = OffsetImage(data.reshape(rows,cols), zoom=0.5,
        #                     zorder=-20, cmap='gray_r')
        box = AnnotationBbox(image, (0.9,0.9), frameon=True)
        ax.add_artist(box)

        
    plt.tight_layout()
    plt.savefig("vsom-mnist-2.pdf")
    plt.show()


    # Display neural and weight maps
    # ------------------------------
    fig = plt.figure(figsize=(16,8))

    # Neuronal space
    # --------------
    ax = plt.subplot(1, 2, 1, aspect=1)
    
    ax.scatter(P[:,0], P[:,1], s=50, edgecolor="k", facecolor="w", linewidth=1.)
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)):
        segments[i] = P[edges[i,0]], P[edges[i,1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=1.)
    ax.add_collection(collection)

    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color="k", linewidth=0.5,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "A", zorder=20,
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    # Weight space
    # ------------
    ax = plt.subplot(1, 2, 2, aspect=1, axisbelow=False)

    cmap = matplotlib.cm.get_cmap('Set3')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=12)
    labels = np.round(10*som.labels).astype(int)
    
    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = PolyCollection(segments, linewidth=0.25, alpha=1.0,
                                edgecolors="k", # cmap(norm(som.labels)),
                                facecolors="w") # cmap(norm(labels)))
    ax.add_collection(collection)

    
    for position, data, label in zip(P, som.codebook, som.labels):
        image = np.zeros((rows,cols,4))
        image[:,:,0] = image[:,:,1] = image[:,:,2] = 0
        image[:,:,3] = data.reshape(rows,cols)
        image = OffsetImage(image, zoom=0.5, zorder=20, interpolation="nearest")
        #image = OffsetImage(data.reshape(rows,cols), zoom=0.5, zorder=-20,
        #                    interpolation="nearest", cmap='gray_r')
        box = AnnotationBbox(image, position, frameon=False)
        ax.add_artist(box)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "B", zorder=20,
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    # Redraw axis because boxes cover it (and zorder doesn't work)
    ax.plot([0,1,1,0,0],[0,0,1,1,0], c='k', lw=.75, clip_on=False, zorder=20)
    
    plt.tight_layout()
    plt.savefig("vsom-mnist-1.pdf")
    plt.show()
    
