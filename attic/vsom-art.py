# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import tqdm
import numpy as np
import scipy.spatial
import networkx as nx
from vsom import VSOM, blue_noise, voronoi, centroid



# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parameters
    # ----------
    seed        = 1
    radius      = 0.0125 # number of neurons ~ 2/(pi*radius**2)
    n_neighbour = 2
    n_samples   = 25000
    n_epochs    = 25000
    sigma       = 0.25, 0.01
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
    print("Generating neuron positions... ", end="", flush=True)
    for i in range(10):
        V = voronoi(P, bbox=[0,1,0,1])
        C = []
        for region in V.filtered_regions:
            vertices = V.vertices[region + [region[0]], :]
            C.append(centroid(vertices))
        P = np.array(C)
    print("done.")

    # Connecticity matrix (C) and distance matrix (D)
    # -----------------------------------------------
    print("Compute distances between neurons... ", end="", flush=True)
    D = scipy.spatial.distance.cdist(P,P)
    print("done.")
    print("Generating topology... ", end="", flush=True)
    sources = np.repeat(np.arange(len(P)),n_neighbour).reshape(len(P),n_neighbour)
    targets = np.argsort(D,axis=1)[:,1:n_neighbour+1]
    edges = np.c_[sources.ravel(), targets.ravel()]
    C = np.zeros(D.shape, dtype=int)
    C[sources,targets] = 1
    print("done.")
    print("Computing distances... ", end="", flush=True)
    lengths = dict(nx.shortest_path_length(nx.Graph(C)))
    distance = np.zeros(D.shape, dtype=int)
    for i in range(len(P)):
        for j in range(len(P)):
            distance[i,j] = lengths[i][j]
    print("done.")
    
    # Train SOM
    # ---------
    som = VSOM((len(P),3), distance)
    samples = np.random.random((n_samples,3))
    som.learn(samples, n_epochs, sigma=sigma, lrate=lrate)



    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    
    # Display neural and weight maps
    # ------------------------------
    fig = plt.figure(figsize=(8,8))

    
    # Weight space
    # ------------
    ax = plt.axes([0,0,1,1], frameon=False, aspect=1)
    # plt.subplot(1, 1, 1, aspect=1)
    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = PolyCollection(segments, linewidth=0,
                                antialiaseds = False,
                                edgecolors=som.codebook,
                                facecolors=som.codebook)
    collection.set_joinstyle("miter")
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])


    plt.tight_layout()
    plt.savefig("vsom-art.pdf", dpi=600)
    plt.savefig("vsom-art.png", dpi=600)
    plt.show()
    
