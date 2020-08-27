# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import scipy.spatial
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection, PolyCollection
from vsom import VSOM, blue_noise, voronoi, centroid



# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parameters
    # ----------
    seed        = 1
    radius      = 0.025 # number of neurons ~ 2/(pi*radius**2)
    n_neighbour = 2
    n_samples   = 25000
    n_epochs    = 10000
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


    # Connecticity matrix (C) and distance matrix (D)
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
    som = VSOM((len(P),1), distance)
    samples = np.random.random((n_samples,1))
    som.learn(samples, n_epochs, sigma=sigma, lrate=lrate)



    # Display activation for 6 primary colors
    # ---------------------------------------
    fig = plt.figure(figsize=(12,8))

    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for i in range(len(values)):
        ax = plt.subplot(2, 3, i+1, aspect=1)
        data = values[i]
        D = -np.sqrt(((som.codebook - data)**2).sum(axis=-1))
        cmap = matplotlib.cm.get_cmap('plasma')
        norm = matplotlib.colors.Normalize(vmin=D.min(), vmax=D.max())
        # norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
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
        
        ax.set_xticks([]), ax.set_yticks([])

        from scipy.interpolate import griddata
        X = np.linspace(0, 1, 512)
        Y = np.linspace(0, 1, 512)
        Z = griddata(P, D, (X[None,:], Y[:,None]), method='nearest')
        ax.contour(X, Y, Z, 8, linewidths=0.5, colors='k', alpha=0.75)

        
    plt.tight_layout()
    plt.savefig("vsom-scalar-2.pdf")
    plt.show()

    
    # Display neural and weight maps
    # ------------------------------
    fig = plt.figure(figsize=(16,8))

    # Neuronal space
    # --------------
    ax = plt.subplot(1, 2, 1, aspect=1)
    # ax.set_title("Neural space")
    
    ax.scatter(P[:,0], P[:,1], s=50, edgecolor="k", facecolor="w", linewidth=1.5)
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)):
        segments[i] = P[edges[i,0]], P[edges[i,1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=1.5)
    ax.add_collection(collection)

    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color="k", linewidth=0.5,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "A",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])


    # Weight space
    # ------------
    ax = plt.subplot(1, 2, 2, aspect=1)
    # ax.set_title("Weight space")
    cmap = matplotlib.cm.get_cmap('plasma')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = PolyCollection(segments, linewidth=1.0,
                                edgecolor=cmap(norm(som.codebook.ravel())),
                                facecolors=cmap(norm(som.codebook.ravel())))
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "B",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    plt.tight_layout()
    plt.savefig("vsom-scalar-1.pdf")
    plt.show()
    
