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
    seed        = 123
    radius      = 0.025 # number of neurons ~ 2/(pi*radius**2)
    n_neighbour = 2

    
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


    
    # Display neural and weight maps
    # ------------------------------
    fig = plt.figure(figsize=(12,4))

    n_neighbour = 2
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
            
    som = VSOM((len(P),3), distance)


    
    # Neuronal space
    # --------------
    ax = plt.subplot(1, 3, 1, aspect=1)
    # ax.set_title("Neural space")
    
    ax.scatter(P[:,0], P[:,1], s=15, edgecolor="k", facecolor="w", linewidth=0.75)
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)):
        segments[i] = P[edges[i,0]], P[edges[i,1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=.75)
    ax.add_collection(collection)


    source = np.argmin(((P-[(0,0)])**2).sum(axis=1))
    target = np.argmin(((P-[(1,1)])**2).sum(axis=1))
    path = nx.shortest_path(nx.Graph(C), source, target)
    print(len(path))
    P_ = P[path]
    ax.scatter(P_[:,0], P_[:,1], s=15, edgecolor="k", facecolor="k", linewidth=0.75)


    
    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color="k", linewidth=0.5,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "A",
                   fontsize=24, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    


    n_neighbour = 3
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
    som = VSOM((len(P),3), distance)



    # Neuronal space
    # --------------
    ax = plt.subplot(1, 3, 2, aspect=1)
    
    ax.scatter(P[:,0], P[:,1], s=15, edgecolor="k", facecolor="w", linewidth=0.75)
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)):
        segments[i] = P[edges[i,0]], P[edges[i,1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=.75)
    ax.add_collection(collection)

    source = np.argmin(((P-[(0,0)])**2).sum(axis=1))
    target = np.argmin(((P-[(1,1)])**2).sum(axis=1))
    path = nx.shortest_path(nx.Graph(C), source, target)
    print(len(path))
    P_ = P[path]
    ax.scatter(P_[:,0], P_[:,1], s=15, edgecolor="k", facecolor="k", linewidth=0.75)

    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color="k", linewidth=0.5,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "B",
                   fontsize=24, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    


    n_neighbour = 4
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
    som = VSOM((len(P),3), distance)


    # Neuronal space
    # --------------
    ax = plt.subplot(1, 3, 3, aspect=1)
    # ax.set_title("Neural space")
    
    ax.scatter(P[:,0], P[:,1], s=15, edgecolor="k", facecolor="w", linewidth=0.75)
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)):
        segments[i] = P[edges[i,0]], P[edges[i,1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=.75)
    ax.add_collection(collection)


    source = np.argmin(((P-[(0,0)])**2).sum(axis=1))
    target = np.argmin(((P-[(1,1)])**2).sum(axis=1))
    path = nx.shortest_path(nx.Graph(C), source, target)
    print(len(path))
    P_ = P[path]
    ax.scatter(P_[:,0], P_[:,1], s=15, edgecolor="k", facecolor="k", linewidth=0.75)

    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color="k", linewidth=0.5,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "C",
                   fontsize=24, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    
    
    plt.tight_layout()
    plt.savefig("vsom-topology-2.pdf")
    plt.show()
    
