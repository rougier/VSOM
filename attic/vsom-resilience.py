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
    n_neighbour = 2
    radius      = 0.05 # number of neurons ~ 2/(pi*radius**2)
    
    # Initialization
    # --------------
    if seed is None:
        seed = np.random.randin(0,1000)
    np.random.seed(seed)
    print("Random seed: {0}".format(seed))
        
    
    # Nice uniform random distribution (blue noise)
    # ---------------------------------------------
    P = blue_noise((1,1), radius=radius)


    n = 25
    
    for i in range(100):
        V = voronoi(P, bbox=[0,1,0,1])
        C = []
        for region in V.filtered_regions:
            vertices = V.vertices[region + [region[0]], :]
            C.append(centroid(vertices))
        P = np.array(C)
    P0, V0 = P, V

    SP = np.random.uniform(0.50, 1.00, (n,2))
    P = np.r_[P, SP]
    for j in range(100):
        V = voronoi(P, bbox=[0,1,0,1])
        for i,region in enumerate(V.filtered_regions):
            vertices = V.vertices[region + [region[0]], :]
            C = centroid(vertices)
            P[np.argmin(((P-C)**2).sum(axis=1))] = C
    P1, V1 = P, V


    P = P0[:-n].copy()
    # P = np.delete(P0, np.random.randint(0,len(P0),n), axis=0)
    for j in range(100):
        V = voronoi(P, bbox=[0,1,0,1])
        for i,region in enumerate(V.filtered_regions):
            vertices = V.vertices[region + [region[0]], :]
            C = centroid(vertices)
            P[np.argmin(((P-C)**2).sum(axis=1))] = C
    P2, V2 = P, V


    print("A: {0} neurons".format(len(P0)))
    print("B: {0} neurons".format(len(P1)))
    print("C: {0} neurons".format(len(P2)))



    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15,10))


    # -------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 1, aspect=1)
    X, Y = P0[:,0], P0[:,1]
    FC = np.zeros((len(P0),4))
    FC[:] = 1,1,1,1
    FC[-n:] = 1,0,0,1
    EC = np.zeros((len(P0),4))
    EC[:] = 0,0,0,1
    EC[-n:] = 1,0,0,1

    ax.scatter(X, Y, s=50, edgecolor=EC, facecolor=FC, linewidth=1.5)

    ax.scatter(SP[:,0], SP[:,1],
               s=50, edgecolor="k", facecolor="k", linewidth=1.5)
    
    segments = []
    for region in V0.filtered_regions:
        segments.append(V0.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color=".75", linewidth=0.5, zorder=-20)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "A",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    # -------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 4, aspect=1)
    D = scipy.spatial.distance.cdist(P0,P0)
    sources = np.repeat(np.arange(len(P0)),n_neighbour).reshape(len(P0),n_neighbour)
    targets = np.argsort(D,axis=1)[:,1:n_neighbour+1]
    edges = np.c_[sources.ravel(), targets.ravel()]
    X, Y = P0[:,0], P0[:,1]
    ax.scatter(X, Y, s=50, edgecolor="k", facecolor="w", linewidth=1.5)
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)):
        segments[i] = P0[edges[i,0]], P0[edges[i,1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=1.5)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "D",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    
    # -------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 2, aspect=1)
    X, Y = P1[:,0], P1[:,1]
    FC = np.zeros((len(P1),4))
    FC[:] = 1,1,1,1
    FC[-n:] = 0,0,0,1
    EC = np.zeros((len(P1),4))
    EC[:] = 0,0,0,1
    EC[-n:] = 0,0,0,1
    ax.scatter(X, Y, s=50, edgecolor=EC, facecolor=FC, linewidth=1.5)
    segments = []
    for region in V1.filtered_regions:
        segments.append(V1.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color="k", linewidth=0.5,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)
    segments = []
    for p0,p1 in zip( np.r_[P0, SP], P1):
        segments.append([p0,p1])
    collection = LineCollection(segments, color="k", linewidth=1.5, zorder=-20)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "B",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])


    # -------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 5, aspect=1)
    D = scipy.spatial.distance.cdist(P1,P1)
    sources = np.repeat(np.arange(len(P1)),n_neighbour).reshape(len(P1),n_neighbour)
    targets = np.argsort(D,axis=1)[:,1:n_neighbour+1]
    edges = np.c_[sources.ravel(), targets.ravel()]
    
    X, Y = P1[:,0], P1[:,1]
    ax.scatter(X, Y, s=50, edgecolor="k", facecolor="w", linewidth=1.5)
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)):
        segments[i] = P1[edges[i,0]], P1[edges[i,1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=1.5)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "E",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])



    # -------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 3, aspect=1)
    X, Y = P2[:,0], P2[:,1]
    ax.scatter(X, Y, s=50, edgecolor="k", facecolor="w", linewidth=1.5)
    segments = []
    for region in V2.filtered_regions:
        segments.append(V2.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color="k", linewidth=0.5,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)
    segments = []
    for p0,p2 in zip(P0[:-n], P2):
        segments.append([p0,p2])
    collection = LineCollection(segments, color="k", linewidth=1.5, zorder=-20)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "C",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])


    # -------------------------------------------------------------------------
    ax = plt.subplot(2, 3, 6, aspect=1)
    D = scipy.spatial.distance.cdist(P2,P2)
    sources = np.repeat(np.arange(len(P2)),n_neighbour).reshape(len(P2),n_neighbour)
    targets = np.argsort(D,axis=1)[:,1:n_neighbour+1]
    edges = np.c_[sources.ravel(), targets.ravel()]
    X, Y = P2[:,0], P2[:,1]
    ax.scatter(X, Y, s=50, edgecolor="k", facecolor="w", linewidth=1.5)
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)):
        segments[i] = P2[edges[i,0]], P2[edges[i,1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=1.5)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "F",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    
    plt.tight_layout()
    plt.savefig("vsom-resilience.pdf")
    plt.show()
    
