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
from spatial import blue_noise, voronoi, centroid, clipped_voronoi

from som import SOM
from plot import weights_3D

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parameters
    seed = 1
    if seed is None:
        seed = np.random.randint(0,1000)
    n = 256
    radius = np.sqrt(2/(n*np.pi))
    n_neighbour = 2

    np.random.seed(seed)
    print("Random seed: {0}".format(seed))

    # Initial Blue noise distribution
    # -------------------------------
    #  P0 : Initial distribution
    #  C0 : Connection matrix
    #  S0 : Edges

    P = blue_noise((1,1), radius=radius)
    for i in range(100): # Lloyd relaxation (100 iterations)
        V = voronoi(P, bbox=[0,1,0,1])
        C = []
        for region in V.filtered_regions:
            vertices = V.vertices[region + [region[0]], :]
            C.append(centroid(vertices))
        P = np.array(C)
    P0, V0 = P, V

    # We reorder P0 such that last points are in bottom right corner
    # It is not strictly necessary but it simplifies lesion code
    # P0 = P0[np.argsort(P0[:,0] + (1-P0[:,1]))]
    # V = voronoi(P0, bbox=[0,1,0,1])
    
    # Computing connection matrix and edges
    D0 = scipy.spatial.distance.cdist(P0, P0)
    C0 = np.zeros(D0.shape, dtype=int)
    S0 = []
    for i in range(len(P0)):
        for j in np.argsort(D0[i])[1:n_neighbour+1]:
            C0[i,j] = 1
            S0.append([P0[i], P0[j]])

    
    # Expanded distribution
    # -------------------------------
    #  P0 + SP : Initial state
    #  P1      : Final state
    n = 25
    SP = np.random.uniform(0.5, 1.00, (n,2))

       
    P = np.r_[P0, SP]
    for j in range(100): # Lloyd relaxation (100 iterations)
        V = voronoi(P, bbox=[0,1,0,1])
        for i,region in enumerate(V.filtered_regions):
            vertices = V.vertices[region + [region[0]], :]
            C = centroid(vertices)
            P[np.argmin(((P-C)**2).sum(axis=1))] = C
    P1, V1 = P, V

    # Computing connection matrix and edges
    D1 = scipy.spatial.distance.cdist(P1, P1)
    C1 = np.zeros(D1.shape, dtype=int)
    S1 = []
    for i in range(len(P1)):
        # Nodes that were already connected
        # They can connect to an old node if it is really closer
        #  or they can connect to a new node it it is closer
        if i < len(P0):
            # Because of internal tests (keeping old node or getting a new one)
            # we cannot guarantee that the index of the new node is not alreay
            # used and we thus test explicitely if we reach the right number.
            # This is also the reason to use 2*n_neighbour instead of
            # n_neighbour.
            count = 0
            for j0,j1 in zip( np.argsort(D0[i])[1:2*n_neighbour+1],
                              np.argsort(D1[i])[1:2*n_neighbour+1]):
                # This test make things works but it might be wrong It was
                # initially a bug but alternatives don't look so good. Here we
                # test if the length of the initial edge has grown by a given
                # factor. If it has grown too muc, we choose a ne closest node
                if j1 > len(P0) or D0[i,j0] < 0.85*D1[i,j0]:
                    j = j1
                else:
                    j = j0
                if C1[i,j] == 0:
                    C1[i,j] = 1
                    S1.append([P1[i], P1[j]])
                    count += 1
                if count == n_neighbour:
                    break

        # New nodes
        # These one have no neighbour yet and can thus connect to any node.
        else:
            for j in np.argsort(D1[i])[1:n_neighbour+1]:
                C1[i,j] = 1
                S1.append([P1[i], P1[j]])


    # Lesioned distribution
    # -------------------------------
    #  P0[:-n] : Initial state
    #  P2 : Final state
    n = 24
    P = P0[:-n].copy()
    for j in range(100): # Lloyd relaxation (100 iterations)
        V = voronoi(P, bbox=[0,1,0,1])
        for i,region in enumerate(V.filtered_regions):
            vertices = V.vertices[region + [region[0]], :]
            C = centroid(vertices)
            P[np.argmin(((P-C)**2).sum(axis=1))] = C
    P2, V2 = P, V

    # Computing connection matrix and edges
    D2 = scipy.spatial.distance.cdist(P2, P2)
    C2 = np.zeros(D2.shape, dtype=int)
    S2 = []
    for i in range(len(P2)):
        # Nodes that were already connected
        # They can connect to an old node if it is really closer
        #  or they can connect to a new node it it is closer
        count = 0
        for j0,j2 in zip( np.argsort(D0[i])[1:2*n_neighbour+1],
                          np.argsort(D2[i])[1:2*n_neighbour+1]):
            if j0 >= len(P2) or D0[i,j0] < 0.75*D2[i,j0]:
                j = j2
            else:
                j = j0
            if C2[i,j] == 0:
                C2[i,j] = 1
                S2.append([P2[i], P2[j]])
                count += 1
            if count == n_neighbour:
                break
                
    
    print("A: {0} neurons".format(len(P0)))
    print("B: {0} neurons".format(len(P1)))
    print("C: {0} neurons".format(len(P2)))



    # -------------------------------------------------------------------------
    n_epochs   = 25000
    sigma      = 0.50, 0.1
    lrate      = 0.50, 0.2
    X,Y = np.random.uniform(0, 1, (50000,3)), None
    som_0 = SOM(size=len(P0), topology="random", neighbours=2, PVC = (P0, V0, C0))
    som_0.voronoi = clipped_voronoi(P0, bbox=[0,1,0,1])
    som_0.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate)

    n_epochs   = 5000
    sigma      = 0.1, 0.01
    lrate      = 0.2, 0.01

    som_1 = SOM(size=len(P1), topology="random", neighbours=2, PVC = (P1, V1, C1))
    som_1.voronoi = clipped_voronoi(P1, bbox=[0,1,0,1])
    som_1.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate, codebook=som_0.codebook)
    
    som_2 = SOM(size=len(P2), topology="random", neighbours=2, PVC = (P2, V2, C2))
    som_2.voronoi = clipped_voronoi(P2, bbox=[0,1,0,1])
    som_2.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate, codebook=som_0.codebook)

    som_0.fit(X, Y, n_epochs, sigma=sigma, lrate=lrate, codebook=som_0.codebook)
        
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(15,15), dpi=50)
    nrow, ncol = 3, 3


    ax = plt.subplot(nrow, ncol,  7, aspect=1)
    weights_3D(ax, som_0)
    text = ax.text(0.05, 0.05, "G",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])

    
    ax = plt.subplot(nrow, ncol,  8, aspect=1)
    weights_3D(ax, som_1)
    text = ax.text(0.05, 0.05, "H",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])

    ax = plt.subplot(nrow, ncol,  9, aspect=1)
    weights_3D(ax, som_2)
    text = ax.text(0.05, 0.05, "I",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])

    
    # --- A ---
    ax = plt.subplot(nrow, ncol,  1, aspect=1)
    X, Y = P0[:,0], P0[:,1]
    FC = np.zeros((len(P0),4))
    FC[:] = 1,1,1,1
    FC[-n:] = 1,0,0,1
    EC = np.zeros((len(P0),4))
    EC[:] = 0,0,0,1
    EC[-n:] = 1,0,0,1

    ax.scatter(X[:10], Y[:10], s=10,
               edgecolor="None", facecolor="black", linewidth=0,zorder=100)

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

    # --- D ---
    ax = plt.subplot(nrow, ncol,  4, aspect=1)
    X, Y = P0[:,0], P0[:,1]
    ax.scatter(X, Y, s=50, edgecolor="k", facecolor="w", linewidth=1.5)
    collection = LineCollection(S0, color="black",
                                linewidth=1.5, zorder=-10, alpha=0.5)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "D",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])


    # --- B ---
    ax = plt.subplot(nrow, ncol,  2, aspect=1)
    X, Y = P1[:,0], P1[:,1]

    ax.scatter(X[:10], Y[:10], s=10,
               edgecolor="None", facecolor="black", linewidth=0,zorder=100)
    
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
    P = np.r_[P0, SP]
    ax.scatter(P[:,0], P[:,1], s=10, lw=0, zorder=-30,
               edgecolor="None", facecolor="black")
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "B",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    # --- E ---
    ax = plt.subplot(nrow, ncol,  5, aspect=1)
    X, Y = P1[:,0], P1[:,1]
    ax.scatter(X, Y, s=50, edgecolor="k", facecolor="w", linewidth=1.5)
    collection = LineCollection(S1, color="black",
                                linewidth=1.5, zorder=-10, alpha=0.5)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "E",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    # --- C ---
    ax = plt.subplot(nrow, ncol,  3, aspect=1)
    X, Y = P2[:,0], P2[:,1]

    ax.scatter(X[:10], Y[:10], s=10,
               edgecolor="None", facecolor="black", linewidth=0,zorder=100)

    
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
    P = P0[:-n]
    ax.scatter(P[:,0], P[:,1], s=10, lw=0, zorder=-30,
               edgecolor="None", facecolor="black")
    collection = LineCollection(segments, color="k", linewidth=1.5, zorder=-20)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "C",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])


    # --- F ---
    ax = plt.subplot(nrow, ncol,  6, aspect=1)
    X, Y = P2[:,0], P2[:,1]
    ax.scatter(X, Y, s=50, edgecolor="k", facecolor="w", linewidth=1.5)
    collection = LineCollection(S2, color="black",
                                linewidth=1.5, zorder=-10, alpha=0.5)
    ax.add_collection(collection)
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "F",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    
    plt.tight_layout()
    plt.savefig("topology-conservation.pdf", dpi=300)
    plt.show()
