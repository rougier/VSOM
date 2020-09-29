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
from spatial import blue_noise, voronoi, centroid



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
    print("Number of neurons: {0} ({1})".format(len(P), n))
    for i in range(100): # Lloyd relaxation (100 iterations)
        V = voronoi(P, bbox=[0,1,0,1])
        C = []
        for region in V.filtered_regions:
            vertices = V.vertices[region + [region[0]], :]
            C.append(centroid(vertices))
        P = np.array(C)
    P0, V0 = P, V

    # Computing connection matrix and edges
    D0 = scipy.spatial.distance.cdist(P0, P0)
    C0 = np.zeros(D0.shape, dtype=int)
    S0 = []
    for i in range(len(P0)):
        for j in np.argsort(D0[i])[1:n_neighbour+1]:
            C0[i,j] = 1
            S0.append([P0[i], P0[j]])

    
    fig = plt.figure( figsize=(18,6))
    ax = plt.subplot(1,3,1, aspect=1)
    ax.scatter(P0[:,0], P0[:,1], facecolor="white", edgecolor="black")
    collection = LineCollection(S0, color="k",
                                linewidth=1.5, zorder=-20, alpha=0.5)
    ax.add_collection(collection)

    
            
    # Expanded distribution
    # -------------------------------
    #  P0 + SP : Initial state
    #  P1      : Final state
    n = 25
    SP = np.random.uniform(0.50, 1.00, (n,2))
    P = np.r_[P, SP]
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


                
    ax = plt.subplot(1,3,2, aspect=1)
    ax.scatter(P1[:len(P0),0], P1[:len(P0),1],
               facecolor="white", edgecolor="black")
    ax.scatter(P1[len(P0):,0], P1[len(P0):,1],
               facecolor="black", edgecolor="black")
    collection = LineCollection(S1, color="k",
                                linewidth=1.5, zorder=-20, alpha=0.5)
    ax.add_collection(collection)



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
                
    ax = plt.subplot(1,3,3, aspect=1)
    ax.scatter(P2[:,0], P2[:,1], facecolor="white", edgecolor="black")
    collection = LineCollection(S2, color="k",
                                linewidth=1.5, zorder=-20, alpha=0.5)
    ax.add_collection(collection)
    plt.show()


    
    # fig = plt.figure( figsize=(8,8))
    # ax = plt.subplot(1,1,1, aspect=1)
    # ax.scatter(P1[:,0], P1[:,1], color="C0")
    # # ax.scatter(SP[:,0], SP[:,1], color="C1")
    # segments = []
    # for p0,p1 in zip( np.r_[P0, SP], P1):
    #     segments.append([p0,p1])
    # collection = LineCollection(segments, color="k", linewidth=1.5, zorder=-20)
    # ax.add_collection(collection)
    # # ax = plt.subplot(1,2,2, aspect=1)
    # # ax.scatter(P1[:len(P0),0], P1[:len(P0),1], color="C0")
    # # ax.scatter(P1[len(P0):,0], P1[len(P0):,1], color="C1")
    # plt.show()


    
    """
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
    """    
