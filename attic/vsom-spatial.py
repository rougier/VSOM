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
    n_neighbour = 3
    n_samples   = 25000
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
    som = VSOM((len(P),2), distance)

    # samples = np.random.uniform(-1, 1, (n_samples,2))
    # samples = np.random.normal(0,.35,(n_samples,2))
    T = np.random.uniform(0.0, 2.0*np.pi, n_samples)
    R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, n_samples))
    samples = np.c_[R*np.cos(T), R*np.sin(T)]
    som.learn(samples, n_epochs, sigma=sigma, lrate=lrate)


    # Display activation for 6 random points
    # --------------------------------------
    indices = np.random.randint(0,len(samples),12)[-6:]
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
        

    plt.tight_layout()
    plt.savefig("vsom-spatial-2.pdf")
    plt.show()


    # Display neural and weight maps
    # ------------------------------
    fig = plt.figure(figsize=(16,8))

    # Neuronal space
    # --------------
    ax = plt.subplot(1, 2, 1, aspect=1)
    
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

    X, Y = som.codebook[:,0], som.codebook[:,1]

    ax.scatter(X, Y, s=30, edgecolor="w", facecolor="k", linewidth=1.0)
    

    ax.scatter(samples[:,0], samples[:,1], s=5,
               edgecolor="None", facecolor="blue", alpha=0.25, zorder=-30)

    # Highlight chosen samples
    S = samples[indices]
    ax.scatter(S[:,0], S[:,1], s=1000, linewidth=0, alpha=.75,
               edgecolor="None", facecolor="white", zorder=30)
    ax.scatter(S[:,0], S[:,1], s=50, linewidth=1.5,
               edgecolor="red", facecolor="white", zorder=40)
    for i in range(len(S)):
        text = ax.text(S[i,0], S[i,1]-0.02, chr(ord("C")+i),
                       color="red", ha = "center", va = "top", zorder=100,
                       fontsize=12, fontweight="bold", transform=ax.transData)
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
                               path_effects.Normal()])


    
    
    segments = np.zeros((len(edges), 2, 2))
    for i in range(len(edges)): 
        segments[i] = som.codebook[edges[i,0]], som.codebook[edges[i,1]]
    collection = LineCollection(segments, linewidth=0.75,
                                color='black', zorder=-10, alpha=1.0)
    ax.add_collection(collection)
    ax.set_xlim(-1,1), ax.set_ylim(-1,1)
    ax.set_xticks([]), ax.set_yticks([])
    text = ax.text(0.05, 0.05, "B",
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

    plt.tight_layout()
    plt.savefig("vsom-spatial-1.pdf")
    plt.show()
    
