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

def gaussian(shape=(16,16), center=(0,0), sigma=(1,1), theta=0):
    A = 1
    x0, y0 = center
    sigma_x, sigma_y = sigma
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    X,Y = np.meshgrid(np.arange(-5,+5,10./shape[0]),np.arange(-5,+5,10./shape[1]))
    return A*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parameters
    # ----------
    seed        = 1
    radius      = 0.035 # number of neurons ~ 2/(pi*radius**2)
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
    rows, cols = 16,16
    samples = np.zeros((n_samples,rows*cols))
    T = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=n_samples)
    S = np.random.uniform(low=0.5, high=2.0, size=n_samples)
    
    for i in range(n_samples):
        samples[i] = gaussian(shape=(rows,cols),
                              sigma=(S[i],2), theta=T[i]).ravel()
    
    som = VSOM((len(P),rows*cols), distance)
    som.learn(samples, n_epochs, sigma=sigma, lrate=lrate)


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
        image = OffsetImage(image, zoom=2.0, zorder=20, interpolation="nearest")
        # image = OffsetImage(data.reshape(rows,cols), zoom=0.5,
        #                     zorder=-20, cmap='gray_r')
        box = AnnotationBbox(image, (0.9,0.9), frameon=True)
        ax.add_artist(box)

        
    plt.tight_layout()
    plt.savefig("vsom-gaussian-2.pdf")
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

    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = PolyCollection(segments, linewidth=0.5, alpha=0.25,
                                edgecolors="k", facecolors="None")
    ax.add_collection(collection)

    
    for position, data, label in zip(P, som.codebook, som.labels):
        image = np.zeros((rows,cols,4))
        image[:,:,0] = image[:,:,1] = image[:,:,2] = 0
        image[:,:,3] = data.reshape(rows,cols)
        image = OffsetImage(image, zoom=1.0, zorder=20, interpolation="nearest")
        # image = OffsetImage(data.reshape(rows,cols), zoom=0.5,
        #                     zorder=-20, cmap='gray_r')
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
    plt.savefig("vsom-gaussian-1.pdf")
    plt.show()
    
